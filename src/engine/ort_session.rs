//! Инференс ONNX-модели BS-Roformer-SW через ort (2.0.0-rc.14, load-dynamic).
//!
//! Дизайн: CUDA EP → CPU fallback **не выполняется молча**. CUDA-сессия
//! строится с `with_disable_cpu_fallback`, и любая ошибка провайдера/графа
//! возвращает `Err` — решение о фолбэке (и логирование) остаётся за вызывающим.
//!
//! Загрузка: `load-dynamic` требует `ORT_DYLIB_PATH` — проверяется до любого
//! вызова ort, иначе ort паникует на первом обращении к API (нет dylib).
//!
//! Окружение: OrtEnv в процессе ровно один — [`ORT_ENV`] создаёт его при
//! первом `OrtSession::new` и переиспользует (ошибка init тоже кэшируется).
//!
//! Потоки: `Session::run` в ort требует `&mut self`, поэтому сессия обёрнута
//! в `Mutex` (Session: Send + Sync) — [`OrtSession`] остаётся `Send + Sync`
//! и отдаёт `run(&self)`.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::engine::{NUM_STEMS, SPEC_BINS, SPEC_FRAMES};
use crate::error::AppError;

/// Имена входов графа (контракт elicwhite/bs-roformer-sw-6stem-onnx).
const INPUT_REAL: &str = "spec_real";
/// Имена входов графа (контракт elicwhite/bs-roformer-sw-6stem-onnx).
const INPUT_IMAG: &str = "spec_imag";
/// Имена выходов графа.
const OUTPUT_REAL: &str = "out_spec_real";
/// Имена выходов графа.
const OUTPUT_IMAG: &str = "out_spec_imag";

/// Форма входного спектра [1, 2, SPEC_BINS, SPEC_FRAMES].
const SHAPE_IN: [usize; 4] = [1, 2, SPEC_BINS, SPEC_FRAMES];
/// Элементов во входном спектре (реальная или мнимая часть).
const IN_LEN: usize = 2 * SPEC_BINS * SPEC_FRAMES;
/// Элементов в выходном спектре одного знака (6 стемов).
const OUT_LEN: usize = NUM_STEMS * IN_LEN;

/// Выбор execution provider для сессии.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ep {
    /// CUDA EP на указанном устройстве; при недоступности — ошибка, не фолбэк.
    Cuda { device_id: i32 },
    /// CPU (дефолтные провайдеры ORT).
    Cpu,
}

/// Сессия ONNX-инференса BS-Roformer-SW.
pub struct OrtSession {
    // ort::Session::run принимает &mut self — Mutex даёт run(&self) наружу.
    session: Mutex<ort::session::Session>,
}

use ort::environment::Environment;

/// Процесс-wide singleton окружения ort. ONNX Runtime допускает ровно один
/// OrtEnv на процесс — повторный `ort::init().build()` отвечает
/// «only one environment is allowed per process» (внутри ort окружение
/// учтено через `Weak` в `OnceLock`: даже после дропа всех хендлов второй
/// `build()` не пройдёт). Поэтому создаём один раз и переиспользуем:
/// это позволяет строить несколько сессий (явный CUDA→CPU fallback
/// в [`crate::engine::native::default_backend_factory`]).
/// Ошибка инициализации кэшируется тоже — повторные попытки получают ту же
/// ошибку, а не «second init».
static ORT_ENV: OnceLock<Result<Environment, String>> = OnceLock::new();

/// Единое окружение ort для всего процесса (аргумент `Session::builder`).
fn ort_env() -> Result<&'static Environment, AppError> {
    ORT_ENV
        .get_or_init(|| ort::init().build().map_err(|e| format!("ort init: {e}")))
        .as_ref()
        .map_err(|msg| AppError::Dsp(msg.clone()))
}

impl OrtSession {
    /// Строит сессию из ONNX-файла с выбранным EP.
    ///
    /// # Errors
    /// [`AppError::Dsp`]: `ORT_DYLIB_PATH` не задан/файл не найден, сбой
    /// инициализации ort, сбой регистрации EP или загрузки модели
    /// (для CUDA — ошибка НЕ маскируется фолбэком на CPU).
    pub fn new(model_path: &Path, ep: Ep) -> Result<Self, AppError> {
        // До ort::init: без dylib ort паникует лениво при первом вызове API.
        let dylib = dylib_path(std::env::var("ORT_DYLIB_PATH").ok().as_deref())?;
        if !dylib.is_file() {
            return Err(AppError::Dsp(format!(
                "ORT_DYLIB_PATH указывает не на файл: {}",
                dylib.display()
            )));
        }

        let env = ort_env()?;

        let builder = match ep {
            Ep::Cuda { device_id } => {
                // SameAsRequested: арена без удвоений (пик VRAM предсказуем);
                // Heuristic: cudnn без бенчмарка всех алгоритмов (лишний VRAM).
                let cuda = ort::ep::CUDA::default()
                    .with_device_id(device_id)
                    .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                    .build()
                    .error_on_failure();
                let builder = ort::session::Session::builder(env).map_err(ort_err)?;
                builder
                    .with_execution_providers([cuda])
                    .map_err(ort_err)?
                    // Запрет молчаливого отката CUDA→CPU внутри сессии:
                    // решение о фолбэке — за default_backend_factory.
                    .with_disable_cpu_fallback()
                    .map_err(ort_err)?
            }
            // CPU: EP не регистрируются. disable_cpu_fallback здесь нельзя:
            // без другого EP ORT отвечает «fallback to CPU EP has been
            // explicitly disabled» — commit CPU-сессии всегда падал бы.
            Ep::Cpu => ort::session::Session::builder(env).map_err(ort_err)?,
        };

        // error_on_failure() возвращает dispatch без Result: реальная ошибка
        // регистрации EP (нет CUDA и т.п.) проявится здесь, на commit.
        let session = builder
            .with_memory_pattern(false)
            .map_err(ort_err)?
            .commit_from_file(model_path)
            .map_err(|e| {
                AppError::Dsp(format!(
                    "ort session commit ({}): {e}",
                    model_path.display()
                ))
            })?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Один инференс: спектры входа → спектры выхода (все f32, плоские).
    ///
    /// Длины входов — ровно `2·1025·345`; выходы — `6·2·1025·345`.
    ///
    /// # Errors
    /// [`AppError::Dsp`]: неверная длина входа или ошибка ort на run/extract.
    pub fn run(
        &self,
        spec_real: &[f32],
        spec_imag: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), AppError> {
        validate_inputs(spec_real, spec_imag)?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| AppError::Dsp(format!("ort session lock poisoned: {e}")))?;

        // from_array требует owned-данные — копия неизбежна (контракт ort).
        let t_real = ort::value::Tensor::from_array((SHAPE_IN, spec_real.to_vec()))
            .map_err(ort_err)?;
        let t_imag = ort::value::Tensor::from_array((SHAPE_IN, spec_imag.to_vec()))
            .map_err(ort_err)?;

        let outputs = session
            .run(ort::inputs![INPUT_REAL => t_real, INPUT_IMAG => t_imag])
            .map_err(ort_err)?;

        let (_, data_real) = outputs[OUTPUT_REAL]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        if data_real.len() != OUT_LEN {
            return Err(AppError::Dsp(format!(
                "{OUTPUT_REAL}: len {} != {OUT_LEN}",
                data_real.len()
            )));
        }
        let (_, data_imag) = outputs[OUTPUT_IMAG]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        if data_imag.len() != OUT_LEN {
            return Err(AppError::Dsp(format!(
                "{OUTPUT_IMAG}: len {} != {OUT_LEN}",
                data_imag.len()
            )));
        }
        Ok((data_real.to_vec(), data_imag.to_vec()))
    }
}

/// `ort::Error` содержит !Send-варианты — в AppError идёт строкой,
/// чтобы AppError оставался Send. R — тип-контекст ошибки (builder/session),
/// на сообщение не влияет.
fn ort_err<R>(e: ort::Error<R>) -> AppError {
    AppError::Dsp(format!("ort: {e}"))
}

/// Чистая проверка `ORT_DYLIB_PATH`: None/пустая строка — ошибка
/// (ort при пустом значении тихо ищет дефолтное имя и паникует позже).
fn dylib_path(env: Option<&str>) -> Result<PathBuf, AppError> {
    match env {
        Some(path) if !path.is_empty() => Ok(PathBuf::from(path)),
        _ => Err(AppError::Dsp(
            "ORT_DYLIB_PATH не задан: укажи путь к libonnxruntime.so \
             (сборка ort с load-dynamic требует dylib ONNX Runtime ≥ 1.31)"
                .to_string(),
        )),
    }
}

/// Чистая валидация длин входных спектров.
fn validate_inputs(spec_real: &[f32], spec_imag: &[f32]) -> Result<(), AppError> {
    if spec_real.len() != IN_LEN {
        return Err(AppError::Dsp(format!(
            "{INPUT_REAL}: len {} != {IN_LEN}",
            spec_real.len()
        )));
    }
    if spec_imag.len() != IN_LEN {
        return Err(AppError::Dsp(format!(
            "{INPUT_IMAG}: len {} != {IN_LEN}",
            spec_imag.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dylib_path_none_is_dsp_error_with_hint() {
        let err = dylib_path(None).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
        assert!(err.to_string().contains("ORT_DYLIB_PATH"));
    }

    #[test]
    fn dylib_path_empty_string_is_dsp_error() {
        // ort трактует пустую строку как «не задано» — повторяем поведение.
        let err = dylib_path(Some("")).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
    }

    #[test]
    fn dylib_path_some_value_is_kept() {
        let path = dylib_path(Some("/opt/ort/libonnxruntime.so")).unwrap();
        assert_eq!(path, PathBuf::from("/opt/ort/libonnxruntime.so"));
    }

    #[test]
    fn validate_accepts_contract_lengths() {
        let ok = vec![0.0f32; IN_LEN];
        assert!(validate_inputs(&ok, &ok).is_ok());
    }

    #[test]
    fn validate_rejects_short_real() {
        let ok = vec![0.0f32; IN_LEN];
        let bad = vec![0.0f32; IN_LEN - 1];
        let err = validate_inputs(&bad, &ok).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
        assert!(err.to_string().contains(INPUT_REAL));
    }

    #[test]
    fn validate_rejects_long_imag() {
        let ok = vec![0.0f32; IN_LEN];
        let bad = vec![0.0f32; IN_LEN + 1];
        let err = validate_inputs(&ok, &bad).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
        assert!(err.to_string().contains(INPUT_IMAG));
    }
}
