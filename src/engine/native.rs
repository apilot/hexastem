//! Нативный движок разделения: трейт [`EngineOps`] поверх DSP-конвейера
//! крейта (`wav` → `resample` → `chunker` → `stft` → инференс → iSTFT →
//! кроссфейд → FLAC 24-bit), без python-subprocess.
//!
//! Эталон семантики — python/separator_driver.py: события протокола
//! (`progress`/`device`/`file_done`/`file_error`/`batch_done`), ретрай
//! per-file 1+1, «model hot» (одна сессия ONNX на батч), раскладка стемов
//! `<output_dir>/<file_stem>/<stem>.flac`.
//!
//! # DI (SOLID)
//!
//! Инференс изолирован трейтом [`InferenceBackend`] ([`OrtSession`] —
//! продакшн, моки — тесты); трейт-объект создаётся фабрикой
//! [`BackendFactory`] ([`default_backend_factory`] — CUDA→CPU fallback,
//! [`NativeEngine::with_backend_factory`] — инжекция в тестах). Весь
//! конвейер, кроме самого инференса, — чистые функции, проверяемые
//! юнит-тестами без GPU.
//!
//! # Прогресс и отмена
//!
//! Прогресс честный, по чанкам: `pct = (files_done + chunks_done/total)
//! / total_files` (`est: false`). Между чанками (пока идёт инференс в
//! воркер-потоке) главный поток тикает оценку `est: true` по скользящему
//! среднему времени чанка (кап 95%, как у драйвера). `on_event` остаётся
//! в вызывающем потоке (он `&mut dyn FnMut`, не `Send`).
//!
//! [`NativeEngine::kill_current_engine`] взводит [`AtomicBool`]; проверка —
//! между чанками (мягкая отмена, флаг-«защёлка»): текущий файл получает
//! `file_error` «cancelled», ретрай не выполняется, оставшиеся файлы
//! получают `file_error` сразу, батч завершается `batch_done`. Экземпляр
//! движка — на один прогон (как `EngineRunner` у TUI), сброс флага не нужен.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::config;
use flacenc::error::{Verify, Verified};
use flacenc::source::MemSource;
use flacenc::encode_with_fixed_block_size;

use crate::engine::chunker::{chunk_signal, merge_chunks};
use crate::engine::ort_session::{Ep, OrtSession};
use crate::engine::resample::resample_stereo;
use crate::engine::stft::Stft;
use crate::engine::wav::read_stereo_f32;
use crate::engine::{CHUNK_SAMPLES, MODEL_SAMPLE_RATE, NUM_STEMS, SPEC_BINS, SPEC_FRAMES, STEM_NAMES};
use crate::error::AppError;
use crate::pipeline::{EngineEvent, EngineOps, StderrSink};

/// Период est-тиков прогресса между чанками (паритет с драйвером).
const EST_TICK: Duration = Duration::from_millis(1000);
/// Оценка не заявляет завершение (как `EST_CAP_PCT` драйвера).
const EST_CAP_PCT: f64 = 95.0;
/// Попыток на файл: 1 проба + 1 ретрай (паритет с драйвером).
const ATTEMPTS_PER_FILE: u32 = 2;
/// Текст мягкой отмены (виден в отчёте пайплайна).
const CANCELLED: &str = "native engine cancelled by user";

// --- бэкенд инференса (DIP) ------------------------------------------------------

/// Инференс чанка: спектры `[1,2,1025,345]` → спектры `[1,6,2,1025,345]`
/// (плоские, freq-major, f32). `Send + Sync` — вызывается из воркер-потока.
pub trait InferenceBackend: Send + Sync {
    /// Один прогон графа: (spec_real, spec_imag) → (out_spec_real, out_spec_imag).
    ///
    /// # Errors
    /// [`AppError::Dsp`]: неверные длины входа или сбой инференса.
    fn run(&self, spec_real: &[f32], spec_imag: &[f32]) -> Result<(Vec<f32>, Vec<f32>), AppError>;
}

impl InferenceBackend for OrtSession {
    fn run(&self, spec_real: &[f32], spec_imag: &[f32]) -> Result<(Vec<f32>, Vec<f32>), AppError> {
        OrtSession::run(self, spec_real, spec_imag)
    }
}

/// blanket: `Arc<B>` несёт трейт своего содержимого (и `Arc<OrtSession>`,
/// и `Arc<dyn InferenceBackend>` идут в конвейер без распаковки).
impl<B: InferenceBackend + ?Sized> InferenceBackend for Arc<B> {
    fn run(&self, spec_real: &[f32], spec_imag: &[f32]) -> Result<(Vec<f32>, Vec<f32>), AppError> {
        (**self).run(spec_real, spec_imag)
    }
}

/// Собранный бэкенд: сессия + имя устройства для события `Device`.
pub struct BuiltBackend {
    /// Сессия инференса («model hot» на весь батч).
    pub backend: Arc<dyn InferenceBackend>,
    /// `cuda` | `cpu` | имя устройства — для события `Device` (шапка TUI).
    pub device: String,
}

/// Фабрика бэкенда (вызывается один раз на батч).
pub type BackendFactory = Arc<dyn Fn(&Path) -> Result<BuiltBackend, AppError> + Send + Sync>;

/// Продакшн-фабрика: CUDA EP (device 0), при недоступности — ЯВНЫЙ fallback
/// на CPU (диагностика в stderr, не молча).
///
/// # Errors
/// [`AppError::Dsp`]: недоступны оба EP (обе ошибки — в тексте сообщения).
pub fn default_backend_factory(model_path: &Path) -> Result<BuiltBackend, AppError> {
    match OrtSession::new(model_path, Ep::Cuda { device_id: 0 }) {
        Ok(session) => Ok(BuiltBackend { backend: Arc::new(session), device: "cuda".to_string() }),
        Err(cuda_err) => {
            eprintln!("[engine] CUDA EP недоступен ({cuda_err}); fallback на CPU EP");
            let session = OrtSession::new(model_path, Ep::Cpu).map_err(|cpu_err| {
                AppError::Dsp(format!("model load failed: CUDA: {cuda_err}; CPU: {cpu_err}"))
            })?;
            Ok(BuiltBackend { backend: Arc::new(session), device: "cpu".to_string() })
        }
    }
}

// --- чистые функции конвейера ----------------------------------------------------

/// Де-интерлив фреймов `[L, R]` в планарные каналы.
fn deinterleave(frames: &[[f32; 2]]) -> [Vec<f32>; 2] {
    let mut left = Vec::with_capacity(frames.len());
    let mut right = Vec::with_capacity(frames.len());
    for frame in frames {
        left.push(frame[0]);
        right.push(frame[1]);
    }
    [left, right]
}

/// Длина спектра одного знака (real/imag) одного канала.
const SPEC_LEN: usize = SPEC_BINS * SPEC_FRAMES;

/// Инференс одного чанка: STFT обоих каналов → бэкенд → iSTFT всех стемов.
///
/// Лейаут входа `[1,2,1025,345]`: канал — ось 1, спектр freq-major
/// (`bin*SPEC_FRAMES + frame`) — конкатенация `[L | R]` без перестановок.
/// Лейаут выхода `[1,6,2,1025,345]`: слайс стема `s`, канала `c` —
/// `out[(s*2+c)*SPEC_LEN .. (s*2+c+1)*SPEC_LEN]`.
///
/// Возвращает `[NUM_STEMS]` планарных стерео-чанков по [`CHUNK_SAMPLES`].
fn separate_chunk(
    backend: &dyn InferenceBackend,
    stft: &Stft,
    chunk_l: &[f32],
    chunk_r: &[f32],
) -> Result<Vec<[Vec<f32>; 2]>, AppError> {
    let (real_l, imag_l) = stft.forward(chunk_l)?;
    let (real_r, imag_r) = stft.forward(chunk_r)?;
    let spec_real: Vec<f32> = real_l.into_iter().chain(real_r).collect();
    let spec_imag: Vec<f32> = imag_l.into_iter().chain(imag_r).collect();

    let (out_real, out_imag) = backend.run(&spec_real, &spec_imag)?;

    let mut stems = Vec::with_capacity(NUM_STEMS);
    for stem in 0..NUM_STEMS {
        let mut pair = [Vec::new(), Vec::new()];
        for (channel, samples) in pair.iter_mut().enumerate() {
            let base = (stem * 2 + channel) * SPEC_LEN;
            let mut chunk = vec![0.0f32; CHUNK_SAMPLES];
            stft.inverse(&out_real[base..base + SPEC_LEN], &out_imag[base..base + SPEC_LEN], &mut chunk)?;
            *samples = chunk;
        }
        stems.push(pair);
    }
    Ok(stems)
}

/// Разделение одного (планарного, в частоте модели) сигнала: чанкер →
/// инференс по чанкам → кроссфейд-склейка. Между чанками проверяется
/// `cancel`; `on_chunk(chunks_done, chunk_secs)` зовётся после каждого
/// чанка (прогресс/est-статистика).
///
/// Возвращает `[NUM_STEMS]` планарных стерео-каналов длины входа.
///
/// # Errors
/// [`AppError::Dsp`]: расхождение длин каналов, пустой вход, сбой
/// инференса/STFT; отмена — [`AppError::Dsp`] с текстом [`CANCELLED`].
fn separate_signal(
    backend: &dyn InferenceBackend,
    stft: &Stft,
    channels: &[Vec<f32>; 2],
    cancel: &AtomicBool,
    mut on_chunk: impl FnMut(usize, f64),
) -> Result<Vec<[Vec<f32>; 2]>, AppError> {
    let num_samples = channels[0].len();
    if channels[1].len() != num_samples {
        return Err(AppError::Dsp(format!(
            "separate_signal: длины каналов расходятся ({} и {})",
            channels[0].len(),
            channels[1].len()
        )));
    }
    if num_samples == 0 {
        return Err(AppError::Dsp("separate_signal: пустой сигнал".to_string()));
    }

    let chunks_l = chunk_signal(&channels[0]);
    let chunks_r = chunk_signal(&channels[1]);
    // Одинаковое число чанков следует из одинаковой длины каналов (chunker
    // детерминирован); assert страхует от регрессии чанкера.
    assert_eq!(chunks_l.len(), chunks_r.len(), "chunker обязан резать каналы поровну");

    // collected[stem][channel] — список чанков стема (для merge_chunks).
    let mut collected: Vec<[Vec<Vec<f32>>; 2]> = (0..NUM_STEMS).map(|_| [Vec::new(), Vec::new()]).collect();
    for (chunk_l, chunk_r) in chunks_l.iter().zip(&chunks_r) {
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::Dsp(CANCELLED.to_string()));
        }
        let started = Instant::now();
        let stems = separate_chunk(backend, stft, chunk_l, chunk_r)?;
        for (acc, stem) in collected.iter_mut().zip(&stems) {
            acc[0].push(stem[0].clone());
            acc[1].push(stem[1].clone());
        }
        on_chunk(collected[0][0].len(), started.elapsed().as_secs_f64());
    }

    collected
        .into_iter()
        .map(|pair| {
            Ok([merge_chunks(&pair[0], num_samples)?, merge_chunks(&pair[1], num_samples)?])
        })
        .collect::<Result<Vec<_>, AppError>>()
}
/// Верифицированный конфиг энкодера flacenc — один на процесс (потокобезопасно).
fn encoder_config() -> Result<&'static Verified<config::Encoder>, AppError> {
    static CONFIG: OnceLock<Result<Verified<config::Encoder>, String>> = OnceLock::new();
    CONFIG
        .get_or_init(|| config::Encoder::default().into_verified().map_err(|(_, e)| e.to_string()))
        .as_ref()
        .map_err(|msg| AppError::Dsp(format!("flac encoder config: {msg}")))
}

/// Пишет один стем во FLAC 24-bit (стерео, частота источника).
/// Квантование — как в `wav.rs` (клип [−1..1], масштаб к 2²³−1, round).
fn write_stem_flac(
    path: &Path,
    sample_rate: u32,
    channels: &[Vec<f32>; 2],
    num_frames: usize,
) -> Result<(), AppError> {
    let full_scale = 8_388_607.0_f32; // 2^23 − 1, паритет с WavBits::I24
    let mut interleaved = Vec::with_capacity(num_frames * 2);
    for frame in 0..num_frames {
        for channel in channels {
            let s = channel.get(frame).copied().unwrap_or(0.0);
            interleaved.push((s.clamp(-1.0, 1.0) * full_scale).round() as i32);
        }
    }
    let vcfg = encoder_config()?;
    let source = MemSource::from_samples(&interleaved, 2, 24, sample_rate as usize);
    let stream = encode_with_fixed_block_size(vcfg, source, vcfg.block_size)
        .map_err(|e| AppError::Dsp(format!("flac encode `{}`: {e}", path.display())))?;
    let mut sink = ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| AppError::Dsp(format!("flac write `{}`: {e}", path.display())))?;
    fs::write(path, sink.as_slice()).map_err(|source| AppError::Io { path: path.to_path_buf(), source })?;
    Ok(())
}

/// Локальные пути стемов файла — раскладка драйвера:
/// `<output_dir>/<file_stem>/<stem>.flac` (сырой stem, без trim).
fn stem_output_paths(output_dir: &Path, file: &Path) -> Vec<PathBuf> {
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    STEM_NAMES
        .iter()
        .map(|name| output_dir.join(&stem).join(format!("{name}.flac")))
        .collect()
}

/// Квантование pct события прогресса (1 знак, паритет с драйвером).
fn round_pct(pct: f64) -> f64 {
    (pct * 10.0).round() / 10.0
}

/// Точный pct по чанкам: (files_done + chunks_done/total_chunks) / total_files.
fn exact_pct(files_done: usize, chunks_done: usize, total_chunks: usize, total_files: usize) -> f64 {
    if total_files == 0 || total_chunks == 0 {
        return 0.0;
    }
    (files_done as f64 + chunks_done as f64 / total_chunks as f64) / total_files as f64 * 100.0
}

// --- прогресс: общее состояние воркера и тикера ----------------------------------

/// Состояние чанк-лупа одного файла (воркер пишет, тикер читает).
struct ChunkState {
    total_chunks: usize,
    chunks_done: usize,
    /// Скользящее среднее времени чанка (кумулятивное); 0 — пока нет данных.
    avg_chunk_secs: f64,
    /// Момент старта текущего (незавершённого) чанка.
    cur_started: Option<Instant>,
}

impl ChunkState {
    fn new(total_chunks: usize) -> Self {
        Self { total_chunks, chunks_done: 0, avg_chunk_secs: 0.0, cur_started: None }
    }

    /// Обновление воркером после завершения чанка (кумулятивное среднее).
    fn record_chunk(&mut self, secs: f64) {
        let done = self.chunks_done as f64;
        self.avg_chunk_secs =
            if done == 0.0 { secs } else { (self.avg_chunk_secs * done + secs) / (done + 1.0) };
        self.chunks_done += 1;
        self.cur_started = Some(Instant::now());
    }

    /// Оценочная доля текущего файла (0..1) по среднему времени чанка:
    /// завершённые чанки + доля «чанка в полёте» (кап по 1.0).
    fn estimated_file_frac(&self) -> f64 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        let in_flight = self
            .cur_started
            .map(|t| (t.elapsed().as_secs_f64() / self.avg_chunk_secs).min(1.0))
            .unwrap_or(0.0);
        (self.chunks_done as f64 + in_flight) / self.total_chunks as f64
    }
}

// --- движок ------------------------------------------------------------------------

/// Нативный движок разделения ([`EngineOps`]): DSP-конвейер крейта + ONNX
/// через [`InferenceBackend`]. Одна сессия на батч («model hot»), ретрай
/// per-file 1+1, мягкая отмена по [`AtomicBool`].
pub struct NativeEngine {
    /// Путь к ONNX-модели (граф bs_roformer_sw_6stem_fp32).
    model_path: PathBuf,
    /// Фабрика бэкенда (продакшн — CUDA→CPU, тесты — моки).
    factory: BackendFactory,
    /// Флаг мягкой отмены (взводится `kill_current_engine`).
    cancelled: Arc<AtomicBool>,
    /// Приёмник `[engine]`-строк (диагностика ретраев). По умолчанию
    /// `eprintln!` (CLI); TUI подставляет канал событий — как у драйвера.
    on_stderr: StderrSink,
}

impl std::fmt::Debug for NativeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEngine").field("model_path", &self.model_path).finish_non_exhaustive()
    }
}

impl NativeEngine {
    /// Движок из пути к ONNX-модели (фабрика — продакшн CUDA→CPU).
    pub fn new(model_path: &Path) -> Self {
        Self {
            model_path: model_path.to_path_buf(),
            factory: Arc::new(default_backend_factory),
            cancelled: Arc::new(AtomicBool::new(false)),
            on_stderr: Arc::new(|line| eprintln!("{line}")),
        }
    }

    /// Builder: подмена фабрики бэкенда (тесты — моки без GPU).
    pub fn with_backend_factory(mut self, factory: BackendFactory) -> Self {
        self.factory = factory;
        self
    }

    /// Builder: приёмник `[engine]`-строк (паритет с [`crate::pipeline::EngineRunner`]).
    pub fn on_stderr(mut self, sink: Box<dyn Fn(&str) + Send + Sync>) -> Self {
        self.on_stderr = Arc::from(sink);
        self
    }
}

impl EngineOps for NativeEngine {
    /// Один вызов на батч: сборка бэкенда → per-file (ретрай 1+1) → `batch_done`.
    ///
    /// Сбой загрузки модели отказывает все файлы (паритет с драйвером);
    /// отказ одного файла батч не останавливает. Отмена (`kill_current_engine`)
    /// завершает текущий файл и все оставшиеся `file_error`-ами.
    fn separate_batch(
        &self,
        files: &[PathBuf],
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(), AppError> {
        if files.is_empty() {
            return Ok(());
        }

        let first = files[0].to_string_lossy().into_owned();
        on_event(progress(&first, "load", 0.0, false));

        let built = match (self.factory)(&self.model_path) {
            Ok(built) => built,
            Err(err) => {
                // Паритет с драйвером: сбой загрузки модели отказывает все файлы.
                let message = format!("model load failed: {err}");
                for file in files {
                    on_event(EngineEvent::FileError {
                        file: file.to_string_lossy().into_owned(),
                        error: message.clone(),
                    });
                }
                on_event(EngineEvent::BatchDone { ok: 0, failed: files.len() as u32 });
                return Ok(());
            }
        };
        on_event(progress(&first, "load", 100.0, false));
        on_event(EngineEvent::Device { name: built.device });

        let stft = Stft::new();
        let (mut ok, mut failed) = (0u32, 0u32);
        for (index, file) in files.iter().enumerate() {
            if self.cancelled.load(Ordering::Relaxed) {
                failed += 1;
                on_event(EngineEvent::FileError {
                    file: file.to_string_lossy().into_owned(),
                    error: CANCELLED.to_string(),
                });
                continue;
            }
            let path = file.to_string_lossy().into_owned();
            on_event(progress(&path, "separate", 0.0, false));
            let (done, last_error) = self.separate_file_with_retry(
                &built.backend,
                &stft,
                file,
                &path,
                index,
                files.len(),
                output_dir,
                on_event,
            );
            if done {
                ok += 1;
            } else {
                failed += 1;
                on_event(EngineEvent::FileError { file: path, error: last_error });
            }
        }
        on_event(EngineEvent::BatchDone { ok, failed });
        Ok(())
    }

    /// Мягкая отмена: флаг проверяется между чанками и перед ретраем.
    fn kill_current_engine(&self) {
        self.cancel_flag().store(true, Ordering::Relaxed);
    }
}

impl NativeEngine {
    /// Общий флаг отмены: тот же, что взводит [`EngineOps::kill_current_engine`]
    /// (TUI/тесты могут держать хендл и гасить чанк-луп извне).
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }
}

fn progress(file: &str, stage: &str, pct: f64, estimated: bool) -> EngineEvent {
    EngineEvent::Progress { file: file.to_string(), stage: stage.to_string(), pct: round_pct(pct), estimated }
}

impl NativeEngine {
    /// Обработка одного файла с ретраем 1+1: `(успех, текст последней ошибки)`.
    /// Отмена прерывает ретрай-цикл сразу (повтор при отмене бессмыслен).
    #[allow(clippy::too_many_arguments)] // плоский контекст: файл + индексы прогресса + колбэк
    fn separate_file_with_retry(
        &self,
        backend: &Arc<dyn InferenceBackend>,
        stft: &Stft,
        file: &Path,
        path: &str,
        files_done: usize,
        total_files: usize,
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> (bool, String) {
        let mut last_error = "unknown error".to_string();
        for attempt in 1..=ATTEMPTS_PER_FILE {
            if self.cancelled.load(Ordering::Relaxed) {
                return (false, CANCELLED.to_string());
            }
            match self.separate_one(backend, stft, file, files_done, total_files, output_dir, on_event) {
                Ok((stems, seconds)) => {
                    on_event(EngineEvent::FileDone { file: path.to_string(), stems, seconds });
                    return (true, String::new());
                }
                Err(err) => {
                    last_error = err.to_string();
                    (self.on_stderr)(&format!(
                        "[engine] attempt {attempt}/{ATTEMPTS_PER_FILE} failed for {}: {last_error}",
                        file.display()
                    ));
                }
            }
        }
        (false, last_error)
    }

    /// Полный конвейер одного файла (одна попытка): WAV → ресемпл → чанк-луп
    /// (воркер-поток + est-тикер) → ресемпл обратно → 6 FLAC 24-bit.
    /// Возвращает (пути стемов, секунды попытки).
    // 8 аргументов — плоский контекст файл+прогресс+колбэк; спрямлять нечего.
    #[allow(clippy::too_many_arguments)]
    fn separate_one(
        &self,
        backend: &Arc<dyn InferenceBackend>,
        stft: &Stft,
        file: &Path,
        files_done: usize,
        total_files: usize,
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(Vec<String>, f64), AppError> {
        let started = Instant::now();
        // 1. Чтение; не-WAV/битый файл — внятная Dsp-ошибка (реальные записи — WAV).
        let (source_rate, frames) = read_stereo_f32(file).map_err(|err| match err {
            AppError::Dsp(msg) => AppError::Dsp(format!(
                "{msg} (native-движок поддерживает только WAV: PCM 16/24/32 бит или float32)"
            )),
            other => other,
        })?;
        let num_frames = frames.len();
        let planar = deinterleave(&frames);

        // 2. Ресемпл в частоту модели (44.1k — passthrough).
        let down: [Vec<f32>; 2] = if source_rate == MODEL_SAMPLE_RATE {
            planar
        } else {
            into_pair(resample_stereo(&planar, source_rate, MODEL_SAMPLE_RATE)?)
        };

        // 3. Чанк-луп в воркер-потоке; главный поток тикает прогресс
        // (on_event не Send — события эмитятся только из вызывающего потока).
        let chunks_total = chunk_signal(&down[0]).len();
        let state = Arc::new(Mutex::new(ChunkState::new(chunks_total)));
        let stems_model_rate = run_with_progress(
            backend,
            stft,
            &down,
            &self.cancelled,
            state,
            files_done,
            total_files,
            &file.to_string_lossy(),
            on_event,
        )?;

        // 4. Стемы обратно в частоту источника (44.1k — passthrough), длина —
        // ровно по исходнику (хвост накачки ресемплера обрезается).
        let stems: Vec<[Vec<f32>; 2]> = if source_rate == MODEL_SAMPLE_RATE {
            stems_model_rate
        } else {
            stems_model_rate
                .into_iter()
                .map(|pair| {
                    let up = into_pair(resample_stereo(&pair, MODEL_SAMPLE_RATE, source_rate)?);
                    let [left, right] = up;
                    Ok([trim_to(left, num_frames), trim_to(right, num_frames)])
                })
                .collect::<Result<Vec<_>, AppError>>()?
        };

        // 5. Запись 6 FLAC 24-bit: <output_dir>/<file_stem>/<stem>.flac.
        let paths = stem_output_paths(output_dir, file);
        for (path, stem) in paths.iter().zip(&stems) {
            if let Some(dir) = path.parent() {
                fs::create_dir_all(dir).map_err(|source| AppError::Io { path: dir.to_path_buf(), source })?;
            }
            write_stem_flac(path, source_rate, stem, num_frames)?;
        }
        let seconds = (started.elapsed().as_secs_f64() * 100.0).round() / 100.0;
        Ok((paths.iter().map(|p| p.to_string_lossy().into_owned()).collect(), seconds))
    }
}

/// Чанк-луп в воркер-потоке + тикер прогресса в главном потоке.
/// Возвращает стемы в частоте модели (или ошибку воркера).
#[allow(clippy::too_many_arguments)]
fn run_with_progress(
    backend: &Arc<dyn InferenceBackend>,
    stft: &Stft,
    channels: &[Vec<f32>; 2],
    cancel: &AtomicBool,
    state: Arc<Mutex<ChunkState>>,
    files_done: usize,
    total_files: usize,
    file: &str,
    on_event: &mut dyn FnMut(EngineEvent),
) -> Result<Vec<[Vec<f32>; 2]>, AppError> {
    let (tx, rx) = std::sync::mpsc::channel();
    let worker_state = Arc::clone(&state);
    // `move` обязателен: tx уходит воркеру — при его панике channel рвётся
    // (Disconnected), а тикер не зацикливается на живом sender'е.
    thread::scope(|scope| {
        scope.spawn(move || {
            let result = separate_signal(backend, stft, channels, cancel, |_chunks_done, secs| {
                if let Ok(mut snapshot) = worker_state.lock() {
                    snapshot.record_chunk(secs);
                }
            });
            let _ = tx.send(result);
        });
        let mut emitted_chunks = 0usize;
        loop {
            match rx.recv_timeout(EST_TICK) {
                Ok(result) => break result,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(AppError::Dsp("native engine worker panicked".to_string()));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let Ok(snapshot) = state.lock() else {
                        continue; // мьютекс отравлен — ниже придёт Disconnected
                    };
                    if snapshot.avg_chunk_secs == 0.0 {
                        continue; // ещё нет завершённых чанков — данных для событий нет
                    }
                    if snapshot.chunks_done != emitted_chunks {
                        // Точный прогресс: граница чанка только что пройдена.
                        emitted_chunks = snapshot.chunks_done;
                        let pct =
                            exact_pct(files_done, snapshot.chunks_done, snapshot.total_chunks, total_files);
                        on_event(progress(file, "separate", pct, false));
                    } else {
                        // Оценка между чанками по скользящему среднему (est, кап 95%).
                        let pct = (files_done as f64 + snapshot.estimated_file_frac())
                            / total_files as f64
                            * 100.0;
                        on_event(progress(file, "separate", pct.min(EST_CAP_PCT), true));
                    }
                }
            }
        }
    })
}

/// `Vec<Vec<f32>>` (длина 2) → `[Vec<f32>; 2]` (контракт resample_stereo).
fn into_pair(mut channels: Vec<Vec<f32>>) -> [Vec<f32>; 2] {
    assert!(channels.len() == 2, "resample_stereo возвращает ровно 2 канала");
    let right = channels.pop().expect("len == 2 проверен");
    let left = channels.pop().expect("len == 2 проверен");
    [left, right]
}

/// Обрезка хвоста ресемплера до длины исходника (паника исключена:
/// `ceil(ceil(n·147/160)·160/147) ≥ n` — ceil растёт монотонно).
fn trim_to(mut samples: Vec<f32>, num_frames: usize) -> Vec<f32> {
    assert!(
        samples.len() >= num_frames,
        "ресемпл вернул {} кадров < {num_frames}",
        samples.len()
    );
    samples.truncate(num_frames);
    samples
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::engine::STEP_SAMPLES;

    /// Детерминированный LCG-сигнал (без rand-крейта).
    fn lcg_signal(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let uniform = ((state >> 33) as f32) / (u32::MAX as f32);
                uniform * 2.0 - 1.0
            })
            .collect()
    }

    const OUT_LEN: usize = NUM_STEMS * 2 * SPEC_LEN;
    const IN_LEN: usize = 2 * SPEC_LEN;

    /// Сигнатура ответа мок-бэкенда: (spec_real, spec_imag) → (out_real, out_imag).
    type ResponseFn = dyn Fn(&[f32], &[f32]) -> (Vec<f32>, Vec<f32>) + Send + Sync;

    /// Мок-бэкенд: ответ из замыкания, счётчик вызовов, отказ после N успехов.
    struct MockBackend {
        response: Box<ResponseFn>,
        calls: AtomicUsize,
        fail_after: usize,
        seen_inputs: Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
    }

    impl MockBackend {
        /// Ответ из замыкания (счётчик вызовов всегда ведётся).
        fn responding(
            f: impl Fn(&[f32], &[f32]) -> (Vec<f32>, Vec<f32>) + Send + Sync + 'static,
        ) -> Self {
            Self {
                response: Box::new(f),
                calls: AtomicUsize::new(0),
                fail_after: usize::MAX,
                seen_inputs: Mutex::new(Vec::new()),
            }
        }

        /// Тишина на выходе.
        fn zeros() -> Self {
            Self::responding(|_, _| (vec![0.0f32; OUT_LEN], vec![0.0f32; OUT_LEN]))
        }

        /// Отказ после `n` успешных вызовов (для ретрай-сценариев).
        fn failing_after(self, n: usize) -> Self {
            Self { fail_after: n, ..self }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl InferenceBackend for MockBackend {
        fn run(&self, spec_real: &[f32], spec_imag: &[f32]) -> Result<(Vec<f32>, Vec<f32>), AppError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen_inputs.lock().expect("inputs lock").push((spec_real.to_vec(), spec_imag.to_vec()));
            if call >= self.fail_after {
                return Err(AppError::Dsp("mock backend forced failure".to_string()));
            }
            Ok((self.response)(spec_real, spec_imag))
        }
    }

    /// Фабрика, отдающая единственный захваченный бэкенд.
    fn factory_of(backend: Arc<dyn InferenceBackend>) -> BackendFactory {
        let holder = Mutex::new(Some(backend));
        Arc::new(move |_| {
            holder
                .lock()
                .expect("factory lock")
                .take()
                .map(|backend| BuiltBackend { backend, device: "mock".to_string() })
                .ok_or_else(|| AppError::Dsp("mock factory exhausted".to_string()))
        })
    }

    // --- чистые функции --------------------------------------------------------

    #[test]
    fn deinterleave_splits_frames_into_planar_channels() {
        let frames = vec![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let [l, r] = deinterleave(&frames);
        assert_eq!(l, vec![0.1, 0.3, 0.5]);
        assert_eq!(r, vec![0.2, 0.4, 0.6]);
    }

    #[test]
    fn exact_pct_formula_matches_spec() {
        assert_eq!(exact_pct(0, 0, 4, 2), 0.0);
        assert_eq!(exact_pct(0, 2, 4, 2), 25.0);
        assert_eq!(exact_pct(0, 4, 4, 2), 50.0);
        assert_eq!(exact_pct(1, 4, 4, 2), 100.0);
    }

    #[test]
    fn round_pct_keeps_one_decimal() {
        assert_eq!(round_pct(12.34), 12.3);
        assert_eq!(round_pct(12.36), 12.4);
    }

    #[test]
    fn stem_output_paths_follow_driver_layout() {
        let paths = stem_output_paths(Path::new("/tmp/out"), Path::new("/work/song.wav"));
        let actual: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let expected: Vec<String> = STEM_NAMES
            .iter()
            .map(|stem| format!("/tmp/out/song/{stem}.flac"))
            .collect();
        assert_eq!(actual, expected, "раскладка и имена — как у python-драйвера");
    }

    #[test]
    fn chunk_state_estimates_fraction_from_rolling_average() {
        let mut state = ChunkState::new(4);
        assert_eq!(state.estimated_file_frac(), 0.0);
        state.record_chunk(1.0);
        assert_eq!(state.chunks_done, 1);
        assert_eq!(state.avg_chunk_secs, 1.0);
        state.record_chunk(1.0);
        assert!((state.avg_chunk_secs - 1.0).abs() < 1e-12, "среднее стабильно");
        let frac = state.estimated_file_frac();
        assert!((0.5..=0.75).contains(&frac), "frac = {frac}");
    }

    // --- separate_chunk: лейаут тензора ------------------------------------------

    #[test]
    fn separate_chunk_input_is_channel_major_stft_concat() {
        let stft = Stft::new();
        let chunk_l: Vec<f32> = (0..CHUNK_SAMPLES).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
        let chunk_r: Vec<f32> = (0..CHUNK_SAMPLES).map(|i| (i as f32 * 0.017).cos() * 0.2).collect();

        let backend = MockBackend::responding(|_, _| (vec![0.0f32; OUT_LEN], vec![0.0f32; OUT_LEN]));
        let _ = separate_chunk(&backend, &stft, &chunk_l, &chunk_r).expect("chunk ok");

        let inputs = backend.seen_inputs.lock().expect("inputs");
        let (real, imag) = &inputs[0];
        assert_eq!(real.len(), IN_LEN, "вход [1,2,1025,345] в плоском виде");
        assert_eq!(imag.len(), IN_LEN);
        // Ось 1 — каналы: [0..SPEC_LEN) = STFT(L), [SPEC_LEN..2*SPEC_LEN) = STFT(R).
        let (expect_l_real, expect_l_imag) = stft.forward(&chunk_l).expect("forward L");
        let (expect_r_real, expect_r_imag) = stft.forward(&chunk_r).expect("forward R");
        assert_eq!(&real[..SPEC_LEN], &expect_l_real[..], "канал 0 = STFT(L)");
        assert_eq!(&real[SPEC_LEN..], &expect_r_real[..], "канал 1 = STFT(R)");
        assert_eq!(&imag[..SPEC_LEN], &expect_l_imag[..]);
        assert_eq!(&imag[SPEC_LEN..], &expect_r_imag[..]);
    }

    #[test]
    fn separate_chunk_slices_output_per_stem_and_channel() {
        // Бэкенд кладёт в каждый слайс (stem, channel) спектр СВОЕГО входного
        // канала с множителем (stem+1): после iSTFT стем s ≈ канал×(s+1) —
        // это проверяет индексацию выхода по обеим осям.
        let stft = Stft::new();
        let chunk_l: Vec<f32> = (0..CHUNK_SAMPLES).map(|i| (i as f32 * 0.011).sin() * 0.25).collect();
        let chunk_r: Vec<f32> = (0..CHUNK_SAMPLES).map(|i| (i as f32 * 0.019).sin() * 0.25).collect();
        let (l_real, l_imag) = stft.forward(&chunk_l).expect("forward");
        let (r_real, r_imag) = stft.forward(&chunk_r).expect("forward");

        let backend = MockBackend::responding(move |_real, _imag| {
            let mut out_real = vec![0.0f32; OUT_LEN];
            let mut out_imag = vec![0.0f32; OUT_LEN];
            for stem in 0..NUM_STEMS {
                let gain = (stem + 1) as f32;
                for (channel, (src_real, src_imag)) in [(&l_real, &l_imag), (&r_real, &r_imag)]
                    .into_iter()
                    .enumerate()
                {
                    let base = (stem * 2 + channel) * SPEC_LEN;
                    for i in 0..SPEC_LEN {
                        out_real[base + i] = src_real[i] * gain;
                        out_imag[base + i] = src_imag[i] * gain;
                    }
                }
            }
            (out_real, out_imag)
        });

        let stems = separate_chunk(&backend, &stft, &chunk_l, &chunk_r).expect("chunk ok");
        assert_eq!(stems.len(), NUM_STEMS);
        for (stem, pair) in stems.iter().enumerate() {
            let gain = (stem + 1) as f32;
            // iSTFT∘STFT ≈ identity (точность roundtrip ~1e-4, см. stft.rs);
            // края (±N_FFT) не проверяем, как в тестах stft.
            for (channel, source) in [&chunk_l, &chunk_r].into_iter().enumerate() {
                let out = &pair[channel];
                assert_eq!(out.len(), CHUNK_SAMPLES);
                for (i, (&a, &b)) in out
                    .iter()
                    .zip(source)
                    .enumerate()
                    .skip(2_048)
                    .take(10_000)
                {
                    assert!(
                        (a - b * gain).abs() < 5e-3 * gain.max(1.0),
                        "stem {stem} ch {channel} pos {i}: {a} vs {}",
                        b * gain
                    );
                }
            }
        }
    }

    // --- separate_signal -----------------------------------------------------------

    #[test]
    fn separate_signal_zeros_backend_yields_silence_full_length() {
        let stft = Stft::new();
        let cancel = AtomicBool::new(false);
        let channels = [lcg_signal(50_000, 7), lcg_signal(50_000, 11)];
        let mut chunks_seen = Vec::new();
        let stems = separate_signal(&MockBackend::zeros(), &stft, &channels, &cancel, |done, _| {
            chunks_seen.push(done)
        })
        .expect("separate ok");
        assert_eq!(stems.len(), NUM_STEMS);
        for pair in &stems {
            assert_eq!(pair[0].len(), 50_000, "длина = длине входа");
            assert_eq!(pair[1].len(), 50_000);
            assert!(pair.iter().all(|c| c.iter().all(|&s| s == 0.0)), "нули → тишина");
        }
        assert_eq!(chunks_seen, vec![1], "один чанк на 50k сэмплов");
    }

    #[test]
    fn separate_signal_returns_input_through_two_chunks() {
        // Бэкенд возвращает каждому (stem, channel) спектр соответствующего
        // входного канала: полный конвейер (чанкер → STFT → iSTFT → кроссфейд)
        // обязан вернуть исходный сигнал во всех 6 стемах.
        let stft = Stft::new();
        let cancel = AtomicBool::new(false);
        let n = CHUNK_SAMPLES + STEP_SAMPLES;
        let mono = lcg_signal(n, 0x5EED);
        let channels = [mono.clone(), mono.clone()];
        let backend = MockBackend::responding(|real, imag| {
            let mut out_real = vec![0.0f32; OUT_LEN];
            let mut out_imag = vec![0.0f32; OUT_LEN];
            for stem in 0..NUM_STEMS {
                for channel in 0..2 {
                    let base = (stem * 2 + channel) * SPEC_LEN;
                    out_real[base..base + SPEC_LEN]
                        .copy_from_slice(&real[channel * SPEC_LEN..(channel + 1) * SPEC_LEN]);
                    out_imag[base..base + SPEC_LEN]
                        .copy_from_slice(&imag[channel * SPEC_LEN..(channel + 1) * SPEC_LEN]);
                }
            }
            (out_real, out_imag)
        });
        let stems = separate_signal(&backend, &stft, &channels, &cancel, |_, _| {}).expect("separate ok");
        assert_eq!(stems.len(), NUM_STEMS);
        // Края чанков (±N_FFT, reflect-пад + OLA-огибающая) шумят — сверяем
        // только «безопасные зоны»: interior неперекрытой части и центр
        // кроссфейда (оба вкладчика в своей надёжной зоне).
        let safe = |i: usize| {
            let local = i % STEP_SAMPLES;
            (8_192..132_300).contains(&local) || (140_492..168_208).contains(&local)
        };
        for pair in &stems {
            for channel in pair {
                for (i, &a) in channel.iter().enumerate().step_by(997) {
                    assert!(
                        !safe(i) || (a - mono[i]).abs() < 2e-3,
                        "pos {i}: {a} vs {}",
                        mono[i]
                    );
                }
            }
        }
    }

    #[test]
    fn separate_signal_mismatched_channels_is_error() {
        let stft = Stft::new();
        let cancel = AtomicBool::new(false);
        let channels = [vec![0.0; 10], vec![0.0; 9]];
        let err =
            separate_signal(&MockBackend::zeros(), &stft, &channels, &cancel, |_, _| {})
                .expect_err("расхождение каналов");
        assert!(matches!(err, AppError::Dsp(_)));
    }
    #[test]
    fn separate_signal_empty_input_is_error() {
        let stft = Stft::new();
        let cancel = AtomicBool::new(false);
        let err = separate_signal(&MockBackend::zeros(), &stft, &[Vec::new(), Vec::new()], &cancel, |_, _| ())
            .expect_err("пустой вход");
        assert!(matches!(err, AppError::Dsp(_)));
    }

    #[test]
    fn separate_signal_cancel_checks_between_chunks() {
        let stft = Stft::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let n = CHUNK_SAMPLES + 1; // ровно 2 чанка
        let mono = lcg_signal(n, 3);
        let channels = [mono.clone(), mono];
        // Отмена взводится после первого чанка → второй не должен стартовать.
        let flag = Arc::clone(&cancel);
        let mut chunks_run = 0usize;
        let result = separate_signal(
            &MockBackend::zeros(),
            &stft,
            &channels,
            &cancel,
            |_, _| {
                chunks_run += 1;
                flag.store(true, Ordering::SeqCst);
            },
        );
        assert!(result.is_err(), "отмена должна прервать конвейер");
        assert!(
            matches!(result.unwrap_err(), AppError::Dsp(msg) if msg.contains("cancelled")),
            "текст отмены — в ошибке"
        );
        assert_eq!(chunks_run, 1, "после отмены новые чанки не запускаются");
    }

    #[test]
    fn separate_signal_backend_error_propagates() {
        let stft = Stft::new();
        let cancel = AtomicBool::new(false);
        let channels = [lcg_signal(1000, 1), lcg_signal(1000, 1)];
        let backend = Arc::new(MockBackend::responding(|_, _| (Vec::new(), Vec::new())).failing_after(0));
        let err = separate_signal(&*backend, &stft, &channels, &cancel, |_, _| ()).expect_err("бэкенд падает");
        assert!(matches!(err, AppError::Dsp(_)));
        assert_eq!(backend.call_count(), 1);
    }

    // --- FLAC-запись ----------------------------------------------------------------

    #[test]
    fn write_stem_flac_roundtrip_24bit_via_claxon() {
        let dir = std::env::temp_dir().join(format!("native_flac_{}", std::process::id()));
        fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("vocals.flac");
        let n = 4800;
        let left: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let right: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).cos() * 0.25).collect();
        write_stem_flac(&path, 48_000, &[left.clone(), right.clone()], n).expect("write flac");

        let mut reader = claxon::FlacReader::open(&path).expect("читаемый FLAC");
        let info = reader.streaminfo();
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.bits_per_sample, 24, "FLAC 24-bit");
        assert_eq!(info.channels, 2);
        assert_eq!(info.samples, Some(n as u64), "StreamInfo — кадры на канал");

        let full_scale = 8_388_607.0_f32;
        for (frame, sample) in reader.samples().enumerate() {
            let sample = sample.expect("sample");
            let src = if frame % 2 == 0 { &left } else { &right };
            let expect = (src[frame / 2].clamp(-1.0, 1.0) * full_scale).round() as i32;
            assert_eq!(sample, expect, "квантование как в wav.rs, frame {frame}");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_stem_flac_pads_missing_tail_with_silence() {
        let dir = std::env::temp_dir().join(format!("native_flac_pad_{}", std::process::id()));
        fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("bass.flac");
        // Канал R короче num_frames: отсутствующий хвост — тишина (страховка
        // от расхождения длин; в основном пути длины выровнены trim_to).
        // 40 кадров: короткие блоки < 16 сэмплов клаксон не читает.
        write_stem_flac(&path, 44_100, &[vec![0.5; 40], vec![-0.5; 33]], 40).expect("write");
        let mut reader = claxon::FlacReader::open(&path).expect("читаемый FLAC");
        let samples: Vec<i32> = reader.samples().map(|s| s.expect("sample")).collect();
        assert_eq!(samples.len(), 80);
        // R-канал: реальные кадры 0..33 (нечётные индексы), хвост — тишина.
        assert!(
            samples.iter().skip(67).step_by(2).all(|&s| s == 0),
            "хвост канала R — тишина"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // --- NativeEngine: e2e на мок-бэкенде ---------------------------------------------

    /// WAV-фикстура (16 бит, стерео) через hound — вне тестируемого кода.
    fn write_wav(path: &Path, rate: u32, samples: &[f32]) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("fixture wav");
        for &s in samples {
            for channel in [s, -s] {
                writer.write_sample((channel * 32_767.0).round() as i16).expect("fixture sample");
            }
        }
        writer.finalize().expect("fixture finalize");
    }

    /// Синус-стерео `seconds` секунд (LR противофазой — ненулевой RMS).
    fn sine_stereo(rate: u32, seconds: f32, freq: f64) -> Vec<f32> {
        let n = (rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| (0.4 * (std::f64::consts::TAU * freq * i as f64 / rate as f64).sin()) as f32)
            .collect()
    }

    /// Прогон батча со сбором всех событий.
    fn collect_events(engine: &NativeEngine, files: &[PathBuf], out: &Path) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        engine.separate_batch(files, out, &mut |event| events.push(event)).expect("batch ok");
        events
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("native_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("tmp dir");
        dir
    }

    #[test]
    fn engine_e2e_44100_passthrough_writes_six_flacs_and_protocol_events() {
        let dir = temp_dir("e2e44k");
        let wav = dir.join("song.wav");
        write_wav(&wav, 44_100, &sine_stereo(44_100, 0.5, 220.0));

        let engine =
            NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory_of(Arc::new(MockBackend::zeros())));
        let events = collect_events(&engine, std::slice::from_ref(&wav), &dir);

        // Протокол: load 0 → load 100 → device → separate 0 → file_done → batch_done.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, EngineEvent::Progress { stage, .. } if stage == "load"))
                .count(),
            2,
            "load 0% и 100%: {events:?}"
        );
        assert!(events.iter().any(|e| matches!(e, EngineEvent::Device { name } if name == "mock")));
        let Some((stems, seconds)) = events.iter().find_map(|e| match e {
            EngineEvent::FileDone { stems, seconds, .. } => Some((stems.clone(), *seconds)),
            _ => None,
        }) else {
            panic!("нет file_done: {events:?}");
        };
        assert!(seconds > 0.0, "секунды попытки фиксируются");
        assert_eq!(stems.len(), 6, "по стему на имя");
        for (i, stem) in stems.iter().enumerate() {
            assert!(Path::new(stem).is_file(), "{stem} записан");
            let name = Path::new(stem).file_name().expect("file name").to_string_lossy().into_owned();
            assert_eq!(name, format!("{}.flac", STEM_NAMES[i]), "имена в порядке STEM_NAMES");
        }
        assert_eq!(
            stems[0],
            dir.join("song").join("bass.flac").to_string_lossy().into_owned(),
            "раскладка <output_dir>/<file_stem>/<stem>.flac"
        );

        // Содержимое: sr, длина и разрядность как в контракте.
        let reader = claxon::FlacReader::open(&stems[0]).expect("flac");
        let info = reader.streaminfo();
        assert_eq!(info.sample_rate, 44_100, "passthrough: sr источника");
        assert_eq!(info.bits_per_sample, 24);
        assert_eq!(info.samples.expect("samples"), 22_050, "0.5 c при 44.1k (кадры на канал)");
        assert_eq!(
            events.last(),
            Some(&EngineEvent::BatchDone { ok: 1, failed: 0 }),
            "финал — batch_done: {events:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_e2e_48000_resamples_back_to_source_rate_and_length() {
        let dir = temp_dir("e2e48k");
        let wav = dir.join("take.wav");
        write_wav(&wav, 48_000, &sine_stereo(48_000, 0.5, 330.0));

        let engine =
            NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory_of(Arc::new(MockBackend::zeros())));
        let events = collect_events(&engine, std::slice::from_ref(&wav), &dir);
        let Some(stems) = events.iter().find_map(|e| match e {
            EngineEvent::FileDone { stems, .. } => Some(stems.clone()),
            _ => None,
        }) else {
            panic!("нет file_done: {events:?}");
        };

        let reader = claxon::FlacReader::open(&stems[0]).expect("flac");
        let info = reader.streaminfo();
        assert_eq!(info.sample_rate, 48_000, "стемы в частоте источника");
        assert_eq!(info.samples.expect("samples"), 24_000, "длительность совпадает с исходником");

        // Ресемпл туда-обратно действительно был: бэкенд получил чанк в 44.1k
        // (24_000·147/160 = 22_050 сэмплов — один чанк).
        let backend = Arc::new(MockBackend::zeros());
        let engine2 = NativeEngine::new(Path::new("/unused.onnx"))
            .with_backend_factory(factory_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>));
        let _ = collect_events(&engine2, &[wav], &dir);
        assert_eq!(backend.call_count(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_retries_failing_file_once_and_reports_file_error() {
        let dir = temp_dir("retry");
        let wav = dir.join("bad.wav");
        write_wav(&wav, 44_100, &sine_stereo(44_100, 0.2, 110.0));

        let backend = Arc::new(MockBackend::zeros().failing_after(0));
        let engine = NativeEngine::new(Path::new("/unused.onnx"))
            .with_backend_factory(factory_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>));
        let events = collect_events(&engine, &[wav], &dir);

        assert_eq!(backend.call_count(), 2, "1 проба + 1 ретрай");
        assert!(matches!(
            events.iter().find(|e| matches!(e, EngineEvent::FileError { .. })),
            Some(EngineEvent::FileError { error, .. }) if error.contains("forced failure")
        ));
        assert_eq!(events.last(), Some(&EngineEvent::BatchDone { ok: 0, failed: 1 }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_recovers_on_second_attempt_after_single_failure() {
        let dir = temp_dir("flaky");
        let wav = dir.join("flaky.wav");
        write_wav(&wav, 44_100, &sine_stereo(44_100, 0.2, 110.0));

        // Первый инференс падает, второй — успех (тишина): файл должен сойтись.
        let backend =
            Arc::new(MockBackend::responding(|_, _| (vec![0.0f32; OUT_LEN], vec![0.0f32; OUT_LEN]))
                .failing_after(1));
        let engine = NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory_of(backend));
        let events = collect_events(&engine, &[wav], &dir);
        assert!(
            events.iter().any(|e| matches!(e, EngineEvent::FileDone { .. })),
            "файл сошёлся со второй попытки: {events:?}"
        );
        assert_eq!(events.last(), Some(&EngineEvent::BatchDone { ok: 1, failed: 0 }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_cancel_before_batch_fails_all_files_without_inference() {
        let dir = temp_dir("cancel");
        let wav = dir.join("x.wav");
        write_wav(&wav, 44_100, &sine_stereo(44_100, 0.2, 110.0));

        let backend = Arc::new(MockBackend::zeros());
        let engine = NativeEngine::new(Path::new("/unused.onnx"))
            .with_backend_factory(factory_of(Arc::clone(&backend) as Arc<dyn InferenceBackend>));
        engine.kill_current_engine();
        let events = collect_events(&engine, &[wav, dir.join("y.wav")], &dir);

        assert_eq!(backend.call_count(), 0, "инференс не запускается после отмены");
        let errors = events.iter().filter(|e| matches!(e, EngineEvent::FileError { .. })).count();
        assert_eq!(errors, 2, "оба файла получили file_error: {events:?}");
        assert_eq!(events.last(), Some(&EngineEvent::BatchDone { ok: 0, failed: 2 }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_cancel_between_chunks_stops_without_retry() {
        let dir = temp_dir("cancel_mid");
        let wav = dir.join("long.wav");
        // 1.25 чанка: cancel взводится внутри первого инференса → второй чанк
        // не стартует, ретрай не выполняется.
        write_wav(&wav, 44_100, &sine_stereo(44_100, 5.0, 110.0));

        let engine = NativeEngine::new(Path::new("/unused.onnx"));
        let engine_flag = engine.cancel_flag(); // флаг самого движка
        let backend = Arc::new(MockBackend::responding(move |_, _| {
            engine_flag.store(true, Ordering::SeqCst);
            (vec![0.0f32; OUT_LEN], vec![0.0f32; OUT_LEN])
        }));
        let engine = engine.with_backend_factory(factory_of(backend.clone()));
        let events = collect_events(&engine, &[wav], &dir);

        assert_eq!(backend.call_count(), 1, "после cancel следующий чанк/ретрай не идут");
        assert!(matches!(
            events.iter().find(|e| matches!(e, EngineEvent::FileError { .. })),
            Some(EngineEvent::FileError { error, .. }) if error.contains("cancelled")
        ));
        assert_eq!(events.last(), Some(&EngineEvent::BatchDone { ok: 0, failed: 1 }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_model_load_failure_fails_whole_batch_like_driver() {
        let dir = temp_dir("loadfail");
        let wav = dir.join("a.wav");
        write_wav(&wav, 44_100, &sine_stereo(44_100, 0.2, 110.0));

        let factory: BackendFactory =
            Arc::new(|_| Err(AppError::Dsp("ORT_DYLIB_PATH не задан".to_string())));
        let engine = NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory);
        let events = collect_events(&engine, &[wav, dir.join("b.wav")], &dir);

        let errors: Vec<&EngineEvent> =
            events.iter().filter(|e| matches!(e, EngineEvent::FileError { .. })).collect();
        assert_eq!(errors.len(), 2, "сбой загрузки отказывает все файлы: {events:?}");
        assert!(matches!(&errors[0], EngineEvent::FileError { error, .. } if error.contains("model load failed")));
        assert_eq!(events.last(), Some(&EngineEvent::BatchDone { ok: 0, failed: 2 }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_non_wav_input_is_clear_dsp_error() {
        let dir = temp_dir("notwav");
        let file = dir.join("notes.wav");
        fs::write(&file, b"definitely not a wav file....").expect("fixture");

        let engine =
            NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory_of(Arc::new(MockBackend::zeros())));
        let events = collect_events(&engine, &[file], &dir);
        assert!(
            matches!(
                events.iter().find(|e| matches!(e, EngineEvent::FileError { .. })),
                Some(EngineEvent::FileError { error, .. }) if error.contains("только WAV")
            ),
            "внятное сообщение про WAV: {events:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_empty_batch_is_noop() {
        let engine =
            NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory_of(Arc::new(MockBackend::zeros())));
        let mut events = Vec::new();
        let result = engine.separate_batch(&[], Path::new("/tmp"), &mut |event| events.push(event));
        assert!(result.is_ok());
        assert!(events.is_empty(), "пустой батч не генерирует событий");
    }

    #[test]
    fn engine_emits_exact_progress_at_chunk_boundaries() {
        let dir = temp_dir("progress");
        let wav = dir.join("p.wav");
        write_wav(&wav, 44_100, &sine_stereo(44_100, 5.0, 220.0)); // 2 чанка

        // Второй чанк держит воркер дольше EST_TICK: тикер гарантированно
        // срабатывает между чанками (иначе гонка «чанки быстрее тика»).
        let call = Arc::new(AtomicUsize::new(0));
        let call_in_backend = Arc::clone(&call);
        let backend = Arc::new(MockBackend::responding(move |_, _| {
            if call_in_backend.fetch_add(1, Ordering::SeqCst) == 1 {
                thread::sleep(Duration::from_millis(1500));
            }
            (vec![0.0f32; OUT_LEN], vec![0.0f32; OUT_LEN])
        }));
        let engine =
            NativeEngine::new(Path::new("/unused.onnx")).with_backend_factory(factory_of(backend));
        let events = collect_events(&engine, &[wav], &dir);
        let separate: Vec<(f64, bool)> = events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Progress { stage, pct, estimated, .. } if stage == "separate" => {
                    Some((*pct, *estimated))
                }
                _ => None,
            })
            .collect();
        assert!(
            separate.iter().any(|(pct, est)| *pct == 0.0 && !est),
            "старт файла — точные 0%: {separate:?}"
        );
        assert!(
            separate.iter().any(|(pct, est)| (*pct - 50.0).abs() < 0.1 && !est),
            "после первого чанка из двух — точные 50%: {separate:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
