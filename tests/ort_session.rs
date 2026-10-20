//! Gated-тесты OrtSession: реальный инференс чанка на модели BS-Roformer-SW,
//! регрессия CPU-фолбэка после провала CUDA.
//!
//! Выполняются ТОЛЬКО при наличии `ORT_DYLIB_PATH` (dylib ONNX Runtime ≥ 1.31)
//! и модели (`HEXASTEM_MODEL_PATH` или дефолт `.models/onnx/`), иначе —
//! graceful-skip. Пометка `#[ignore]`: нужны GPU + 668 MB модель, в обычный
//! `cargo test` не входят.
//!
//! Запуск (CUDA, GPU 0):
//! ```text
//! SESS=<...>/2026-10-14-native-rust-engine
//! ORT_DYLIB_PATH=$SESS/onnxruntime-linux-x64-gpu_cuda12-1.31.0/lib/libonnxruntime.so \
//! LD_LIBRARY_PATH="$SESS/cudart12/nvidia/cuda_runtime/lib:<nvidia/*/lib из venv>" \
//! cargo test --release --test ort_session -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use hexastem::engine::ort_session::{Ep, OrtSession};

/// Контракт входа: 2·1025·345; выхода: 6·2·1025·345.
const IN_LEN: usize = 2 * 1025 * 345;
const OUT_LEN: usize = 6 * IN_LEN;

/// Путь к модели: env-override или дефолт крейта; None → тест пропустится.
fn model_path() -> Option<PathBuf> {
    if let Ok(from_env) = std::env::var("HEXASTEM_MODEL_PATH") {
        let p = PathBuf::from(from_env);
        return p.is_file().then_some(p);
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".models/onnx/bs_roformer_sw_6stem_fp32.onnx");
    p.is_file().then_some(p)
}

/// Детерминированный вход (не нули): checksum подтверждает реальный прогон графа.
fn deterministic_input() -> Vec<f32> {
    (0..IN_LEN)
        .map(|k| (((k % 997) as f32 / 997.0) - 0.5) * 0.1)
        .collect()
}

#[test]
#[ignore = "нужны GPU (CUDA), модель ONNX и ORT_DYLIB_PATH — запуск: cargo test --release -- --ignored"]
fn cuda_inference_chunk_deterministic_and_finite() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: модель не найдена (HEXASTEM_MODEL_PATH или .models/onnx/)");
        return;
    };
    if std::env::var("ORT_DYLIB_PATH").map_or(true, |v| v.is_empty()) {
        eprintln!("SKIP: ORT_DYLIB_PATH не задан");
        return;
    }

    let session = OrtSession::new(&model, Ep::Cuda { device_id: 0 })
        .expect("CUDA-сессия (проверь ORT_DYLIB_PATH/LD_LIBRARY_PATH/драйвер)");

    let real = deterministic_input();
    let imag = vec![0.0f32; IN_LEN];

    let start = Instant::now();
    let (r1, i1) = session.run(&real, &imag).expect("первый прогон");
    let elapsed = start.elapsed().as_secs_f64();

    let (r2, i2) = session.run(&real, &imag).expect("второй прогон");

    assert_eq!(r1.len(), OUT_LEN);
    assert_eq!(i1.len(), OUT_LEN);
    assert!(
        r1.iter().chain(&i1).all(|v| v.is_finite()),
        "не-finite значения в выходе"
    );
    // Детерминизм: два прогона одного входа — бит-в-бит идентичный выход.
    assert_eq!(r1, r2, "out_spec_real недетерминирован");
    assert_eq!(i1, i2, "out_spec_imag недетерминирован");

    let checksum: f32 = r1.iter().sum();
    eprintln!(
        "cuda chunk: {elapsed:.3} s (audio 4.0 s, RTF {:.3}), checksum={checksum:.4}",
        elapsed / 4.0
    );
}

/// Регрессия боя: явный фолбэк `default_backend_factory`. Первая попытка
/// CUDA падает (Pascal: `CUBLAS_STATUS_ARCH_MISMATCH` на commit) — вторая
/// попытка CPU обязана пройти в том же процессе. Раньше падала: каждый
/// `OrtSession::new` звал `ort::init()`, а OrtEnv в процессе ровно один.
#[test]
#[ignore = "нужны модель ONNX и ORT_DYLIB_PATH (GPU не обязателен) — запуск: cargo test --release -- --ignored"]
fn cpu_session_succeeds_after_failed_cuda_attempt() {
    let Some(model) = model_path() else {
        eprintln!("SKIP: модель не найдена (HEXASTEM_MODEL_PATH или .models/onnx/)");
        return;
    };
    if std::env::var("ORT_DYLIB_PATH").map_or(true, |v| v.is_empty()) {
        eprintln!("SKIP: ORT_DYLIB_PATH не задан");
        return;
    }

    match OrtSession::new(&model, Ep::Cuda { device_id: 0 }) {
        // RTX-машина: CUDA успешен — сценарий фолбэка здесь неприменим.
        Ok(_) => eprintln!("SKIP: CUDA-сессия успешна — фолбэк-сценарий неприменим"),
        Err(cuda_err) => {
            eprintln!("CUDA EP отклонён: {cuda_err}; пробую CPU-фолбэк (тот же процесс)");
            let _session =
                OrtSession::new(&model, Ep::Cpu).expect("CPU-сессия после провала CUDA");
            eprintln!("cpu fallback: сессия построена после провала CUDA — OK");
        }
    }
}
