//! DSP-ядро нативного движка разделения (без Python).
//!
//! Конвейер чанка (эталон — elicwhite/bs-roformer-web, см.
//! `docs/tasks/2026-10-14-native-rust-engine.md`):
//! WAV → (ресемпл в 44.1k) → чанки 176400 → STFT → ONNX → iSTFT →
//! кроссфейд-склейка → (ресемпл обратно) → WAV/FLAC.
//!
//! Модули независимы и чистые: `wav` (I/O), `resample` (rubato),
//! `stft` (rustfft), `chunker` (нарезка/склейка).

pub mod chunker;
pub mod native;
pub mod ort_session;
pub mod resample;
pub mod stft;
pub mod wav;

/// Длина чанка: 4 c при 44.1 kHz (pipeline.js:19).
pub const CHUNK_SAMPLES: usize = 176_400;
/// Шаг чанка: 75% перекрытия 25% (pipeline.js:20).
pub const STEP_SAMPLES: usize = 132_300;
/// Перекрытие соседних чанков (25%).
pub const OVERLAP_SAMPLES: usize = 44_100;
/// Размер FFT (pipeline.js:20).
pub const N_FFT: usize = 2_048;
/// Хоп STFT (pipeline.js:21).
pub const HOP_LENGTH: usize = 512;
/// Длина окна = N_FFT (pipeline.js:22).
pub const WIN_LENGTH: usize = 2_048;
/// Частотных бинов: N_FFT/2 + 1.
pub const SPEC_BINS: usize = 1_025;
/// Временных кадров STFT: CHUNK/HOP + 1 (RoPE зафиксирован на T=345).
pub const SPEC_FRAMES: usize = 345;
/// Частота дискретизации модели.
pub const MODEL_SAMPLE_RATE: u32 = 44_100;
/// Число стемов.
pub const NUM_STEMS: usize = 6;
/// Порядок стемов на оси 1 выхода графа (pipeline.js:24).
pub const STEM_NAMES: [&str; NUM_STEMS] = ["bass", "drums", "other", "vocals", "guitar", "piano"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_layout_is_consistent() {
        assert_eq!(CHUNK_SAMPLES - STEP_SAMPLES, OVERLAP_SAMPLES);
        assert_eq!(CHUNK_SAMPLES / HOP_LENGTH + 1, SPEC_FRAMES);
        assert_eq!(N_FFT / 2 + 1, SPEC_BINS);
        assert_eq!(WIN_LENGTH, N_FFT);
    }
}
