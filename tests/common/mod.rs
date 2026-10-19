//! Общий загрузчик golden-фикстур `reference.npz` для интеграционных тестов.
//!
//! Файл лежит в `.models/` (gitignore) и скачивается с HF
//! (elicwhite/bs-roformer-sw-6stem-onnx); при отсутствии тесты пропускаются.

#![allow(dead_code)]

use ndarray::{Array, Ix3, Ix4, Ix5, OwnedRepr};
use ndarray_npy::NpzReader;
use std::path::{Path, PathBuf};

/// Путь к reference.npz относительно корня крейта.
pub fn reference_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".models/onnx/reference.npz")
}

/// True, если фикстуры недоступны (тест должен пропуститься с предупреждением).
pub fn reference_missing() -> bool {
    let p = reference_path();
    if p.exists() {
        return false;
    }
    eprintln!("SKIP: {} not found (скачай с HF elicwhite/bs-roformer-sw-6stem-onnx)", p.display());
    true
}

/// Загруженные массивы reference.npz (все f32, кроме скаляра `samples`).
pub struct Reference {
    /// [1, 2, 176400] — входной чанк.
    pub input_audio: Array<f32, Ix3>,
    /// [1, 2, 1025, 345] — STFT real (PyTorch).
    pub spec_real: Array<f32, Ix4>,
    /// [1, 2, 1025, 345] — STFT imag (PyTorch).
    pub spec_imag: Array<f32, Ix4>,
    /// [1, 6, 2, 1025, 345] — выход графа real.
    pub out_spec_real: Array<f32, Ix5>,
    /// [1, 6, 2, 1025, 345] — выход графа imag.
    pub out_spec_imag: Array<f32, Ix5>,
    /// [1, 6, 2, 176400] — istft(out_real, out_imag) по PyTorch, по стему.
    pub ref_audio: Array<f32, Ix4>,
    /// Скаляр 176400 (длина валидной части ref_audio).
    pub samples: i64,
}

impl Reference {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = reference_path();
        let mut npz = NpzReader::new(std::fs::File::open(&path)?)?;
        Ok(Self {
            input_audio: npz.by_name::<OwnedRepr<f32>, Ix3>("input_audio")?,
            spec_real: npz.by_name::<OwnedRepr<f32>, Ix4>("spec_real")?,
            spec_imag: npz.by_name::<OwnedRepr<f32>, Ix4>("spec_imag")?,
            out_spec_real: npz.by_name::<OwnedRepr<f32>, Ix5>("out_spec_real")?,
            out_spec_imag: npz.by_name::<OwnedRepr<f32>, Ix5>("out_spec_imag")?,
            ref_audio: npz.by_name::<OwnedRepr<f32>, Ix4>("ref_audio")?,
            samples: npz.by_name::<OwnedRepr<i64>, ndarray::Ix0>("samples")?.into_scalar(),
        })
    }
}
