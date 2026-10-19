//! STFT/iSTFT: periodic hann 2048/512, reflect-паддинг (center=True), OLA.
//!
//! T7. Эталон: `torch.stft(center=True, normalized=False)` — reflect-паддинг
//! N_FFT/2 по краям чанка, периодическое окно hann `0.5*(1-cos(2*pi*i/2048))`
//! (знаменатель N, не N-1); iSTFT — OLA по окну², нормировка огибающей с
//! порогом 1e-8, срез [1024, 1024+176400) из padded 178176.
//!
//! Лейаут спектра — freq-major C-порядок тензора `[1025, 345]`:
//! `index = bin * SPEC_FRAMES + frame` (байтово совпадает с входом/выходом
//! ONNX-графа `[1, C, 1025, 345]` — zero-copy без перестановок).
//!
//! Нормировка обратного FFT: rustfft 6.4.1 НЕ нормирует выход (см. доки
//! `Normalization` в его lib.rs), поэтому множитель `1/N_FFT` применяется
//! вручную на кадр (эквивалент `windowInvN` JS-эталона stft-worker.js).

use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

use crate::error::AppError;

use super::{CHUNK_SAMPLES, HOP_LENGTH, N_FFT, SPEC_BINS, SPEC_FRAMES};

/// Полуширина center-паддинга (N_FFT/2).
const PAD: usize = N_FFT / 2;
/// Длина overlap-add буфера: (T-1)*hop + n_fft = 344*512 + 2048 = 178176.
const PADDED_LEN: usize = (SPEC_FRAMES - 1) * HOP_LENGTH + N_FFT;
/// Порог огибающей OLA: ниже — множитель 0 (stft-worker.js:138).
const ENV_THRESHOLD: f32 = 1e-8;

/// STFT/iSTFT чанка 176400 сэмплов (torch.stft/istft center=True).
///
/// Планы FFT и окно фиксированы константами движка — структура дорога в
/// создании и дешева в клонировании `Arc`; держать одну на пайплайн.
pub struct Stft {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// 1/Σₜ window²[(i - t*hop)⁺] по padded-буферу, 0 ниже порога.
    inv_window_sum: Vec<f32>,
}

impl Default for Stft {
    fn default() -> Self {
        Self::new()
    }
}

impl Stft {
    /// Планирует FFT-2048 и предвычисляет periodic-hann + огибающую OLA.
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(N_FFT);
        let inverse = planner.plan_fft_inverse(N_FFT);
        // periodic hann: знаменатель N, не N-1 (совпадает с torch.hann_window).
        let window: Vec<f32> = (0..N_FFT)
            .map(|i| {
                (0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / N_FFT as f64).cos())) as f32
            })
            .collect();
        let mut window_sum = vec![0.0f32; PADDED_LEN];
        for t in 0..SPEC_FRAMES {
            for (i, &w) in window.iter().enumerate() {
                window_sum[t * HOP_LENGTH + i] += w * w;
            }
        }
        let inv_window_sum = window_sum
            .iter()
            .map(|&s| if s > ENV_THRESHOLD { 1.0 / s } else { 0.0 })
            .collect();
        Self {
            forward,
            inverse,
            window,
            inv_window_sum,
        }
    }

    /// STFT канала чанка → (real, imag), по `[SPEC_BINS * SPEC_FRAMES]`
    /// в freq-major лейауте (`bin * SPEC_FRAMES + frame`).
    pub fn forward(&self, chunk: &[f32]) -> Result<(Vec<f32>, Vec<f32>), AppError> {
        if chunk.len() != CHUNK_SAMPLES {
            return Err(AppError::Dsp(format!(
                "stft forward: ожидался чанк {CHUNK_SAMPLES} сэмплов, получено {}",
                chunk.len()
            )));
        }
        let mut real = vec![0.0f32; SPEC_BINS * SPEC_FRAMES];
        let mut imag = vec![0.0f32; SPEC_BINS * SPEC_FRAMES];
        let mut frame = vec![Complex::new(0.0f32, 0.0); N_FFT];
        for t in 0..SPEC_FRAMES {
            for (i, slot) in frame.iter_mut().enumerate() {
                // frameStart = t*hop - n_fft/2 может выходить за края → reflect.
                let src = reflect_index((t as isize) * HOP_LENGTH as isize + i as isize - PAD as isize);
                *slot = Complex::new(chunk[src] * self.window[i], 0.0);
            }
            self.forward.process(&mut frame);
            for (k, x) in frame.iter().take(SPEC_BINS).enumerate() {
                real[k * SPEC_FRAMES + t] = x.re;
                imag[k * SPEC_FRAMES + t] = x.im;
            }
        }
        Ok((real, imag))
    }

    /// iSTFT спектра (freq-major, как `forward`) в `out` (176400 сэмплов):
    /// полный спектр 2048 (зеркальные конъюгаты), ifft без нормировки rustfft
    /// → множитель window/2048, OLA, деление на огибающую, срез центра.
    pub fn inverse(
        &self,
        spec_real: &[f32],
        spec_imag: &[f32],
        out: &mut [f32],
    ) -> Result<(), AppError> {
        let spec_len = SPEC_BINS * SPEC_FRAMES;
        if spec_real.len() != spec_len || spec_imag.len() != spec_len {
            return Err(AppError::Dsp(format!(
                "stft inverse: ожидался спектр {spec_len} значений real/imag, получено {}/{}",
                spec_real.len(),
                spec_imag.len()
            )));
        }
        if out.len() != CHUNK_SAMPLES {
            return Err(AppError::Dsp(format!(
                "stft inverse: ожидался выход {CHUNK_SAMPLES} сэмплов, получено {}",
                out.len()
            )));
        }
        let mut padded = vec![0.0f32; PADDED_LEN];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); N_FFT];
        for t in 0..SPEC_FRAMES {
            // Бин 0 и Nyquist (1024) вещественные; бины 1..1023 — с зеркальными
            // конъюгатами в 2047..1025. Мнимая часть DC/Nyquist игнорируется
            // (как в JS-эталоне: там лежит только re).
            spectrum[0] = Complex::new(spec_real[t], 0.0);
            for k in 1..SPEC_BINS - 1 {
                let re = spec_real[k * SPEC_FRAMES + t];
                let im = spec_imag[k * SPEC_FRAMES + t];
                spectrum[k] = Complex::new(re, im);
                spectrum[N_FFT - k] = Complex::new(re, -im);
            }
            spectrum[SPEC_BINS - 1] = Complex::new(spec_real[(SPEC_BINS - 1) * SPEC_FRAMES + t], 0.0);
            self.inverse.process(&mut spectrum);
            // rustfft НЕ нормирует ifft — 1/N_FFT вручную, вместе с окном.
            for (i, x) in spectrum.iter().enumerate() {
                padded[t * HOP_LENGTH + i] += x.re * self.window[i] / N_FFT as f32;
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = padded[PAD + i] * self.inv_window_sum[PAD + i];
        }
        Ok(())
    }
}

/// torch-отражение индекса от края сигнала (период 2*(len-1)),
/// stft-worker.js:29-35; корректно для i в диапазоне кадра.
fn reflect_index(i: isize) -> usize {
    let len = CHUNK_SAMPLES as isize;
    let period = 2 * (len - 1);
    let i = i.rem_euclid(period);
    if i < len {
        i as usize
    } else {
        (period - i) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_periodic_hann() {
        let stft = Stft::new();
        assert_eq!(stft.window.len(), N_FFT);
        assert_eq!(stft.window[0], 0.0);
        // periodic (деление на N, не N-1): w[2047] ≈ 4.7e-6 > 0 — не symmetric hann.
        assert!(stft.window[N_FFT - 1] > 0.0, "w[2047] должен быть > 0");
        assert!(stft.window[N_FFT - 1] < 1e-5);
        assert!((stft.window[N_FFT / 2] - 1.0).abs() < 1e-6);
        assert!((stft.window[N_FFT / 4] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_reconstructs_chunk_center() {
        let stft = Stft::new();
        let chunk: Vec<f32> = (0..CHUNK_SAMPLES)
            .map(|i| (i as f32 * 0.01).sin() * 0.5 + (i as f32 * 0.0003).cos() * 0.3)
            .collect();
        let (real, imag) = stft.forward(&chunk).expect("forward ok");
        let mut out = vec![0.0f32; CHUNK_SAMPLES];
        stft.inverse(&real, &imag, &mut out).expect("inverse ok");
        // Края (±N_FFT) не проверяем; центр обязан восстановиться точно.
        let edge = N_FFT;
        let max_diff = chunk[edge..CHUNK_SAMPLES - edge]
            .iter()
            .zip(&out[edge..CHUNK_SAMPLES - edge])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "roundtrip max abs diff = {max_diff}");
    }

    #[test]
    fn inv_window_sum_is_positive_over_output_region() {
        // В срезе результата [1024, 1024+176400) огибающая OLA обязана превышать
        // порог 1e-8, иначе на краях чанка выход обнулится.
        let stft = Stft::new();
        for (i, &inv) in stft
            .inv_window_sum
            .iter()
            .enumerate()
            .take(PAD + CHUNK_SAMPLES)
            .skip(PAD)
        {
            assert!(inv > ENV_THRESHOLD && inv.is_finite(), "нулевая огибающая при i={i}");
        }
    }

    #[test]
    fn forward_rejects_wrong_length() {
        let stft = Stft::new();
        let err = stft.forward(&vec![0.0; 100]).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
    }

    #[test]
    fn inverse_rejects_wrong_lengths() {
        let stft = Stft::new();
        let spec_len = SPEC_BINS * SPEC_FRAMES;
        let mut out = vec![0.0f32; CHUNK_SAMPLES];
        let err = stft
            .inverse(&vec![0.0; spec_len], &vec![0.0; spec_len - 1], &mut out)
            .unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
        let err = stft
            .inverse(&vec![0.0; spec_len], &vec![0.0; spec_len], &mut [0.0; 10])
            .unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
    }
}
