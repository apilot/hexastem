//! Ресемплинг: 48k→44.1k и 44.1k→48k (rubato 5), passthrough для 44.1k.
//!
//! T6. Стемы пишутся в sample rate источника (решение владельца):
//! down 44.1k → модель → up обратно. 96k — опционально.

use crate::error::AppError;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};

/// Желаемый размер чанка обработки, кадров (рекомендация документации rubato:
/// саб-чанки ~256 кадров дают малую задержку при хорошем качестве фильтра).
const FFT_CHUNK_FRAMES: usize = 1024;

/// Поддерживаемые частоты дискретизации.
const SUPPORTED_RATES: [u32; 2] = [44_100, 48_000];

/// Ресемплирует планарный стерео-буфер из `src_rate` в `dst_rate` (офлайн, весь буфер).
///
/// Контракт:
/// - возвращает 2 канала f32 (планарно);
/// - длина результата — `ceil(n * dst / src)` (целочисленный ceil; лишний кадр
///   от fp-арифметики rubato обрезается, см. тело функции);
/// - `src == dst` → passthrough (побитовое клонирование, без обработки);
/// - поддерживаются 44100 и 48000 Гц; прочие (включая 96k, даже при равенстве
///   частот) → `AppError::Dsp`;
/// - внутренняя задержка ресемплера компенсируется автоматически: `process_all`
///   срезает `output_delay` кадров начала, поэтому выход фазово совпадает с входом.
pub fn resample_stereo(
    channels: &[Vec<f32>; 2],
    src_rate: u32,
    dst_rate: u32,
) -> Result<Vec<Vec<f32>>, AppError> {
    for rate in [src_rate, dst_rate] {
        if !SUPPORTED_RATES.contains(&rate) {
            return Err(AppError::Dsp(format!(
                "resample_stereo: частота {rate} Гц не поддерживается (допустимо 44100/48000)"
            )));
        }
    }

    // Passthrough: побитовое клонирование без обработки.
    if src_rate == dst_rate {
        return Ok(channels.to_vec());
    }

    let n = channels[0].len();
    if channels[1].len() != n {
        return Err(AppError::Dsp(format!(
            "resample_stereo: длины каналов расходятся ({} и {})",
            channels[0].len(),
            channels[1].len()
        )));
    }
    if n == 0 {
        return Ok(vec![Vec::new(), Vec::new()]);
    }

    // FFT-ресемплер rubato: окно BlackmanHarris2, саб-чанки ~256 кадров
    // (малая задержка при хорошем подавлении алиасинга). Режим FixedSync::Input —
    // офлайн-обработка фиксированными входными чанками с переменным выходом.
    let mut resampler = Fft::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        FFT_CHUNK_FRAMES,
        2,
        FixedSync::Input,
    )
    .map_err(|e| AppError::Dsp(format!("resample_stereo: создание ресемплера: {e}")))?;

    let input = SequentialSliceOfVecs::new(channels, 2, n)
        .map_err(|e| AppError::Dsp(format!("resample_stereo: входной буфер: {e}")))?;

    // process_all перекачивает весь буфер чанками и срезает внутреннюю задержку
    // (output_delay) — компенсация фазы выполняется самим rubato.
    let out = resampler
        .process_all(&input, n, None)
        .map_err(|e| AppError::Dsp(format!("resample_stereo: обработка: {e}")))?;

    // Де-интерлив: layout InterleavedOwned — data[кадр * 2 + канал].
    // rubato возвращает fp-ceil(ratio * n); из-за ошибки f64 (например,
    // 160/147 * 44100) это бывает на 1 кадр больше целочисленного ceil —
    // лишний хвост накачки тишиной обрезаем до точного ceil(n * dst / src).
    let want = expected_len(n, src_rate, dst_rate);
    let interleaved = out.take_data();
    if interleaved.len() / 2 < want {
        return Err(AppError::Dsp(format!(
            "resample_stereo: rubato вернул {} кадров, требуется минимум {want}",
            interleaved.len() / 2
        )));
    }
    let interleaved = &interleaved[..want * 2];

    // Отдельная аллокация на канал: vec![x; 2] клонировал бы Vec без capacity.
    let mut result: Vec<Vec<f32>> = (0..2).map(|_| Vec::with_capacity(want)).collect();
    for (ch, dst) in result.iter_mut().enumerate() {
        dst.extend(interleaved[ch..].iter().step_by(2).copied());
    }
    Ok(result)
}

/// Ожидаемая длина результата: ceil(n * dst / src) в целочисленной арифметике.
fn expected_len(n: usize, src: u32, dst: u32) -> usize {
    (n as u64 * dst as u64).div_ceil(src as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Синус `freq` Гц амплитуды `amp` на сетке частоты `rate`, `n` отсчётов.
    fn sine(rate: u32, n: usize, amp: f32, freq: f64) -> Vec<f32> {
        (0..n)
            .map(|i| (amp as f64 * (TAU as f64 * freq * i as f64 / rate as f64).sin()) as f32)
            .collect()
    }

    /// Планарный стерео-буфер из одинаковых моно-дорожек.
    fn stereo(mono: Vec<f32>) -> [Vec<f32>; 2] {
        [mono.clone(), mono]
    }

    /// SNR (дБ) факта против эталона. Края обрезаются по `edge` отсчёта с обеих
    /// сторон (переходные процессы фильтра), среди целочисленных сдвигов
    /// ±`max_shift` берётся лучший (выравнивание фазы). Возвращает (SNR, сдвиг).
    fn best_snr(actual: &[f32], reference: &[f32], edge: usize, max_shift: isize) -> (f64, isize) {
        let n = actual.len().min(reference.len()) as isize;
        let mut best = (f64::NEG_INFINITY, 0isize);
        for shift in -max_shift..=max_shift {
            let mut sig_sq = 0.0f64;
            let mut err_sq = 0.0f64;
            for i in edge as isize..(n - edge as isize) {
                let a = actual[i as usize] as f64;
                let r = reference[(i - shift) as usize] as f64;
                sig_sq += a * a;
                err_sq += (a - r) * (a - r);
            }
            let snr = 10.0 * (sig_sq / err_sq).log10();
            if snr > best.0 {
                best = (snr, shift);
            }
        }
        best
    }

    #[test]
    fn passthrough_44100_is_bitwise_identical() {
        let input = stereo(sine(44_100, 1000, 0.5, 1000.0));
        let input_copy = input.clone();
        let out = resample_stereo(&input, 44_100, 44_100).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], input_copy[0]);
        assert_eq!(out[1], input_copy[1]);
    }

    #[test]
    fn downsample_48k_to_44k1_snr_at_least_60db() {
        let n = 48_000;
        let src = stereo(sine(48_000, n, 0.5, 1000.0));
        let out = resample_stereo(&src, 48_000, 44_100).unwrap();
        let want = expected_len(n, 48_000, 44_100);
        assert_eq!(out[0].len(), want);
        assert_eq!(out[1].len(), want);
        let reference = sine(44_100, want, 0.5, 1000.0);
        for (ch, channel) in out.iter().enumerate() {
            let (snr, shift) = best_snr(channel, &reference, 64, 4);
            assert!(
                snr >= 60.0,
                "channel {ch}: SNR {snr:.2} dB < 60 dB (shift {shift})"
            );
        }
    }

    #[test]
    fn upsample_44k1_to_48k_snr_at_least_60db() {
        let n = 44_100;
        let src = stereo(sine(44_100, n, 0.5, 1000.0));
        let out = resample_stereo(&src, 44_100, 48_000).unwrap();
        let want = expected_len(n, 44_100, 48_000);
        assert_eq!(out[0].len(), want);
        assert_eq!(out[1].len(), want);
        let reference = sine(48_000, want, 0.5, 1000.0);
        for (ch, channel) in out.iter().enumerate() {
            let (snr, shift) = best_snr(channel, &reference, 64, 4);
            assert!(
                snr >= 60.0,
                "channel {ch}: SNR {snr:.2} dB < 60 dB (shift {shift})"
            );
        }
    }

    #[test]
    fn roundtrip_48_to_44k1_to_48_length_and_snr() {
        let n = 48_000;
        let src = stereo(sine(48_000, n, 0.5, 1000.0));
        let down = resample_stereo(&src, 48_000, 44_100).unwrap();
        let arr: [Vec<f32>; 2] = [down[0].clone(), down[1].clone()];
        let up = resample_stereo(&arr, 44_100, 48_000).unwrap();
        assert_eq!(up[0].len(), n);
        assert_eq!(up[1].len(), n);
        for ch in 0..2 {
            let (snr, shift) = best_snr(&up[ch], &src[ch], 64, 4);
            assert!(
                snr >= 55.0,
                "channel {ch}: roundtrip SNR {snr:.2} dB < 55 dB (shift {shift})"
            );
        }
    }

    #[test]
    fn unsupported_rate_returns_dsp_error() {
        let src = stereo(sine(96_000, 100, 0.5, 1000.0));
        let err = resample_stereo(&src, 96_000, 44_100).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
        // 96k не поддерживается даже при равенстве частот (passthrough только
        // для 44.1k/48k).
        let err = resample_stereo(&src, 96_000, 96_000).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
    }

    #[test]
    fn mismatched_channel_lengths_return_dsp_error() {
        let src = [vec![0.0f32; 100], vec![0.0f32; 99]];
        let err = resample_stereo(&src, 48_000, 44_100).unwrap_err();
        assert!(matches!(err, AppError::Dsp(_)));
    }
}
