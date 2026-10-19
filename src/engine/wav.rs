//! WAV I/O для нативного движка: чтение в планарное стерео f32, запись PCM 16/24.
//!
//! Чтение ([read_stereo_f32]) — стриминговое, итератором сэмплов `hound`
//! (файл не грузится сырыми байтами, как и в `dedup.rs`). Поддержка:
//! PCM int 16/24/32 бит и IEEE float 32; mono дублируется в стерео,
//! больше двух каналов — [`AppError::Dsp`]. Нормировка целых — деление на
//! максимум знакового диапазона (−32768 → −1.0, 32767 → +0.99997), как в
//! `decode_samples` из `dedup.rs`; float проходит как есть, без клипа.
//!
//! Запись ([write_stereo_f32]) — PCM int 16/24: сэмплы клипуются к [−1..1],
//! масштабируются к максимуму положительного диапазона (32767 / 8 388 607)
//! и округляются к ближайшему целому (`f32::round`, половинки — от нуля);
//! −1.0 даёт −32767, минимум диапазона недостижим (асимметрия как в
//! фиксурах `dedup.rs`) — зато +1.0 попадает ровно в потолок без клипа.

use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::error::AppError;

/// Разрядность PCM при записи WAV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavBits {
    /// 16 бит на сэмпл.
    I16,
    /// 24 бита на сэмпл.
    I24,
}

impl WavBits {
    /// Бит на сэмпл для `WavSpec`.
    fn bits_per_sample(self) -> u16 {
        match self {
            WavBits::I16 => 16,
            WavBits::I24 => 24,
        }
    }

    /// Максимум положительного знакового диапазона (масштаб при записи).
    fn full_scale(self) -> f32 {
        match self {
            WavBits::I16 => 32_767.0,
            WavBits::I24 => 8_388_607.0,
        }
    }
}

/// Читает WAV в планарное стерео f32 (−1..1): (частота, фреймы [L, R]).
/// Поддержка: PCM int 16/24/32 бит, IEEE float 32; mono → дубликат в оба
/// канала, прочее число каналов и разрядность — [`AppError::Dsp`].
/// Неполный хвостовой фрейм битого файла отбрасывается (как в `dedup.rs`).
pub fn read_stereo_f32(path: &Path) -> Result<(u32, Vec<[f32; 2]>), AppError> {
    let mut reader = WavReader::open(path).map_err(|e| wav_err(path, e))?;
    let spec = reader.spec();
    if !matches!(spec.channels, 1 | 2) {
        return Err(AppError::Dsp(format!(
            "`{}`: {} каналов, поддерживаются только моно и стерео",
            path.display(),
            spec.channels
        )));
    }
    if spec.sample_format == SampleFormat::Int && !matches!(spec.bits_per_sample, 16 | 24 | 32) {
        return Err(AppError::Dsp(format!(
            "`{}`: PCM {} бит не поддерживается",
            path.display(),
            spec.bits_per_sample
        )));
    }
    let frames = match spec.sample_format {
        // 24-бит читается как i32 со знаковым расширением (API hound).
        SampleFormat::Int => {
            collect_frames(reader.samples::<i32>(), spec.channels, int_to_f32(spec.bits_per_sample))
        }
        SampleFormat::Float => collect_frames(reader.samples::<f32>(), spec.channels, |v| v),
    }
    .map_err(|e| wav_err(path, e))?;
    Ok((spec.sample_rate, frames))
}

/// Сборка чередующихся сэмплов в фреймы [L, R]; mono дублируется в оба канала.
fn collect_frames<S>(
    samples: impl Iterator<Item = hound::Result<S>>,
    channels: u16,
    to_f32: impl Fn(S) -> f32,
) -> Result<Vec<[f32; 2]>, hound::Error> {
    let mut frames = Vec::new();
    let mut pending = [0.0_f32; 2];
    let mut filled = 0_usize;
    for sample in samples {
        pending[filled] = to_f32(sample?);
        filled += 1;
        if filled == channels as usize {
            // Mono дублируется в оба канала явно: pending переиспользуется,
            // и без этого R остался бы от прошлого фрейма.
            if channels == 1 {
                pending[1] = pending[0];
            }
            frames.push(pending);
            filled = 0;
        }
    }
    Ok(frames)
}

/// Нормировка целого сэмплa 16/24/32 бит в −1..1 делением на 2^(бит−1):
/// −32768 → −1.0, максимум диапазона → чуть меньше +1.0.
fn int_to_f32(bits: u16) -> impl Fn(i32) -> f32 {
    let full = (1_u32 << (bits - 1)) as f32;
    move |v| v as f32 / full
}

/// Пишет стерео PCM WAV 16/24 бит. Каждый сэмпл клипуется к [−1..1],
/// масштабируется к максимуму диапазона ([`WavBits`]) и округляется к
/// ближайшему целому (половинки — от нуля).
pub fn write_stereo_f32(
    path: &Path,
    sample_rate: u32,
    bits: WavBits,
    samples: &[[f32; 2]],
) -> Result<(), AppError> {
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: bits.bits_per_sample(),
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).map_err(|e| wav_err(path, e))?;
    let full = bits.full_scale();
    for frame in samples {
        for &channel in frame {
            let scaled = channel.clamp(-1.0, 1.0) * full;
            // 24-бит пишется i32: значения уже полного диапазона (API hound).
            match bits {
                WavBits::I16 => writer.write_sample(scaled.round() as i16),
                WavBits::I24 => writer.write_sample(scaled.round() as i32),
            }
            .map_err(|e| wav_err(path, e))?;
        }
    }
    writer.finalize().map_err(|e| wav_err(path, e))?;
    Ok(())
}

/// Маппинг ошибки `hound`: I/O — в [`AppError::Io`], формат и содержимое —
/// в [`AppError::Dsp`].
fn wav_err(path: &Path, err: hound::Error) -> AppError {
    match err {
        hound::Error::IoError(source) => AppError::Io { path: path.to_path_buf(), source },
        other => AppError::Dsp(format!("`{}`: wav: {other}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use hound::{SampleFormat, WavSpec, WavWriter};

    use super::*;

    /// Счётчик уникальных имён фикстур (тесты в крейте идут параллельно).
    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

    /// Путь временной фикстуры: pid + счётчик — параллельные тесты не пересекаются.
    fn temp_wav(tag: &str) -> PathBuf {
        let n = NEXT_FIXTURE.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir()
            .join(format!("hexastem_wav_{}_{}_{tag}.wav", std::process::id(), n))
    }

    /// Удаление фикстуры в конце теста (best effort).
    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    /// Фикстура int-WAV, записанная hound'ом напрямую (не через тестируемый код).
    fn write_int_wav(path: &Path, channels: u16, bits: u16, samples: &[i32]) {
        let spec = WavSpec {
            channels,
            sample_rate: 44_100,
            bits_per_sample: bits,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).expect("fixture wav");
        for &s in samples {
            writer.write_sample(s).expect("fixture sample");
        }
        writer.finalize().expect("fixture finalize");
    }

    /// Фикстура IEEE float WAV.
    fn write_float_wav(path: &Path, channels: u16, samples: &[f32]) {
        let spec = WavSpec {
            channels,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).expect("fixture wav");
        for &s in samples {
            writer.write_sample(s).expect("fixture sample");
        }
        writer.finalize().expect("fixture finalize");
    }

    // --- roundtrip: write → read -------------------------------------------------

    #[test]
    fn roundtrip_i16_keeps_quantum_steps_and_clips() {
        let path = temp_wav("rt16");
        let step = 1.0 / 32_767.0;
        let input = [0.0_f32, step, -step, 0.25, -0.25, 1.0, -1.0, 1.5, -1.5];
        let frames: Vec<[f32; 2]> = input.iter().map(|&v| [v, -v]).collect();

        write_stereo_f32(&path, 44_100, WavBits::I16, &frames).expect("write");
        let (rate, read) = read_stereo_f32(&path).expect("read");
        cleanup(&path);

        assert_eq!(rate, 44_100);
        assert_eq!(read.len(), input.len());
        // Допуск 1.5 кванта чтения: полкванта — округление при записи,
        // ещё квант — асимметрия нормировки (потолок 32767/32768, не 1.0).
        let tol = 1.5 / 32_768.0;
        for (i, (frame, &v)) in read.iter().zip(&input).enumerate() {
            let expect = v.clamp(-1.0, 1.0);
            assert!(
                (frame[0] - expect).abs() <= tol,
                "фрейм {i} L: {} против {}",
                frame[0],
                expect
            );
            assert!(
                (frame[1] + expect).abs() <= tol,
                "фрейм {i} R: {} против {}",
                frame[1],
                -expect
            );
        }
        // Клип к потолку: 1.5 → ровно максимум 16-битного диапазона.
        // Фрейм = [v, −v]: фрейм 7 (v=+1.5) клипует L в плюс, фрейм 8
        // (v=−1.5) клипует L в минус, а его R=+1.5 — в плюс.
        assert_eq!(read[7][0], 32_767.0 / 32_768.0, "клип +1.5 (L)");
        assert_eq!(read[8][0], -32_767.0 / 32_768.0, "клип −1.5 (L)");
        assert_eq!(read[8][1], 32_767.0 / 32_768.0, "клип +1.5 (R = −v)");
    }

    #[test]
    fn roundtrip_i24_keeps_quantum_steps_and_clips() {
        let path = temp_wav("rt24");
        let step = 1.0 / 8_388_607.0;
        let input = [0.0_f32, step, -step, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
        let frames: Vec<[f32; 2]> = input.iter().map(|&v| [v, v]).collect();

        write_stereo_f32(&path, 44_100, WavBits::I24, &frames).expect("write");
        let (rate, read) = read_stereo_f32(&path).expect("read");
        cleanup(&path);

        assert_eq!(rate, 44_100);
        assert_eq!(read.len(), input.len());
        // Допуск 1.5 кванта чтения: полкванта — округление при записи,
        // ещё квант — асимметрия нормировки (потолок 8388607/8388608, не 1.0).
        let tol = 1.5 / 8_388_608.0;
        for (i, (frame, &v)) in read.iter().zip(&input).enumerate() {
            let expect = v.clamp(-1.0, 1.0);
            assert!(
                (frame[0] - expect).abs() <= tol,
                "фрейм {i} L: {} против {}",
                frame[0],
                expect
            );
            assert!(
                (frame[1] - expect).abs() <= tol,
                "фрейм {i} R: {} против {}",
                frame[1],
                expect
            );
        }
        assert_eq!(read[7][0], 8_388_607.0 / 8_388_608.0, "клип +2.0");
        assert_eq!(read[8][0], -8_388_607.0 / 8_388_608.0, "клип −2.0");
    }

    // --- чтение чужих форматов (фикстуры через hound) ----------------------------

    #[test]
    fn reads_32bit_int_wav_normalized() {
        let path = temp_wav("int32");
        // Полнодиапазонные значения: 0, +2^30, −2^30, максимум, минимум.
        let raw = [0_i32, 1 << 30, -(1 << 30), i32::MAX, i32::MIN];
        let interleaved: Vec<i32> =
            raw.iter().flat_map(|&v| [v, v]).collect();
        write_int_wav(&path, 2, 32, &interleaved);

        let (rate, read) = read_stereo_f32(&path).expect("read");
        cleanup(&path);

        assert_eq!(rate, 44_100);
        assert_eq!(read.len(), raw.len());
        let expect = [0.0_f32, 0.5, -0.5, 1.0, -1.0];
        for (i, frame) in read.iter().enumerate() {
            assert!((frame[0] - expect[i]).abs() < 1e-6, "фрейм {i}: {}", frame[0]);
            assert!((frame[1] - expect[i]).abs() < 1e-6, "фрейм {i}: {}", frame[1]);
        }
    }

    #[test]
    fn reads_float32_wav_as_is() {
        let path = temp_wav("f32");
        // Float проходит без клипа — даже за пределами [−1..1].
        let raw = [0.5_f32, -0.75, 1.5, -1.5];
        let interleaved: Vec<f32> = raw.iter().flat_map(|&v| [v, -v]).collect();
        write_float_wav(&path, 2, &interleaved);

        let (rate, read) = read_stereo_f32(&path).expect("read");
        cleanup(&path);

        assert_eq!(rate, 48_000);
        assert_eq!(read.len(), raw.len());
        for (i, frame) in read.iter().enumerate() {
            assert_eq!(frame[0], raw[i], "фрейм {i} L");
            assert_eq!(frame[1], -raw[i], "фрейм {i} R");
        }
    }

    #[test]
    fn mono_wav_is_duplicated_to_stereo() {
        let path = temp_wav("mono");
        let raw = [0_i32, 16_384, -16_384];
        write_int_wav(&path, 1, 16, &raw);

        let (_, read) = read_stereo_f32(&path).expect("read");
        cleanup(&path);

        assert_eq!(read.len(), raw.len());
        for (i, frame) in read.iter().enumerate() {
            let expect = f32::from(raw[i] as i16) / 32_768.0;
            assert_eq!(frame[0], expect, "фрейм {i} L");
            assert_eq!(frame[1], expect, "фрейм {i} R — mono дублируется");
        }
    }

    // --- ошибки ------------------------------------------------------------------

    #[test]
    fn not_a_wav_file_is_a_dsp_error() {
        let path = temp_wav("notwav");
        fs::write(&path, b"this is definitely not a wav file....").expect("fixture");
        let err = read_stereo_f32(&path).expect_err("не-WAV должен дать ошибку");
        cleanup(&path);
        assert!(matches!(err, AppError::Dsp(_)), "{err:?}");
    }

    #[test]
    fn truncated_wav_is_an_error() {
        let path = temp_wav("trunc");
        // Валидный WAV, обрезанный посреди заголовка. Hound оборачивает
        // усечение данных в IoError(kind: Other) — отличить его от системной
        // I/O-ошибки нельзя, поэтому контракт только «любой вариант AppError».
        let full = temp_wav("trunc_full");
        write_int_wav(&full, 2, 16, &[0, 0, 0, 0]);
        let bytes = fs::read(&full).expect("fixture bytes");
        fs::write(&path, &bytes[..30]).expect("truncated fixture");
        let err = read_stereo_f32(&path).expect_err("битый файл должен дать ошибку");
        cleanup(&path);
        cleanup(&full);
        assert!(matches!(err, AppError::Dsp(_) | AppError::Io { .. }), "{err:?}");
    }

    #[test]
    fn more_than_two_channels_are_rejected() {
        let path = temp_wav("quad");
        write_int_wav(&path, 4, 16, &[0_i32; 8]);
        let err = read_stereo_f32(&path).expect_err("4 канала должны дать ошибку");
        cleanup(&path);
        assert!(matches!(err, AppError::Dsp(_)), "{err:?}");
    }
}
