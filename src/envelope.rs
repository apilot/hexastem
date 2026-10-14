//! Полная огибающая громкости PCM WAV по каналам (waveform-прогресс в TUI).
//!
//! Потоковое чтение поверх RIFF-парсера [`crate::dedup::parse_wav`]: файл
//! НЕ читается целиком — записи рекордера ~600 МБ, в памяти живут только
//! блок чтения (~сотни КБ) и результат O(columns). Пик колонки канала —
//! max |сэмпл| этого канала по фреймам колонки; нормировка обеспечивается
//! декодом PCM (−1..1 полной шкалы), дополнительной перенормировки нет:
//! высота колонки отражает абсолютную громкость файла.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::dedup::{parse_wav, read_samples};

/// Размер блока чтения (сек): сотни КБ памяти, не весь файл.
const READ_BLOCK_SECS: u64 = 1;

/// Ошибка вычисления огибающей.
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    /// Файл не PCM WAV (не RIFF/WAVE, нет `data`, float-формат, 8 бит…).
    #[error("not a PCM WAV file")]
    NotPcm,
    /// Ошибка чтения файла (открытие, обрыв данных, seek).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Огибающая пиков по каналам: `columns` значений 0..1 на канал.
///
/// Стерео-карта TUI («в зеркале», решение владельца 2026-10-14): `left`
/// рисуется вверх от центра карты, `right` — вниз. Моно-файл (парсер
/// принимает 1–2 канала) заполняет оба канала одинаково — единственный
/// канал является и L, и R, «пустой» половины карты нет.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StereoEnvelope {
    /// Пики левого канала.
    pub left: Vec<f32>,
    /// Пики правого канала.
    pub right: Vec<f32>,
}

/// Полная огибающая пиков локального файла: `columns` значений 0..1 на канал.
///
/// Колонка `i` — пик фреймов `[i·frames/columns, (i+1)·frames/columns)`;
/// остаток от деления распределяется формулой `i = frame·columns/frames`,
/// поэтому все колонки заполнены при любой длине. Файл короче `columns`
/// фреймов → хвостовые колонки 0.0 (без паники). `columns == 0` → пустые
/// каналы. Ошибки: не PCM WAV → [`EnvelopeError::NotPcm`], I/O → `Io`.
pub fn wav_stereo_envelope(path: &Path, columns: usize) -> Result<StereoEnvelope, EnvelopeError> {
    let mut file = File::open(path)?;
    stereo_envelope_from_reader(&mut file, columns)
}

/// Огибающая из произвольного `Read + Seek` (тесты — на синтетике в памяти).
pub fn stereo_envelope_from_reader<R: Read + Seek>(
    reader: &mut R,
    columns: usize,
) -> Result<StereoEnvelope, EnvelopeError> {
    if columns == 0 {
        return Ok(StereoEnvelope::default());
    }
    let Some(info) = parse_wav(reader) else {
        return Err(EnvelopeError::NotPcm);
    };
    let mut envelope =
        StereoEnvelope { left: vec![0.0_f32; columns], right: vec![0.0_f32; columns] };
    let channels = usize::from(info.channels);
    reader.seek(SeekFrom::Start(info.data_offset))?;
    let (mut raw, mut samples) = (Vec::new(), Vec::new());
    let (mut frame, mut left) = (0_u64, info.frames);
    while left > 0 {
        let frames = (READ_BLOCK_SECS * u64::from(info.sample_rate)).min(left);
        read_samples(reader, frames * info.block_align(), info.bits, &mut raw, &mut samples)?;
        left -= frames;
        // Канал 0 → L, канал 1 → R (многоканальные: front L/R); моно —
        // один канал в обе стороны. Колонка — равномерное распределение
        // фреймов по колонкам (остаток не сбивается в конец).
        for chunk in samples.chunks(channels) {
            let column = frame * columns as u64 / info.frames;
            let l_peak = chunk.first().map(|s| s.abs()).unwrap_or_default();
            let index = column as usize;
            envelope.left[index] = envelope.left[index].max(l_peak);
            let r_peak = if channels == 1 {
                l_peak
            } else {
                chunk.get(1).map(|s| s.abs()).unwrap_or_default()
            };
            envelope.right[index] = envelope.right[index].max(r_peak);
            frame += 1;
        }
    }
    Ok(envelope)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::dedup::wav_fix;

    /// Уникальная локальная директория теста (параллельные не пересекаются).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("audio_split_envelope_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Пик колонки с допуском на шум квантования 16 бит (~3e-5).
    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
    }

    #[test]
    fn constant_amplitude_sets_column_height() {
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &vec![0.5; 8_000 * 2]);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 8).expect("valid wav");

        assert_eq!(envelope.left.len(), 8);
        assert_eq!(envelope.right.len(), 8);
        for peak in envelope.left.iter().chain(&envelope.right) {
            assert_close(*peak, 0.5);
        }
    }

    #[test]
    fn silence_gives_zero_columns() {
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &vec![0.0; 8_000 * 2]);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 16).expect("valid wav");

        assert_eq!(envelope.left.len(), 16);
        assert!(envelope.left.iter().chain(&envelope.right).all(|p| *p == 0.0));
    }

    #[test]
    fn sine_column_peak_matches_amplitude() {
        // Полные периоды синуса в каждой колонке: пик колонки ≈ амплитуде.
        let rate = 8_000_u32;
        let samples: Vec<f32> = (0..rate * 3)
            .map(|i| 0.7 * (i as f64 / f64::from(rate) * 220.0).sin() as f32)
            .collect();
        let bytes = wav_fix::encode_wav(rate, 16, 1, &samples);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 10).expect("valid wav");

        assert_eq!(envelope.left.len(), 10);
        for peak in &envelope.left {
            assert!(*peak > 0.69 && *peak <= 0.71, "sine column peak {peak}");
        }
    }

    #[test]
    fn column_count_matches_requested_columns() {
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &vec![0.25; 8_000 * 3]);

        for columns in [1, 7, 100] {
            let envelope =
                stereo_envelope_from_reader(&mut Cursor::new(bytes.clone()), columns).unwrap();
            assert_eq!(envelope.left.len(), columns, "columns={columns}");
            assert_eq!(envelope.right.len(), columns, "columns={columns}");
        }
    }

    #[test]
    fn short_file_pads_missing_columns_with_zero() {
        // 4 моно-фрейма против 16 колонок: фреймы распределяются равномерно
        // (col = frame·columns/frames → колонки 0, 4, 8, 12), остальные — нули.
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &[0.6, -0.6, 0.3, -0.3]);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 16).expect("valid wav");

        assert_eq!(envelope.left.len(), 16);
        assert_close(envelope.left[0], 0.6);
        assert_close(envelope.left[4], 0.6);
        assert_close(envelope.left[8], 0.3);
        assert_close(envelope.left[12], 0.3);
        for (i, peak) in envelope.left.iter().enumerate() {
            if !matches!(i, 0 | 4 | 8 | 12) {
                assert_eq!(*peak, 0.0, "колонка {i} без фреймов — нулевая");
            }
        }
    }

    #[test]
    fn non_pcm_input_is_an_error() {
        let junk = Cursor::new(b"wav-bytes".to_vec());
        assert!(matches!(
            stereo_envelope_from_reader(&mut junk.clone(), 8),
            Err(EnvelopeError::NotPcm)
        ));
        // Не-WAV байты правильного размера тоже отклоняются парсером.
        assert!(matches!(
            stereo_envelope_from_reader(&mut Cursor::new(vec![0_u8; 100]), 8),
            Err(EnvelopeError::NotPcm)
        ));
    }

    #[test]
    fn missing_file_is_io_error() {
        let path = temp_dir("missing").join("nope.wav");

        assert!(matches!(wav_stereo_envelope(&path, 8), Err(EnvelopeError::Io(_))));
    }

    /// Асимметричный стерео-файл: пики L и R различаются и попадают в свои
    /// каналы (суть стерео-огибающей — прежняя mono-max семантика ушла).
    #[test]
    fn asymmetric_stereo_separates_channel_peaks() {
        let stereo: Vec<f32> = [0.2_f32, 0.8].iter().cycle().take(8_000 * 2).copied().collect();
        let bytes = wav_fix::encode_wav(8_000, 16, 2, &stereo);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 4).expect("valid wav");

        for peak in &envelope.left {
            assert_close(*peak, 0.2,);
        }
        for peak in &envelope.right {
            assert_close(*peak, 0.8);
        }
        assert_ne!(envelope.left, envelope.right, "каналы не склеиваются в max");
    }

    /// Моно-файл — один канал в обе стороны: карта не рисует «пустой R».
    #[test]
    fn mono_file_fills_both_channels_alike() {
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &vec![0.4; 8_000]);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 5).expect("valid wav");

        assert_eq!(envelope.left, envelope.right);
        assert!(envelope.right.iter().all(|p| (p - 0.4).abs() < 1e-3));
    }

    #[test]
    fn zero_columns_returns_empty_without_panic() {
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &[0.5, -0.5]);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 0).unwrap();

        assert_eq!(envelope, StereoEnvelope::default(), "пустые каналы, без паники");
    }

    #[test]
    fn reading_across_block_boundaries_is_transparent() {
        // 3 с: [0..2) тише 0.2, [2..3) громче 0.9; блок чтения 1 с — переход
        // 0.2→0.9 ровно на границе блоков, пик не должен «перетекать».
        let rate = 8_000_u32;
        let samples: Vec<f32> = (0..rate * 3)
            .map(|i| if i < 2 * rate { 0.2 } else { 0.9 })
            .collect();
        let bytes = wav_fix::encode_wav(rate, 16, 1, &samples);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 6).expect("valid wav");

        for peak in &envelope.left[..4] {
            assert_close(*peak, 0.2);
        }
        for peak in &envelope.left[4..] {
            assert_close(*peak, 0.9);
        }
    }

    #[test]
    fn uneven_frames_fill_every_column() {
        // 10 фреймов на 3 колонки: остаток распределяется, пустых колонок нет.
        let samples: Vec<f32> = (0..10).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let bytes = wav_fix::encode_wav(8_000, 16, 1, &samples);

        let envelope = stereo_envelope_from_reader(&mut Cursor::new(bytes), 3).expect("valid wav");

        assert_eq!(envelope.left.len(), 3);
        assert!(envelope.left.iter().all(|p| *p > 0.4), "все колонки заполнены: {:?}", envelope.left);
    }

    /// Большой файл считается в spawn_blocking и не блокирует async-рантайм:
    /// тики цикла идут, пока огибающая ещё считается. Длительность — в логе
    /// теста (замер стримингового чтения; 644 МБ тот же кодовый путь).
    #[tokio::test]
    async fn big_file_envelope_keeps_runtime_responsive() {
        let dir = temp_dir("big");
        let path = dir.join("big.wav");
        let rate = 44_100_u32;
        // 60 с стерео 16 бит ≈ 10 МБ — чтение заведомо дольше тика 1 мс.
        let samples: Vec<f32> = (0..rate * 60)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                (0.6 * (t * 220.0).sin() + 0.3 * (t * 331.0).sin()) as f32
            })
            .collect();
        let stereo: Vec<f32> = samples.iter().flat_map(|s| [*s, *s * 0.5]).collect();
        wav_fix::write_wav(&path, rate, 16, 2, &stereo);
        let size_mb = std::fs::metadata(&path).unwrap().len() / (1024 * 1024);

        let started = Instant::now();
        let env_path = path.clone();
        let task = tokio::task::spawn_blocking(move || wav_stereo_envelope(&env_path, 100));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
        let mut ticks = 0_u32;
        while !task.is_finished() {
            let _ = tx.send(());
            if tokio::time::timeout(Duration::from_millis(1), rx.recv()).await.is_ok() {
                ticks += 1;
            }
        }
        let elapsed = started.elapsed();
        let envelope = task.await.unwrap().expect("valid wav fixture");

        assert_eq!(envelope.left.len(), 100);
        assert_eq!(envelope.right.len(), 100);
        assert!(ticks > 0, "рантайм отвечал во время чтения ({ticks} тиков)");
        println!(
            "envelope: {size_mb} MB in {elapsed:?} (streaming), runtime stayed responsive ({ticks} ticks)"
        );
    }
}
