//! Чанкер: нарезка на 176400 с шагом 132300, zero-pad хвоста,
//! линейный комплементарный кроссфейд 25% при склейке.
//!
//! T8. Кроссфейд: вес i/OVERLAP (сумма ровно 1), без нормировки краёв;
//! после склейки срез по числу сэмплов источника.
//!
//! Эталон: elicwhite/bs-roformer-web, pipeline.js:66–138
//! (см. `.tmp/sessions/2026-10-14-native-rust-engine/t1-reference-math.md`, §2 и §6).

use super::{CHUNK_SAMPLES, OVERLAP_SAMPLES, STEP_SAMPLES};
use crate::error::AppError;

/// Нарезка сигнала на чанки по [`CHUNK_SAMPLES`] с шагом [`STEP_SAMPLES`].
///
/// Последний неполный кусок дополняется нулями до полной длины чанка
/// (pipeline.js:107–116). Пустой вход → пустой вектор.
pub fn chunk_signal(samples: &[f32]) -> Vec<Vec<f32>> {
    if samples.is_empty() {
        return Vec::new();
    }
    // Эталон (pipeline.js:67): numSegments = max(1, ceil((N - OVERLAP) / STEP)).
    let num_segments = 1.max(samples.len().saturating_sub(OVERLAP_SAMPLES).div_ceil(STEP_SAMPLES));
    let mut chunks = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let start = seg * STEP_SAMPLES;
        // start < N по построению; хвост короче CHUNK дополняется нулями.
        let len = CHUNK_SAMPLES.min(samples.len().saturating_sub(start));
        let mut chunk = vec![0.0f32; CHUNK_SAMPLES];
        chunk[..len].copy_from_slice(&samples[start..start + len]);
        chunks.push(chunk);
    }
    chunks
}

/// Склейка чанков с линейным комплементарным кроссфейдом в перекрытии:
/// `out = prev*(1−t) + next*t`, где `t = j/OVERLAP` — сумма весов ровно 1,
/// без нормировки краёв (pipeline.js:52–64, 118–138).
///
/// Возвращает первые `num_samples` сэмплов склеенного сигнала.
///
/// Ошибки ([`AppError::Dsp`]): пустой список чанков; длина любого чанка
/// не равна [`CHUNK_SAMPLES`].
pub fn merge_chunks(chunks: &[Vec<f32>], num_samples: usize) -> Result<Vec<f32>, AppError> {
    if chunks.is_empty() {
        return Err(AppError::Dsp("merge_chunks: пустой список чанков".into()));
    }
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.len() != CHUNK_SAMPLES {
            return Err(AppError::Dsp(format!(
                "merge_chunks: длина чанка {i} = {}, ожидается {CHUNK_SAMPLES}",
                chunk.len()
            )));
        }
    }

    let last = chunks.len() - 1;
    let mut out = vec![0.0f32; num_samples];
    for (seg, chunk) in chunks.iter().enumerate() {
        let start = seg * STEP_SAMPLES;
        for (i, &sample) in chunk.iter().enumerate() {
            let pos = start + i;
            if pos >= num_samples {
                break; // нулевой pad последнего чанка за пределами сигнала не нужен
            }
            // Веса эталона mixSegment (pipeline.js:169–171): fadeIn для входа
            // не-первого чанка, fadeOut для выхода не-последнего. Ветви
            // взаимоисключающие, т.к. OVERLAP (44100) < CHUNK − OVERLAP (132300).
            let mut w = 1.0f32;
            if seg > 0 && i < OVERLAP_SAMPLES {
                w = i as f32 / OVERLAP_SAMPLES as f32; // fadeIn[j] = j / OVERLAP
            }
            if seg < last && i >= CHUNK_SAMPLES - OVERLAP_SAMPLES {
                // fadeOut[j] = 1 - j / OVERLAP, j = i - (CHUNK - OVERLAP)
                w = 1.0 - (i - (CHUNK_SAMPLES - OVERLAP_SAMPLES)) as f32 / OVERLAP_SAMPLES as f32;
            }
            out[pos] += sample * w;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детерминированный LCG (PCG-подобный множитель) — без внешнего rand-крейта.
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

    #[test]
    fn exact_chunk_length_yields_single_chunk() {
        let samples = vec![0.5f32; CHUNK_SAMPLES];
        let chunks = chunk_signal(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), CHUNK_SAMPLES);
        assert_eq!(chunks[0].as_slice(), samples.as_slice());
    }

    #[test]
    fn shorter_input_zero_padded_to_single_chunk() {
        let samples: Vec<f32> = (0..STEP_SAMPLES).map(|i| i as f32 * 0.001).collect();
        let chunks = chunk_signal(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), CHUNK_SAMPLES);
        // Реальная часть = вход; хвост чанка после pad — нули.
        assert_eq!(&chunks[0][..STEP_SAMPLES], samples.as_slice());
        assert!(
            chunks[0][STEP_SAMPLES..].iter().all(|&s| s == 0.0),
            "хвост после pad обязан быть нулевым"
        );
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_signal(&[]).is_empty());
    }

    #[test]
    fn one_sample_overflow_spawns_second_chunk() {
        let samples = vec![0.25f32; CHUNK_SAMPLES + 1];
        let chunks = chunk_signal(&samples);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_slice(), &samples[..CHUNK_SAMPLES]);
        // Второй чанк: 44101 реальных сэмплов (со стороны перекрытия) + нулевой pad.
        let real = CHUNK_SAMPLES + 1 - STEP_SAMPLES; // 44101
        assert!(chunks[1][..real].iter().all(|&s| s == 0.25));
        assert!(chunks[1][real..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn chunk_plus_step_input_yields_two_full_chunks() {
        let n = CHUNK_SAMPLES + STEP_SAMPLES; // 308700
        let samples: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
        let chunks = chunk_signal(&samples);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_slice(), &samples[..CHUNK_SAMPLES]);
        // Второй чанк полный, начинается со смещения STEP.
        assert_eq!(chunks[1].as_slice(), &samples[STEP_SAMPLES..]);
    }

    #[test]
    fn merge_of_constant_chunks_keeps_constant_in_overlap() {
        // 2+ чанка: в перекрытии 44100 весов fadeOut+fadeIn суммируются в 1.
        let samples = vec![0.75f32; CHUNK_SAMPLES + STEP_SAMPLES];
        let chunks = chunk_signal(&samples);
        assert_eq!(chunks.len(), 2);
        let merged = merge_chunks(&chunks, samples.len()).expect("валидный merge");
        assert_eq!(merged.len(), samples.len());
        for (i, &s) in merged.iter().enumerate() {
            assert!((s - 0.75).abs() < 1e-6, "pos {i}: {s} != 0.75");
        }
    }

    #[test]
    fn chunk_merge_roundtrip_recovers_lcg_signal() {
        // Property: chunk_signal → merge_chunks ≈ исходный сигнал (f32-округление кроссфейда).
        let n = 500_000;
        let samples = lcg_signal(n, 0x2026_1014);
        let chunks = chunk_signal(&samples);
        assert_eq!(chunks.len(), 4); // ceil((500000-44100)/132300) = 4
        assert!(chunks.iter().all(|c| c.len() == CHUNK_SAMPLES));

        let merged = merge_chunks(&chunks, n).expect("валидный merge");
        assert_eq!(merged.len(), n);
        for (i, (a, b)) in merged.iter().zip(&samples).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "pos {i}: merged {a} vs source {b}"
            );
        }
    }

    #[test]
    fn merge_rejects_wrong_chunk_length() {
        let bad = vec![vec![0.0f32; CHUNK_SAMPLES], vec![0.0f32; CHUNK_SAMPLES - 1]];
        let err = merge_chunks(&bad, CHUNK_SAMPLES).expect_err("длина чанка неверна");
        assert!(matches!(err, AppError::Dsp(_)), "ожидалась Dsp-ошибка: {err:?}");
    }

    #[test]
    fn merge_rejects_empty_chunks() {
        let err = merge_chunks(&[], 100).expect_err("пустой список чанков");
        assert!(matches!(err, AppError::Dsp(_)), "ожидалась Dsp-ошибка: {err:?}");
    }
}
