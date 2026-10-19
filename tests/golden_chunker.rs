//! Golden-тест чанкера против reference.npz: `input_audio[c]` (ровно 176400
//! сэмплов) → `chunk_signal` даёт 1 чанк, побитово идентичный входу;
//! `merge_chunks` восстанавливает вход побитово (1 чанк → все веса 1).
//! При отсутствии reference.npz тест пропускается.

mod common;

use hexastem::engine::chunker::{chunk_signal, merge_chunks};

#[test]
fn golden_chunker_roundtrip_bitwise_matches_reference() {
    if common::reference_missing() {
        return;
    }
    let r = common::Reference::load().expect("reference.npz должен читаться");
    for channel in 0..2 {
        let plane = r.input_audio.index_axis(ndarray::Axis(0), 0);
        let row = plane.index_axis(ndarray::Axis(0), channel);
        let input = row.as_slice().expect("input_audio[c] contiguous");
        assert_eq!(input.len(), 176_400);

        // Нарезка: ровно 1 чанк, побитово идентичный входу.
        let chunks = chunk_signal(input);
        assert_eq!(chunks.len(), 1, "channel {channel}: ожидался 1 чанк");
        assert_eq!(chunks[0].len(), 176_400);
        assert!(
            chunks[0].iter().zip(input).all(|(a, b)| a.to_bits() == b.to_bits()),
            "channel {channel}: чанк обязан побитово совпадать с input_audio"
        );

        // Склейка: побитовое восстановление входа.
        let merged = merge_chunks(&chunks, input.len()).expect("валидный merge");
        assert_eq!(merged.len(), input.len());
        assert!(
            merged.iter().zip(input).all(|(a, b)| a.to_bits() == b.to_bits()),
            "channel {channel}: merge обязан побитово восстанавливать input_audio"
        );
    }
}
