//! Golden-тесты STFT/iSTFT против reference.npz (эталон PyTorch, wrapper.py).
//!
//! forward: input_audio[c] → (real, imag) ≈ spec_real/imag[c], max abs diff < 1e-3.
//! inverse: out_spec_* по стемам 0..6 → ≈ ref_audio[s], max abs diff < 1e-3.

mod common;

use hexastem::engine::stft::Stft;
use ndarray::s;

/// Порог паритета репозитория (validate_onnx.py:41).
const TOL: f32 = 1e-3;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn forward_matches_pytorch_spec() {
    if common::reference_missing() {
        return;
    }
    let r = common::Reference::load().expect("reference.npz должен читаться");
    let stft = Stft::new();
    for c in 0..2 {
        let chunk: Vec<f32> = r.input_audio.slice(s![0, c, ..]).iter().copied().collect();
        let (real, imag) = stft.forward(&chunk).expect("forward ok");
        // freq-major [1025, 345] C-порядка — плоско сравниваем со срезом тензора.
        let ref_real = r.spec_real.slice(s![0, c, .., ..]);
        let ref_imag = r.spec_imag.slice(s![0, c, .., ..]);
        let dr = max_abs_diff(&real, ref_real.as_slice().expect("стандартный лейаут"));
        let di = max_abs_diff(&imag, ref_imag.as_slice().expect("стандартный лейаут"));
        eprintln!("forward ch{c}: max abs diff real = {dr:.3e}, imag = {di:.3e}");
        assert!(dr < TOL, "ch{c}: spec_real max abs diff = {dr}");
        assert!(di < TOL, "ch{c}: spec_imag max abs diff = {di}");
    }
}

#[test]
fn inverse_matches_pytorch_ref_audio() {
    if common::reference_missing() {
        return;
    }
    let r = common::Reference::load().expect("reference.npz должен читаться");
    let stft = Stft::new();
    for stem in 0..6 {
        for c in 0..2 {
            let real: Vec<f32> = r
                .out_spec_real
                .slice(s![0, stem, c, .., ..])
                .iter()
                .copied()
                .collect();
            let imag: Vec<f32> = r
                .out_spec_imag
                .slice(s![0, stem, c, .., ..])
                .iter()
                .copied()
                .collect();
            let mut out = vec![0.0f32; 176_400];
            stft.inverse(&real, &imag, &mut out).expect("inverse ok");
            let d = max_abs_diff(&out, r.ref_audio.slice(s![0, stem, c, ..]).as_slice().unwrap());
            eprintln!("inverse stem {stem} ch{c}: max abs diff = {d:.3e}");
            assert!(d < TOL, "stem {stem} ch{c}: ref_audio max abs diff = {d}");
        }
    }
}
