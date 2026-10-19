//! Smoke-тест golden-фикстур: reference.npz читается, шейпы совпадают с контрактом модели.

mod common;

#[test]
fn reference_fixture_loads_and_shapes_match_contract() {
    if common::reference_missing() {
        return;
    }
    let r = common::Reference::load().expect("reference.npz должен читаться");
    assert_eq!(r.input_audio.shape(), &[1, 2, 176_400]);
    assert_eq!(r.spec_real.shape(), &[1, 2, 1025, 345]);
    assert_eq!(r.spec_imag.shape(), &[1, 2, 1025, 345]);
    assert_eq!(r.out_spec_real.shape(), &[1, 6, 2, 1025, 345]);
    assert_eq!(r.out_spec_imag.shape(), &[1, 6, 2, 1025, 345]);
    assert_eq!(r.ref_audio.shape(), &[1, 6, 2, 176_400]);
    assert_eq!(r.samples, 176_400);
}
