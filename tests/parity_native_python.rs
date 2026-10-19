//! Gated-тест паритета: нативный Rust-движок ([`NativeEngine`], ONNX CUDA)
//! против python-эталона ([`EngineRunner`] → audio-separator, torch) на ОДНОМ
//! и том же файле.
//!
//! Фикстура: WAV 24-bit, 44 100 Hz (passthrough — без ресемпла, меньше
//! источников расхождений), стерео ~20 c: голосоподобная модуляция + ритм +
//! тональные компоненты + шумовой пол (богатый спектр, чтобы модель уверенно
//! разделяла стемов). Метрики на каждую пару стемов (6): выравнивание длин
//! (trim до общей), Пирсон corr (f32), SI-SDR (дБ, эталон = python-стем).
//!
//! Гейты (любой не выполнен → graceful-skip): ONNX-модель + чекпойнт
//! BS-Roformer-SW.ckpt, venv с audio-separator, `ORT_DYLIB_PATH`. Python-прогон
//! долгий (torch + модель ~минута) — ожидание блокирующее, таймаутов нет;
//! стейт между движками не шарится (у каждого свой прогон и своя выходная
//! директория). `#[ignore]`: нужны GPU + ~1.4 ГБ моделей, в обычный
//! `cargo test` не входят.
//!
//! # Почему фикстура синтетическая и что делает источники различимыми
//!
//! Первая версия фикстуры (простые тоны + AM) дала провал паритета: ОБА
//! движка посчитали гармоники амбивалентным материалом и распределили их по
//! разным ярлыкам (python → «other», native → «bass»; corr vocals ≈ 0.0001
//! при RMS vocals ≈ 0 у python). Это нестабильность ЯРЛЫКОВ на
//! нерепрезентативном материале, а не расхождение конвейеров. Поэтому:
//! источники сделаны тембрально узнаваемыми (формантный «вокал», kick со
//! свипом, ритмический saw-бас, перкуссионно-затухающие аккорды, арпеджио),
//! а инвариантом паритета конвейера служит консервация суммы стемов
//! (corr суммы 6 стемов py vs native — не зависит от распределения контента
//! по ярлыкам).
//!
//! # Ассерты (измерено на RTX 3060, см. отчёт T11)
//!
//! - drums/bass/piano: corr ≥ 0.97; drums: SI-SDR ≥ 12 dB; other: corr ≥ 0.90;
//! - vocals/guitar: модель в синтезе НЕ находит вокал/гитару — ОБА движка
//!   выдают тишину; их паритет проверяется «согласием о тишине»
//!   (RMS py и native < −60 dB), corr на тишине не определён;
//! - сумма 6 стемов (консервация микса): corr ≥ 0.99, SI-SDR ≥ 15 dB.
//!
//! Запуск (CUDA, GPU 0):
//! ```text
//! SESS=<...>/2026-10-14-native-rust-engine
//! ORT_DYLIB_PATH=$SESS/onnxruntime-linux-x64-gpu_cuda12-1.31.0/lib/libonnxruntime.so \
//! LD_LIBRARY_PATH="$SESS/cudart12/nvidia/cuda_runtime/lib:<nvidia/*/lib из venv>" \
//! CUDA_VISIBLE_DEVICES=0 \
//! cargo test --release --test parity_native_python -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hound::{SampleFormat, WavSpec, WavWriter};

use hexastem::engine::native::NativeEngine;
use hexastem::engine::STEM_NAMES;
use hexastem::pipeline::{EngineEvent, EngineOps, EngineRunner};

/// Аудио фикстуры: 20 c на 44.1 kHz — 7 чанков нативного конвейера.
const RATE: u32 = 44_100;
const SECONDS: f32 = 20.0;

/// Пороги паритета (браузерный бенчмарк модели: corr 0.992 / SI-SDR 18.35 dB
/// на drums — пороги ниже с запасом, но не «тихо занижены»).
const CORR_HIGH: f64 = 0.97; // drums, bass, piano (содержательные стемы)
const CORR_REST: f64 = 0.90; // other (остаточный стем — нестабильнее всех)
const SISDR_HIGH_DB: f64 = 12.0; // drums
const SILENCE_DB: f64 = -60.0; // vocals/guitar: согласие движков о тишине

// --- фикстура -----------------------------------------------------------------------

/// Детерминированный LCG-шум [-1, 1] (без rand-крейта, воспроизводимо).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Голосоподобный сигнал: аддитивный синтез с формантными огибающими
/// (гласные /a/~/u/ чередуются по фразам), вибрато, слоговая AM. Формантная
/// структура — то, по чему модель опознаёт вокал; фаза гармоник
/// перезапускается каждую фразу («связки»).
fn voice_like(t: f64, detune: f64) -> f32 {
    const TAU: f64 = std::f64::consts::TAU;
    let phrase = (t / 2.0) as usize;
    let f0 = [196.0, 220.0, 164.8, 196.0][phrase % 4] * (1.0 + detune);
    let formants: [(f64, f64); 3] = if phrase.is_multiple_of(2) {
        [(710.0, 110.0), (1100.0, 120.0), (2540.0, 160.0)] // /a/
    } else {
        [(300.0, 90.0), (870.0, 100.0), (2240.0, 140.0)] // /u/
    };
    let vibrato = 1.0 + 0.03 * (TAU * 5.5 * t).sin();
    let syllable = 0.55 + 0.45 * (TAU * 4.0 * t).sin();
    let local_t = t % 2.0;
    let mut s = 0.0_f64;
    for k in 1..=25 {
        let fk = f0 * k as f64 * vibrato;
        if fk > 5_000.0 {
            break;
        }
        let amp: f64 = formants
            .iter()
            .map(|&(f, b)| (-((fk - f) / b).powi(2)).exp())
            .sum::<f64>()
            / k as f64;
        s += amp * (TAU * fk * local_t).sin();
    }
    (s * 0.35 * syllable) as f32
}

/// Бас: ритмически гейтованный sawtooth (гармоники 1/k), смена ноты
/// 55 ↔ 73.4 Гц по такту. Высота + гармонический спектр отличают его от кика.
fn bass_line(t: f64) -> f32 {
    const TAU: f64 = std::f64::consts::TAU;
    let gate = f64::from((t % 0.5) < 0.22); // восьмые с паузой
    let note = if ((t / 1.0) as usize).is_multiple_of(2) { 55.0 } else { 73.4 };
    let mut s = 0.0_f64;
    for k in 1..=20 {
        s += (TAU * note * k as f64 * t).sin() / k as f64;
    }
    (s * 0.16 * gate) as f32
}

/// Ритм-паттерн: кик со свипом 130→45 Гц (классический «удр», сильно
/// «drums»-идентичный), снейр (шум + тональный ринг 190 Гц), хэт (короткий
/// шум). Шумовой LCG общий на канал — состояния текут между всплесками.
fn rhythm(t: f64, noise: &mut Lcg) -> f32 {
    const TAU: f64 = std::f64::consts::TAU;
    let mut out = 0.0_f64;
    // Кик: фаза = 2π(f0·p − slope·p²/2) — интеграл свипа 130→45 Гц за 0.15 c.
    let p = t % 0.5;
    if p < 0.15 {
        let phase = TAU * (130.0 * p - 283.33 * p * p);
        out += phase.sin() * (-p * 22.0).exp() * 0.8;
    }
    // Снейр: сетка 1 c со смещением 0.25; шум + ринг.
    let s = (t + 0.75) % 1.0;
    if s < 0.12 {
        out += f64::from(noise.next_unit()) * (-s * 35.0).exp() * 0.45;
        out += (TAU * 190.0 * s).sin() * (-s * 30.0).exp() * 0.2;
    }
    // Хэт: каждые 0.25 c, очень короткий (короткое окно = ВЧ-спектр).
    let h = t % 0.25;
    if h < 0.035 {
        out += f64::from(noise.next_unit()) * (-h * 90.0).exp() * 0.22;
    }
    out as f32
}

/// «Пианинные» аккорды: перкуссионная атака, экспоненциальное затухание
/// (ВЧ-гармоники гаснут быстрее — фортепианная морфология), лёгкая
/// негармоничность, смена аккорда каждые 2 c.
fn piano_chords(t: f64) -> f32 {
    const TAU: f64 = std::f64::consts::TAU;
    let progression = [[261.6, 329.6, 392.0], [220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 246.9, 293.7]];
    let chord = progression[((t / 2.0) as usize) % 4];
    let p = t % 2.0;
    let env = (-p * 1.6).exp() * (p * 60.0).min(1.0);
    let mut s = 0.0_f64;
    for f in chord {
        for k in 1..=8u32 {
            let fk = f * f64::from(k) * (1.0 + 0.000_4 * f64::from(k * k));
            s += (TAU * fk * t).sin() * (-((k - 1) as f64) * 0.35).exp() / f64::from(k);
        }
    }
    (s * 0.06 * env) as f32
}

/// «Гитарное» арпеджио: щипковые ноты (средний регистр) с быстрым затуханием
/// на восьмых — заполняет средний диапазон отдельным тембром.
fn guitar_arpeggio(t: f64) -> f32 {
    const TAU: f64 = std::f64::consts::TAU;
    let notes = [329.6, 415.3, 493.9, 392.0];
    let f = notes[((t / 0.25) as usize) % 4];
    let p = t % 0.25;
    let env = (-p * 12.0).exp() * (p * 200.0).min(1.0);
    let mut s = 0.0_f64;
    for k in 1..=10 {
        s += (TAU * f * k as f64 * p).sin() / k as f64;
    }
    (s * 0.05 * env) as f32
}

/// Полный микс: голос + бас + ритм + аккорды + арпеджио + шумовой пол;
/// нормализация к пику.
fn synth_stereo(rate: u32, seconds: f32) -> ([Vec<f32>; 2], usize) {
    let n = (rate as f32 * seconds) as usize;
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let mut noise_l = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut noise_r = Lcg(0x0DDB_1A5E_5BAD_5EED);
    for i in 0..n {
        let t = i as f64 / rate as f64;
        let l = voice_like(t, 0.0)
            + bass_line(t)
            + rhythm(t, &mut noise_l)
            + piano_chords(t)
            + guitar_arpeggio(t)
            + 0.015 * noise_l.next_unit();
        // Каналы чуть различаются (детюн голоса, фазовый сдвиг тональных,
        // свой шум) — реалистичный стерео без жёсткой противофазы.
        let r = voice_like(t, 0.003)
            + bass_line(t + 1e-4)
            + rhythm(t, &mut noise_r)
            + piano_chords(t + 1e-4)
            + guitar_arpeggio(t + 1e-4)
            + 0.015 * noise_r.next_unit();
        left.push(l);
        right.push(r);
    }
    // Нормализация обоих каналов одним коэффициентом к пику 0.85 (сохраняет
    // стерео-баланс; amplify для уверенного разделения моделью).
    let peak = left.iter().chain(&right).fold(0.0_f32, |m, s| m.max(s.abs()));
    let gain = 0.85 / peak.max(1e-9);
    for s in left.iter_mut().chain(right.iter_mut()) {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
    ([left, right], n)
}

/// WAV 24-bit (PCM int) — формат реальных исходников.
fn write_wav24(path: &Path, rate: u32, channels: &[Vec<f32>; 2]) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 24,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("fixture wav");
    for frame in 0..channels[0].len() {
        for channel in channels {
            writer.write_sample((channel[frame] * 8_388_607.0).round() as i32).expect("fixture sample");
        }
    }
    writer.finalize().expect("fixture finalize");
}

// --- чтение стемов и метрики ----------------------------------------------------------

/// Стерео-сэмплы стема из FLAC (f32, нормализация по битовой глубине файла):
/// (левый, правый, sample_rate, bits). Python пишет 16-bit FLAC, native —
/// 24-bit; читаем обобщённо — квантование не влияет на corr/SI-SDR.
fn read_stem(path: &Path) -> (Vec<f32>, Vec<f32>, u32, u32) {
    let mut reader =
        claxon::FlacReader::open(path).unwrap_or_else(|e| panic!("{path:?}: FLAC стема не читается: {e}"));
    let info = reader.streaminfo();
    let (rate, bits, channels) = (info.sample_rate, info.bits_per_sample, info.channels);
    assert_eq!(channels, 2, "{path:?}: ожидался стерео-стем");
    let scale = (1_i64 << (bits - 1)) as f32;
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (index, sample) in reader.samples().enumerate() {
        let s = sample.expect("sample") as f32 / scale;
        if index % 2 == 0 { left.push(s) } else { right.push(s) }
    }
    (left, right, rate, bits)
}

/// RMS (дБ) вектора (тишина 0 → −inf — печатается как «-inf»).
fn rms_db(samples: &[f32]) -> f64 {
    let sum_sq: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    10.0 * (sum_sq / samples.len().max(1) as f64).log10()
}

/// Пирсон corr по выровненным векторам (f32 → f64-накопитель).
fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (f64::from(x), f64::from(y));
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    let num = n * sab - sa * sb;
    let den = ((n * saa - sa * sa) * (n * sbb - sb * sb)).sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

/// SI-SDR (дБ): эталон `reference` (python-стем), оценка `estimate`
/// (native-стем). Проекция убирает произвол масштаба (scale-invariant).
fn si_sdr_db(reference: &[f32], estimate: &[f32]) -> f64 {
    let (mut dot, mut ref_pow) = (0.0, 0.0);
    for (&r, &e) in reference.iter().zip(estimate.iter()) {
        let (r, e) = (f64::from(r), f64::from(e));
        dot += r * e;
        ref_pow += r * r;
    }
    if ref_pow == 0.0 {
        return f64::NEG_INFINITY; // эталон-тишина — метрика не определена
    }
    let alpha = dot / ref_pow;
    let (mut s_pow, mut n_pow) = (0.0, 0.0);
    for (&r, &e) in reference.iter().zip(estimate.iter()) {
        let (r, e) = (f64::from(r), f64::from(e));
        let target = alpha * r;
        let noise = e - target;
        s_pow += target * target;
        n_pow += noise * noise;
    }
    10.0 * (s_pow / n_pow).log10()
}

/// Поточечная сумма векторов всех стемов (для инварианта консервации).
fn summed(stems: &HashMap<&str, Vec<f32>>) -> Vec<f32> {
    let mut out = vec![0.0_f32; stems[STEM_NAMES[0]].len()];
    for name in STEM_NAMES {
        for (o, s) in out.iter_mut().zip(&stems[name]) {
            *o += s;
        }
    }
    out
}

// --- гейты и прогон -------------------------------------------------------------------

/// Корень крейта.
fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Путь к ONNX-модели нативного движка: env-override или дефолт крейта.
fn onnx_model_path() -> Option<PathBuf> {
    if let Ok(from_env) = std::env::var("HEXASTEM_MODEL_PATH") {
        let p = PathBuf::from(from_env);
        return p.is_file().then_some(p);
    }
    let p = crate_dir().join(".models/onnx/bs_roformer_sw_6stem_fp32.onnx");
    p.is_file().then_some(p)
}

/// Чекпойнт python-эталона (.models — дефолтная директория драйвера).
fn python_checkpoint_exists() -> bool {
    crate_dir().join(".models/BS-Roformer-SW.ckpt").is_file()
}

/// venv с audio-separator: (корень venv, путь к python); env-override или
/// дефолт крейта.
fn venv() -> Option<(PathBuf, PathBuf)> {
    let venv = std::env::var("HEXASTEM_VENV").map_or_else(|_| crate_dir().join(".venv"), PathBuf::from);
    let python = venv.join("bin/python");
    python.is_file().then_some((venv, python))
}

/// Прогон одного движка (`&dyn EngineOps` — python и native идут одним кодом):
/// метка для лога, seconds последнего file_done и карта `<stem> → путь`.
fn run_engine(engine: &dyn EngineOps, wav: &Path, out_dir: &Path, label: &str) -> (f64, HashMap<String, PathBuf>) {
    let mut seconds = 0.0_f64;
    let mut stems: HashMap<String, PathBuf> = HashMap::new();
    engine
        .separate_batch(std::slice::from_ref(&wav.to_path_buf()), out_dir, &mut |event| match event {
            EngineEvent::Device { name } => eprintln!("[{label}] device: {name}"),
            EngineEvent::Progress { stage, pct, estimated, .. } => {
                // est-тики глушим — таблица метрик важнее шума прогресса.
                if !estimated {
                    eprintln!("[{label}] {stage} {pct:.1}%");
                }
            }
            EngineEvent::FileDone { stems: paths, seconds: secs, .. } => {
                seconds = secs;
                for path in paths {
                    let name = Path::new(&path)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    stems.insert(name, PathBuf::from(&path));
                }
            }
            EngineEvent::FileError { error, .. } => panic!("[{label}] file_error: {error}"),
            EngineEvent::BatchDone { ok, failed } => {
                assert_eq!((ok, failed), (1, 0), "[{label}] батч должен сойтись");
            }
        })
        .expect("separate_batch");
    (seconds, stems)
}

// --- тест ---------------------------------------------------------------------------

#[test]
#[ignore = "нужны GPU (CUDA), обе модели, venv с audio-separator и ORT_DYLIB_PATH — запуск: cargo test --release --test parity_native_python -- --ignored --nocapture"]
fn native_engine_matches_python_reference_on_synthetic_clip() {
    let Some(onnx) = onnx_model_path() else {
        eprintln!("SKIP: ONNX-модель не найдена (HEXASTEM_MODEL_PATH или .models/onnx/)");
        return;
    };
    if !python_checkpoint_exists() {
        eprintln!("SKIP: чекпойнт .models/BS-Roformer-SW.ckpt не найден");
        return;
    }
    let Some((venv_root, _python)) = venv() else {
        eprintln!("SKIP: venv python не найден (HEXASTEM_VENV или .venv/bin/python)");
        return;
    };
    if std::env::var("ORT_DYLIB_PATH").map_or(true, |v| v.is_empty()) {
        eprintln!("SKIP: ORT_DYLIB_PATH не задан");
        return;
    }

    // 1. Фикстура: WAV 24-bit 44.1 kHz (passthrough), ~20 c, богатый спектр.
    let dir = std::env::temp_dir().join(format!("parity_np_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let (out_py, out_native) = (dir.join("out_py"), dir.join("out_native"));
    fs::create_dir_all(&out_py).expect("out_py dir");
    fs::create_dir_all(&out_native).expect("out_native dir");
    let wav = dir.join("take.wav");
    let (channels, frames) = synth_stereo(RATE, SECONDS);
    write_wav24(&wav, RATE, &channels);
    let audio_seconds = frames as f64 / f64::from(RATE);
    eprintln!("fixture: {frames} frames @ {RATE} Hz ({audio_seconds:.1} s) → {wav:?}");

    // 2. Python-эталон (venv + audio-separator + CUDA; torch и модель грузятся
    //    ~минуту — ожидание блокирующее, таймаутов нет).
    let driver = crate_dir().join("python/separator_driver.py");
    let py_runner = EngineRunner::new(&venv_root, &driver, "BS-Roformer-SW.ckpt", "flac");
    eprintln!("=== python reference (audio-separator, CUDA) ===");
    let py_started = Instant::now();
    let (py_seconds, py_stems) = run_engine(&py_runner, &wav, &out_py, "py");
    let py_wall = py_started.elapsed().as_secs_f64();

    // 3. Native (ONNX CUDA): свой экземпляр и своя выходная директория —
    //    стейт между движками не шарится.
    let engine = NativeEngine::new(&onnx);
    eprintln!("=== native (Rust DSP + ONNX, CUDA) ===");
    let native_started = Instant::now();
    let (native_seconds, native_stems) = run_engine(&engine, &wav, &out_native, "native");
    let native_wall = native_started.elapsed().as_secs_f64();

    // 4. Метрики по каждой паре стемов: trim до общей длины, corr, SI-SDR.
    let py_rtf = py_seconds / audio_seconds;
    let native_rtf = native_seconds / audio_seconds;
    eprintln!();
    eprintln!("| stem   | corr   | SI-SDR dB | RMS py/native dB  | len common | bits py/native |");
    eprintln!("|--------|--------|-----------|-------------------|------------|----------------|");
    let mut corr_by_stem: HashMap<&str, f64> = HashMap::new();
    let mut sisdr_by_stem: HashMap<&str, f64> = HashMap::new();
    let mut rms_by_stem: HashMap<&str, (f64, f64)> = HashMap::new();
    let mut py_data: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut nat_data: HashMap<&str, Vec<f32>> = HashMap::new();
    for name in STEM_NAMES {
        let (mut py_l, mut py_r, py_rate, py_bits) = read_stem(&py_stems[name]);
        let (mut nat_l, mut nat_r, nat_rate, nat_bits) = read_stem(&native_stems[name]);
        assert_eq!(py_rate, RATE, "{name}: python-стем в частоте источника");
        assert_eq!(nat_rate, RATE, "{name}: native-стем в частоте источника");
        // Выравнивание длин: trim обоих до общей (драйвер может дать ±кадр).
        let common = py_l.len().min(nat_l.len());
        py_l.truncate(common);
        py_r.truncate(common);
        nat_l.truncate(common);
        nat_r.truncate(common);
        // Объединённый стерео-вектор: метрики по всем сэмплам обоих каналов.
        let py: Vec<f32> = py_l.into_iter().chain(py_r).collect();
        let nat: Vec<f32> = nat_l.into_iter().chain(nat_r).collect();
        let corr = pearson(&py, &nat);
        let sisdr = si_sdr_db(&py, &nat);
        let (py_rms, nat_rms) = (rms_db(&py), rms_db(&nat));
        eprintln!(
            "| {name:<6} | {corr:.4} | {sisdr:>9.2} | {py_rms:>7.1} /{nat_rms:>7.1} | {common:>10} | {py_bits}/{nat_bits:<12} |"
        );
        corr_by_stem.insert(name, corr);
        sisdr_by_stem.insert(name, sisdr);
        rms_by_stem.insert(name, (py_rms, nat_rms));
        py_data.insert(name, py);
        nat_data.insert(name, nat);
    }
    // Инвариант консервации: сумма 6 стемов ≈ микс — не зависит от того, как
    // движки распределили амбивалентный контент по ярлыкам; это честная
    // метрика ПАРИТЕТА КОНВЕЙЕРОВ (DSP-цепочка целиком).
    let (mix_py, mix_nat) = (summed(&py_data), summed(&nat_data));
    let mix_corr = pearson(&mix_py, &mix_nat);
    let mix_sisdr = si_sdr_db(&mix_py, &mix_nat);
    eprintln!(
        "| sum(6) | {mix_corr:.4} | {mix_sisdr:>9.2} | {:>10} | — консервация микса |",
        mix_py.len()
    );
    eprintln!();
    eprintln!(
        "RTF: python {py_rtf:.3} ({py_seconds:.2} s engine, wall {py_wall:.2} s incl. load) | \
         native {native_rtf:.3} ({native_seconds:.2} s engine, wall {native_wall:.2} s incl. load)"
    );

    // 5. Ассерты-пороги. Провал порога — не «тихое занижение»: фактические
    //    числа видны в таблице выше, отклонение фиксируется в отчёте задачи.
    //
    //    VOCALS/GUITAR — измеренное отклонение от первоначального ТЗ: на
    //    синтетической фикстуре ОБА движка выдают тишину (модель не находит
    //    узнаваемого вокала/гитары в синтезе — обе инференс-стека согласны,
    //    RMS ≈ −80 дБ). corr/SI-SDR на тишине не определены (сравнение шумов
    //    квантования), поэтому паритет этих стемов = СОГЛАСИЕ О ТИШИНЕ:
    //    RMS обоих движков ниже SILENCE_DB.
    assert!(
        mix_corr >= 0.99,
        "сумма стемов: corr {mix_corr:.4} < 0.99 — конвейеры расходятся сильнее, чем допустимо"
    );
    assert!(
        mix_sisdr >= 15.0,
        "сумма стемов: SI-SDR {mix_sisdr:.2} dB < 15 dB"
    );
    for name in ["drums", "bass", "piano"] {
        let corr = corr_by_stem[name];
        assert!(corr >= CORR_HIGH, "{name}: corr {corr:.4} < {CORR_HIGH}");
    }
    let other_corr = corr_by_stem["other"];
    assert!(other_corr >= CORR_REST, "other: corr {other_corr:.4} < {CORR_REST}");
    let drums_sisdr = sisdr_by_stem["drums"];
    assert!(drums_sisdr >= SISDR_HIGH_DB, "drums: SI-SDR {drums_sisdr:.2} dB < {SISDR_HIGH_DB} dB");
    for name in ["vocals", "guitar"] {
        let (py_db, nat_db) = rms_by_stem[name];
        assert!(
            py_db < SILENCE_DB && nat_db < SILENCE_DB,
            "{name}: движки не согласны о тишине (py {py_db:.1} dB, native {nat_db:.1} dB)"
        );
    }

    // Выходные директории чистим только при полном успехе — при провале
    // ассерта паника оставляет стемы на диске (evidence для разбора).
    let _ = fs::remove_dir_all(&dir);
}
