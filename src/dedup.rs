//! Дедупликация дублей-записей рекордера (стадия между download и движком).
//!
//! На рекордере появляются файлы-дубли: одна и та же запись, но одна дорожка
//! громче (цифровой гейн + лимитер). Реальный кейс (утверждён владельцем
//! 2026-10-13): `unknown 2` (чистая, пик −6.6 дБФС, крест-фактор ~28 дБ) и
//! `unknown 3` (+11.1 дБ, пережата лимитером, пик −0.32 дБФС, крест ~23 дБ):
//! corr нормированных огибающих 0.985–0.999 в окнах начало/середина/конец
//! (по 30 с: старт+60 с, середина, конец−90 с, фрейм 50 мс), сдвиг 0 мс,
//! длительность совпадает до фрейма.
//!
//! Методика: PCM WAV читается потоково — RIFF-заголовок, 3 окна по 30 с и
//! блок-скан всего файла на клип; файлы ~600 МБ в память не помещаются.
//! Дубль: равное число фреймов И corr пик-огибающих (фрейм 50 мс,
//! `max(|L|,|R|)`) ≥ [`CORR_THRESHOLD`] в каждом окне при прямом
//! выравнивании (сдвиг 0).
//!
//! Правило хранителя (утверждено владельцем): 1) клипующие (пик у потолка)
//! исключаются; 2) различие глобального крест-фактора файла
//! (dbfs глобального пика − медиана медианных RMS трёх окон) > 2 дБ →
//! хранитель более динамичный, даже если тише; 3) иначе — более громкий (RMS).
//! Крест-фактор — именно глобальный: окна покрывают лишь 90 с записи, и в
//! кейсе unknown 2/3 средний по окнам крест давал 21.6 против 20.8 (Δ 0.8 ≤ 2)
//! и лимитерованная копия ошибочно становилась хранителем; глобальный даёт
//! 28.7 против 23.9 (Δ 4.8 > 2).
//!
//! std only, без новых крейтов; кроме [`analyze_file`] всё — чистые функции.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// --- пороги и размеры (ссылка на кейс unknown 2/3) --------------------------------

/// Порог корреляции нормированных огибающих в каждом окне. Реальные дубли
/// из кейса unknown 2/3 дают 0.985–0.999; 0.97 отделяет их от разного
/// контента (там corr, как правило, < 0.5) с запасом.
pub const CORR_THRESHOLD: f32 = 0.97;

/// Минимальное различие крест-фактора (дБ), при котором хранителем становится
/// более динамичный файл, а не более громкий (правило владельца: > 2 дБ).
pub const CREST_MARGIN_DB: f32 = 2.0;

/// Потолок пика (нормированная амплитуда): `true_peak ≥` значения трактуется
/// как «пик у потолка» — признак клипа/лимитера при исключении кандидатов.
pub const PEAK_CEILING: f32 = 0.999;

/// Минимальная серия подряд идущих сэмплов у потолка, фиксируемая как клип
/// (для отчёта; порог решения — [`ClipStats::is_clipping`]).
pub const MIN_CLIP_RUN: u32 = 3;

/// Размер фрейма огибающей (мс).
pub const ENV_MS: u64 = 50;

/// Длительность окна сравнения (сек).
pub const WINDOW_SECS: u64 = 30;

/// Число окон сравнения: старт+60 с, середина, конец−90 с.
pub const WINDOW_COUNT: usize = 3;

/// Отступ первого окна от начала записи (сек) — пропускаем шум включения.
pub const WINDOW_LEAD_SECS: u64 = 60;

/// Отступ последнего окна от конца записи (сек; окно 30 с ⇒ старт за 120 с
/// до конца).
pub const WINDOW_TAIL_SECS: u64 = 90;

/// Размер блока чтения при скане клипа (сек) — файл целиком не читается.
pub const SCAN_BLOCK_SECS: u64 = 30;

/// Размер порции чтения окна (сек) — сотни КБ памяти, не весь файл.
const READ_CHUNK_SECS: u64 = 1;

/// Формат PCM в `fmt `-чанке (целые сэмплы).
const PCM_FORMAT: u16 = 1;

/// Поддерживаемые разрядности (вход с рекордера всегда PCM WAV; KISS).
const SUPPORTED_BITS: [u16; 3] = [16, 24, 32];

/// Поддерживаемое число каналов.
const SUPPORTED_CHANNELS: [u16; 2] = [1, 2];

/// Пол RMS (−120 дБФС): защита математики крест-фактора от полной тишины.
const RMS_FLOOR: f32 = 1e-6;

// --- модели данных -----------------------------------------------------------------

/// Параметры PCM WAV из RIFF-заголовка.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    /// Частота дискретизации, Гц.
    pub sample_rate: u32,
    /// Разрядность (16/24/32 бит, целые).
    pub bits: u16,
    /// Число каналов (1/2).
    pub channels: u16,
    /// Число фреймов данных (фрейм = сэмпл всех каналов).
    pub frames: u64,
    /// Абсолютное смещение чанка `data` в файле.
    pub data_offset: u64,
    /// Размер данных (обрезан по длине файла, если заголовок врёт).
    pub data_size: u64,
}

impl WavInfo {
    /// Байт на фрейм (все каналы).
    pub fn block_align(&self) -> u64 {
        block_align_bytes(self.channels, self.bits)
    }
}

/// Байт на фрейм: каналы × байты на сэмпл.
fn block_align_bytes(channels: u16, bits: u16) -> u64 {
    u64::from(channels) * u64::from(bits) / 8
}

/// Клип-статистика всего файла (глобальный блок-скан).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipStats {
    /// Истинный пик: max |сэмпл| в нормировке 0..1.
    pub true_peak: f32,
    /// Доля сэмплов |x| ≥ [`PEAK_CEILING`].
    pub clipped_share: f64,
    /// Максимальная серия подряд сэмплов |x| ≥ [`PEAK_CEILING`].
    pub max_clip_run: u32,
}

impl ClipStats {
    /// Клипующий файл: пик у потолка (правило хранителя, п. 1 —
    /// «исключить клипующие (пики у потолка)»). Доля/серия у потолка
    /// физически невозможны без такого пика, поэтому решение — по пику.
    pub fn is_clipping(&self) -> bool {
        self.true_peak >= PEAK_CEILING
    }
}

/// Анализ одного окна (30 с): огибающая пиков по фреймам 50 мс + RMS.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowData {
    /// Пик-огибающая: max(|L|,|R|) по фреймам 50 мс (нормировка 0..1).
    pub envelope: Vec<f32>,
    /// Пик окна (max по огибающей).
    pub peak: f32,
    /// Медианный RMS окна (по фреймам 50 мс).
    pub median_rms: f32,
}

/// Полный анализ файла для дедупликации: заголовок + окна + клип-скан.
#[derive(Debug, Clone, PartialEq)]
pub struct FileAnalysis {
    /// Параметры WAV.
    pub info: WavInfo,
    /// Анализы трёх окон.
    pub windows: [WindowData; WINDOW_COUNT],
    /// Клип-статистика всего файла.
    pub clip: ClipStats,
}

/// Глобальный крест-фактор файла (дБ): dbfs истинного пика всего файла
/// (из блок-скана клипа) минус медиана медианных RMS трёх окон (в дБ).
/// Не оконный: окна покрывают лишь 90 с, и лимитер может не проявиться
/// в их пиках, тогда как блок-скан видит весь файл (кейс unknown 2/3).
fn global_crest(a: &FileAnalysis) -> f32 {
    let mut rms_db: Vec<f32> = a
        .windows
        .iter()
        .map(|w| ratio_db(w.median_rms.max(RMS_FLOOR), 1.0))
        .collect();
    ratio_db(a.clip.true_peak.max(RMS_FLOOR), 1.0) - median(&mut rms_db)
}

/// Средний медианный RMS файла по окнам — метрика громкости хранителя.
fn mean_loudness(a: &FileAnalysis) -> f32 {
    a.windows.iter().map(|w| w.median_rms).sum::<f32>() / WINDOW_COUNT as f32
}

/// Результат сравнения пары анализов (a — первый, b — второй).
#[derive(Debug, Clone, PartialEq)]
pub struct PairVerdict {
    /// Совпадение контента (дубль) по порогам.
    pub is_duplicate: bool,
    /// Corr нормированных огибающих по окнам [старт, середина, конец].
    pub corr: [f32; WINDOW_COUNT],
    /// Гейн b относительно a (дБ; положительный = b громче); медиана по окнам.
    pub gain_db: f32,
    /// Глобальный крест-фактор a, дБ ([global_crest]).
    pub crest_a_db: f32,
    /// Глобальный крест-фактор b, дБ ([global_crest]).
    pub crest_b_db: f32,
    /// Клип-статистика a (для отчёта).
    pub clip_a: ClipStats,
    /// Клип-статистика b (для отчёта).
    pub clip_b: ClipStats,
}

/// Причина выбора хранителя (для отчёта).
#[derive(Debug, Clone, PartialEq)]
pub enum KeeperReason {
    /// Кандидат в кластере один — сравнение не требовалось.
    SoleCandidate,
    /// Прочие кандидаты клипуют — хранитель единственный «чистый».
    OnlyNonClipped,
    /// Клипующие исключены; хранитель динамичнее лучшего из остальных
    /// на `margin_db` (НЕ лимитерованный, даже если тише).
    CleanerDynamics { margin_db: f32 },
    /// Крест-факторы близки; хранитель громче лучшего из остальных
    /// на `margin_db` (по RMS).
    Louder { margin_db: f32 },
    /// Все кандидаты клипуют — хранитель с максимальным крест-фактором.
    AllClipped { margin_db: f32 },
}

/// Выбор хранителя: индекс в переданном списке + причина (для отчёта).
#[derive(Debug, Clone, PartialEq)]
pub struct KeeperChoice {
    /// Индекс хранителя в переданном списке кандидатов.
    pub index: usize,
    /// Почему выбран именно он.
    pub reason: KeeperReason,
}

/// Совпадение внутри группы: дубль → его хранитель.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateMatch {
    /// Индекс дубля в группе.
    pub duplicate: usize,
    /// Индекс хранителя в группе.
    pub kept: usize,
    /// Метрики сравнения пары (хранитель, дубль).
    pub verdict: PairVerdict,
}

/// Итог разбора группы кандидатов: кто движку, кто дубль.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupResolution {
    /// Хранители кластеров (обрабатываются движком).
    pub keepers: Vec<usize>,
    /// Совпадений не нашлось — уникальные (обрабатываются движком).
    pub unique: Vec<usize>,
    /// Дубли с указанием хранителя (снимаются с обработки).
    pub duplicates: Vec<DuplicateMatch>,
}

// --- разбор WAV --------------------------------------------------------------------

/// Потоковый разбор RIFF/WAVE: ищет `fmt ` и `data`, остальные чанки
/// пропускает seek-ом (чанки `bext`/`LIST` рекордера не мешают).
/// `None` — не PCM WAV: не RIFF/WAVE, нет `fmt`/`data`, формат не PCM,
/// биты/каналы вне поддержки, данные пустые. Такой файл считается
/// уникальным (с примечанием в отчёте) — вход с рекордера всегда PCM WAV.
pub fn parse_wav<R: Read + Seek>(reader: &mut R) -> Option<WavInfo> {
    let mut riff = [0_u8; 12];
    reader.read_exact(&mut riff).ok()?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return None;
    }
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (формат, каналы, Гц, биты)
    let mut data: Option<(u64, u64)> = None; // (смещение, размер)
    while fmt.is_none() || data.is_none() {
        let mut header = [0_u8; 8];
        // Обрыв потока до нужных чанков → заголовок неполный, не наш файл.
        reader.read_exact(&mut header).ok()?;
        let size = u32::from_le_bytes(header[4..8].try_into().ok()?) as u64;
        match &header[0..4] {
            b"fmt " if size >= 16 => {
                fmt = Some(read_fmt(reader)?);
                skip_chunk(reader, size - 16)?;
            }
            b"data" => {
                let offset = reader.stream_position().ok()?;
                data = Some((offset, size));
                skip_chunk(reader, size)?;
            }
            _ => skip_chunk(reader, size)?,
        }
    }
    let (format, channels, sample_rate, bits) = fmt?;
    if format != PCM_FORMAT
        || !SUPPORTED_BITS.contains(&bits)
        || !SUPPORTED_CHANNELS.contains(&channels)
        || sample_rate == 0
    {
        return None;
    }
    let (data_offset, declared) = data?;
    // Недописанный рекордером хвост: размер data обрезаем по длине файла.
    let file_len = reader.seek(SeekFrom::End(0)).ok()?;
    let data_size = declared.min(file_len.saturating_sub(data_offset));
    if data_size == 0 {
        return None;
    }
    let frames = data_size / block_align_bytes(channels, bits);
    Some(WavInfo { sample_rate, bits, channels, frames, data_offset, data_size })
}

/// Чтение обязательных полей `fmt `-чанка (PCM: формат, каналы, Гц, биты).
fn read_fmt<R: Read>(reader: &mut R) -> Option<(u16, u16, u32, u16)> {
    let mut buf = [0_u8; 16];
    reader.read_exact(&mut buf).ok()?;
    Some((
        u16::from_le_bytes(buf[0..2].try_into().ok()?),
        u16::from_le_bytes(buf[2..4].try_into().ok()?),
        u32::from_le_bytes(buf[4..8].try_into().ok()?),
        u16::from_le_bytes(buf[14..16].try_into().ok()?),
    ))
}

/// Пропуск тела чанка с выравниванием RIFF на чётное число байт.
fn skip_chunk<R: Read + Seek>(reader: &mut R, size: u64) -> Option<()> {
    let padded = size + (size & 1);
    reader.seek(SeekFrom::Current(padded as i64)).ok().map(|_| ())
}

// --- чтение и анализ ---------------------------------------------------------------

/// Полный анализ локального файла. `Ok(None)` — не PCM WAV (файл считается
/// уникальным); ошибки I/O отдаются вызывающему.
pub fn analyze_file(path: &Path) -> std::io::Result<Option<FileAnalysis>> {
    let mut file = File::open(path)?;
    let Some(info) = parse_wav(&mut file) else { return Ok(None) };
    let windows = [
        read_window(&mut file, &info, window_start(&info, 0))?,
        read_window(&mut file, &info, window_start(&info, 1))?,
        read_window(&mut file, &info, window_start(&info, 2))?,
    ];
    let clip = scan_clip(&mut file, &info)?;
    Ok(Some(FileAnalysis { info, windows, clip }))
}

/// Старт окна (в фреймах): [старт+60 с, середина−15 с, конец−120 с],
/// прижатый к [0, frames − окно]; файл короче окна — окно с нуля.
fn window_start(info: &WavInfo, window: usize) -> u64 {
    let rate = u64::from(info.sample_rate);
    let win = WINDOW_SECS * rate;
    let raw = [
        WINDOW_LEAD_SECS * rate,
        (info.frames / 2).saturating_sub(win / 2),
        info.frames.saturating_sub((WINDOW_TAIL_SECS + WINDOW_SECS) * rate),
    ];
    raw[window].min(info.frames.saturating_sub(win))
}

/// Сэмплов на один фрейм огибающей (50 мс, все каналы).
fn env_point_samples(info: &WavInfo) -> u64 {
    u64::from(info.sample_rate) * ENV_MS / 1000 * u64::from(info.channels)
}

/// Чтение окна: пик-огибающая по фреймам 50 мс + пик + медианный RMS.
/// Читает секунду за секундой — в памяти только порции данных, не весь файл.
fn read_window<R: Read + Seek>(
    reader: &mut R,
    info: &WavInfo,
    start_frame: u64,
) -> std::io::Result<WindowData> {
    let rate = u64::from(info.sample_rate);
    let len = (WINDOW_SECS * rate).min(info.frames.saturating_sub(start_frame));
    reader.seek(SeekFrom::Start(info.data_offset + start_frame * info.block_align()))?;
    let mut envelope = Vec::new();
    let mut frame_rms = Vec::new();
    let mut raw = Vec::new();
    let mut samples = Vec::new();
    let mut point = PointAcc::new(env_point_samples(info));
    let mut left = len;
    while left > 0 {
        let frames = (READ_CHUNK_SECS * rate).min(left);
        read_samples(reader, frames * info.block_align(), info.bits, &mut raw, &mut samples)?;
        left -= frames;
        for &sample in &samples {
            if let Some((peak, rms)) = point.push(sample) {
                envelope.push(peak);
                frame_rms.push(rms);
            }
        }
    }
    if let Some((peak, rms)) = point.flush() {
        envelope.push(peak);
        frame_rms.push(rms);
    }
    let peak = envelope.iter().copied().fold(0.0_f32, f32::max);
    Ok(WindowData { envelope, peak, median_rms: median(&mut frame_rms) })
}

/// Аккумулятор одного фрейма огибающей (50 мс): пик и RMS по сэмплам.
struct PointAcc {
    /// Сэмплов во фрейме.
    len: u64,
    peak: f32,
    sq: f64,
    n: u64,
}

impl PointAcc {
    fn new(len: u64) -> Self {
        Self { len, peak: 0.0, sq: 0.0, n: 0 }
    }

    /// Добавить сэмпл; на границе фрейма отдаёт (пик, RMS) и сбрасывается.
    fn push(&mut self, sample: f32) -> Option<(f32, f32)> {
        let abs = sample.abs();
        self.peak = self.peak.max(abs);
        self.sq += f64::from(sample * sample);
        self.n += 1;
        if self.n < self.len { None } else { self.take() }
    }

    /// Закрыть неполный хвостовой фрейм (короткое окно/файл).
    fn flush(&mut self) -> Option<(f32, f32)> {
        if self.n == 0 { None } else { self.take() }
    }

    fn take(&mut self) -> Option<(f32, f32)> {
        let point = (self.peak, (self.sq / self.n as f64).sqrt() as f32);
        *self = Self::new(self.len);
        Some(point)
    }
}

/// Глобальный скан файла блоками по [`SCAN_BLOCK_SECS`]: истинный пик,
/// доля сэмплов у потолка, максимальная серия у потолка.
fn scan_clip<R: Read + Seek>(reader: &mut R, info: &WavInfo) -> std::io::Result<ClipStats> {
    let rate = u64::from(info.sample_rate);
    reader.seek(SeekFrom::Start(info.data_offset))?;
    let (mut peak, mut clipped, mut total) = (0.0_f32, 0_u64, 0_u64);
    let (mut run, mut best_run) = (0_u32, 0_u32);
    let mut raw = Vec::new();
    let mut samples = Vec::new();
    let mut left = info.frames;
    while left > 0 {
        let frames = (SCAN_BLOCK_SECS * rate).min(left);
        read_samples(reader, frames * info.block_align(), info.bits, &mut raw, &mut samples)?;
        left -= frames;
        for &sample in &samples {
            let abs = sample.abs();
            peak = peak.max(abs);
            total += 1;
            if abs >= PEAK_CEILING {
                clipped += 1;
                run += 1;
                best_run = best_run.max(run);
            } else {
                run = 0;
            }
        }
    }
    Ok(ClipStats {
        true_peak: peak,
        clipped_share: clipped as f64 / total.max(1) as f64,
        max_clip_run: best_run,
    })
}

/// Чтение `bytes` байт данных и декодирование в f32 (буферы переиспользуются).
fn read_samples<R: Read>(
    reader: &mut R,
    bytes: u64,
    bits: u16,
    raw: &mut Vec<u8>,
    out: &mut Vec<f32>,
) -> std::io::Result<()> {
    raw.clear();
    raw.resize(bytes as usize, 0);
    reader.read_exact(raw)?;
    decode_samples(raw, bits, out);
    Ok(())
}

/// Декод PCM int (16/24/32) в нормированные f32 (−1..1, деление на максимум
/// знакового диапазона); неполный хвостовой сэмпл отбрасывается. Биты
/// валидированы при разборе заголовка, иная разрядность недостижима.
fn decode_samples(bytes: &[u8], bits: u16, out: &mut Vec<f32>) {
    out.clear();
    match bits {
        16 => out.extend(
            bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0),
        ),
        24 => out.extend(bytes.chunks_exact(3).map(|b| {
            let v = b[0] as i32 | ((b[1] as i32) << 8) | ((b[2] as i8 as i32) << 16);
            v as f32 / 8_388_608.0
        })),
        32 => out.extend(bytes.chunks_exact(4).map(|b| {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
        })),
        _ => {}
    }
}

// --- сравнение ---------------------------------------------------------------------

/// Сравнение пары: дубль при равном числе фреймов И corr нормированных
/// огибающих ≥ [`CORR_THRESHOLD`] в каждом из трёх окон (прямое
/// выравнивание, сдвиг 0). Чистая функция — работает на готовых анализах.
pub fn compare_pair(a: &FileAnalysis, b: &FileAnalysis) -> PairVerdict {
    let corr =
        std::array::from_fn(|w| envelope_corr(&a.windows[w].envelope, &b.windows[w].envelope));
    PairVerdict {
        is_duplicate: a.info.frames == b.info.frames && corr.iter().all(|c| *c >= CORR_THRESHOLD),
        corr,
        gain_db: median(&mut gain_windows(a, b)),
        crest_a_db: global_crest(a),
        crest_b_db: global_crest(b),
        clip_a: a.clip,
        clip_b: b.clip,
    }
}

/// Гейн b относительно a по окнам (дБ, по медианным RMS); окна с тишиной
/// (RMS = 0) в расчёт не идут.
fn gain_windows(a: &FileAnalysis, b: &FileAnalysis) -> Vec<f32> {
    a.windows
        .iter()
        .zip(&b.windows)
        .filter(|(wa, wb)| wa.median_rms > 0.0 && wb.median_rms > 0.0)
        .map(|(wa, wb)| ratio_db(wb.median_rms, wa.median_rms))
        .collect()
}

/// Корреляция Пирсона нормированных (по максимуму) огибающих. Пустые,
/// полностью тихие огибающие или разная длина → 0 (не дубль).
fn envelope_corr(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (max_a, max_b) = (max_abs(a), max_abs(b));
    if max_a <= 0.0 || max_b <= 0.0 {
        return 0.0;
    }
    pearson(&normalized(a, max_a), &normalized(b, max_b))
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0_f32, |m, v| m.max(*v))
}

/// Нормировка огибающей к максимуму 1.0.
fn normalized(values: &[f32], max: f32) -> Vec<f64> {
    values.iter().map(|v| f64::from(*v / max)).collect()
}

/// Коэффициент корреляции Пирсона (в f64 для точности); вырождение → 0.
fn pearson(x: &[f64], y: &[f64]) -> f32 {
    let n = x.len() as f64;
    let (mean_x, mean_y) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (xi, yi) in x.iter().zip(y) {
        let (dx, dy) = (*xi - mean_x, *yi - mean_y);
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 0.0 || var_y <= 0.0 {
        return 0.0;
    }
    (cov / (var_x.sqrt() * var_y.sqrt())) as f32
}

/// 20·log10(числитель/знаменатель); числитель 0 → −∞ (тишина), знаменатель
/// 0 невозможен: пол [`RMS_FLOOR`] ставят вызывающие.
fn ratio_db(num: f32, den: f32) -> f32 {
    if num <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * (num / den).log10()
    }
}

/// Медиана (чётная длина — среднее двух центральных); пустой вход → 0.
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

// --- выбор хранителя и разбор группы ------------------------------------------------

/// Правило хранителя (владелец, 2026-10-13) для кластера дублей.
/// Порядок: клипующие (пик у потолка) исключаются; затем различие
/// глобального крест-фактора ([global_crest]) больше [`CREST_MARGIN_DB`] →
/// более динамичный, даже если тише; иначе — более громкий (RMS).
/// Клипуют все → максимум крест-фактора.
pub fn select_keeper(candidates: &[&FileAnalysis]) -> KeeperChoice {
    if candidates.len() == 1 {
        return KeeperChoice { index: 0, reason: KeeperReason::SoleCandidate };
    }
    let crest = |i: usize| global_crest(candidates[i]);
    let loud = |i: usize| mean_loudness(candidates[i]);
    let clean: Vec<usize> =
        (0..candidates.len()).filter(|&i| !candidates[i].clip.is_clipping()).collect();
    let pool: Vec<usize> = if clean.is_empty() { (0..candidates.len()).collect() } else { clean };
    if pool.len() == 1 {
        return KeeperChoice { index: pool[0], reason: KeeperReason::OnlyNonClipped };
    }
    let (best, margin) = top_margin(&pool, crest);
    if margin > CREST_MARGIN_DB {
        let reason = if pool.len() == candidates.len() && candidates.iter().all(|c| c.clip.is_clipping()) {
            KeeperReason::AllClipped { margin_db: margin }
        } else {
            KeeperReason::CleanerDynamics { margin_db: margin }
        };
        return KeeperChoice { index: best, reason };
    }
    let (loudest, _) = top_margin(&pool, loud);
    let rest: Vec<usize> = pool.iter().copied().filter(|&i| i != loudest).collect();
    let (second, _) = top_margin(&rest, loud);
    KeeperChoice {
        index: loudest,
        reason: KeeperReason::Louder { margin_db: ratio_db(loud(loudest), loud(second)) },
    }
}

/// Лучший по метрике и его отрыв от второго; при равенстве — меньший
/// индекс (детерминизм для отчёта). Единственный элемент — отрыв 0.
fn top_margin(indices: &[usize], metric: impl Fn(usize) -> f32) -> (usize, f32) {
    let mut ranked: Vec<usize> = indices.to_vec();
    ranked.sort_by(|a, b| metric(*b).total_cmp(&metric(*a)).then(a.cmp(b)));
    let margin = if ranked.len() >= 2 { metric(ranked[0]) - metric(ranked[1]) } else { 0.0 };
    (ranked[0], margin)
}

/// Разбор группы кандидатов (одинаковые data_size и frames): попарные
/// сравнения → кластеры дублей (DSU) → хранитель по правилу на кластер.
pub fn resolve_group(analyses: &[FileAnalysis]) -> GroupResolution {
    let n = analyses.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for (i, a) in analyses.iter().enumerate() {
        for (k, b) in analyses[i + 1..].iter().enumerate() {
            if compare_pair(a, b).is_duplicate {
                dsu_union(&mut parent, i, i + 1 + k);
            }
        }
    }
    let mut clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        clusters.entry(dsu_find(&mut parent, i)).or_default().push(i);
    }
    // Детерминизм отчёта: кластеры по минимальному индексу, члены по возрастанию.
    let mut roots: Vec<usize> = clusters.keys().copied().collect();
    roots.sort_unstable();
    let mut resolution = GroupResolution::default();
    for root in roots {
        let members = &clusters[&root];
        if members.len() == 1 {
            resolution.unique.push(members[0]);
            continue;
        }
        let refs: Vec<&FileAnalysis> = members.iter().map(|&m| &analyses[m]).collect();
        let choice = select_keeper(&refs);
        let kept = members[choice.index];
        resolution.keepers.push(kept);
        for &dup in members.iter().filter(|&&m| m != kept) {
            resolution.duplicates.push(DuplicateMatch {
                duplicate: dup,
                kept,
                verdict: compare_pair(&analyses[kept], &analyses[dup]),
            });
        }
    }
    resolution
}

/// Минимальный DSU: корень множества (сжатие пути пополам).
fn dsu_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Объединение множеств: больший корень подвешивается к меньшему (детерминизм).
fn dsu_union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (dsu_find(parent, a), dsu_find(parent, b));
    if ra != rb {
        let (root, child) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[child] = root;
    }
}

// --- данные для отчёта --------------------------------------------------------------

/// Запись отчёта о дедупликации: пара дубль → хранитель.
#[derive(Debug, Clone, PartialEq)]
pub struct PairReport {
    /// Имя файла-дубля.
    pub duplicate: String,
    /// Имя хранителя.
    pub kept: String,
    /// Corr по окнам [старт, середина, конец].
    pub corr: [f32; WINDOW_COUNT],
    /// Гейн дубля относительно хранителя, дБ.
    pub gain_db: f32,
    /// Крест-фактор хранителя, дБ.
    pub crest_kept_db: f32,
    /// Крест-фактор дубля, дБ.
    pub crest_dup_db: f32,
    /// Хранитель клипует (пик у потолка).
    pub kept_clipping: bool,
    /// Дубль клипует (пик у потолка).
    pub dup_clipping: bool,
    /// Дубль переименован в `DUPLICATE_*` (хранитель дошёл до Ok).
    pub renamed: bool,
    /// Текст ошибки rename (если переименование не удалось).
    pub rename_error: Option<String>,
}

/// Сводка стадии дедупликации для отчёта.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DedupSummary {
    /// Пары дубль → хранитель.
    pub pairs: Vec<PairReport>,
    /// Файлы с нечитаемым WAV-заголовком (считаны уникальными).
    pub unparseable: Vec<String>,
}

#[cfg(test)]
pub(crate) mod wav_fix {
    //! Генератор WAV-фикстур для тестов (чистый Rust, std only): детерминированные
    //! сигналы + энкодер PCM 16/24/32. Реальные WAV в тестах не используются.

    use std::fs;
    use std::path::Path;

    /// Энкодер PCM WAV (format=1) в байты: стерео/моно, 16/24/32 бит.
    pub fn encode_wav(rate: u32, bits: u16, channels: u16, samples: &[f32]) -> Vec<u8> {
        let sample_bytes = (bits / 8) as usize;
        let data_size = (samples.len() * sample_bytes) as u32;
        let block_align = (channels * bits / 8) as u32;
        let mut bytes = Vec::with_capacity(44 + data_size as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * block_align).to_le_bytes()); // byte_rate
        bytes.extend_from_slice(&(block_align as u16).to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            encode_sample(*sample, bits, &mut bytes);
        }
        bytes
    }

    /// Запись WAV-фикстуры на диск.
    pub fn write_wav(path: &Path, rate: u32, bits: u16, channels: u16, samples: &[f32]) {
        fs::write(path, encode_wav(rate, bits, channels, samples)).expect("write wav fixture");
    }

    /// Сэмпл → little-endian PCM целое (нормировка на максимум диапазона).
    fn encode_sample(sample: f32, bits: u16, out: &mut Vec<u8>) {
        let clamped = sample.clamp(-1.0, 1.0);
        match bits {
            16 => out.extend_from_slice(&((clamped * 32_767.0) as i16).to_le_bytes()),
            24 => {
                let v = (clamped * 8_388_607.0) as i32;
                out.extend_from_slice(&[v as u8, (v >> 8) as u8, (v >> 16) as u8]);
            }
            32 => out.extend_from_slice(&((clamped * 2_147_483_647.0) as i32).to_le_bytes()),
            _ => panic!("unsupported bit depth for fixture: {bits}"),
        }
    }

    /// Детерминированная «запись»: тональные вспышки с паузами (гейт 12.5%
    /// времени) — высокий крест-фактор, как у живой записи с рекордера.
    pub fn burst_signal(rate: u32, seconds: u32) -> Vec<f32> {
        (0..u64::from(rate) * u64::from(seconds))
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let gate = if t % 2.0 < 0.25 { 1.0 } else { 0.001 };
                let tone = 0.7 * (t * 220.0).sin() + 0.2 * (t * 331.0).sin();
                (tone * gate) as f32
            })
            .collect()
    }

    /// Запись с другим контентом (для негативного кейса сравнения).
    pub fn other_signal(rate: u32, seconds: u32) -> Vec<f32> {
        (0..u64::from(rate) * u64::from(seconds))
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let gate = if t % 3.0 < 0.4 { 1.0 } else { 0.001 };
                let tone = 0.6 * (t * 442.0).sin();
                (tone * gate) as f32
            })
            .collect()
    }

    /// «Гейн + лимитер»: умножение с жёсткой нормализацией к потолку —
    /// так на рекордере появляется громкая пережатая копия записи.
    pub fn gain_limited(samples: &[f32], gain: f32, ceiling: f32) -> Vec<f32> {
        samples.iter().map(|s| (s * gain).clamp(-ceiling, ceiling)).collect()
    }

    /// Жёсткий клиппинг к ±1.0 (пики у потолка, длинные серии у потолка).
    pub fn hard_clipped(samples: &[f32], gain: f32) -> Vec<f32> {
        samples.iter().map(|s| (s * gain).clamp(-1.0, 1.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    use super::*;

    /// Уникальная локальная директория теста (параллельные не пересекаются).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("audio_split_dedup_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Анализ WAV-фикстуры с диска (паника = ошибка тест-фикстуры).
    fn analysis_of(path: &Path) -> FileAnalysis {
        analyze_file(path).expect("analysis io").expect("valid wav fixture")
    }

    /// Частота 8 кГц: компактные фикстуры при полных 3 окнах методики.
    const RATE: u32 = 8_000;
    /// Длина «записи» в тестах с полными окнами: 300 с = 5 мин.
    const SECONDS: u32 = 300;

    // --- парсер заголовка -------------------------------------------------------

    #[test]
    fn parse_wav_reads_valid_24bit_stereo_header() {
        let samples = vec![0.5, -0.5, 0.25, -0.25, 0.0, 0.0]; // 3 стерео-фрейма
        let bytes = wav_fix::encode_wav(44_100, 24, 2, &samples);

        let info = parse_wav(&mut Cursor::new(bytes)).expect("valid wav");

        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.bits, 24);
        assert_eq!(info.channels, 2);
        assert_eq!(info.frames, 3);
        assert_eq!(info.data_size, 18);
        assert_eq!(info.data_offset, 44);
    }

    #[test]
    fn parse_wav_rejects_broken_and_foreign_bytes() {
        let good = wav_fix::encode_wav(44_100, 16, 1, &[0.1, -0.1]);
        // Обрыв внутри заголовка и чанков.
        for cut in [0, 4, 12, 30, 40] {
            assert_eq!(parse_wav(&mut Cursor::new(good[..cut].to_vec())), None, "cut at {cut}");
        }
        // Совсем не WAV.
        assert_eq!(parse_wav(&mut Cursor::new(b"not a wav at all.....".to_vec())), None);
    }

    #[test]
    fn parse_wav_rejects_non_pcm_format_and_unsupported_layout() {
        let samples = [0.1_f32, -0.1];
        // format=3 (IEEE float) — не PCM.
        let mut float_wav = wav_fix::encode_wav(44_100, 16, 1, &samples);
        float_wav[20..22].copy_from_slice(&3_u16.to_le_bytes());
        assert_eq!(parse_wav(&mut Cursor::new(float_wav)), None, "float format must be rejected");
        // Поддерживаются только 1/2 канала: ручной заголовок с 4 каналами.
        assert_eq!(parse_wav(&mut Cursor::new(plain_wav_bytes(44_100, 16, 4))), None, "4 канала");
        // И только 16/24/32 бит: ручной заголовок с 8 бит.
        assert_eq!(parse_wav(&mut Cursor::new(plain_wav_bytes(44_100, 8, 1))), None, "8 бит");
    }

    /// Минимальный валидный WAV-заголовок (44 байта, без данных) с заданной
    /// разрядностью и числом каналов — для кейсов вне поддержки фикстурой.
    fn plain_wav_bytes(rate: u32, bits: u16, channels: u16) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&64_u32.to_le_bytes());
        bytes
    }

    #[test]
    fn parse_wav_finds_fmt_even_after_data_chunk() {
        // Редкий порядок чанков: data раньше fmt.
        let body = wav_fix::encode_wav(8_000, 16, 1, &[0.5, -0.5]);
        let samples_bytes = &body[44..];
        let mut reordered = Vec::new();
        reordered.extend_from_slice(b"RIFF");
        reordered.extend_from_slice(&((36 + samples_bytes.len()) as u32).to_le_bytes());
        reordered.extend_from_slice(b"WAVE");
        reordered.extend_from_slice(b"data");
        reordered.extend_from_slice(&(samples_bytes.len() as u32).to_le_bytes());
        reordered.extend_from_slice(samples_bytes);
        reordered.extend_from_slice(&body[12..36]); // чанк "fmt " целиком

        let info = parse_wav(&mut Cursor::new(reordered)).expect("data-before-fmt wav");

        assert_eq!(info.frames, 2);
        assert_eq!(info.bits, 16);
    }

    #[test]
    fn parse_wav_clamps_data_size_to_file_length() {
        let mut bytes = wav_fix::encode_wav(8_000, 16, 1, &[0.5, -0.5]);
        // Заголовок заявляет 1 МБ данных, файл короче (недописанная запись).
        bytes[40..44].copy_from_slice(&1_000_000_u32.to_le_bytes());

        let info = parse_wav(&mut Cursor::new(bytes)).expect("truncated wav");

        assert_eq!(info.data_size, 4, "declared size clamped to real bytes");
        assert_eq!(info.frames, 2);
    }

    // --- декодирование ------------------------------------------------------------

    #[test]
    fn decode_samples_handles_16_24_32_bit() {
        let mut out = Vec::new();
        decode_samples(&0_i16.to_le_bytes(), 16, &mut out);
        assert!((out[0]).abs() < 1e-6, "нулевое int16 → 0.0");
        decode_samples(&(-32_768_i16).to_le_bytes(), 16, &mut out);
        assert!((out[0] + 1.0).abs() < 1e-6, "min int16 → −1.0");
        decode_samples(&32_767_i16.to_le_bytes(), 16, &mut out);
        assert!(out[0] > 0.999, "max int16 → ~1.0");

        decode_samples(
            &[0x00, 0x00, 0x80, 0x00, 0x00, 0x40], // −1.0 и +0.5 в 24 бит
            24,
            &mut out,
        );
        assert!((out[0] + 1.0).abs() < 1e-6, "24-bit sign extension: {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "24-bit +0.5: {}", out[1]);

        decode_samples(&(-2_147_483_648_i32).to_le_bytes(), 32, &mut out);
        assert!((out[0] + 1.0).abs() < 1e-6);
        // Неподдерживаемая разрядность не декодирует ничего (недостижимо).
        decode_samples(&[0, 0, 0], 8, &mut out);
        assert!(out.is_empty());
    }

    // --- флагманский сценарий: чистая и гейн-копия (кейс unknown 2/3) --------------

    #[test]
    fn gain_copy_is_duplicate_and_clean_record_wins_by_crest() {
        let dir = temp_dir("flagship");
        let clean = wav_fix::burst_signal(RATE, SECONDS);
        let loud = wav_fix::gain_limited(&clean, 6.31, 0.96); // +16 дБ с лимитером
        let (clean_path, loud_path) = (dir.join("clean.wav"), dir.join("loud.wav"));
        wav_fix::write_wav(&clean_path, RATE, 16, 1, &clean);
        wav_fix::write_wav(&loud_path, RATE, 16, 1, &loud);

        let clean_a = analysis_of(&clean_path);
        let loud_a = analysis_of(&loud_path);

        assert_eq!(clean_a.info.frames, loud_a.info.frames, "длительность до фрейма");
        assert!(clean_a.clip.true_peak < 0.95, "чистая запись без клипа: {}", clean_a.clip.true_peak);
        assert!(loud_a.clip.true_peak >= 0.9, "копия пережита к потолку");

        let verdict = compare_pair(&clean_a, &loud_a);
        assert!(verdict.is_duplicate, "corr по окнам: {:?}", verdict.corr);
        assert!(verdict.corr.iter().all(|c| *c >= 0.98), "corr: {:?}", verdict.corr);
        assert!(verdict.gain_db > 1.0, "копия громче: +{:.1} дБ", verdict.gain_db);
        let crest_margin = verdict.crest_a_db - verdict.crest_b_db;
        assert!(
            crest_margin > 2.0,
            "лимитер срезает крест-фактор: {:.1} против {:.1}",
            verdict.crest_a_db,
            verdict.crest_b_db
        );

        // Правило хранителя: держим более динамичного, даже если он тише.
        let choice = select_keeper(&[&clean_a, &loud_a]);
        assert_eq!(choice.index, 0, "хранитель — чистая запись");
        assert!(
            matches!(choice.reason, KeeperReason::CleanerDynamics { margin_db } if margin_db > 2.0),
            "{:?}",
            choice.reason
        );
        assert!(mean_loudness(&clean_a) < mean_loudness(&loud_a), "хранитель тише — и всё равно он");
    }

    #[test]
    fn close_crest_falls_back_to_louder() {
        let dir = temp_dir("louder");
        // Обе копии в пределах 1.0: гейн не должен упираться в потолок,
        // иначе это уже лимитер и крест-факторы разъедутся. Амплитуды
        // не крошечные — шум квантования 16 бит не искажает крест-фактор.
        let quiet: Vec<f32> = wav_fix::burst_signal(RATE, 10).iter().map(|s| s * 0.5).collect();
        let loud: Vec<f32> = quiet.iter().map(|s| s * 2.0).collect(); // тот же крест-фактор
        let (quiet_path, loud_path) = (dir.join("q.wav"), dir.join("l.wav"));
        wav_fix::write_wav(&quiet_path, RATE, 16, 1, &quiet);
        wav_fix::write_wav(&loud_path, RATE, 16, 1, &loud);

        let (quiet_a, loud_a) = (analysis_of(&quiet_path), analysis_of(&loud_path));
        let verdict = compare_pair(&quiet_a, &loud_a);
        assert!(verdict.is_duplicate, "масштабирование — тоже дубль: {:?}", verdict.corr);
        assert!(
            (verdict.crest_a_db - verdict.crest_b_db).abs() < 0.5,
            "у масштабированных копий крест-фактор совпадает: a={:.3} b={:.3} rms_q={:.6} rms_l={:.6}",
            verdict.crest_a_db,
            verdict.crest_b_db,
            quiet_a.windows[0].median_rms,
            loud_a.windows[0].median_rms
        );

        let choice = select_keeper(&[&quiet_a, &loud_a]);
        assert_eq!(choice.index, 1, "динамика равна — держим более громкий");
        assert!(
            matches!(choice.reason, KeeperReason::Louder { margin_db } if margin_db > 5.0),
            "crest_q={:.3} crest_l={:.3} reason={:?}",
            global_crest(&quiet_a),
            global_crest(&loud_a),
            choice.reason
        );
    }

    #[test]
    fn different_content_is_not_a_duplicate() {
        let dir = temp_dir("diff");
        let (a_path, b_path) = (dir.join("a.wav"), dir.join("b.wav"));
        wav_fix::write_wav(&a_path, RATE, 16, 1, &wav_fix::burst_signal(RATE, 10));
        wav_fix::write_wav(&b_path, RATE, 16, 1, &wav_fix::other_signal(RATE, 10));

        let verdict = compare_pair(&analysis_of(&a_path), &analysis_of(&b_path));

        assert!(!verdict.is_duplicate, "разный контент: {:?}", verdict.corr);
        assert!(verdict.corr.iter().all(|c| *c < CORR_THRESHOLD));
    }

    #[test]
    fn one_frame_difference_is_not_a_duplicate() {
        let dir = temp_dir("frame");
        let full = wav_fix::burst_signal(RATE, 10);
        let (a_path, b_path) = (dir.join("a.wav"), dir.join("b.wav"));
        wav_fix::write_wav(&a_path, RATE, 16, 1, &full);
        wav_fix::write_wav(&b_path, RATE, 16, 1, &full[..full.len() - 1]);

        let (a, b) = (analysis_of(&a_path), analysis_of(&b_path));
        assert_eq!(b.info.frames, a.info.frames - 1);

        let verdict = compare_pair(&a, &b);
        assert!(!verdict.is_duplicate, "разница даже в 1 фрейм — не дубль");
    }

    // --- правило хранителя: клипующие ------------------------------------------------

    #[test]
    fn clipping_candidate_is_excluded_from_keeper_pool() {
        let dir = temp_dir("clipping");
        let clean = wav_fix::burst_signal(RATE, 10);
        let clipped = wav_fix::hard_clipped(&clean, 1.2); // пики у потолка
        let (clean_path, clip_path) = (dir.join("clean.wav"), dir.join("clip.wav"));
        wav_fix::write_wav(&clean_path, RATE, 16, 1, &clean);
        wav_fix::write_wav(&clip_path, RATE, 16, 1, &clipped);

        let (clean_a, clip_a) = (analysis_of(&clean_path), analysis_of(&clip_path));
        assert!(clip_a.clip.is_clipping(), "пик у потолка: {}", clip_a.clip.true_peak);
        assert!(clip_a.clip.max_clip_run >= MIN_CLIP_RUN, "серия у потолка");
        assert!(!clean_a.clip.is_clipping());

        // Контент совпадает — это дубль, но хранителем клипующий не станет.
        let verdict = compare_pair(&clean_a, &clip_a);
        assert!(verdict.is_duplicate, "клип не ломает совпадение контента: {:?}", verdict.corr);

        let choice = select_keeper(&[&clip_a, &clean_a]);
        assert_eq!(choice.index, 1, "клипующий исключён, хранитель — чистый");
        assert_eq!(choice.reason, KeeperReason::OnlyNonClipped);
    }

    #[test]
    fn all_clipped_pool_picks_max_crest() {
        let dir = temp_dir("allclipped");
        let clean = wav_fix::burst_signal(RATE, 10);
        let (mild_path, hard_path) = (dir.join("mild.wav"), dir.join("hard.wav"));
        wav_fix::write_wav(&mild_path, RATE, 16, 1, &wav_fix::hard_clipped(&clean, 1.5));
        wav_fix::write_wav(&hard_path, RATE, 16, 1, &wav_fix::hard_clipped(&clean, 8.0));

        let (mild, hard) = (analysis_of(&mild_path), analysis_of(&hard_path));
        assert!(mild.clip.is_clipping() && hard.clip.is_clipping());

        let choice = select_keeper(&[&mild, &hard]);
        assert_eq!(choice.index, 0, "менее пережатый = динамичнее");
        assert!(matches!(choice.reason, KeeperReason::AllClipped { margin_db } if margin_db > 0.0));
    }

    // --- разбор группы ----------------------------------------------------------------

    // --- регрессия кейса unknown 2/3: крест-фактор файла — глобальный ---------

    /// «Запись» кейса unknown 2: тихие вспышки (пик 0.0765, медианный RMS
    /// межвспышечных кадров ≈ −67 дБ; пол между вспышками 0.01 — заведомо
    /// выше шума квантования 16 бит, чтобы +11 дБ гейна давали ровно +11 дБ
    /// по медианным RMS) и громкий участок с `loud_from_secs` с глобальным
    /// пиком 0.468 ≈ −6.6 дБФС. Для 300 с громкий участок [210..300) лежит
    /// ВНЕ всех трёх окон — как в реальном кейсе: окна лимитер не видят,
    /// глобальный блок-скан видит. Поэтому средний оконный крест-фактор
    /// чистой и гейн-копии расходится меньше чем на 2 дБ (старая логика
    /// отдавала хранителя лимитерованной копии по правилу «громче»),
    /// а глобальный (пик файла − медиана медианных RMS окон) — больше.
    fn case_signal(seconds: u32, loud_from_secs: usize) -> Vec<f32> {
        let loud_from = loud_from_secs * RATE as usize;
        (0..RATE as usize * seconds as usize)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                let gate = if t % 2.0 < 0.25 { 1.0 } else { 0.01 };
                let amp = if i >= loud_from { 0.52 } else { 0.085 };
                let tone = 0.7 * (t * 220.0).sin() + 0.2 * (t * 331.0).sin();
                (tone * gate * amp) as f32
            })
            .collect()
    }

    #[test]
    fn gain_copy_of_quiet_record_keeps_clean_by_global_crest() {
        let dir = temp_dir("regress");
        // 40 с, громкий хвост [35..40): +11.0 дБ с ограничением пиков
        // до 0.96 (−0.36 дБФС), как у кейса unknown 3.
        let clean = case_signal(40, 35);
        let loud = wav_fix::gain_limited(&clean, 3.548, 0.96);
        let (clean_path, loud_path) = (dir.join("clean.wav"), dir.join("loud.wav"));
        wav_fix::write_wav(&clean_path, RATE, 16, 1, &clean);
        wav_fix::write_wav(&loud_path, RATE, 16, 1, &loud);

        let (clean_a, loud_a) = (analysis_of(&clean_path), analysis_of(&loud_path));
        assert_eq!(clean_a.info.frames, loud_a.info.frames);
        assert!(clean_a.clip.true_peak < 0.5, "тихая чистая: {}", clean_a.clip.true_peak);
        assert!(loud_a.clip.true_peak >= 0.95, "копия ограничена к потолку");
        assert!(
            clean_a.clip.true_peak > loud_a.clip.true_peak / 3.0,
            "глобальные пики расходятся слабо — лимитер виден по динамике"
        );

        let verdict = compare_pair(&clean_a, &loud_a);
        assert!(verdict.is_duplicate, "гейн-копия — дубль: {:?}", verdict.corr);
        let global_margin = verdict.crest_a_db - verdict.crest_b_db;
        assert!(global_margin > CREST_MARGIN_DB, "глобальный крест расходится > 2 дБ: {global_margin:.2}");

        // Правило хранителя: динамичнее = чистая, хоть она и тише.
        let choice = select_keeper(&[&clean_a, &loud_a]);
        assert_eq!(choice.index, 0, "хранитель — чистая (глобальный крест)");
        assert!(
            matches!(choice.reason, KeeperReason::CleanerDynamics { margin_db } if margin_db > CREST_MARGIN_DB),
            "{:?}",
            choice.reason
        );
    }

    #[test]
    fn group_of_three_with_limiter_copies_keeps_clean_by_global_crest() {
        let dir = temp_dir("regress_trio");
        // 300 с, громкий хвост [210..300) вне всех трёх окон: копии клипуются
        // только там, корреляция по окнам чистая, лимитер виден лишь
        // глобальному скану — точное зеркало кейса unknown 2/3.
        let clean = case_signal(300, 210);
        // +11 дБ и +16 дБ — обе пережаты лимитером на 0.96.
        let mild = wav_fix::gain_limited(&clean, 3.548, 0.96);
        let hard = wav_fix::gain_limited(&clean, 6.31, 0.96);
        let (a, b, c) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("c.wav"));
        wav_fix::write_wav(&a, RATE, 16, 1, &clean);
        wav_fix::write_wav(&b, RATE, 16, 1, &mild);
        wav_fix::write_wav(&c, RATE, 16, 1, &hard);
        let analyses = vec![analysis_of(&a), analysis_of(&b), analysis_of(&c)];

        let refs: Vec<&FileAnalysis> = analyses.iter().collect();
        let choice = select_keeper(&refs);
        assert_eq!(choice.index, 0, "хранитель — чистая, а не самая громкая копия");
        assert!(
            matches!(choice.reason, KeeperReason::CleanerDynamics { margin_db } if margin_db > CREST_MARGIN_DB),
            "{:?}",
            choice.reason
        );

        let resolution = resolve_group(&analyses);
        assert_eq!(resolution.keepers, vec![0], "хранитель — чистая запись");
        assert!(resolution.unique.is_empty());
        let dups: Vec<(usize, usize)> =
            resolution.duplicates.iter().map(|m| (m.duplicate, m.kept)).collect();
        assert_eq!(dups, vec![(1, 0), (2, 0)], "обе лимитерованные копии сняты");
    }

    #[test]
    fn group_of_three_yields_one_keeper_and_two_duplicates() {
        let dir = temp_dir("trio");
        let clean = wav_fix::burst_signal(RATE, 10);
        let (a, b, c) = (
            dir.join("a.wav"),
            dir.join("b.wav"),
            dir.join("c.wav"),
        );
        wav_fix::write_wav(&a, RATE, 16, 1, &clean);
        wav_fix::write_wav(&b, RATE, 16, 1, &wav_fix::gain_limited(&clean, 3.16, 0.96));
        wav_fix::write_wav(&c, RATE, 16, 1, &wav_fix::gain_limited(&clean, 6.31, 0.96));
        let analyses = vec![analysis_of(&a), analysis_of(&b), analysis_of(&c)];

        let resolution = resolve_group(&analyses);

        assert_eq!(resolution.keepers, vec![0], "хранитель — чистая запись");
        assert!(resolution.unique.is_empty());
        let dups: Vec<(usize, usize)> =
            resolution.duplicates.iter().map(|m| (m.duplicate, m.kept)).collect();
        assert_eq!(dups, vec![(1, 0), (2, 0)], "оба гейн-дубля сняты за одним хранителем");
        assert!(resolution.duplicates.iter().all(|m| m.verdict.is_duplicate));
    }

    #[test]
    fn group_of_unrelated_files_has_no_duplicates() {
        let dir = temp_dir("unrelated");
        let (a, b) = (dir.join("a.wav"), dir.join("b.wav"));
        wav_fix::write_wav(&a, RATE, 16, 1, &wav_fix::burst_signal(RATE, 10));
        wav_fix::write_wav(&b, RATE, 16, 1, &wav_fix::other_signal(RATE, 10));
        let analyses = vec![analysis_of(&a), analysis_of(&b)];

        let resolution = resolve_group(&analyses);

        assert!(resolution.keepers.is_empty());
        assert_eq!(resolution.unique, vec![0, 1]);
        assert!(resolution.duplicates.is_empty());
    }

    // --- анализ файла: крайние случаи ---------------------------------------------------

    #[test]
    fn analyze_file_returns_none_for_non_wav_bytes() {
        let dir = temp_dir("nonwav");
        let path = dir.join("junk.wav");
        std::fs::write(&path, b"wav-bytes").expect("junk");

        assert!(matches!(analyze_file(&path), Ok(None)), "не-PCM → None, файл уникален");
    }

    #[test]
    fn analyze_file_handles_file_shorter_than_one_window() {
        let dir = temp_dir("short");
        let path = dir.join("tiny.wav");
        wav_fix::write_wav(&path, RATE, 16, 1, &wav_fix::burst_signal(RATE, 3));

        let analysis = analysis_of(&path);

        // Все три окна прижаты к [0, длина файла) и непусты.
        assert!(analysis.windows.iter().all(|w| !w.envelope.is_empty()));
        assert!(analysis.windows.iter().all(|w| w.median_rms > 0.0));
    }

    #[test]
    fn analyze_file_reads_24bit_stereo() {
        let dir = temp_dir("bit24");
        let path = dir.join("s.wav");
        let left = wav_fix::burst_signal(RATE, 10);
        let stereo: Vec<f32> = left.iter().flat_map(|s| [*s, *s * 0.5]).collect();
        wav_fix::write_wav(&path, RATE, 24, 2, &stereo);

        let analysis = analysis_of(&path);

        assert_eq!(analysis.info.bits, 24);
        assert_eq!(analysis.info.channels, 2);
        assert!(analysis.clip.true_peak > 0.5, "пик по обоим каналам: {}", analysis.clip.true_peak);
    }

    // --- корреляция: вырожденные случаи ---------------------------------------------------

    #[test]
    fn envelope_corr_degenerate_cases_are_zero() {
        assert_eq!(envelope_corr(&[], &[]), 0.0, "пустые огибающие");
        assert_eq!(envelope_corr(&[1.0, 2.0], &[1.0, 2.0, 3.0]), 0.0, "разная длина");
        assert_eq!(envelope_corr(&[0.0, 0.0], &[1.0, 2.0]), 0.0, "тишина в одном файле");
        assert!((envelope_corr(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-5);
        assert!(envelope_corr(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) < -0.99, "антикорреляция = −1");
        assert!(envelope_corr(&[1.0, 2.0, 3.0], &[2.0, 1.0, 2.0]).abs() < 1e-5, "нет связи → 0");
    }

    #[test]
    fn median_handles_even_odd_and_empty() {
        let mut even = [3.0_f32, 1.0, 4.0, 2.0];
        assert!((median(&mut even) - 2.5).abs() < 1e-6);
        let mut odd = [3.0_f32, 1.0, 2.0];
        assert!((median(&mut odd) - 2.0).abs() < 1e-6);
        assert_eq!(median(&mut []), 0.0);
    }
}
