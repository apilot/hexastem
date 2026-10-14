//! Виджеты TUI: чистые функции рендера над собственными view-структурами.
//!
//! Контракт subtask 02: `render_*(frame, area, &view)` не имеет side-effects
//! и не знает про `App`/раннер — seq 03 отображает `EngineEvent`→view,
//! композиция кадра — `render_screen` (кадр: шапка → табы → FILES+DETAILS →
//! footer). Карта координат для мыши: [`tab_at`] / [`file_row_at`] — те же
//! зоны, что у рендера, поэтому клик и отрисовка не расходятся.
//!
//! Ограничения: min 80×24 (проверяется выше по стеку, `tui::MIN_WIDTH`);
//! ширина < [`FULL_LAYOUT_MIN_WIDTH`] — compact-режим (таблица без DETAILS).

use std::ops::Range;
use std::sync::Arc;

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Painter, Shape};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table, TableState, Widget};
use ratatui::Frame;

use crate::dedup::{KeeperReason, PairReport};
use crate::pipeline::EXPECTED_STEM_COUNT;
use crate::tui::i18n::StrTable;
use crate::tui::log::{LogBuffer, LogEntry};

// --- глифы и геометрия ----------------------------------------------------------

/// Уровни громкости для спарклайна GPU: 0 — пусто, 7 — полный блок.
const WAVE_BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// Пикселей брайль-сетки в ячейке канваса (горизонталь, вертикаль).
const BRAILLE_PX: (usize, usize) = (2, 4);
/// Высота брайль-карты: 2–3 текстовые строки = 8–12 пиксельных рядов.
const WAVE_CANVAS_ROWS_MIN: u16 = 2;
const WAVE_CANVAS_ROWS_MAX: u16 = 3;
/// Статус-зона с текущим файлом: строка индикатора + карта (2 ряда) + подпись.
const STATUS_TALL_ROWS: u16 = 4;
const BAR_FILLED: char = '█';
const BAR_EMPTY: char = '░';
const PROGRESS_BAR_WIDTH: usize = 12;
const GPU_BAR_WIDTH: usize = 6;
/// Колонок спарклайна утилизации GPU в шапке (btop-стиль).
const GPU_SPARK_WIDTH: usize = 20;
/// Колонок мини-бара глобального прогресса в футере.
const FILES_BAR_WIDTH: usize = 6;
/// Ниже этой ширины DETAILS-колонка скрывается (compact-режим).
const FULL_LAYOUT_MIN_WIDTH: u16 = 100;
/// Ширина DETAILS-колонки: подпись waveform (40 символов) + рамка + запас.
const DETAILS_WIDTH: u16 = 44;
const TAB_COUNT: usize = 4;

const STYLE_DIM: Style = Style::new().fg(Color::DarkGray);
const STYLE_FILL: Style = Style::new().fg(Color::Yellow);
const STYLE_EMPTY: Style = Style::new().fg(Color::DarkGray);
const STYLE_HEADER: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);
/// Заливка пройденной части карты (палитра UI: зелёный статуса «ГОТ»).
const STYLE_WAVE_FILL: Color = Color::Green;
/// Серые пики непройденной части карты.
const STYLE_WAVE_PEAK: Color = Color::DarkGray;

// --- бейджи ----------------------------------------------------------------------

/// Статус файла в таблице. Текст + цвет (`color-not-only`): бейдж читается
/// и без различения цветов.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBadge {
    /// Обрабатывается.
    Run,
    /// Готов: стемы на NAS, исходник переименован.
    Ok,
    /// Ошибка.
    Fail,
    /// Пропущен (resume по `CONVERTED_`).
    Skip,
    /// Дубль (решение дедупликации).
    Dup,
}

impl FileBadge {
    /// Текст бейджа (локализованный; фикс. ширина — на рендере).
    pub fn label(self, tr: &StrTable) -> &'static str {
        match self {
            Self::Run => tr.badge_run,
            Self::Ok => tr.badge_ok,
            Self::Fail => tr.badge_fail,
            Self::Skip => tr.badge_skip,
            Self::Dup => tr.badge_dup,
        }
    }

    /// Ячейка бейджа: ФИКСИРОВАННАЯ ширина 4 в обеих локалях — колонка
    /// таблицы не «пляшет» (английские метки короче — добиваются пробелом).
    pub fn cell(self, tr: &StrTable) -> String {
        format!("{:<4}", self.label(tr))
    }

    /// Цвет статуса (дополняет текст, не заменяет его).
    pub fn style(self) -> Style {
        let color = match self {
            Self::Run => Color::Yellow,
            Self::Ok => Color::Green,
            Self::Fail => Color::Red,
            Self::Skip => Color::Cyan,
            Self::Dup => Color::Magenta,
        };
        Style::new().fg(color).add_modifier(Modifier::BOLD)
    }
}

// --- view-структуры --------------------------------------------------------------

/// Шапка: подключение + модель. `gpu: None` → «GPU: n/a» (поллер — seq 06).
#[derive(Debug, Clone, Default)]
pub struct HeaderView {
    /// Хост SMB.
    pub host: String,
    /// Имя шары.
    pub share: String,
    /// Удалённая директория прогона.
    pub remote_dir: String,
    /// Модель разделения (`sw` | `6s`).
    pub model: String,
    /// Снимок GPU-поллера, когда он поднят.
    pub gpu: Option<GpuView>,
    /// Прогресс загрузки модели (`Some` — видимый индикатор «модель: NN%»,
    /// гаснет по достижении 100%).
    pub model_pct: Option<f64>,
}

/// Снимок GPU для шапки (заполняет seq 06 из nvidia-smi).
#[derive(Debug, Clone, Default)]
pub struct GpuView {
    /// Имя GPU.
    pub name: String,
    /// Занято VRAM, GiB.
    pub used_gib: f32,
    /// Всего VRAM, GiB.
    pub total_gib: f32,
    /// Утилизация, % (`None` — nvidia-smi не даёт: показываем «n/a»).
    pub util_pct: Option<f64>,
    /// История утилизации (%) для спарклайна (btop-стиль).
    pub util_history: Vec<u8>,
}

/// Строка таблицы файлов (UI-модель; источник — маппинг EngineEvent, seq 03).
#[derive(Debug, Clone)]
pub struct FileRowView {
    /// Статус-бейдж.
    pub badge: FileBadge,
    /// Имя файла.
    pub name: String,
    /// Текущая стадия (`scan`/`download`/`separate`/…).
    pub stage: String,
    /// Прогресс отдельной стадии, 0..=100.
    pub pct: f64,
    /// `true` — pct оценочный (драйвер): UI помечает тильдой «~NN%».
    pub estimated: bool,
    /// Готовых стемов (`Some(x)` — счётчик известен); `None` — неизвестен:
    /// колонка STEMS рисует «—», а не врущее «0/6».
    pub stems_done: Option<u8>,
    /// Секунд с входа в текущую стадию (`None` — таймер не идёт).
    pub stage_elapsed: Option<f64>,
    /// Фаза неопределённого прогресса (тики): `Some` — анимированный бар
    /// вместо «мёртвых 0%» на стадиях download/upload/rename.
    pub indeterminate: Option<u64>,
}

/// Состояние стема в чеклисте Details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemState {
    /// Готов (`✓`).
    Done,
    /// В работе (`▸`).
    Current,
    /// Ожидает (`·`).
    Pending,
}

/// Один стем чеклиста Details.
#[derive(Debug, Clone)]
pub struct StemView {
    /// Имя стема (`vocals`, `drums`, …).
    pub name: String,
    /// Состояние.
    pub state: StemState,
}

/// Огибающая громкости (брайль-версия, решение владельца 2026-10-14):
/// симметричная пиксельная карта в Canvas `Marker::Braille`, серые пики
/// непройденной части, цветная заливка пройденной.
#[derive(Debug, Clone, Default)]
pub struct WaveformView {
    /// Прогресс separate, 0..=100.
    pub pct: f64,
    /// Длительность трека, сек (`None` — метки времени скрыты).
    pub duration_secs: Option<f64>,
    /// `true` — pct оценочный: подпись с тильдой «~NN%».
    pub estimated: bool,
    /// Пики по колонкам 0.0..=1.0 (seq 03, `wav_envelope`); пусто → fallback-бар.
    pub peaks: Arc<[f32]>,
}

/// Панель Details выбранного файла.
#[derive(Debug, Clone, Default)]
pub struct DetailsView {
    /// Имя файла.
    pub name: String,
    /// Текущая стадия.
    pub stage: String,
    /// Источник (путь/имя исходника).
    pub source: String,
    /// Оценка остатка, сек.
    pub eta_secs: Option<f64>,
    /// Формат выходных стемов (`flac`).
    pub format: String,
    /// Чеклист стемов `✓/▸/·`.
    pub stems: Vec<StemView>,
    /// Последние события (порядок: старые → новые).
    pub events: Vec<String>,
    /// Waveform-прогресс.
    pub waveform: WaveformView,
}

/// Footer: подсказка клавиш + сводка прогона.
#[derive(Debug, Clone, Default)]
pub struct FooterView {
    /// Сводка справа (`OK 3 · FAIL 1 · DUP 2 · ETA 04:12`).
    pub summary: String,
}

/// Индикатор текущего файла (статус-строка над контентом, «крупно»).
#[derive(Debug, Clone, Default)]
pub struct CurrentFileView {
    /// Имя файла.
    pub name: String,
    /// Стадия.
    pub stage: String,
    /// Процент стадии (`None` — стадия без процента: индeterminate-бар).
    pub pct: Option<f64>,
    /// `true` — pct оценочный: метка «~NN%» вместо «NN%».
    pub estimated: bool,
    /// Фаза анимации (тики): спиннер + бегущий бар.
    pub phase: u64,
    /// Секунд в текущей стадии.
    pub stage_elapsed: f64,
    /// Огибающая текущего файла: брайль-карта на всю ширину статус-зоны
    /// (пустые пики → fallback-бар под строкой индикатора).
    pub waveform: WaveformView,
}

/// Табы основного экрана (клики мышью — [`tab_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    /// Таблица файлов.
    #[default]
    Files,
    /// Дедупликация.
    Dedup,
    /// Журнал.
    Log,
    /// Справка.
    Help,
}

/// Порядок и подписи табов (индекс = клавиша 1..4).
pub const TABS: [Tab; TAB_COUNT] = [Tab::Files, Tab::Dedup, Tab::Log, Tab::Help];

impl Tab {
    /// Подпись таба с номером клавиши (локализованная).
    pub fn label(self, tr: &StrTable) -> &'static str {
        match self {
            Self::Files => tr.tab_files,
            Self::Dedup => tr.tab_dedup,
            Self::Log => tr.tab_log,
            Self::Help => tr.tab_help,
        }
    }

    /// Таб по индексу клавиши (0-based).
    pub fn from_index(index: usize) -> Option<Self> {
        TABS.get(index).copied()
    }
}

/// Взгляд на весь кадр (собирает seq 03+ из состояния `App`).
#[derive(Debug, Clone, Default)]
pub struct ScreenView {
    /// Шапка.
    pub header: HeaderView,
    /// Активный таб.
    pub tab: Tab,
    /// Строки таблицы файлов.
    pub files: Vec<FileRowView>,
    /// Индекс выбранной строки таблицы.
    pub selected: Option<usize>,
    /// Панель Details (`None` — ничего не выбрано).
    pub details: Option<DetailsView>,
    /// Footer.
    pub footer: FooterView,
    /// Пары дедупликации последнего прогона (таб Dedup).
    pub dedup_pairs: Vec<PairReport>,
    /// Файлы с нечитаемым заголовком (таб Dedup, примечание).
    pub dedup_unparseable: Vec<String>,
    /// Журнал событий (таб Log).
    pub log: LogBuffer,
    /// Режим ввода фильтра журнала: текст и позиция курсора в символах.
    pub log_filter_input: Option<(String, usize)>,
    /// Баннер над контентом (приоритетнее current): «Прерывание…».
    pub banner: Option<String>,
    /// Индикатор текущего файла в статус-строке (`None` — не показывать).
    pub current: Option<CurrentFileView>,
}

/// Прямоугольники зон кадра (рендер и мышь используют одни значения).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenZones {
    /// Шапка.
    pub header: Rect,
    /// Полоса табов.
    pub tabs: Rect,
    /// Статус-строка: баннер отмены или индикатор текущего файла.
    pub status: Rect,
    /// FILES (+ DETAILS в полном режиме).
    pub content: Rect,
    /// Footer.
    pub footer: Rect,
}

// --- чистые хелперы (unit-тестируемые) ------------------------------------------

/// `MM:SS` (минуты не заворачиваются: 3600 c → `60:00`); отрицательное → `00:00`.
pub fn format_mmss(secs: f64) -> String {
    let total = secs.max(0.0).floor() as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Прогресс → (обработано, осталось) сек; `pct` обрезается в 0..=100.
pub fn split_progress(duration_secs: f64, pct: f64) -> (f64, f64) {
    let duration = duration_secs.max(0.0);
    let done = duration * pct.clamp(0.0, 100.0) / 100.0;
    (done, duration - done)
}

/// Прогресс-бар из полуколонок: `████░░░░` (unicode-safe, моно-ширина).
pub fn progress_bar(pct: f64, width: usize) -> String {
    let (filled, empty) = progress_bar_parts(pct, width);
    format!("{filled}{empty}")
}

/// Части прогресс-бара для раздельной стилизации (заполнено/остаток).
pub fn progress_bar_parts(pct: f64, width: usize) -> (String, String) {
    let filled = fill_columns(pct, width);
    (
        BAR_FILLED.to_string().repeat(filled),
        BAR_EMPTY.to_string().repeat(width - filled),
    )
}

/// Число заполненных колонок: `floor(pct% × width)`, граница — на колонке.
fn fill_columns(pct: f64, width: usize) -> usize {
    let clamped = pct.clamp(0.0, 100.0);
    ((clamped / 100.0) * width as f64).floor() as usize
}

/// Уровень спарклайна: значение 0..=100 → индекс блока `▁..█`.
fn spark_level(value: u8) -> usize {
    let scaled = f64::from(value.min(100)) / 100.0 * (WAVE_BLOCKS.len() - 1) as f64;
    scaled.round() as usize
}

/// Спарклайн истории (btop-стиль): последние `width` значений, каждый — блок
/// `▁▂▃▄▅▆▇█` по величине; недостающие колонки слева — пробелы.
pub fn sparkline(history: &[u8], width: usize) -> String {
    let skip = history.len().saturating_sub(width);
    let bars: String = history[skip..].iter().map(|v| WAVE_BLOCKS[spark_level(*v)]).collect();
    format!("{}{bars}", " ".repeat(width - (history.len() - skip)))
}

/// Глобальная полоса футера по завершённым/всего: «файлы 1/3 ██░░░░ 33%».
/// Заполнение округляется (не floor): 1/3 даёт 2 из 6 колонок, как в btop.
pub fn files_progress(tr: &StrTable, done: usize, total: usize) -> String {
    if total == 0 {
        return format!("{} 0/0", tr.files_word);
    }
    let pct = done as f64 / total as f64 * 100.0;
    let filled = (pct / 100.0 * FILES_BAR_WIDTH as f64).round() as usize;
    let bar = format!(
        "{}{}",
        BAR_FILLED.to_string().repeat(filled),
        BAR_EMPTY.to_string().repeat(FILES_BAR_WIDTH - filled),
    );
    format!("{} {done}/{total} {bar} {pct:.0}%", tr.files_word)
}

/// Неопределённый прогресс (download/upload без процента): бегущее окно
/// заполненных колонок; `phase` — номер тика (анимация при перерисовке).
pub fn indeterminate_bar(phase: u64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let fill = (width / 4).clamp(1, width);
    let pos = (phase as usize) % (width + fill);
    (0..width)
        .map(|col| if (pos..pos + fill).contains(&col) { BAR_FILLED } else { BAR_EMPTY })
        .collect()
}

/// ETA прогона: среднее время на завершённый файл × остаток.
/// `None` — считать нечем (ни одного завершённого) или всё завершено.
pub fn eta_secs(elapsed_secs: f64, done: usize, total: usize) -> Option<f64> {
    if done == 0 || total == 0 || done >= total {
        return None;
    }
    Some(elapsed_secs / done as f64 * (total - done) as f64)
}

/// Ячейка стадии с тикающим таймером: `download · 00:12` (без таймера — как есть).
pub fn stage_cell(stage: &str, stage_elapsed: Option<f64>) -> String {
    match stage_elapsed {
        Some(secs) => format!("{stage} · {}", format_mmss(secs)),
        None => stage.to_string(),
    }
}

/// Кадры спиннера для статус-строки (анимация по тикам).
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Символ спиннера по фазе тиков.
pub fn spinner(phase: u64) -> char {
    SPINNER[(phase as usize) % SPINNER.len()]
}

/// Амплитуда пиксельной колонки 0..=1: сжатие — max по окну `peaks`,
/// растяжение — ближайший сэмпл (переиспользует ресемплинг текстовой карты).
fn pixel_amplitude(peaks: &[f32], px: usize, px_w: usize) -> f32 {
    column_peak(peaks, px, px_w).clamp(0.0, 1.0)
}

/// Пиксельный след колонки: (верх, низ) симметрично горизонтальной оси
/// карты; `None` — тишина. Половина высоты следа `ceil(a × рядов/2)`:
/// даже шёпот даёт видимые точки, максимум занимает все ряды.
fn trace_span(amplitude: f32, rows_px: usize) -> Option<(usize, usize)> {
    let center = rows_px / 2;
    let half = (amplitude.clamp(0.0, 1.0) * center as f32).ceil() as usize;
    (half > 0).then_some((center - half, center + half - 1))
}

/// Вертикальные брайль-следы огибающей для колонок `range` пиксельной сетки.
/// Растеризация на лету из `peaks` — без промежуточных буферов точек.
struct EnvelopeTrace<'a> {
    peaks: &'a [f32],
    px_w: usize,
    px_h: usize,
    range: Range<usize>,
    color: Color,
}

impl Shape for EnvelopeTrace<'_> {
    fn draw(&self, painter: &mut Painter) {
        for px in self.range.clone() {
            let Some((top, bottom)) =
                trace_span(pixel_amplitude(self.peaks, px, self.px_w), self.px_h)
            else {
                continue;
            };
            for py in top..=bottom {
                painter.paint(px, py, self.color);
            }
        }
    }
}

/// Пик колонки: сжатие — max по окну, растяжение — ближайший сэмпл.
fn column_peak(peaks: &[f32], col: usize, columns: usize) -> f32 {
    if peaks.is_empty() {
        return 0.0;
    }
    let start = col * peaks.len() / columns;
    if peaks.len() <= columns {
        return peaks[start];
    }
    let end = ((col + 1) * peaks.len() / columns).max(start + 1);
    peaks[start..end.min(peaks.len())].iter().copied().fold(0.0_f32, f32::max)
}

// --- рендеры ---------------------------------------------------------------------

/// Шапка: `audio_split ─ //host/share ─ dir ─ модель ─ GPU [бар] mem`.
pub fn render_header(frame: &mut Frame, area: Rect, view: &HeaderView, tr: &StrTable) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" audio_split ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let line = Line::from(header_spans(view, tr));
    frame.render_widget(Paragraph::new(line), inner);
}

/// Спаны шапки: подключение, директория, модель (+загрузка), GPU-слот.
fn header_spans(view: &HeaderView, tr: &StrTable) -> Vec<Span<'static>> {
    let mut spans = vec![
        Span::styled("audio_split", STYLE_HEADER),
        Span::raw(" ─ "),
        Span::styled(format!("//{}/{}", view.host, view.share), Style::new().fg(Color::Cyan)),
        Span::raw(" ─ "),
        Span::raw(view.remote_dir.clone()),
        Span::raw(" ─ "),
        Span::styled(view.model.clone(), Style::new().add_modifier(Modifier::BOLD)),
    ];
    if let Some(pct) = view.model_pct {
        // Глобальная загрузка модели: видна только пока идёт (до 100%).
        spans.push(Span::raw(" ─ "));
        spans.push(Span::styled((tr.model_loading)(pct), STYLE_FILL));
    }
    spans.push(Span::raw(" ─ "));
    spans.extend(gpu_spans(view.gpu.as_ref(), tr));
    spans
}

/// GPU-слот: `None` → `GPU: n/a` (graceful без поллера); иначе утилизация
/// «NN%» + спарклайн истории (btop-стиль) + VRAM-бар `▓ X/Y GiB`. Без данных
/// утилизации — «n/a», VRAM остаётся (поллер мог не дать util_percent).
fn gpu_spans(gpu: Option<&GpuView>, tr: &StrTable) -> Vec<Span<'static>> {
    let Some(gpu) = gpu else {
        return vec![Span::styled(tr.gpu_na.to_string(), STYLE_DIM)];
    };
    let mut spans = vec![Span::styled("GPU ", STYLE_DIM)];
    match gpu.util_pct {
        Some(pct) => {
            spans.push(Span::styled(format!("{pct:.0}%"), STYLE_FILL));
            spans.push(Span::styled(
                format!(" {} ", sparkline(&gpu.util_history, GPU_SPARK_WIDTH)),
                STYLE_FILL,
            ));
        }
        None => spans.push(Span::styled("n/a ", STYLE_DIM)),
    }
    spans.push(Span::styled("▓ ", STYLE_FILL));
    let pct = if gpu.total_gib > 0.0 {
        f64::from(gpu.used_gib) / f64::from(gpu.total_gib) * 100.0
    } else {
        0.0
    };
    let (filled, empty) = progress_bar_parts(pct, GPU_BAR_WIDTH);
    spans.push(Span::styled(filled, STYLE_FILL));
    spans.push(Span::styled(empty, STYLE_EMPTY));
    spans.push(Span::styled(format!(" {:.1}/{:.1}GiB", gpu.used_gib, gpu.total_gib), STYLE_DIM));
    spans
}

/// Полоса табов: 4 равные зоны, активный подсвечен (зоны = [`tab_at`]).
pub fn render_tab_bar(frame: &mut Frame, area: Rect, selected: Tab, tr: &StrTable) {
    let chunks = Layout::horizontal([Constraint::Fill(1); TAB_COUNT]).split(area);
    for (tab, chunk) in TABS.iter().zip(chunks.iter()) {
        let style = if *tab == selected {
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            STYLE_DIM
        };
        let line = Line::from(tab.label(tr)).style(style).alignment(Alignment::Center);
        frame.render_widget(Paragraph::new(line), *chunk);
    }
}

/// Таблица файлов; пустой список → empty-state подсказка.
pub fn render_files(frame: &mut Frame, area: Rect, files: &[FileRowView], selected: Option<usize>, tr: &StrTable) {
    if files.is_empty() {
        render_files_empty(frame, area, tr);
        return;
    }
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.files_title);
    let header = Row::new([
        tr.th_st,
        tr.th_stage,
        tr.th_file,
        tr.th_progress,
        tr.th_stems,
    ])
    .style(STYLE_HEADER);
    let rows: Vec<Row> = files.iter().map(|view| file_row(view, tr)).collect();
    let widths = [
        Constraint::Length(5),
        Constraint::Length(16),
        Constraint::Min(8),
        Constraint::Length(18),
        Constraint::Length(6),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(block)
        .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
        .highlight_symbol("▸ ");
    let mut state = TableState::default();
    state.select(selected);
    frame.render_stateful_widget(table, area, &mut state);
}

/// Empty-state таблицы: «нет файлов — Esc → настройка».
fn render_files_empty(frame: &mut Frame, area: Rect, tr: &StrTable) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.files_title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let hint = Line::from(tr.no_files_hint)
        .style(STYLE_DIM)
        .alignment(Alignment::Center);
    frame.render_widget(Paragraph::new(hint), inner);
}

/// Строка таблицы: бейдж / стадия(+таймер) / имя / прогресс или
/// индeterminate-бар / стемы (моно-цифры).
fn file_row(view: &FileRowView, tr: &StrTable) -> Row<'static> {
    let badge = Cell::from(Span::styled(view.badge.cell(tr), view.badge.style()));
    let stage = Cell::from(Span::styled(stage_cell(&view.stage, view.stage_elapsed), STYLE_DIM));
    let name = Cell::from(view.name.clone());
    let progress = match view.indeterminate {
        // Стадия без процента: анимированный бар вместо «мёртвых 0%».
        Some(phase) => Cell::from(Line::from(vec![
            Span::styled(indeterminate_bar(phase, PROGRESS_BAR_WIDTH), STYLE_FILL),
            Span::styled("  ··", STYLE_DIM),
        ])),
        None => {
            let (filled, empty) = progress_bar_parts(view.pct, PROGRESS_BAR_WIDTH);
            // Оценочный процент честно помечаем тильдой (~63%), фактический — как раньше.
            let label = if view.estimated {
                format!("~{:.0}%", view.pct)
            } else {
                format!(" {:>3.0}%", view.pct)
            };
            Cell::from(Line::from(vec![
                Span::styled(filled, STYLE_FILL),
                Span::styled(empty, STYLE_EMPTY),
                Span::raw(label),
            ]))
        }
    };
    // Счётчик стемов: неизвестен — «—», а не врущее «0/6» (честная колонка).
    let stems = Cell::from(stems_cell(view.stems_done, tr));
    Row::new([badge, stage, name, progress, stems])
}

/// Ячейка счётчика стемов: `x/6`; пока счётчик неизвестен — «—»
/// (честная колонка: нет данных — не рисуем «0/6»).
fn stems_cell(done: Option<u8>, tr: &StrTable) -> String {
    match done {
        Some(count) => format!("{count}/{EXPECTED_STEM_COUNT}"),
        None => tr.stems_unknown.to_string(),
    }
}

/// Панель Details выбранного файла; `None` → пустое состояние.
pub fn render_details_or_empty(frame: &mut Frame, area: Rect, view: Option<&DetailsView>, tr: &StrTable) {
    match view {
        Some(details) => render_details(frame, area, details, tr),
        None => render_details_empty(frame, area, tr),
    }
}

/// Details: заголовок, стадия/ETA, источник, формат, стемы, события + waveform.
pub fn render_details(frame: &mut Frame, area: Rect, view: &DetailsView, tr: &StrTable) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.details_title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    // Waveform — фикс. блок снизу (брайль-карта 3 ряда + подпись + метки),
    // текст — сверху.
    let chunks = Layout::vertical([Constraint::Min(1), Constraint::Length(5)]).split(inner);
    frame.render_widget(
        Paragraph::new(details_lines(view, tr, chunks[0].height as usize)),
        chunks[0],
    );
    render_waveform(frame, chunks[1], &view.waveform, tr);
}

/// Строки текстовой части Details (без waveform).
fn details_lines(view: &DetailsView, tr: &StrTable, max_lines: usize) -> Vec<Line<'static>> {
    let eta = match view.eta_secs {
        Some(secs) => format!(" · ETA {}", format_mmss(secs)),
        None => String::new(),
    };
    let mut lines = vec![
        Line::from(view.name.clone()).style(STYLE_HEADER),
        Line::from(format!("{} {}{eta}", tr.details_stage, view.stage)),
        Line::from(format!("{} {}", tr.details_source, view.source)),
        Line::from(format!("{} {}", tr.details_format, view.format)),
        stems_line(&view.stems),
        Line::from(tr.details_events).style(STYLE_DIM),
    ];
    // События — последние, сколько влезает (новые снизу).
    let free = max_lines.saturating_sub(lines.len());
    let visible = view.events.len().min(free);
    lines.extend(
        view.events[view.events.len() - visible..]
            .iter()
            .map(|event| Line::from(format!("  {event}")).style(STYLE_DIM)),
    );
    lines
}

/// Чеклист стемов одной строкой: `✓ vocals  ▸ drums  · bass …`.
fn stems_line(stems: &[StemView]) -> Line<'static> {
    let spans: Vec<Span> = stems
        .iter()
        .flat_map(|stem| {
            let (glyph, style) = match stem.state {
                StemState::Done => ("✓", Style::new().fg(Color::Green)),
                StemState::Current => ("▸", Style::new().fg(Color::Yellow)),
                StemState::Pending => ("·", STYLE_DIM),
            };
            let name = stem.name.clone();
            [Span::styled(format!("{glyph} {name}"), style), Span::raw("  ")]
        })
        .collect();
    Line::from(spans)
}

/// Пустое состояние Details (ничего не выбрано).
fn render_details_empty(frame: &mut Frame, area: Rect, tr: &StrTable) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.details_title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let hint = Line::from(tr.details_empty).style(STYLE_DIM).alignment(Alignment::Center);
    frame.render_widget(Paragraph::new(hint), inner);
}

/// Waveform-прогресс: подпись, брайль-карта огибающей (Canvas 2–3 строки),
/// метки времени. Без пиков или в тесную высоту — прежний fallback.
pub fn render_waveform(frame: &mut Frame, area: Rect, view: &WaveformView, tr: &StrTable) {
    if area.is_empty() {
        return;
    }
    if !view.peaks.is_empty() {
        if let Some((canvas, label, labels)) = wave_areas(area, view.duration_secs.is_some()) {
            frame.render_widget(envelope_canvas(view, canvas), canvas);
            frame.render_widget(Paragraph::new(waveform_label_line(view, tr)), label);
            if let Some(labels_area) = labels {
                let duration = view.duration_secs.unwrap_or_default();
                frame.render_widget(
                    Paragraph::new(time_labels(duration, labels_area.width as usize)),
                    labels_area,
                );
            }
            return;
        }
    }
    render_waveform_fallback(frame, area, view, tr);
}

/// Разметка виджета: брайль-карта (2–3 ряда), под ней подпись, ниже метки
/// времени, когда влезают; `None` — на карту даже минимума высоты нет
/// (fallback).
fn wave_areas(area: Rect, with_labels: bool) -> Option<(Rect, Rect, Option<Rect>)> {
    if area.height < WAVE_CANVAS_ROWS_MIN + 1 {
        return None;
    }
    let labels = with_labels && area.height >= WAVE_CANVAS_ROWS_MIN + 2;
    let canvas_rows = (area.height - 1 - u16::from(labels)).min(WAVE_CANVAS_ROWS_MAX);
    let rows = Layout::vertical([
        Constraint::Length(canvas_rows),
        Constraint::Length(1),
        Constraint::Length(u16::from(labels)),
    ])
    .split(area);
    Some((rows[0], rows[1], labels.then_some(rows[2])))
}

/// Брайль-канвас: серые пики всей карты, поверх — цветная заливка пройденной
/// части. Один слой: ячейка брайля хранит один цвет, порядок отрисовки
/// (серые → цвет) красит граничную ячейку заливкой — граница с точностью
/// пиксельной колонки.
fn envelope_canvas<'a>(view: &'a WaveformView, area: Rect) -> impl Widget + 'a {
    let px_w = usize::from(area.width) * BRAILLE_PX.0;
    let px_h = usize::from(area.height) * BRAILLE_PX.1;
    let fill = fill_columns(view.pct, px_w);
    let peaks: &[f32] = &view.peaks;
    Canvas::default()
        .marker(Marker::Braille)
        .x_bounds([0.0, px_w.saturating_sub(1) as f64])
        .y_bounds([0.0, px_h.saturating_sub(1) as f64])
        .paint(move |ctx| {
            let gray = EnvelopeTrace { peaks, px_w, px_h, range: fill..px_w, color: STYLE_WAVE_PEAK };
            let fill = EnvelopeTrace { peaks, px_w, px_h, range: 0..fill, color: STYLE_WAVE_FILL };
            ctx.draw(&gray);
            ctx.draw(&fill);
        })
}

/// Fallback (нет пиков / тесно): подпись + обычный прогресс-бар (+ метки).
fn render_waveform_fallback(frame: &mut Frame, area: Rect, view: &WaveformView, tr: &StrTable) {
    let mut lines = vec![waveform_label_line(view, tr)];
    if area.height > 1 {
        let (filled, empty) = progress_bar_parts(view.pct, area.width as usize);
        lines.push(Line::from(vec![
            Span::styled(filled, STYLE_FILL),
            Span::styled(empty, STYLE_EMPTY),
        ]));
    }
    if area.height > 2 {
        if let Some(duration) = view.duration_secs {
            lines.push(time_labels(duration, area.width as usize));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

/// Подпись «[~]NN%[ · обработано MM:SS · осталось MM:SS]».
fn waveform_label_line(view: &WaveformView, tr: &StrTable) -> Line<'static> {
    Line::from(waveform_label(view, tr)).style(Style::new().fg(Color::Cyan))
}

fn waveform_label(view: &WaveformView, tr: &StrTable) -> String {
    let mark = if view.estimated { "~" } else { "" };
    match view.duration_secs {
        Some(duration) => {
            let (done, left) = split_progress(duration, view.pct);
            format!(
                "{mark}{:.0}% · {} {} · {} {}",
                view.pct,
                tr.wf_processed,
                format_mmss(done),
                tr.wf_left,
                format_mmss(left)
            )
        }
        None => format!("{mark}{:.0}%", view.pct),
    }
}

/// Метки времени по краям: `00:00` … длительность.
fn time_labels(duration: f64, width: usize) -> Line<'static> {
    let right = format_mmss(duration);
    let pad = " ".repeat(width.saturating_sub(5 + right.len()));
    Line::from(vec![
        Span::styled("00:00", STYLE_DIM),
        Span::raw(pad),
        Span::styled(right, STYLE_DIM),
    ])
}

/// Footer: подсказки клавиш слева, сводка справа (не влезает — режется).
pub fn render_footer(frame: &mut Frame, area: Rect, view: &FooterView, tr: &StrTable) {
    if area.is_empty() {
        return;
    }
    let hints_width = tr.keyhints.chars().count();
    let mut spans = Vec::new();
    if view.summary.is_empty() {
        // Нечего показывать справа — место целиком подсказкам клавиш.
        spans.push(Span::styled(tr.keyhints.to_string(), Style::new().fg(Color::Gray)));
    } else {
        // Сводка важнее подсказок: не влезают обе — только сводка.
        let both_fit = area.width as usize > hints_width + view.summary.chars().count();
        if both_fit {
            spans.push(Span::styled(tr.keyhints.to_string(), Style::new().fg(Color::Gray)));
            let gap = (area.width as usize).saturating_sub(hints_width + view.summary.chars().count());
            spans.push(Span::raw(" ".repeat(gap.max(1))));
        }
        spans.push(Span::styled(view.summary.clone(), Style::new().fg(Color::Cyan)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// --- табы Dedup / Log / Help и confirm-диалог ------------------------------------

/// Человекочитаемая причина выбора хранителя. Единственный источник текста:
/// используют таблица Dedup и журнал (mod.rs).
pub fn keeper_reason_label(reason: &KeeperReason, tr: &StrTable) -> String {
    match reason {
        KeeperReason::SoleCandidate => tr.reason_sole.to_string(),
        KeeperReason::OnlyNonClipped => tr.reason_no_clip.to_string(),
        KeeperReason::CleanerDynamics { margin_db } => (tr.reason_dynamic)(f64::from(*margin_db)),
        KeeperReason::Louder { margin_db } => (tr.reason_louder)(f64::from(*margin_db)),
        KeeperReason::AllClipped { margin_db } => (tr.reason_all_clipped)(f64::from(*margin_db)),
    }
}

/// Corr по окнам: `0.99/1.00/0.99` (tabular-цифры, моноширинный шрифт).
fn corr_cells(corr: &[f32]) -> String {
    corr.iter().map(|c| format!("{c:.2}")).collect::<Vec<_>>().join("/")
}

/// Флаг клипа пары: текст + цвет (`color-not-only`). `K` — хранитель,
/// `D` — дубль, `KD` — оба, `—` — никто.
fn clip_flag(kept_clipping: bool, dup_clipping: bool) -> (&'static str, Style) {
    match (kept_clipping, dup_clipping) {
        (false, false) => ("—", STYLE_DIM),
        (true, false) => ("K", Style::new().fg(Color::Red)),
        (false, true) => ("D", Style::new().fg(Color::Red)),
        (true, true) => ("KD", Style::new().fg(Color::Red)),
    }
}

/// Таблица дедупликации (таб Dedup): пары дубль → хранитель; пустой итог —
/// заглушка; под таблицей — примечание о нечитаемых WAV.
pub fn render_dedup(
    frame: &mut Frame,
    area: Rect,
    pairs: &[PairReport],
    unparseable: &[String],
    tr: &StrTable,
) {
    let (content, note) = split_dedup_note(area, unparseable);
    if pairs.is_empty() {
        render_dedup_empty(frame, content, tr);
    } else {
        render_dedup_table(frame, content, pairs, tr);
    }
    if let (Some(note_area), false) = (note, unparseable.is_empty()) {
        let names = unparseable.join(", ");
        let text = (tr.unparseable_note)(&names);
        frame.render_widget(Paragraph::new(text).style(STYLE_FILL), note_area);
    }
}

/// Место под примечание: одна строка внизу, если есть нечитаемые файлы.
fn split_dedup_note(area: Rect, unparseable: &[String]) -> (Rect, Option<Rect>) {
    if unparseable.is_empty() {
        return (area, None);
    }
    let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(area);
    (rows[0], Some(rows[1]))
}

fn render_dedup_empty(frame: &mut Frame, area: Rect, tr: &StrTable) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.dedup_title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    frame.render_widget(Paragraph::new(tr.dedup_empty).style(STYLE_DIM), inner);
}

fn render_dedup_table(frame: &mut Frame, area: Rect, pairs: &[PairReport], tr: &StrTable) {
    let header = Row::new([
        tr.th_dup,
        tr.th_kept,
        tr.th_corr,
        tr.th_gain,
        tr.th_crest,
        tr.th_clip,
        tr.th_ren,
        tr.th_reason,
    ])
    .style(STYLE_HEADER);
    let rows: Vec<Row> = pairs.iter().map(|pair| dedup_row(pair, tr)).collect();
    let widths = [
        Constraint::Length(14),
        Constraint::Length(14),
        Constraint::Length(15),
        Constraint::Length(7),
        Constraint::Length(10),
        Constraint::Length(4),
        Constraint::Length(3),
        Constraint::Min(10),
    ];
    frame.render_widget(Table::new(rows, widths).header(header), area);
}

fn dedup_row(pair: &PairReport, tr: &StrTable) -> Row<'static> {
    let (clip, clip_style) = clip_flag(pair.kept_clipping, pair.dup_clipping);
    let (renamed, renamed_style) = if pair.rename_error.is_some() {
        ("!", Style::new().fg(Color::Red))
    } else if pair.renamed {
        ("✓", STYLE_DIM)
    } else {
        ("—", STYLE_DIM)
    };
    Row::new([
        Cell::from(pair.duplicate.clone()),
        Cell::from(pair.kept.clone()),
        Cell::from(corr_cells(&pair.corr)),
        Cell::from(format!("{:+.1}", pair.gain_db)),
        Cell::from(format!("{:.1}/{:.1}", pair.crest_kept_db, pair.crest_dup_db)),
        Cell::from(Span::styled(clip, clip_style)),
        Cell::from(Span::styled(renamed, renamed_style)),
        Cell::from(Span::styled(keeper_reason_label(&pair.reason, tr), STYLE_DIM)),
    ])
}

/// Журнал (таб Log): хвост буфера (автоскролл при scroll == 0), прокрутка
/// вверх, строка статуса/фильтра внизу. Пустой буфер — заглушка.
pub fn render_log(
    frame: &mut Frame,
    area: Rect,
    log: &LogBuffer,
    filter_input: Option<&(String, usize)>,
    tr: &StrTable,
) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.log_title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner);
    render_log_lines(frame, rows[0], log, tr);
    render_log_status(frame, rows[1], log, filter_input, tr);
}

/// Окно журнала: `scroll` строк от хвоста вверх; 0 — последние строки.
fn render_log_lines(frame: &mut Frame, area: Rect, log: &LogBuffer, tr: &StrTable) {
    if log.is_empty() {
        frame.render_widget(Paragraph::new(tr.log_empty).style(STYLE_DIM), area);
        return;
    }
    let visible = log.visible();
    if visible.is_empty() {
        frame.render_widget(Paragraph::new(tr.log_no_match).style(STYLE_DIM), area);
        return;
    }
    let height = area.height as usize;
    let start = visible.len().saturating_sub(height + log.scroll());
    let end = (start + height).min(visible.len());
    let lines: Vec<Line> = visible[start..end].iter().map(|e| log_entry_line(e)).collect();
    frame.render_widget(Paragraph::new(lines), area);
}

/// Строка журнала: dim-время + текст события.
fn log_entry_line(entry: &LogEntry) -> Line<'static> {
    Line::from(vec![
        Span::styled(entry.time.clone(), STYLE_DIM),
        Span::raw("  "),
        Span::raw(entry.message.clone()),
    ])
}

/// Нижняя строка таба Log: ввод фильтра → активный фильтр → подсказки.
fn render_log_status(
    frame: &mut Frame,
    area: Rect,
    log: &LogBuffer,
    filter_input: Option<&(String, usize)>,
    tr: &StrTable,
) {
    let line = match filter_input {
        Some((text, cursor)) => filter_input_line(text, *cursor, tr),
        None => match log.filter() {
            Some(filter) => Line::from(vec![
                Span::raw((tr.log_filter_active)(filter, log.visible_len(), log.len())),
                Span::styled(tr.log_filter_edit_hints.to_string(), STYLE_DIM),
            ]),
            None => log_hints_line(log, tr),
        },
    };
    frame.render_widget(Paragraph::new(line), area);
}

/// Ввод фильтра: курсор `▌` вставлен по позиции (unicode-safe).
fn filter_input_line(text: &str, cursor: usize, tr: &StrTable) -> Line<'static> {
    let mut value: String = text.chars().take(cursor).collect();
    value.push('▌');
    value.extend(text.chars().skip(cursor));
    Line::from(vec![
        Span::raw(tr.log_filter_label.to_string()),
        Span::styled(value, STYLE_FILL),
        Span::styled(tr.log_filter_apply_hints.to_string(), STYLE_DIM),
    ])
}

fn log_hints_line(log: &LogBuffer, tr: &StrTable) -> Line<'static> {
    let mut spans = vec![Span::styled(tr.log_hints.to_string(), STYLE_DIM)];
    if log.scroll() > 0 {
        spans.push(Span::styled((tr.log_scroll_back)(log.scroll()), STYLE_FILL));
    }
    Line::from(spans)
}

/// Строка справки: клавиша (подсвечена) — действие.
fn help_line(key: &str, desc: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {key:<10}"), STYLE_FILL),
        Span::raw(" "),
        Span::raw(desc.to_string()),
    ])
}

/// Справка: оверлей со всеми кейбиндами (истина — по коду app.rs/mod.rs:
/// Tab-навигация, таб Log, confirm выхода, попап запуска).
pub fn render_help(frame: &mut Frame, area: Rect, tr: &StrTable) {
    let popup = centered(area, 86, 88);
    let block = Block::bordered().border_type(BorderType::Rounded).title(tr.help_title);
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if inner.is_empty() {
        return;
    }
    let lines: Vec<Line> = help_lines(tr)
        .into_iter()
        .map(|(key, desc)| help_line(key, desc))
        .chain(std::iter::once(Line::from(Span::styled(
            tr.help_clip_legend,
            STYLE_DIM,
        ))))
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

/// Пары «клавиша — описание» справки (клавиши языконезависимы).
fn help_lines(tr: &StrTable) -> Vec<(&'static str, &'static str)> {
    vec![
        ("[1–4]/Tab", tr.help_tabs),
        ("?", tr.help_self),
        ("Enter/s", tr.help_start),
        ("Enter", tr.help_new_cycle),
        ("p", tr.help_pause),
        ("r", tr.help_retry),
        ("x", tr.help_clear_files),
        ("↑/↓, j/k", tr.help_select),
        ("f", tr.help_filter),
        ("c", tr.help_clear_log),
        ("PgUp/PgDn", tr.help_scroll),
        ("l", tr.help_lang),
        ("Enter", tr.help_confirm),
        ("Esc", tr.help_close),
        ("q/Esc", tr.help_quit),
        ("m/o", tr.help_reserved),
    ]
}

/// Модальный диалог выхода во время прогона (поверх всего кадра).
pub fn render_confirm(frame: &mut Frame, area: Rect, tr: &StrTable) {
    let popup = centered(area, 70, 44);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(STYLE_FILL)
        .title(tr.confirm_title);
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if inner.is_empty() {
        return;
    }
    let lines = vec![
        Line::from(tr.confirm_question),
        Line::from(String::new()),
        Line::from(Span::styled(tr.confirm_soft_head, STYLE_FILL)),
        Line::from(Span::styled(tr.confirm_soft_body, STYLE_FILL)),
        Line::from(String::new()),
        Line::from(vec![
            Span::styled("Enter", STYLE_FILL),
            Span::raw(tr.confirm_soft_action),
            Span::styled("Esc/q", STYLE_FILL),
            Span::raw(tr.confirm_stay),
        ]),
        Line::from(Span::styled(tr.confirm_force_head, STYLE_DIM)),
        Line::from(Span::styled(tr.confirm_force_body, STYLE_DIM)),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// Центрированный прямоугольник в процентах от площади (паттерн popup.rs).
fn centered(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let side_y = (100 - percent_y) / 2;
    let side_x = (100 - percent_x) / 2;
    let rows = Layout::vertical([
        Constraint::Percentage(side_y),
        Constraint::Percentage(percent_y),
        Constraint::Percentage(side_y),
    ])
    .split(area);
    Layout::horizontal([
        Constraint::Percentage(side_x),
        Constraint::Percentage(percent_x),
        Constraint::Percentage(side_x),
    ])
    .split(rows[1])[1]
}

// --- композиция кадра и карта для мыши ------------------------------------------

/// Полный кадр: шапка → табы → статус-строка → контент активного таба → footer.
pub fn render_screen(frame: &mut Frame, area: Rect, view: &ScreenView, tr: &StrTable) {
    let zones = compute_zones(area, status_zone_height(view.current.is_some()));
    render_header(frame, zones.header, &view.header, tr);
    render_tab_bar(frame, zones.tabs, view.tab, tr);
    render_status_line(frame, zones.status, view.banner.as_deref(), view.current.as_ref(), tr);
    match view.tab {
        Tab::Files => render_files_tab(frame, zones, view, tr),
        Tab::Dedup => {
            render_dedup(frame, zones.content, &view.dedup_pairs, &view.dedup_unparseable, tr);
        }
        Tab::Log => render_log(frame, zones.content, &view.log, view.log_filter_input.as_ref(), tr),
        // Фон — таблица файлов, справка поверх.
        Tab::Help => {
            render_files_tab(frame, zones, view, tr);
            render_help(frame, zones.content, tr);
        }
    }
    render_footer(frame, zones.footer, &view.footer, tr);
}

/// Классический вид таба Files: таблица + DETAILS в полном режиме.
fn render_files_tab(frame: &mut Frame, zones: ScreenZones, view: &ScreenView, tr: &StrTable) {
    let has_details = view.details.is_some();
    let (files_area, details_area) = content_split(zones.content, has_details);
    render_files(frame, files_area, &view.files, view.selected, tr);
    render_details_or_empty(frame, details_area, view.details.as_ref(), tr);
}

/// Статус-зона: баннер отмены (приоритет, вся зона) либо индикатор текущего
/// файла — строка «спиннер ▶ имя ─ стадия ─ бар NN%» + брайль-карта огибающей
/// на всю ширину с подписью (зона 4 ряда, см. [`status_zone_height`]).
fn render_status_line(
    frame: &mut Frame,
    area: Rect,
    banner: Option<&str>,
    current: Option<&CurrentFileView>,
    tr: &StrTable,
) {
    if let Some(text) = banner {
        let line = Line::from(Span::styled(format!("! {text}"), STYLE_FILL));
        frame.render_widget(Paragraph::new(line), area);
        return;
    }
    let Some(current) = current else {
        return;
    };
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
    render_current_line(frame, rows[0], current);
    render_waveform(frame, rows[1], &current.waveform, tr);
}

/// Строка индикатора: спиннер, имя, стадия с таймером, бар/индeterminate, %.
fn render_current_line(frame: &mut Frame, area: Rect, current: &CurrentFileView) {
    let (bar, pct_label) = match current.pct {
        Some(pct) if current.estimated => {
            (progress_bar(pct, PROGRESS_BAR_WIDTH), format!("~{pct:.0}%"))
        }
        Some(pct) => (progress_bar(pct, PROGRESS_BAR_WIDTH), format!("{pct:>3.0}%")),
        None => (indeterminate_bar(current.phase, PROGRESS_BAR_WIDTH), "··".to_string()),
    };
    let line = Line::from(vec![
        Span::styled(spinner(current.phase).to_string(), STYLE_FILL),
        Span::raw(" ▶ "),
        Span::styled(current.name.clone(), Style::new().add_modifier(Modifier::BOLD)),
        Span::raw(" ─ "),
        Span::styled(stage_cell(&current.stage, Some(current.stage_elapsed)), STYLE_DIM),
        Span::raw(" ─ "),
        Span::styled(bar, STYLE_FILL),
        Span::styled(format!(" {pct_label}"), STYLE_DIM),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// Зона таблицы файлов внутри контента (мышь — [`file_row_at`] по ней же);
/// `has_details` — полный режим с DETAILS-панелью справа.
pub fn files_table_area(content: Rect, has_details: bool) -> Rect {
    content_split(content, has_details).0
}

/// Высота статус-зоны: с индикатором текущего файла — индикатор + карта +
/// подпись ([`STATUS_TALL_ROWS`]); иначе одна строка (баннер/пусто).
/// Значение общее для рендера и карты кликов мыши.
pub fn status_zone_height(has_current: bool) -> u16 {
    if has_current {
        STATUS_TALL_ROWS
    } else {
        1
    }
}

/// Зоны кадра: шапка 3, табы 1, статус-зона (1 или 4), контент Min(1), footer 1.
pub fn compute_zones(area: Rect, status_height: u16) -> ScreenZones {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(status_height.max(1)),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(area);
    ScreenZones {
        header: rows[0],
        tabs: rows[1],
        status: rows[2],
        content: rows[3],
        footer: rows[4],
    }
}

/// Разделение контента: полный режим (≥100 колонок) — FILES + DETAILS.
fn content_split(area: Rect, has_details: bool) -> (Rect, Rect) {
    if !has_details || area.width < FULL_LAYOUT_MIN_WIDTH {
        return (area, Rect::default());
    }
    let chunks =
        Layout::horizontal([Constraint::Min(30), Constraint::Length(DETAILS_WIDTH)]).split(area);
    (chunks[0], chunks[1])
}

/// Таб под координатой X (клик мышью); зоны те же, что в [`render_tab_bar`].
pub fn tab_at(x: u16, tabs_area: Rect) -> Option<Tab> {
    if tabs_area.is_empty() {
        return None;
    }
    let chunks = Layout::horizontal([Constraint::Fill(1); TAB_COUNT]).split(tabs_area);
    TABS.iter().copied().find(|tab| {
        let chunk = chunks[tab_index(*tab)];
        x >= chunk.x && x < chunk.x + chunk.width
    })
}

/// Индекс таба в [`TABS`] (для сопоставления с зонами layout).
fn tab_index(tab: Tab) -> usize {
    TABS.iter().position(|t| *t == tab).unwrap_or_default()
}

/// Строка таблицы под координатой Y (рамка + заголовок учитываются);
/// валидность индекса против `files.len()` — на вызывающей стороне.
pub fn file_row_at(y: u16, table_area: Rect) -> Option<usize> {
    if table_area.width == 0 || table_area.height < 3 {
        return None;
    }
    let first_row_y = table_area.y + 2;
    if y < first_row_y || y >= table_area.y + table_area.height - 1 {
        return None;
    }
    Some((y - first_row_y) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::i18n::{self, Lang};
    use ratatui::Terminal;

    /// Дефолтная (русская) таблица строк для golden-рендеров.
    fn tr() -> &'static i18n::StrTable {
        i18n::tr(Lang::Ru)
    }
    use ratatui::backend::TestBackend;

    /// Рендер произвольной функции в буфер (текст + доступ к ячейкам).
    fn render_frame<F>(width: u16, height: u16, draw: F) -> Terminal<TestBackend>
    where
        F: FnOnce(&mut Frame, Rect),
    {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, frame.area())).unwrap();
        terminal
    }

    /// Весь текст буфера одной строкой.
    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal.backend().buffer().content().iter().map(|cell| cell.symbol()).collect()
    }

    /// Цвет ячейки буфера.
    fn fg_at(terminal: &Terminal<TestBackend>, x: u16, y: u16) -> Color {
        terminal.backend().buffer()[(x, y)].fg
    }

    // --- форматтеры и чистые функции ---

    #[test]
    fn format_mmss_zero_and_bounds() {
        assert_eq!(format_mmss(0.0), "00:00");
        assert_eq!(format_mmss(59.9), "00:59");
        assert_eq!(format_mmss(60.0), "01:00");
        assert_eq!(format_mmss(-5.0), "00:00");
    }

    #[test]
    fn format_mmss_non_integer_duration() {
        assert_eq!(format_mmss(125.4), "02:05");
        assert_eq!(format_mmss(3661.0), "61:01");
    }

    #[test]
    fn split_progress_zero_fifty_hundred() {
        assert_eq!(split_progress(200.0, 0.0), (0.0, 200.0));
        assert_eq!(split_progress(200.0, 50.0), (100.0, 100.0));
        assert_eq!(split_progress(200.0, 100.0), (200.0, 0.0));
    }

    #[test]
    fn split_progress_clamps_pct() {
        assert_eq!(split_progress(200.0, 150.0), (200.0, 0.0));
        assert_eq!(split_progress(200.0, -10.0), (0.0, 200.0));
    }

    #[test]
    fn progress_bar_fifty_percent_is_half() {
        assert_eq!(progress_bar(50.0, 10), "█████░░░░░");
        assert_eq!(progress_bar(0.0, 4), "░░░░");
        assert_eq!(progress_bar(100.0, 4), "████");
    }

    #[test]
    fn progress_bar_edge_inputs() {
        assert_eq!(progress_bar(0.0, 0), "");
        assert_eq!(progress_bar(150.0, 3), "███");
        assert_eq!(progress_bar(-5.0, 3), "░░░");
    }

    #[test]
    fn trace_span_silence_is_empty() {
        assert_eq!(trace_span(0.0, 8), None, "тишина — точек нет");
        assert_eq!(trace_span(-0.5, 12), None, "отрицательное клампится в 0");
    }

    #[test]
    fn trace_span_full_peak_covers_all_rows() {
        assert_eq!(trace_span(1.0, 8), Some((0, 7)), "полная амплитуда — все ряды");
        assert_eq!(trace_span(1.5, 12), Some((0, 11)), "клампится в 1.0");
    }

    #[test]
    fn trace_span_is_symmetric_around_midline() {
        // Половина амплитуды: центр 4 из 8 → ряды 2..=5 (симметрично оси 3.5).
        assert_eq!(trace_span(0.5, 8), Some((2, 5)));
        // 12 рядов: центр 6 → 3..=8.
        assert_eq!(trace_span(0.5, 12), Some((3, 8)));
        // Шёпот: ceil даёт видимый след из одной пары точек вокруг оси.
        assert_eq!(trace_span(0.01, 8), Some((3, 4)));
    }

    #[test]
    fn pixel_amplitude_empty_peaks_is_silence() {
        assert_eq!(pixel_amplitude(&[], 0, 10), 0.0);
    }

    #[test]
    fn pixel_amplitude_compress_takes_window_max() {
        // 4 пика на 2 пиксельные колонки: max по окну.
        let peaks = [0.1, 0.9, 0.2, 0.3];
        assert_eq!(pixel_amplitude(&peaks, 0, 2), 0.9);
        assert_eq!(pixel_amplitude(&peaks, 1, 2), 0.3);
    }

    #[test]
    fn pixel_amplitude_stretch_takes_nearest_sample() {
        // 2 пика на 4 пиксельные колонки: ближайший сэмпл.
        let peaks = [0.5, 0.25];
        assert_eq!(pixel_amplitude(&peaks, 0, 4), 0.5);
        assert_eq!(pixel_amplitude(&peaks, 1, 4), 0.5);
        assert_eq!(pixel_amplitude(&peaks, 2, 4), 0.25);
        assert_eq!(pixel_amplitude(&peaks, 3, 4), 0.25);
    }

    #[test]
    fn badge_labels_are_fixed_text() {
        let ru = tr();
        assert_eq!(FileBadge::Run.label(ru), "ПРЦ");
        assert_eq!(FileBadge::Ok.label(ru), "ГОТ");
        assert_eq!(FileBadge::Fail.label(ru), "ОШБ");
        assert_eq!(FileBadge::Skip.label(ru), "ПРОП");
        assert_eq!(FileBadge::Dup.label(ru), "ДУБЛ");

        let en = i18n::tr(Lang::En);
        assert_eq!(FileBadge::Run.label(en), "RUN");
        assert_eq!(FileBadge::Ok.label(en), "OK");
        assert_eq!(FileBadge::Fail.label(en), "FAIL");
        assert_eq!(FileBadge::Skip.label(en), "SKIP");
        assert_eq!(FileBadge::Dup.label(en), "DUP");
    }

    #[test]
    fn badge_cells_are_padded_to_four_columns_in_both_locales() {
        for lang in [Lang::Ru, Lang::En] {
            let table = i18n::tr(lang);
            for badge in [FileBadge::Run, FileBadge::Ok, FileBadge::Fail, FileBadge::Skip, FileBadge::Dup] {
                assert_eq!(badge.cell(table).chars().count(), 4, "{lang:?}: {badge:?}");
            }
        }
    }

    #[test]
    fn badge_colors_are_distinct() {
        let badges = [FileBadge::Run, FileBadge::Ok, FileBadge::Fail, FileBadge::Skip, FileBadge::Dup];
        for (i, a) in badges.iter().enumerate() {
            for b in &badges[i + 1..] {
                assert_ne!(a.style().fg, b.style().fg, "{a:?} и {b:?} одного цвета");
            }
        }
    }

    // --- карта координат для мыши ---

    #[test]
    fn tab_at_hits_four_equal_zones() {
        let area = Rect::new(0, 10, 80, 1);
        assert_eq!(tab_at(10, area), Some(Tab::Files));
        assert_eq!(tab_at(25, area), Some(Tab::Dedup));
        assert_eq!(tab_at(45, area), Some(Tab::Log));
        assert_eq!(tab_at(79, area), Some(Tab::Help));
    }

    #[test]
    fn tab_at_outside_area_is_none() {
        let area = Rect::new(5, 0, 40, 1);
        assert_eq!(tab_at(4, area), None);
        assert_eq!(tab_at(45, area), None);
        assert_eq!(tab_at(10, Rect::default()), None);
    }

    #[test]
    fn file_row_at_maps_inner_rows() {
        let area = Rect::new(0, 0, 50, 10);
        assert_eq!(file_row_at(0, area), None); // верхняя рамка
        assert_eq!(file_row_at(1, area), None); // заголовок таблицы
        assert_eq!(file_row_at(2, area), Some(0));
        assert_eq!(file_row_at(5, area), Some(3));
        assert_eq!(file_row_at(9, area), None); // нижняя рамка
        assert_eq!(file_row_at(2, Rect::new(0, 0, 50, 2)), None);
    }

    #[test]
    fn zones_layout_is_stable() {
        let zones = compute_zones(Rect::new(0, 0, 80, 24), 1);
        assert_eq!(zones.header, Rect::new(0, 0, 80, 3));
        assert_eq!(zones.tabs, Rect::new(0, 3, 80, 1));
        assert_eq!(zones.status, Rect::new(0, 4, 80, 1));
        assert_eq!(zones.content, Rect::new(0, 5, 80, 18));
        assert_eq!(zones.footer, Rect::new(0, 23, 80, 1));
    }

    #[test]
    fn zones_grow_status_zone_for_current_file() {
        let zones = compute_zones(Rect::new(0, 0, 80, 24), status_zone_height(true));
        assert_eq!(zones.status, Rect::new(0, 4, 80, 4), "индикатор + карта + подпись");
        assert_eq!(zones.content, Rect::new(0, 8, 80, 15), "контент ужался, но жив");
        assert_eq!(status_zone_height(false), 1, "без текущего файла — одна строка");
    }

    // --- инфографика: таймеры/ETA/анимация (багфикс-волна G) ---

    #[test]
    fn indeterminate_bar_moves_window_and_loops() {
        let width = 12;
        assert_ne!(indeterminate_bar(0, width), indeterminate_bar(1, width), "анимация по фазе");
        // Окно = width/4; период = width + fill → фаза циклится.
        let fill = width / 4;
        assert_eq!(
            indeterminate_bar(0, width).chars().filter(|c| *c == BAR_FILLED).count(),
            fill
        );
        assert_eq!(indeterminate_bar(0, width), indeterminate_bar((width + fill) as u64, width));
        assert_eq!(indeterminate_bar(0, 0), "");
    }

    #[test]
    fn eta_secs_is_average_times_remaining() {
        assert_eq!(eta_secs(20.0, 0, 10), None, "до первого завершённого ETA нет");
        assert_eq!(eta_secs(20.0, 10, 10), None, "всё завершено");
        assert_eq!(eta_secs(20.0, 2, 0), None, "пустой план");
        let eta = eta_secs(20.0, 2, 5).expect("ETA по двум завершённым");
        assert!((eta - 30.0).abs() < 1e-9, "10 c/файл × 3 остатка");
    }

    #[test]
    fn stage_cell_appends_ticking_timer() {
        assert_eq!(stage_cell("download", None), "download");
        assert_eq!(stage_cell("download", Some(12.4)), "download · 00:12");
        assert_eq!(stage_cell("separate", Some(252.0)), "separate · 04:12");
    }

    #[test]
    fn spinner_cycles_through_frames() {
        assert_eq!(spinner(0), spinner(10), "полный оборот — 10 кадров");
        assert_ne!(spinner(0), spinner(1));
    }

    #[test]
    fn header_shows_model_load_indicator_only_while_active() {
        let loading = HeaderView { model: "sw".into(), model_pct: Some(37.4), ..HeaderView::default() };
        let text = buffer_text(&render_frame(100, 3, |f, a| render_header(f, a, &loading, tr())));
        assert!(text.contains("модель: 37%"), "глобальная загрузка модели видна");

        let idle = HeaderView { model: "sw".into(), ..HeaderView::default() };
        let text = buffer_text(&render_frame(100, 3, |f, a| render_header(f, a, &idle, tr())));
        assert!(!text.contains("модель:"), "вне загрузки слот скрыт");
    }

    #[test]
    fn header_model_loading_label_in_both_locales() {
        let loading = HeaderView { model: "sw".into(), model_pct: Some(37.4), ..HeaderView::default() };
        let text = buffer_text(&render_frame(100, 3, |f, a| render_header(f, a, &loading, tr())));
        assert!(text.contains("модель: 37%"), "ru: {text}");

        let en = i18n::tr(Lang::En);
        let text = buffer_text(&render_frame(100, 3, |f, a| render_header(f, a, &loading, en)));
        assert!(text.contains("model: 37%"), "en: {text}");
        assert!(!text.contains("модель:"), "ru-строка ушла: {text}");
    }

    #[test]
    fn status_line_shows_current_file_and_banner_takes_priority() {
        let current = CurrentFileView {
            name: "a.wav".into(),
            stage: "download".into(),
            pct: None,
            estimated: false,
            phase: 2,
            stage_elapsed: 5.0,
            waveform: WaveformView::default(),
        };
        let text = buffer_text(&render_frame(100, 1, |f, a| render_status_line(f, a, None, Some(&current), tr())));
        assert!(text.contains("a.wav"));
        assert!(text.contains("download · 00:05"), "таймер стадии");
        assert!(text.contains("··"), "стадия без процента — индeterminate-бар");

        let text = buffer_text(&render_frame(100, 1, |f, a| {
            render_status_line(f, a, Some("Прерывание… текущий файл доигрывается"), Some(&current), tr())
        }));
        assert!(text.contains("Прерывание…"), "баннер приоритетен");
        assert!(!text.contains("a.wav"), "индикатор файла скрыт под баннером");
    }

    #[test]
    fn status_line_marks_estimated_pct_with_tilde() {
        let base = |pct: Option<f64>, estimated: bool| CurrentFileView {
            name: "a.wav".into(),
            stage: "separate".into(),
            pct,
            estimated,
            phase: 0,
            stage_elapsed: 0.0,
            waveform: WaveformView::default(),
        };
        let text = buffer_text(&render_frame(100, 1, |f, a| {
            render_status_line(f, a, None, Some(&base(Some(42.0), true)), tr())
        }));
        assert!(text.contains("~42%"), "оценка — с тильдой: {text}");

        let text = buffer_text(&render_frame(100, 1, |f, a| {
            render_status_line(f, a, None, Some(&base(Some(42.0), false)), tr())
        }));
        assert!(text.contains(" 42%"), "фактический — без тильды: {text}");
    }

    #[test]
    fn status_line_renders_waveform_below_indicator_full_width() {
        let current = CurrentFileView {
            name: "a.wav".into(),
            stage: "separate".into(),
            pct: Some(50.0),
            estimated: false,
            phase: 0,
            stage_elapsed: 3.0,
            waveform: WaveformView {
                pct: 50.0,
                duration_secs: None,
                estimated: false,
                peaks: flat_peaks(80, 0.75),
            },
        };
        // Зона 4 ряда: строка индикатора + карта 2 ряда + подпись.
        let terminal = render_frame(80, 4, |f, a| render_status_line(f, a, None, Some(&current), tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("▶ a.wav"), "индикатор в первой строке: {text}");
        assert!(text.contains("50%"), "подпись под картой: {text}");
        // Карта — брайль-точки на всю ширину; заливка слева, серые пики справа.
        assert_eq!(fg_at(&terminal, 5, 2), Color::Green);
        assert_eq!(fg_at(&terminal, 75, 2), Color::DarkGray);
    }

    #[test]
    fn status_line_without_peaks_shows_fallback_bar_under_indicator() {
        let current = CurrentFileView {
            name: "a.wav".into(),
            stage: "download".into(),
            pct: Some(0.0),
            estimated: false,
            phase: 0,
            stage_elapsed: 1.0,
            waveform: WaveformView::default(),
        };
        let terminal = render_frame(80, 4, |f, a| render_status_line(f, a, None, Some(&current), tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains('░'), "fallback-бар вместо карты: {text}");
    }

    #[test]
    fn status_line_banner_hides_waveform() {
        let current = CurrentFileView {
            name: "a.wav".into(),
            stage: "separate".into(),
            pct: Some(50.0),
            estimated: false,
            phase: 0,
            stage_elapsed: 3.0,
            waveform: WaveformView {
                pct: 50.0,
                duration_secs: None,
                estimated: false,
                peaks: flat_peaks(80, 0.75),
            },
        };
        let terminal = render_frame(80, 4, |f, a| {
            render_status_line(f, a, Some("Прерывание…"), Some(&current), tr())
        });
        assert!(buffer_text(&terminal).contains("Прерывание…"));
        assert_ne!(fg_at(&terminal, 5, 2), Color::Green, "карта скрыта под баннером");
    }

    #[test]
    fn table_stage_column_shows_timer_and_indeterminate_bar() {
        let rows = vec![FileRowView {
            badge: FileBadge::Run,
            name: "a.wav".into(),
            stage: "download".into(),
            pct: 0.0,
            estimated: false,
            stems_done: None,
            stage_elapsed: Some(65.0),
            indeterminate: Some(3),
        }];
        let text = buffer_text(&render_frame(100, 8, |f, a| render_files(f, a, &rows, None, tr())));
        assert!(text.contains("download · 01:05"));
        assert!(text.contains("··"), "индeterminate-бар");
        assert!(!text.contains("  0%"), "мёртвых 0% больше нет");
    }

    // --- golden-рендеры ---
    fn header_view() -> HeaderView {
        HeaderView {
            host: "nas.local".into(),
            share: "media".into(),
            remote_dir: "/recordings".into(),
            model: "sw".into(),
            gpu: None,
            model_pct: None,
        }
    }

    fn file_row_view(badge: FileBadge, name: &str, pct: f64, stems: Option<u8>) -> FileRowView {
        FileRowView {
            badge,
            name: name.into(),
            stage: "separate".into(),
            pct,
            estimated: false,
            stems_done: stems,
            stage_elapsed: None,
            indeterminate: None,
        }
    }

    /// Синтетическая огибающая: все колонки одинаковые — детерминированный golden.
    fn flat_peaks(columns: usize, peak: f32) -> Arc<[f32]> {
        Arc::from(vec![peak; columns])
    }

    #[test]
    fn header_renders_connection_model_and_gpu_placeholder() {
        let terminal = render_frame(80, 3, |f, area| render_header(f, area, &header_view(), tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("audio_split"));
        assert!(text.contains("//nas.local/media"));
        assert!(text.contains("/recordings"));
        assert!(text.contains("sw"));
        assert!(text.contains("GPU: n/a"));
    }

    #[test]
    fn header_renders_gpu_bar_when_poller_present() {
        let mut view = header_view();
        view.gpu = Some(GpuView {
            name: "RTX".into(),
            used_gib: 2.1,
            total_gib: 6.0,
            util_pct: Some(63.0),
            util_history: vec![10, 40, 63, 90],
        });
        // Широкий кадр: спарклайн (20) + VRAM должны помещаться в одну строку.
        let terminal = render_frame(140, 3, |f, area| render_header(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        // 2.1/6.0 = 35% → 2 из 6 полуколонок заполнены.
        assert!(text.contains("GPU 63%"), "утилизация процентом: {text}");
        assert!(text.contains("▂▄▅▇"), "спарклайн по истории [10,40,63,90]: {text}");
        assert!(text.contains("▓"), "разделитель VRAM-бара: {text}");
        assert!(text.contains("██░░░░"));
        assert!(text.contains("2.1/6.0GiB"));
        assert!(!text.contains("n/a"));
    }

    #[test]
    fn header_gpu_without_utilization_shows_na_but_keeps_vram() {
        let mut view = header_view();
        view.gpu = Some(GpuView {
            name: "RTX".into(),
            used_gib: 1.0,
            total_gib: 8.0,
            util_pct: None,
            util_history: Vec::new(),
        });
        let terminal = render_frame(120, 3, |f, area| render_header(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("GPU n/a "), "утилизация неизвестна: {text}");
        assert!(text.contains("▓"), "VRAM-бар остаётся: {text}");
        assert!(text.contains("1.0/8.0GiB"));
    }

    #[test]
    fn sparkline_maps_values_to_blocks_right_aligned() {
        assert_eq!(sparkline(&[], 4), "    ", "пустая история — пробелы");
        assert_eq!(sparkline(&[100], 4), "   █", "хвост выравнивается вправо");
        assert_eq!(sparkline(&[0, 50, 100], 4), " ▁▅█", "уровень 50 → round(3.5) → ▅");
        assert_eq!(sparkline(&[10, 20, 30, 40], 2), "▃▄", "берутся последние width значений");
        assert_eq!(sparkline(&[255], 4), "   █", "значения выше 100 клампятся");
    }

    #[test]
    fn files_progress_bar_over_done_total() {
        assert_eq!(files_progress(tr(), 0, 3), "файлы 0/3 ░░░░░░ 0%");
        assert_eq!(files_progress(tr(), 1, 3), "файлы 1/3 ██░░░░ 33%");
        assert_eq!(files_progress(tr(), 3, 3), "файлы 3/3 ██████ 100%");
        assert_eq!(files_progress(tr(), 0, 0), "файлы 0/0", "пустой план — без бара");
    }

    #[test]
    fn table_marks_estimated_pct_with_tilde() {
        let mut row = file_row_view(FileBadge::Run, "a.wav", 63.0, None);
        row.estimated = true;
        let terminal = render_frame(80, 10, |f, area| render_files(f, area, &[row], None, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("~63%"), "оценка помечается тильдой: {text}");

        let real = file_row_view(FileBadge::Run, "a.wav", 63.0, None);
        let terminal = render_frame(80, 10, |f, area| render_files(f, area, &[real], None, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains(" 63%"), "фактический — без тильды: {text}");
    }

    #[test]
    fn table_renders_all_five_badges_and_numbers() {
        let files = [
            file_row_view(FileBadge::Run, "a.wav", 63.0, None),
            file_row_view(FileBadge::Ok, "b.wav", 100.0, Some(6)),
            file_row_view(FileBadge::Fail, "c.wav", 0.0, None),
            file_row_view(FileBadge::Skip, "d.wav", 0.0, None),
            file_row_view(FileBadge::Dup, "e.wav", 0.0, None),
        ];
        let terminal = render_frame(80, 10, |f, area| render_files(f, area, &files, None, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("ПРЦ"));
        assert!(text.contains("ГОТ"));
        assert!(text.contains("ОШБ"));
        assert!(text.contains("ПРОП"));
        assert!(text.contains("ДУБЛ"));
        assert!(text.contains(" 63%"));
        assert!(text.contains("6/6"), "известный счётчик — числом");
        assert!(text.contains("—"), "неизвестный счётчик — «—», не «0/6»");
        assert!(text.contains("████████████"));
    }

    #[test]
    fn stems_cell_is_dash_until_count_known() {
        assert_eq!(stems_cell(None, tr()), "—");
        assert_eq!(stems_cell(Some(0), tr()), "0/6");
        assert_eq!(stems_cell(Some(6), tr()), "6/6");
    }

    #[test]
    fn table_selected_row_is_marked_with_arrow() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 10.0, None)];
        let terminal = render_frame(60, 8, |f, area| render_files(f, area, &files, Some(0), tr()));
        assert!(buffer_text(&terminal).contains("▸"));
        let unselected = render_frame(60, 8, |f, area| render_files(f, area, &files, None, tr()));
        assert!(!buffer_text(&unselected).contains("▸"));
    }

    #[test]
    fn empty_table_shows_hint() {
        let terminal = render_frame(60, 8, |f, area| render_files(f, area, &[], None, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("нет файлов — Esc → настройка"));
        assert!(!text.contains("ПРЦ"));
    }

    #[test]
    fn waveform_at_zero_percent_all_dim() {
        let view = WaveformView { pct: 0.0, duration_secs: Some(120.0), estimated: false, peaks: flat_peaks(80, 0.75) };
        let terminal = render_frame(80, 5, |f, area| render_waveform(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("0% · обработано 00:00 · осталось 02:00"));
        // Ни одной залитой колонки на рядах карты (y=1..3).
        for y in 1..4 {
            assert_ne!(fg_at(&terminal, 0, y), Color::Green);
            assert_ne!(fg_at(&terminal, 79, y), Color::Green);
        }
    }

    #[test]
    fn waveform_at_fifty_percent_boundary_on_column() {
        let view = WaveformView { pct: 50.0, duration_secs: Some(120.0), estimated: false, peaks: flat_peaks(80, 0.75) };
        let terminal = render_frame(80, 5, |f, area| render_waveform(f, area, &view, tr()));
        assert!(buffer_text(&terminal).contains("50% · обработано 01:00 · осталось 01:00"));
        // Граница заливки в пикселях канваса (2 px на колонку): 40 колонок
        // залиты, 41-я — серая.
        assert_eq!(fg_at(&terminal, 39, 2), Color::Green);
        assert_eq!(fg_at(&terminal, 40, 2), Color::DarkGray);
    }

    #[test]
    fn waveform_at_hundred_percent_all_bright() {
        let view = WaveformView { pct: 100.0, duration_secs: Some(120.0), estimated: false, peaks: flat_peaks(80, 0.75) };
        let terminal = render_frame(80, 5, |f, area| render_waveform(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("100% · обработано 02:00 · осталось 00:00"));
        assert_eq!(fg_at(&terminal, 0, 2), Color::Green);
        assert_eq!(fg_at(&terminal, 79, 2), Color::Green);
        assert!(text.contains("00:00"));
        assert!(text.contains("02:00"));
    }

    #[test]
    fn waveform_renders_braille_dots_full_resolution() {
        // Карта — брайль-символы: полная амплитуда в средней строке канваса
        // закрашивает все 2×4 точки ячейки → «⣿»; в ячейку колонки попадают
        // обе пиксельные колонки (2× разрешение по горизонтали).
        let view = WaveformView { pct: 0.0, duration_secs: None, estimated: false, peaks: flat_peaks(80, 1.0) };
        let terminal = render_frame(80, 4, |f, area| render_waveform(f, area, &view, tr()));
        let symbol_at = |x: u16, y: u16| terminal.backend().buffer()[(x, y)].symbol().to_string();
        assert_eq!(symbol_at(0, 2), "⣿", "полный след — все точки ячейки");
        assert_eq!(symbol_at(40, 2), "⣿", "карта тянется на всю ширину");
        assert_eq!(symbol_at(0, 1), "⣿", "полная амплитуда закрашивает и верхний ряд");
    }

    #[test]
    fn waveform_silence_renders_no_dots() {
        // Тихие колонки дают пустую ячейку (след симметрично сжат в точку).
        let view = WaveformView { pct: 0.0, duration_secs: None, estimated: false, peaks: flat_peaks(80, 0.0) };
        let terminal = render_frame(80, 4, |f, area| render_waveform(f, area, &view, tr()));
        let symbol_at = |x: u16, y: u16| terminal.backend().buffer()[(x, y)].symbol().to_string();
        assert_eq!(symbol_at(0, 2), " ", "тишина — точек нет");
        assert!(buffer_text(&terminal).contains("0%"));
    }

    #[test]
    fn waveform_label_marks_estimated_with_tilde() {
        let estimated = WaveformView { pct: 42.0, duration_secs: None, estimated: true, peaks: Vec::new().into() };
        let terminal = render_frame(40, 2, |f, area| render_waveform(f, area, &estimated, tr()));
        assert!(buffer_text(&terminal).contains("~42%"));

        let real = WaveformView { pct: 42.0, duration_secs: None, estimated: false, peaks: Vec::new().into() };
        let text = buffer_text(&render_frame(40, 2, |f, area| render_waveform(f, area, &real, tr())));
        assert!(text.contains("42%"), "фактический — без тильды: {text}");
        assert!(!text.contains('~'), "никаких тильд у фактического: {text}");
    }

    #[test]
    fn waveform_narrow_height_degrades_to_two_canvas_rows() {
        // 3 ряда: карта 2 ряда (8 пиксельных) + подпись под ней; метки времени
        // уходят первыми.
        let view = WaveformView { pct: 50.0, duration_secs: Some(120.0), estimated: false, peaks: flat_peaks(40, 0.75) };
        let terminal = render_frame(40, 3, |f, area| render_waveform(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("50% ·"), "подпись под картой: {text}");
        assert_eq!(fg_at(&terminal, 5, 1), Color::Green, "карта жива: залитые колонки");
        assert!(!text.contains("00:00"), "метки времени не влезли — скрыты");
    }

    #[test]
    fn waveform_falls_back_to_plain_bar_without_peaks() {
        let view = WaveformView { pct: 40.0, duration_secs: Some(120.0), estimated: false, peaks: Vec::new().into() };
        let terminal = render_frame(40, 5, |f, area| render_waveform(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains('█'));
        assert!(text.contains('░'));
        assert!(text.contains("40% · обработано 00:48 · осталось 01:12"));
    }

    #[test]
    fn waveform_too_short_area_falls_back_without_panic() {
        let view = WaveformView { pct: 50.0, duration_secs: None, estimated: false, peaks: flat_peaks(40, 0.75) };
        render_frame(40, 1, |f, area| render_waveform(f, area, &view, tr()));
        render_frame(0, 0, |f, area| render_waveform(f, area, &view, tr()));
    }

    #[test]
    fn details_renders_stage_eta_stems_and_events() {
        let view = DetailsView {
            name: "a.wav".into(),
            stage: "separate".into(),
            source: "//nas/media/a.wav".into(),
            eta_secs: Some(252.0),
            format: "flac".into(),
            stems: vec![
                StemView { name: "vocals".into(), state: StemState::Done },
                StemView { name: "drums".into(), state: StemState::Current },
                StemView { name: "bass".into(), state: StemState::Pending },
            ],
            events: vec!["download ok".into(), "separate 45%".into()],
            waveform: WaveformView::default(),
        };
        let terminal = render_frame(50, 14, |f, area| render_details(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("DETAILS"));
        assert!(text.contains("a.wav"));
        assert!(text.contains("стадия: separate · ETA 04:12"));
        assert!(text.contains("источник: //nas/media/a.wav"));
        assert!(text.contains("формат: flac"));
        assert!(text.contains("✓"));
        assert!(text.contains("▸"));
        assert!(text.contains("·"));
        assert!(text.contains("separate 45%"));
    }

    #[test]
    fn details_empty_state_when_nothing_selected() {
        let terminal = render_frame(50, 10, |f, area| render_details_or_empty(f, area, None, tr()));
        assert!(buffer_text(&terminal).contains("нет выбранного файла"));
    }

    #[test]
    fn footer_renders_keyhints_and_summary() {
        let view = FooterView { summary: "OK 3 · FAIL 1 · DUP 2".into() };
        // 100 колонок: подсказки (60) + сводка (21) влезают с зазором.
        let terminal = render_frame(100, 1, |f, area| render_footer(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("[s]тарт"));
        assert!(text.contains("[p]ауза"));
        assert!(text.contains("[r]етрай"));
        assert!(text.contains("[m]одель"));
        assert!(text.contains("[o]тчёт"));
        assert!(text.contains("[?]справка"));
        assert!(text.contains("[q]выход"));
        assert!(text.contains("OK 3 · FAIL 1 · DUP 2"));
    }

    fn screen_view(files: Vec<FileRowView>, details: Option<DetailsView>) -> ScreenView {
        ScreenView {
            header: header_view(),
            tab: Tab::Files,
            files,
            selected: Some(0),
            details,
            footer: FooterView { summary: "ETA 04:12".into() },
            ..Default::default()
        }
    }

    #[test]
    fn screen_idle_renders_full_frame_with_empty_state() {
        let view = screen_view(Vec::new(), None);
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("audio_split"));
        assert!(text.contains("[1]Файлы"));
        assert!(text.contains("[4]Справка"));
        assert!(text.contains("нет файлов — Esc → настройка"));
        assert!(text.contains("[q]выход"));
        assert!(text.contains("ETA 04:12"));
    }

    #[test]
    fn screen_running_compact_hides_details_panel() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 63.0, Some(2))];
        let view = screen_view(files.into(), None);
        let terminal = render_frame(80, 24, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("ФАЙЛЫ"));
        assert!(text.contains("ПРЦ"));
        // Compact-режим: DETAILS-колонка скрыта на 80 колонках.
        assert!(!text.contains("DETAILS"));
        assert!(!text.contains("нет выбранного файла"));
    }

    #[test]
    fn screen_running_full_shows_details_with_waveform() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 50.0, Some(3))];
        let details = DetailsView {
            name: "a.wav".into(),
            stage: "separate".into(),
            source: "//nas/media/a.wav".into(),
            eta_secs: Some(60.0),
            format: "flac".into(),
            stems: vec![StemView { name: "vocals".into(), state: StemState::Done }],
            events: vec!["separate 50%".into()],
            waveform: WaveformView {
                pct: 50.0,
                duration_secs: Some(120.0),
                estimated: false,
                peaks: flat_peaks(34, 0.6),
            },
        };
        let view = screen_view(files.into(), Some(details));
        let terminal = render_frame(110, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("DETAILS"));
        assert!(text.contains("50% · обработано 01:00 · осталось 01:00"));
        assert!(text.contains("стадия: separate · ETA 01:00"));
    }

    #[test]
    fn screen_running_shows_waveform_in_status_zone() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 50.0, Some(3))];
        let mut view = screen_view(files.into(), None);
        view.current = Some(CurrentFileView {
            name: "a.wav".into(),
            stage: "separate".into(),
            pct: Some(50.0),
            estimated: false,
            phase: 0,
            stage_elapsed: 3.0,
            waveform: WaveformView {
                pct: 50.0,
                duration_secs: None,
                estimated: false,
                peaks: flat_peaks(80, 0.75),
            },
        });
        let terminal = render_frame(80, 24, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("▶ a.wav"), "индикатор текущего файла: {text}");
        assert!(text.contains("50%"), "подпись карты в статус-зоне: {text}");
        // Карта на всю ширину: слева заливка, справа серые пики.
        assert_eq!(fg_at(&terminal, 5, 6), Color::Green);
        assert_eq!(fg_at(&terminal, 75, 6), Color::DarkGray);
    }

    #[test]
    fn screen_finished_renders_summary_and_badges() {
        let files = [
            file_row_view(FileBadge::Ok, "a.wav", 100.0, Some(6)),
            file_row_view(FileBadge::Fail, "b.wav", 0.0, None),
            file_row_view(FileBadge::Dup, "c.wav", 0.0, None),
        ];
        let mut view = screen_view(files.into(), None);
        view.footer.summary = "OK 1 · FAIL 1 · DUP 1".into();
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("OK 1 · FAIL 1 · DUP 1"));
        assert!(text.contains("FAIL"));
        assert!(text.contains("DUP"));
    }

    #[test]
    fn screen_too_small_area_renders_without_panic() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 63.0, Some(1))];
        let view = screen_view(files.into(), None);
        // Ниже минимума 80×24 — рендер не паникует; сводка важнее подсказок,
        // в 40 колонок влезает только она (новое правило приоритета footer).
        let terminal = render_frame(40, 12, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("audio_split"));
        assert!(text.contains("ETA 04:12"), "сводка видна даже в узком кадре");
    }

    #[test]
    fn degenerate_areas_do_not_panic() {
        let view = screen_view(Vec::new(), None);
        // Нулевой и усечённый размеры не должны паниковать при сборке кадра.
        render_frame(0, 0, |f, area| render_screen(f, area, &view, tr()));
        render_frame(10, 2, |f, area| render_screen(f, area, &view, tr()));
    }

    // --- табы Dedup / Log / Help и confirm (seq 05) ------------------------------

    use crate::dedup::KeeperReason;
    use crate::tui::log::LogBuffer;

    /// Пара дедупликации для таба Dedup (числа из CLI-кейса unknown 2/3).
    fn dedup_pair() -> PairReport {
        PairReport {
            duplicate: "unknown 3.wav".into(),
            kept: "unknown 2.wav".into(),
            corr: [0.985, 0.999, 0.991],
            gain_db: 11.1,
            crest_kept_db: 28.2,
            crest_dup_db: 23.0,
            kept_clipping: false,
            dup_clipping: false,
            renamed: true,
            rename_error: None,
            reason: KeeperReason::Louder { margin_db: 11.1 },
        }
    }

    /// Журнал с двумя записями.
    fn log_fixture() -> LogBuffer {
        let mut log = LogBuffer::default();
        log.push("00:00:01", "download a.wav");
        log.push("00:00:02", "separate b.wav");
        log
    }

    /// Экран с нестандартным табом/данными поверх базовой фикстуры.
    fn tab_view(tab: Tab, fill: impl FnOnce(&mut ScreenView)) -> ScreenView {
        let mut view = ScreenView { tab, ..ScreenView::default() };
        view.header = header_view();
        view.footer = FooterView { summary: String::new() };
        fill(&mut view);
        view
    }

    #[test]
    fn keeper_reason_labels_cover_all_variants() {
        assert_eq!(keeper_reason_label(&KeeperReason::SoleCandidate, tr()), "единственный кандидат");
        assert_eq!(keeper_reason_label(&KeeperReason::OnlyNonClipped, tr()), "без клипа");
        assert_eq!(
            keeper_reason_label(&KeeperReason::CleanerDynamics { margin_db: 3.25 }, tr()),
            "динамичнее на 3.2 дБ"
        );
        assert_eq!(keeper_reason_label(&KeeperReason::Louder { margin_db: 11.1 }, tr()), "громче на 11.1 дБ");
        assert_eq!(
            keeper_reason_label(&KeeperReason::AllClipped { margin_db: 0.5 }, tr()),
            "макс. crest при клипе (+0.5 дБ)"
        );
    }

    #[test]
    fn corr_cells_format_three_windows() {
        assert_eq!(corr_cells(&[0.985, 0.999, 0.991]), "0.99/1.00/0.99");
    }

    #[test]
    fn dedup_tab_renders_pairs_table() {
        let view = tab_view(Tab::Dedup, |v| {
            v.dedup_pairs = vec![dedup_pair()];
            v.dedup_unparseable = vec!["broken.wav".into()];
        });
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);

        assert!(text.contains("ДУБЛ"), "заголовок");
        assert!(text.contains("ХРАН"));
        assert!(text.contains("CORR"));
        assert!(text.contains("ПРИЧИНА"));
        assert!(text.contains("unknown 3.wav"), "файл-дубль");
        assert!(text.contains("unknown 2.wav"), "хранитель");
        assert!(text.contains("0.99/1.00/0.99"), "corr по окнам");
        assert!(text.contains("+11.1"), "gain со знаком");
        assert!(text.contains("28.2/23.0"), "crest пары");
        assert!(text.contains("✓"), "переименован");
        assert!(text.contains("громче на 11.1 дБ"), "решение с причиной");
        assert!(text.contains("не разобраны как WAV: broken.wav"), "примечание");
    }

    #[test]
    fn dedup_empty_state_shows_hint() {
        let view = tab_view(Tab::Dedup, |_| {});
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        assert!(buffer_text(&terminal).contains("дублей не найдено"));
    }

    #[test]
    fn dedup_clip_flags_text_and_rename_error() {
        let mut pair = dedup_pair();
        pair.kept_clipping = true;
        pair.dup_clipping = true;
        pair.rename_error = Some("EACCES".into());
        let view = tab_view(Tab::Dedup, |v| v.dedup_pairs = vec![pair]);

        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);

        assert!(text.contains("KD"), "клипуют оба (текст, не только цвет)");
        assert!(text.contains("!"), "ошибка rename видна текстом");
    }

    #[test]
    fn log_tab_renders_entries_with_timestamps() {
        let view = tab_view(Tab::Log, |v| v.log = log_fixture());
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);

        assert!(text.contains("Журнал"), "заголовок");
        assert!(text.contains("00:00:01"), "timestamp");
        assert!(text.contains("download a.wav"));
        assert!(text.contains("separate b.wav"));
        assert!(text.contains("f — фильтр"), "подсказки");
    }

    #[test]
    fn log_empty_state_shows_hint() {
        let view = tab_view(Tab::Log, |_| {});
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        assert!(buffer_text(&terminal).contains("журнал пуст"));
    }

    #[test]
    fn log_filtered_to_nothing_shows_hint() {
        let view = tab_view(Tab::Log, |v| {
            v.log = log_fixture();
            v.log.set_filter(Some("нет-таких".into()));
        });
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("нет записей по фильтру"));
        assert!(text.contains("фильтр: \"нет-таких\""));
    }

    #[test]
    fn log_filter_input_renders_cursor_in_text() {
        let view = tab_view(Tab::Log, |v| v.log_filter_input = Some(("do".into(), 1)));
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);
        assert!(text.contains("фильтр:"), "строка ввода");
        assert!(text.contains("d▌o"), "курсор внутри текста");
        assert!(text.contains("Enter — применить"));
    }

    #[test]
    fn log_window_follows_tail_and_scrolls_up() {
        let mut log = LogBuffer::default();
        for i in 0..30 {
            log.push("00:00:00", format!("msg-{i:02}"));
        }
        let mut view = tab_view(Tab::Log, |v| v.log = log.clone());

        let tail = render_frame(80, 24, |f, area| render_screen(f, area, &view, tr()));
        let tail_text = buffer_text(&tail);
        assert!(tail_text.contains("msg-29"), "автоскролл: хвост виден");
        assert!(!tail_text.contains("msg-00"), "голова за окном");

        log.scroll_up(100);
        view.log = log;
        let head = render_frame(80, 24, |f, area| render_screen(f, area, &view, tr()));
        let head_text = buffer_text(&head);
        assert!(head_text.contains("msg-00"), "прокрутка вверх: голова видна");
        assert!(head_text.contains("от хвоста"), "индикатор отступа");
    }

    #[test]
    fn help_overlay_renders_all_keybind_groups() {
        let view = tab_view(Tab::Help, |_| {});
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view, tr()));
        let text = buffer_text(&terminal);

        assert!(text.contains("Справка"));
        assert!(text.contains("[1–4]/Tab"), "табы");
        assert!(text.contains("эта справка"), "?");
        assert!(text.contains("настройки прогона"), "Enter/s");
        assert!(text.contains("пауза"), "p");
        assert!(text.contains("повтор"), "r");
        assert!(text.contains("фильтр журнала"), "f");
        assert!(text.contains("очистить журнал"), "c");
        assert!(text.contains("PgUp/PgDn"), "прокрутка");
        assert!(text.contains("подтверждение отмены"), "q/Esc в Running");
        assert!(text.contains("зарезервировано"), "m/o");
        assert!(text.contains("язык интерфейса"), "l — переключатель языка");
        assert!(text.contains("K/D — клипует"), "легенда CLIP");
    }

    #[test]
    fn confirm_renders_warning_and_both_actions() {
        let terminal = render_frame(100, 30, |f, area| render_confirm(f, area, tr()));
        let text = buffer_text(&terminal);

        assert!(text.contains("Подтверждение"));
        assert!(text.contains("Прервать прогон и выйти?"));
        assert!(text.contains("границе файлов"), "предупреждение");
        assert!(text.contains("доигран раннером"));
        assert!(text.contains("Esc/q — остаться"));
        assert!(text.contains("мягко: доиграть текущий файл"));
        assert!(text.contains("форс"), "форс-ветка описана");
        assert!(text.contains("без rename"), "resume-оговорка");
    }

    #[test]
    fn confirm_and_help_survive_degenerate_areas() {
        render_frame(0, 0, |f, area| render_confirm(f, area, tr()));
        render_frame(2, 2, |f, area| render_confirm(f, area, tr()));
        render_frame(2, 2, |f, area| render_help(f, area, tr()));
    }
}
