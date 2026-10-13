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

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table, TableState};
use ratatui::Frame;

use crate::dedup::{KeeperReason, PairReport};
use crate::pipeline::EXPECTED_STEM_COUNT;
use crate::tui::log::{LogBuffer, LogEntry};

// --- глифы и геометрия ----------------------------------------------------------

/// Уровни громкости ASCII-огибающей: 0 — пусто, 7 — полный блок.
const WAVE_BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// Уровней громкости на строку огибающей (3 строки × 8 блоков = 24).
const WAVE_ROWS: usize = 3;
const BAR_FILLED: char = '█';
const BAR_EMPTY: char = '░';
const PROGRESS_BAR_WIDTH: usize = 12;
const GPU_BAR_WIDTH: usize = 6;
/// Ниже этой ширины DETAILS-колонка скрывается (compact-режим).
const FULL_LAYOUT_MIN_WIDTH: u16 = 100;
/// Ширина DETAILS-колонки: подпись waveform (40 символов) + рамка + запас.
const DETAILS_WIDTH: u16 = 44;
const TAB_COUNT: usize = 4;
/// Подсказки клавиш footer (кейбинды утверждены владельцем, ТЗ).
pub const KEYHINTS: &str = "[s]tart [p]ause [r]etry [m]odel [o]pen report [?]help [q]uit";

const STYLE_DIM: Style = Style::new().fg(Color::DarkGray);
const STYLE_FILL: Style = Style::new().fg(Color::Yellow);
const STYLE_EMPTY: Style = Style::new().fg(Color::DarkGray);
const STYLE_HEADER: Style = Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD);

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
    /// Текст фиксированной ширины 4 — колонка таблицы не «пляшет».
    pub fn label(self) -> &'static str {
        match self {
            Self::Run => "RUN",
            Self::Ok => "OK",
            Self::Fail => "FAIL",
            Self::Skip => "SKIP",
            Self::Dup => "DUP",
        }
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
    /// Утилизация, %.
    pub util_pct: f64,
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
    /// Прогресс separate, 0..=100.
    pub pct: f64,
    /// Готовых стемов (из [`EXPECTED_STEM_COUNT`]).
    pub stems_done: u8,
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

/// ASCII-огибающая громкости в Details (решение владельца 2026-10-13):
/// 3 строки блоков `▁▂▃▄▅▆▇█`, заливка по прогрессу, подпись с временем.
#[derive(Debug, Clone, Default)]
pub struct WaveformView {
    /// Прогресс separate, 0..=100.
    pub pct: f64,
    /// Длительность трека, сек (`None` — метки времени скрыты).
    pub duration_secs: Option<f64>,
    /// Пики по колонкам 0.0..=1.0 (seq 03, `wav_envelope`); пусто → fallback-бар.
    pub peaks: Vec<f32>,
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
    /// Подпись таба с номером клавиши.
    pub fn label(self) -> &'static str {
        match self {
            Self::Files => "[1]Files",
            Self::Dedup => "[2]Dedup",
            Self::Log => "[3]Log",
            Self::Help => "[4]Help",
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
}

/// Прямоугольники зон кадра (рендер и мышь используют одни значения).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenZones {
    /// Шапка.
    pub header: Rect,
    /// Полоса табов.
    pub tabs: Rect,
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

/// Три строки огибающей: колонки ресемплятся из `peaks` (max по окну при
/// сжатии, растяжение при нехватке); пустой пик — пробел.
pub fn waveform_rows(peaks: &[f32], columns: usize) -> [String; WAVE_ROWS] {
    let mut rows: [String; WAVE_ROWS] = Default::default();
    for col in 0..columns {
        let level = peak_level(column_peak(peaks, col, columns));
        for (r, row) in rows.iter_mut().enumerate() {
            row.push(row_char(level, r));
        }
    }
    rows
}

/// Уровень колонки 0..=24 (0 — тишина, 24 — максимум).
fn peak_level(peak: f32) -> usize {
    let peak = peak.clamp(0.0, 1.0);
    if peak <= 0.0 {
        0
    } else {
        ((peak * 24.0).ceil() as usize).clamp(1, 24)
    }
}

/// Символ строки `row` (0 — верхняя) для колонки уровня `level`:
/// бар растёт снизу, строка показывает свой диапазон из 8 блоков.
fn row_char(level: usize, row: usize) -> char {
    let row_top = 24 - 8 * row;
    let row_bottom = row_top - 7;
    if level < row_bottom {
        ' '
    } else {
        WAVE_BLOCKS[(level - row_bottom).min(7)]
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
pub fn render_header(frame: &mut Frame, area: Rect, view: &HeaderView) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" audio_split ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let line = Line::from(header_spans(view));
    frame.render_widget(Paragraph::new(line), inner);
}

/// Спаны шапки: подключение, директория, модель, GPU-слот.
fn header_spans(view: &HeaderView) -> Vec<Span<'static>> {
    let mut spans = vec![
        Span::styled("audio_split", STYLE_HEADER),
        Span::raw(" ─ "),
        Span::styled(format!("//{}/{}", view.host, view.share), Style::new().fg(Color::Cyan)),
        Span::raw(" ─ "),
        Span::raw(view.remote_dir.clone()),
        Span::raw(" ─ "),
        Span::styled(view.model.clone(), Style::new().add_modifier(Modifier::BOLD)),
        Span::raw(" ─ "),
    ];
    spans.extend(gpu_spans(view.gpu.as_ref()));
    spans
}

/// GPU-слот: `None` → `GPU: n/a` (graceful без поллера), иначе бар + VRAM.
fn gpu_spans(gpu: Option<&GpuView>) -> Vec<Span<'static>> {
    let Some(gpu) = gpu else {
        return vec![Span::styled("GPU: n/a", STYLE_DIM)];
    };
    let pct = if gpu.total_gib > 0.0 {
        f64::from(gpu.used_gib) / f64::from(gpu.total_gib) * 100.0
    } else {
        0.0
    };
    let (filled, empty) = progress_bar_parts(pct, GPU_BAR_WIDTH);
    vec![
        Span::styled("GPU ", STYLE_DIM),
        Span::styled(filled, STYLE_FILL),
        Span::styled(empty, STYLE_EMPTY),
        Span::styled(format!(" {:.1}/{:.1}GiB", gpu.used_gib, gpu.total_gib), STYLE_DIM),
    ]
}

/// Полоса табов: 4 равные зоны, активный подсвечен (зоны = [`tab_at`]).
pub fn render_tab_bar(frame: &mut Frame, area: Rect, selected: Tab) {
    let chunks = Layout::horizontal([Constraint::Fill(1); TAB_COUNT]).split(area);
    for (tab, chunk) in TABS.iter().zip(chunks.iter()) {
        let style = if *tab == selected {
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            STYLE_DIM
        };
        let line = Line::from(tab.label()).style(style).alignment(Alignment::Center);
        frame.render_widget(Paragraph::new(line), *chunk);
    }
}

/// Таблица файлов; пустой список → empty-state подсказка.
pub fn render_files(frame: &mut Frame, area: Rect, files: &[FileRowView], selected: Option<usize>) {
    if files.is_empty() {
        render_files_empty(frame, area);
        return;
    }
    let block = Block::bordered().border_type(BorderType::Rounded).title(" FILES ");
    let header = Row::new(["ST", "STAGE", "FILE", "PROGRESS", "STEMS"]).style(STYLE_HEADER);
    let rows: Vec<Row> = files.iter().map(file_row).collect();
    let widths = [
        Constraint::Length(5),
        Constraint::Length(9),
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
fn render_files_empty(frame: &mut Frame, area: Rect) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" FILES ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let hint = Line::from("нет файлов — Esc → настройка")
        .style(STYLE_DIM)
        .alignment(Alignment::Center);
    frame.render_widget(Paragraph::new(hint), inner);
}

/// Строка таблицы: бейдж / стадия / имя / прогресс / стемы (моно-цифры).
fn file_row(view: &FileRowView) -> Row<'static> {
    let badge = Cell::from(Span::styled(view.badge.label(), view.badge.style()));
    let stage = Cell::from(Span::styled(view.stage.clone(), STYLE_DIM));
    let name = Cell::from(view.name.clone());
    let (filled, empty) = progress_bar_parts(view.pct, PROGRESS_BAR_WIDTH);
    let progress = Cell::from(Line::from(vec![
        Span::styled(filled, STYLE_FILL),
        Span::styled(empty, STYLE_EMPTY),
        Span::raw(format!(" {:>3.0}%", view.pct)),
    ]));
    let stems = Cell::from(format!("{}/{}", view.stems_done, EXPECTED_STEM_COUNT));
    Row::new([badge, stage, name, progress, stems])
}

/// Панель Details выбранного файла; `None` → пустое состояние.
pub fn render_details_or_empty(frame: &mut Frame, area: Rect, view: Option<&DetailsView>) {
    match view {
        Some(details) => render_details(frame, area, details),
        None => render_details_empty(frame, area),
    }
}

/// Details: заголовок, стадия/ETA, источник, формат, стемы, события + waveform.
pub fn render_details(frame: &mut Frame, area: Rect, view: &DetailsView) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" DETAILS ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    // Waveform — фикс. блок снизу (подпись + 3 строки + метки), текст — сверху.
    let chunks = Layout::vertical([Constraint::Min(1), Constraint::Length(5)]).split(inner);
    frame.render_widget(Paragraph::new(details_lines(view, chunks[0].height as usize)), chunks[0]);
    render_waveform(frame, chunks[1], &view.waveform);
}

/// Строки текстовой части Details (без waveform).
fn details_lines(view: &DetailsView, max_lines: usize) -> Vec<Line<'static>> {
    let eta = match view.eta_secs {
        Some(secs) => format!(" · ETA {}", format_mmss(secs)),
        None => String::new(),
    };
    let mut lines = vec![
        Line::from(view.name.clone()).style(STYLE_HEADER),
        Line::from(format!("стадия: {}{eta}", view.stage)),
        Line::from(format!("источник: {}", view.source)),
        Line::from(format!("формат: {}", view.format)),
        stems_line(&view.stems),
        Line::from("события:").style(STYLE_DIM),
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
fn render_details_empty(frame: &mut Frame, area: Rect) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" DETAILS ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let hint = Line::from("нет выбранного файла").style(STYLE_DIM).alignment(Alignment::Center);
    frame.render_widget(Paragraph::new(hint), inner);
}

/// Waveform-прогресс: подпись, 3 строки огибающей (или fallback-бар), метки.
pub fn render_waveform(frame: &mut Frame, area: Rect, view: &WaveformView) {
    if area.is_empty() {
        return;
    }
    frame.render_widget(Paragraph::new(waveform_lines(view, area.width as usize)), area);
}

/// Строки waveform-виджета: подпись с %/временем, бар-строки, метки 00:00.
fn waveform_lines(view: &WaveformView, width: usize) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(waveform_label(view)).style(Style::new().fg(Color::Cyan))];
    if view.peaks.is_empty() {
        // Fallback: без огибающей — обычный прогресс-бар.
        let (filled, empty) = progress_bar_parts(view.pct, width);
        lines.push(Line::from(vec![
            Span::styled(filled, STYLE_FILL),
            Span::styled(empty, STYLE_EMPTY),
        ]));
    } else {
        let fill = fill_columns(view.pct, width);
        for row in waveform_rows(&view.peaks, width) {
            lines.push(split_fill_line(row, fill));
        }
    }
    if let Some(duration) = view.duration_secs {
        lines.push(time_labels(duration, width));
    }
    lines
}

/// Подпись «NN% · обработано MM:SS · осталось MM:SS» (или просто `NN%`).
fn waveform_label(view: &WaveformView) -> String {
    match view.duration_secs {
        Some(duration) => {
            let (done, left) = split_progress(duration, view.pct);
            format!("{:.0}% · обработано {} · осталось {}", view.pct, format_mmss(done), format_mmss(left))
        }
        None => format!("{:.0}%", view.pct),
    }
}

/// Строка огибающей: заполненные колонки ярко, остаток dim.
fn split_fill_line(row: String, fill: usize) -> Line<'static> {
    let chars: Vec<char> = row.chars().collect();
    let bright: String = chars.iter().take(fill).collect();
    let dim: String = chars.iter().skip(fill).collect();
    Line::from(vec![Span::styled(bright, STYLE_FILL), Span::styled(dim, STYLE_EMPTY)])
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
pub fn render_footer(frame: &mut Frame, area: Rect, view: &FooterView) {
    if area.is_empty() {
        return;
    }
    let mut spans = vec![Span::styled(KEYHINTS, Style::new().fg(Color::Gray))];
    if !view.summary.is_empty() {
        let gap = (area.width as usize).saturating_sub(KEYHINTS.len() + view.summary.len());
        // Сводка прижата вправо; при нехватке ширины — минимум один пробел.
        spans.push(Span::raw(" ".repeat(gap.max(1))));
        spans.push(Span::styled(view.summary.clone(), Style::new().fg(Color::Cyan)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// --- табы Dedup / Log / Help и confirm-диалог ------------------------------------

/// Человекочитаемая причина выбора хранителя. Единственный источник текста:
/// используют таблица Dedup и журнал (mod.rs).
pub fn keeper_reason_label(reason: &KeeperReason) -> String {
    match reason {
        KeeperReason::SoleCandidate => "единственный кандидат".to_string(),
        KeeperReason::OnlyNonClipped => "без клипа".to_string(),
        KeeperReason::CleanerDynamics { margin_db } => format!("динамичнее на {margin_db:.1} дБ"),
        KeeperReason::Louder { margin_db } => format!("громче на {margin_db:.1} дБ"),
        KeeperReason::AllClipped { margin_db } => {
            format!("макс. crest при клипе (+{margin_db:.1} дБ)")
        }
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
pub fn render_dedup(frame: &mut Frame, area: Rect, pairs: &[PairReport], unparseable: &[String]) {
    let (content, note) = split_dedup_note(area, unparseable);
    if pairs.is_empty() {
        render_dedup_empty(frame, content);
    } else {
        render_dedup_table(frame, content, pairs);
    }
    if let (Some(note_area), false) = (note, unparseable.is_empty()) {
        let text = format!("! не разобраны как WAV: {}", unparseable.join(", "));
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

fn render_dedup_empty(frame: &mut Frame, area: Rect) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" Dedup ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    frame.render_widget(Paragraph::new("дублей не найдено").style(STYLE_DIM), inner);
}

fn render_dedup_table(frame: &mut Frame, area: Rect, pairs: &[PairReport]) {
    let header = Row::new(["DUP", "KEPT", "CORR", "GAIN дБ", "CREST дБ", "CLIP", "REN", "REASON"])
        .style(STYLE_HEADER);
    let rows: Vec<Row> = pairs.iter().map(dedup_row).collect();
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

fn dedup_row(pair: &PairReport) -> Row<'static> {
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
        Cell::from(Span::styled(keeper_reason_label(&pair.reason), STYLE_DIM)),
    ])
}

/// Журнал (таб Log): хвост буфера (автоскролл при scroll == 0), прокрутка
/// вверх, строка статуса/фильтра внизу. Пустой буфер — заглушка.
pub fn render_log(
    frame: &mut Frame,
    area: Rect,
    log: &LogBuffer,
    filter_input: Option<&(String, usize)>,
) {
    let block = Block::bordered().border_type(BorderType::Rounded).title(" Log ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.is_empty() {
        return;
    }
    let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner);
    render_log_lines(frame, rows[0], log);
    render_log_status(frame, rows[1], log, filter_input);
}

/// Окно журнала: `scroll` строк от хвоста вверх; 0 — последние строки.
fn render_log_lines(frame: &mut Frame, area: Rect, log: &LogBuffer) {
    if log.is_empty() {
        frame.render_widget(
            Paragraph::new("журнал пуст — события прогона появятся здесь").style(STYLE_DIM),
            area,
        );
        return;
    }
    let visible = log.visible();
    if visible.is_empty() {
        frame.render_widget(Paragraph::new("нет записей по фильтру").style(STYLE_DIM), area);
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
) {
    let line = match filter_input {
        Some((text, cursor)) => filter_input_line(text, *cursor),
        None => match log.filter() {
            Some(filter) => Line::from(vec![
                Span::raw(format!(
                    "фильтр: \"{filter}\" — {} из {} · ",
                    log.visible_len(),
                    log.len()
                )),
                Span::styled("f — изменить · c — очистить", STYLE_DIM),
            ]),
            None => log_hints_line(log),
        },
    };
    frame.render_widget(Paragraph::new(line), area);
}

/// Ввод фильтра: курсор `▌` вставлен по позиции (unicode-safe).
fn filter_input_line(text: &str, cursor: usize) -> Line<'static> {
    let mut value: String = text.chars().take(cursor).collect();
    value.push('▌');
    value.extend(text.chars().skip(cursor));
    Line::from(vec![
        Span::raw("фильтр: "),
        Span::styled(value, STYLE_FILL),
        Span::styled("  Enter — применить · Esc — отмена", STYLE_DIM),
    ])
}

fn log_hints_line(log: &LogBuffer) -> Line<'static> {
    let mut spans = vec![Span::styled(
        "f — фильтр · c — очистить · PgUp/PgDn — прокрутка".to_string(),
        STYLE_DIM,
    )];
    if log.scroll() > 0 {
        spans.push(Span::styled(format!("  ↑{} от хвоста", log.scroll()), STYLE_FILL));
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
pub fn render_help(frame: &mut Frame, area: Rect) {
    let popup = centered(area, 86, 88);
    let block = Block::bordered().border_type(BorderType::Rounded).title(" Справка ");
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if inner.is_empty() {
        return;
    }
    let lines = vec![
        help_line("[1–4]/Tab", "табы: Files, Dedup, Log, Help"),
        help_line("?", "эта справка (повторно — назад)"),
        help_line("Enter/s", "настройки прогона (Idle)"),
        help_line("Enter", "новый цикл (Finished)"),
        help_line("p", "пауза между файлами (Running)"),
        help_line("r", "повтор выбранного упавшего файла"),
        help_line("↑/↓, j/k", "выбор файла в таблице"),
        help_line("f", "фильтр журнала по подстроке (Log)"),
        help_line("c", "очистить журнал (Log)"),
        help_line("PgUp/PgDn", "прокрутка журнала (Log)"),
        help_line("Enter", "подтвердить в диалогах и попапе"),
        help_line("Esc", "закрыть попап/справку/диалог"),
        help_line("q/Esc", "выход; в Running — подтверждение отмены"),
        help_line("m/o", "модель/отчёт — зарезервировано"),
        Line::from(Span::styled(
            " CLIP: K/D — клипует хранитель/дубль · REN: ✓ — переименован",
            STYLE_DIM,
        )),
    ];
    frame.render_widget(Paragraph::new(lines), inner);
}

/// Модальный диалог выхода во время прогона (поверх всего кадра).
pub fn render_confirm(frame: &mut Frame, area: Rect) {
    let popup = centered(area, 70, 34);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(STYLE_FILL)
        .title(" Подтверждение ");
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if inner.is_empty() {
        return;
    }
    let lines = vec![
        Line::from("Прервать прогон и выйти?"),
        Line::from(String::new()),
        Line::from(Span::styled("Отмена применяется на границе файлов:", STYLE_FILL)),
        Line::from(Span::styled("текущий файл будет доигран раннером.", STYLE_FILL)),
        Line::from(String::new()),
        Line::from(vec![
            Span::styled("Enter", STYLE_FILL),
            Span::raw(" — прервать и выйти · "),
            Span::styled("Esc", STYLE_FILL),
            Span::raw(" — остаться"),
        ]),
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

/// Полный кадр: шапка → табы → контент активного таба → footer.
pub fn render_screen(frame: &mut Frame, area: Rect, view: &ScreenView) {
    let zones = compute_zones(area);
    render_header(frame, zones.header, &view.header);
    render_tab_bar(frame, zones.tabs, view.tab);
    match view.tab {
        Tab::Files => render_files_tab(frame, zones, view),
        Tab::Dedup => {
            render_dedup(frame, zones.content, &view.dedup_pairs, &view.dedup_unparseable);
        }
        Tab::Log => render_log(frame, zones.content, &view.log, view.log_filter_input.as_ref()),
        // Фон — таблица файлов, справка поверх.
        Tab::Help => {
            render_files_tab(frame, zones, view);
            render_help(frame, zones.content);
        }
    }
    render_footer(frame, zones.footer, &view.footer);
}

/// Классический вид таба Files: таблица + DETAILS в полном режиме.
fn render_files_tab(frame: &mut Frame, zones: ScreenZones, view: &ScreenView) {
    let (files_area, details_area) = content_split(zones.content, view.details.is_some());
    render_files(frame, files_area, &view.files, view.selected);
    render_details_or_empty(frame, details_area, view.details.as_ref());
}

/// Зоны кадра: шапка 3, табы 1, контент Min(1), footer 1.
pub fn compute_zones(area: Rect) -> ScreenZones {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(area);
    ScreenZones {
        header: rows[0],
        tabs: rows[1],
        content: rows[2],
        footer: rows[3],
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
    use ratatui::Terminal;
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
    fn waveform_rows_column_count_matches() {
        let rows = waveform_rows(&[0.5, 0.8], 7);
        assert!(rows.iter().all(|row| row.chars().count() == 7));
    }

    #[test]
    fn waveform_level_mapping_half_peak() {
        // 0.5 → уровень 12 из 24: верх пуст, середина ▄, низ █.
        let rows = waveform_rows(&[0.5], 1);
        let cell = |r: usize| rows[r].chars().next().unwrap();
        assert_eq!((cell(0), cell(1), cell(2)), (' ', '▄', '█'));
    }

    #[test]
    fn waveform_silence_is_blank_and_max_is_full_block() {
        let silence = waveform_rows(&[0.0], 1);
        assert!(silence.iter().all(|row| row.starts_with(' ')));
        let max = waveform_rows(&[1.0], 1);
        assert!(max.iter().all(|row| row.starts_with('█')));
    }

    #[test]
    fn waveform_quiet_peak_bottom_block() {
        let rows = waveform_rows(&[0.01], 1);
        assert_eq!(rows[2].chars().next(), Some('▁'));
        assert_eq!(rows[0].chars().next(), Some(' '));
    }

    #[test]
    fn waveform_resample_takes_window_max() {
        let rows = waveform_rows(&[0.1, 0.9, 0.2, 0.3], 2);
        let col = |r: usize, c: usize| rows[r].chars().nth(c).unwrap();
        // Колонка 0 — max(0.1, 0.9) = 0.9: уровень 22 → верх ▆.
        assert_eq!(col(0, 0), '▆');
        assert_eq!(col(1, 0), '█');
        // Колонка 1 — max(0.2, 0.3) = 0.3: уровень 8 → только низ █.
        assert_eq!(col(1, 1), ' ');
        assert_eq!(col(2, 1), '█');
    }

    #[test]
    fn waveform_short_input_no_panic() {
        let rows = waveform_rows(&[0.5], 50);
        assert!(rows.iter().all(|row| row.chars().count() == 50));
    }

    #[test]
    fn badge_labels_are_fixed_text() {
        assert_eq!(FileBadge::Run.label(), "RUN");
        assert_eq!(FileBadge::Ok.label(), "OK");
        assert_eq!(FileBadge::Fail.label(), "FAIL");
        assert_eq!(FileBadge::Skip.label(), "SKIP");
        assert_eq!(FileBadge::Dup.label(), "DUP");
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
        let zones = compute_zones(Rect::new(0, 0, 80, 24));
        assert_eq!(zones.header, Rect::new(0, 0, 80, 3));
        assert_eq!(zones.tabs, Rect::new(0, 3, 80, 1));
        assert_eq!(zones.content, Rect::new(0, 4, 80, 19));
        assert_eq!(zones.footer, Rect::new(0, 23, 80, 1));
    }

    // --- golden-рендеры ---

    fn header_view() -> HeaderView {
        HeaderView {
            host: "nas.local".into(),
            share: "media".into(),
            remote_dir: "/recordings".into(),
            model: "sw".into(),
            gpu: None,
        }
    }

    fn file_row_view(badge: FileBadge, name: &str, pct: f64, stems: u8) -> FileRowView {
        FileRowView {
            badge,
            name: name.into(),
            stage: "separate".into(),
            pct,
            stems_done: stems,
        }
    }

    /// Синтетическая огибающая: все колонки одинаковые — детерминированный golden.
    fn flat_peaks(columns: usize, peak: f32) -> Vec<f32> {
        vec![peak; columns]
    }

    #[test]
    fn header_renders_connection_model_and_gpu_placeholder() {
        let terminal = render_frame(80, 3, |f, area| render_header(f, area, &header_view()));
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
            util_pct: 63.0,
        });
        let terminal = render_frame(80, 3, |f, area| render_header(f, area, &view));
        let text = buffer_text(&terminal);
        // 2.1/6.0 = 35% → 2 из 6 полуколонок заполнены.
        assert!(text.contains("GPU ██░░░░"));
        assert!(text.contains("2.1/6.0GiB"));
        assert!(!text.contains("n/a"));
    }

    #[test]
    fn table_renders_all_five_badges_and_numbers() {
        let files = [
            file_row_view(FileBadge::Run, "a.wav", 63.0, 0),
            file_row_view(FileBadge::Ok, "b.wav", 100.0, 6),
            file_row_view(FileBadge::Fail, "c.wav", 0.0, 0),
            file_row_view(FileBadge::Skip, "d.wav", 0.0, 0),
            file_row_view(FileBadge::Dup, "e.wav", 0.0, 0),
        ];
        let terminal = render_frame(80, 10, |f, area| render_files(f, area, &files, None));
        let text = buffer_text(&terminal);
        assert!(text.contains("RUN"));
        assert!(text.contains("OK"));
        assert!(text.contains("FAIL"));
        assert!(text.contains("SKIP"));
        assert!(text.contains("DUP"));
        assert!(text.contains(" 63%"));
        assert!(text.contains("6/6"));
        assert!(text.contains("████████████"));
    }

    #[test]
    fn table_selected_row_is_marked_with_arrow() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 10.0, 0)];
        let terminal = render_frame(60, 8, |f, area| render_files(f, area, &files, Some(0)));
        assert!(buffer_text(&terminal).contains("▸"));
        let unselected = render_frame(60, 8, |f, area| render_files(f, area, &files, None));
        assert!(!buffer_text(&unselected).contains("▸"));
    }

    #[test]
    fn empty_table_shows_hint() {
        let terminal = render_frame(60, 8, |f, area| render_files(f, area, &[], None));
        let text = buffer_text(&terminal);
        assert!(text.contains("нет файлов — Esc → настройка"));
        assert!(!text.contains("RUN"));
    }

    #[test]
    fn waveform_at_zero_percent_all_dim() {
        let view = WaveformView { pct: 0.0, duration_secs: Some(120.0), peaks: flat_peaks(80, 0.75) };
        let terminal = render_frame(80, 5, |f, area| render_waveform(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("0% · обработано 00:00 · осталось 02:00"));
        // Ни одной яркой колонки во второй строке огибающей (y=1).
        assert_ne!(fg_at(&terminal, 0, 1), Color::Yellow);
        assert_ne!(fg_at(&terminal, 79, 1), Color::Yellow);
    }

    #[test]
    fn waveform_at_fifty_percent_boundary_on_column() {
        let view = WaveformView { pct: 50.0, duration_secs: Some(120.0), peaks: flat_peaks(80, 0.75) };
        let terminal = render_frame(80, 5, |f, area| render_waveform(f, area, &view));
        assert!(buffer_text(&terminal).contains("50% · обработано 01:00 · осталось 01:00"));
        // Граница заливки ровно на полуколонке: 40 заполнено, 41-я — dim.
        assert_eq!(fg_at(&terminal, 39, 1), Color::Yellow);
        assert_eq!(fg_at(&terminal, 40, 1), Color::DarkGray);
    }

    #[test]
    fn waveform_at_hundred_percent_all_bright() {
        let view = WaveformView { pct: 100.0, duration_secs: Some(120.0), peaks: flat_peaks(80, 0.75) };
        let terminal = render_frame(80, 5, |f, area| render_waveform(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("100% · обработано 02:00 · осталось 00:00"));
        assert_eq!(fg_at(&terminal, 0, 1), Color::Yellow);
        assert_eq!(fg_at(&terminal, 79, 1), Color::Yellow);
        assert!(text.contains("00:00"));
        assert!(text.contains("02:00"));
    }

    #[test]
    fn waveform_falls_back_to_plain_bar_without_peaks() {
        let view = WaveformView { pct: 40.0, duration_secs: Some(120.0), peaks: Vec::new() };
        let terminal = render_frame(40, 5, |f, area| render_waveform(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains('█'));
        assert!(text.contains('░'));
        assert!(text.contains("40% · обработано 00:48 · осталось 01:12"));
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
        let terminal = render_frame(50, 14, |f, area| render_details(f, area, &view));
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
        let terminal = render_frame(50, 10, |f, area| render_details_or_empty(f, area, None));
        assert!(buffer_text(&terminal).contains("нет выбранного файла"));
    }

    #[test]
    fn footer_renders_keyhints_and_summary() {
        let view = FooterView { summary: "OK 3 · FAIL 1 · DUP 2".into() };
        // 100 колонок: подсказки (60) + сводка (21) влезают с зазором.
        let terminal = render_frame(100, 1, |f, area| render_footer(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("[s]tart"));
        assert!(text.contains("[p]ause"));
        assert!(text.contains("[r]etry"));
        assert!(text.contains("[m]odel"));
        assert!(text.contains("[o]pen report"));
        assert!(text.contains("[?]help"));
        assert!(text.contains("[q]uit"));
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
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("audio_split"));
        assert!(text.contains("[1]Files"));
        assert!(text.contains("[4]Help"));
        assert!(text.contains("нет файлов — Esc → настройка"));
        assert!(text.contains("[q]uit"));
        assert!(text.contains("ETA 04:12"));
    }

    #[test]
    fn screen_running_compact_hides_details_panel() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 63.0, 2)];
        let view = screen_view(files.into(), None);
        let terminal = render_frame(80, 24, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("FILES"));
        assert!(text.contains("RUN"));
        // Compact-режим: DETAILS-колонка скрыта на 80 колонках.
        assert!(!text.contains("DETAILS"));
        assert!(!text.contains("нет выбранного файла"));
    }

    #[test]
    fn screen_running_full_shows_details_with_waveform() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 50.0, 3)];
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
                peaks: flat_peaks(34, 0.6),
            },
        };
        let view = screen_view(files.into(), Some(details));
        let terminal = render_frame(110, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("DETAILS"));
        assert!(text.contains("50% · обработано 01:00 · осталось 01:00"));
        assert!(text.contains("стадия: separate · ETA 01:00"));
    }

    #[test]
    fn screen_finished_renders_summary_and_badges() {
        let files = [
            file_row_view(FileBadge::Ok, "a.wav", 100.0, 6),
            file_row_view(FileBadge::Fail, "b.wav", 0.0, 0),
            file_row_view(FileBadge::Dup, "c.wav", 0.0, 0),
        ];
        let mut view = screen_view(files.into(), None);
        view.footer.summary = "OK 1 · FAIL 1 · DUP 1".into();
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("OK 1 · FAIL 1 · DUP 1"));
        assert!(text.contains("FAIL"));
        assert!(text.contains("DUP"));
    }

    #[test]
    fn screen_too_small_area_renders_without_panic() {
        let files = [file_row_view(FileBadge::Run, "a.wav", 63.0, 1)];
        let view = screen_view(files.into(), None);
        // Ниже минимума 80×24 — рендер не паникует; footer обрезается по ширине,
        // поэтому проверяем подсказку, попадающую в первые 40 колонок.
        let terminal = render_frame(40, 12, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("audio_split"));
        assert!(text.contains("[s]tart"));
    }

    #[test]
    fn degenerate_areas_do_not_panic() {
        let view = screen_view(Vec::new(), None);
        // Нулевой и усечённый размеры не должны паниковать при сборке кадра.
        render_frame(0, 0, |f, area| render_screen(f, area, &view));
        render_frame(10, 2, |f, area| render_screen(f, area, &view));
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
        assert_eq!(keeper_reason_label(&KeeperReason::SoleCandidate), "единственный кандидат");
        assert_eq!(keeper_reason_label(&KeeperReason::OnlyNonClipped), "без клипа");
        assert_eq!(
            keeper_reason_label(&KeeperReason::CleanerDynamics { margin_db: 3.25 }),
            "динамичнее на 3.2 дБ"
        );
        assert_eq!(keeper_reason_label(&KeeperReason::Louder { margin_db: 11.1 }), "громче на 11.1 дБ");
        assert_eq!(
            keeper_reason_label(&KeeperReason::AllClipped { margin_db: 0.5 }),
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
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);

        assert!(text.contains("DUP"), "заголовок");
        assert!(text.contains("KEPT"));
        assert!(text.contains("CORR"));
        assert!(text.contains("REASON"));
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
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        assert!(buffer_text(&terminal).contains("дублей не найдено"));
    }

    #[test]
    fn dedup_clip_flags_text_and_rename_error() {
        let mut pair = dedup_pair();
        pair.kept_clipping = true;
        pair.dup_clipping = true;
        pair.rename_error = Some("EACCES".into());
        let view = tab_view(Tab::Dedup, |v| v.dedup_pairs = vec![pair]);

        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);

        assert!(text.contains("KD"), "клипуют оба (текст, не только цвет)");
        assert!(text.contains("!"), "ошибка rename видна текстом");
    }

    #[test]
    fn log_tab_renders_entries_with_timestamps() {
        let view = tab_view(Tab::Log, |v| v.log = log_fixture());
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);

        assert!(text.contains("Log"), "заголовок");
        assert!(text.contains("00:00:01"), "timestamp");
        assert!(text.contains("download a.wav"));
        assert!(text.contains("separate b.wav"));
        assert!(text.contains("f — фильтр"), "подсказки");
    }

    #[test]
    fn log_empty_state_shows_hint() {
        let view = tab_view(Tab::Log, |_| {});
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        assert!(buffer_text(&terminal).contains("журнал пуст"));
    }

    #[test]
    fn log_filtered_to_nothing_shows_hint() {
        let view = tab_view(Tab::Log, |v| {
            v.log = log_fixture();
            v.log.set_filter(Some("нет-таких".into()));
        });
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
        let text = buffer_text(&terminal);
        assert!(text.contains("нет записей по фильтру"));
        assert!(text.contains("фильтр: \"нет-таких\""));
    }

    #[test]
    fn log_filter_input_renders_cursor_in_text() {
        let view = tab_view(Tab::Log, |v| v.log_filter_input = Some(("do".into(), 1)));
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
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

        let tail = render_frame(80, 24, |f, area| render_screen(f, area, &view));
        let tail_text = buffer_text(&tail);
        assert!(tail_text.contains("msg-29"), "автоскролл: хвост виден");
        assert!(!tail_text.contains("msg-00"), "голова за окном");

        log.scroll_up(100);
        view.log = log;
        let head = render_frame(80, 24, |f, area| render_screen(f, area, &view));
        let head_text = buffer_text(&head);
        assert!(head_text.contains("msg-00"), "прокрутка вверх: голова видна");
        assert!(head_text.contains("от хвоста"), "индикатор отступа");
    }

    #[test]
    fn help_overlay_renders_all_keybind_groups() {
        let view = tab_view(Tab::Help, |_| {});
        let terminal = render_frame(100, 30, |f, area| render_screen(f, area, &view));
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
        assert!(text.contains("K/D — клипует"), "легенда CLIP");
    }

    #[test]
    fn confirm_renders_warning_and_both_actions() {
        let terminal = render_frame(100, 30, render_confirm);
        let text = buffer_text(&terminal);

        assert!(text.contains("Подтверждение"));
        assert!(text.contains("Прервать прогон и выйти?"));
        assert!(text.contains("границе файлов"), "предупреждение");
        assert!(text.contains("доигран раннером"));
        assert!(text.contains("Esc — остаться"));
        assert!(text.contains("Enter — прервать и выйти"));
    }

    #[test]
    fn confirm_and_help_survive_degenerate_areas() {
        render_frame(0, 0, render_confirm);
        render_frame(2, 2, render_confirm);
        render_frame(2, 2, render_help);
    }
}
