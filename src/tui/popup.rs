//! Попап старта прогона: view + рендер поверх основного экрана.
//!
//! Контракт как у `widgets`: чистые функции рендера над view-структурой —
//! без знания об `App` и раннере (состояние → view маппирует `app.rs`,
//! композицию кадра делает `mod.rs`: `render_screen` + оверлей попапа).
//!
//! Вставка из буфера: crossterm-событие `Event::Paste` (bracketed paste —
//! drag&drop пути из ФМ приходит текстом) попадает в АКТИВНОЕ ТЕКСТОВОЕ
//! ПОЛЕ попапа: курсор из [`cursor_position`] показывает точку вставки,
//! окно скролла (`visible_value`) держит её в кадре. Сам приём Paste —
//! в `app.rs` (`App::on_paste`), рендер здесь отвечает за то, чтобы поле
//! ввода и курсор были видны.
//!
//! Поля: Host / Share / Remote dir / Local dir (текст с курсором),
//! Model (sw|6s), Dedup и Local mode (переключатели пробелом). Таб/Shift-Tab
//! — фокус, Enter — проверка и старт, Esc — отмена. Для сфокусированного
//! текстового поля ставится терминальный курсор; длинные значения скроллятся
//! так, чтобы курсор оставался виден.

use crossterm::event::KeyCode;

use ratatui::layout::Rect;
use ratatui::layout::Position;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};
use ratatui::Frame;

use crate::config::Model;
use crate::tui::app::{Focus, ValidationStatus};

/// Ширина колонки подписей (`Remote dir: `).
const LABEL_WIDTH: usize = 12;
/// Размер попапа (обрезается по размеру терминала).
const POPUP_WIDTH: u16 = 68;
const POPUP_HEIGHT: u16 = 13;

const STYLE_DIM: Style = Style::new().fg(Color::DarkGray);
const STYLE_HINT: Style = Style::new().fg(Color::Gray);
const STYLE_ACTIVE: Style = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);
const STYLE_INFO: Style = Style::new().fg(Color::Cyan);
const STYLE_OK: Style = Style::new().fg(Color::Green);
const STYLE_WARN: Style = Style::new().fg(Color::Yellow);

/// Значение текстового поля (снимок из [`crate::tui::app::TextField`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextFieldView {
    /// Текст поля.
    pub value: String,
    /// Курсор в символах.
    pub cursor: usize,
}

/// Вид попапа запуска: значения полей + фокус + статус валидации.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopupView {
    pub host: TextFieldView,
    pub share: TextFieldView,
    pub remote_dir: TextFieldView,
    pub local_dir: TextFieldView,
    /// Модель разделения (переключатель sw|6s).
    pub model: Model,
    /// Дедупликация включена.
    pub dedup: bool,
    /// Локальная папка вместо SMB.
    pub local_mode: bool,
    /// Активный элемент попапа.
    pub focus: Focus,
    /// Статус валидации.
    pub validation: ValidationStatus,
    /// Статус-строка распознанного SMB/GVFS-paste (`None` — нет).
    pub hint: Option<String>,
}

/// Прямоугольник попапа: по центру экрана, не больше [`POPUP_WIDTH`]×[`POPUP_HEIGHT`].
pub fn popup_area(area: Rect) -> Rect {
    let width = POPUP_WIDTH.min(area.width);
    let height = POPUP_HEIGHT.min(area.height);
    Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    }
}

/// Рендер попапа поверх экрана: очистка фона, рамка, поля, статус, курсор.
pub fn render_popup(frame: &mut Frame, area: Rect, view: &PopupView) {
    let popup = popup_area(area);
    frame.render_widget(Clear, popup);
    let block = Block::bordered().title(" Запуск прогона ");
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if inner.is_empty() {
        return;
    }
    let rows = field_rows(inner, view);
    for (row, line) in rows.iter().enumerate() {
        let y = inner.y + row as u16;
        if y >= inner.y + inner.height {
            break;
        }
        frame.render_widget(Paragraph::new(line.clone()), Rect { height: 1, ..strip_y(inner, y) });
    }
    if let Some(position) = cursor_position(inner, view) {
        frame.set_cursor_position(position);
    }
}

fn strip_y(inner: Rect, y: u16) -> Rect {
    Rect { x: inner.x, y, width: inner.width, height: 1 }
}

/// Строки попапа: поля, пустая строка (или статус SMB/GVFS-paste), статус
/// валидации, две строки подсказок.
fn field_rows(inner: Rect, view: &PopupView) -> Vec<Line<'static>> {
    let value_width = (inner.width as usize).saturating_sub(LABEL_WIDTH);
    let hint_row = view
        .hint
        .as_ref()
        .map(|text| Line::from(Span::styled(text.clone(), STYLE_INFO)))
        .unwrap_or_default();
    let mut rows = vec![
        text_row("Host:", &view.host, view.focus == Focus::Host, value_width),
        text_row("Share:", &view.share, view.focus == Focus::Share, value_width),
        text_row("Remote dir:", &view.remote_dir, view.focus == Focus::RemoteDir, value_width),
        text_row("Local dir:", &view.local_dir, view.focus == Focus::LocalDir, value_width),
        model_row(view),
        toggle_row("Dedup:", view.dedup, "вкл", "выкл", view.focus == Focus::Dedup),
        toggle_row(
            "Local mode:",
            view.local_mode,
            "локальная папка",
            "smb-шара",
            view.focus == Focus::LocalMode,
        ),
        hint_row,
        status_row(&view.validation),
        Line::from(Span::styled("Tab/Shift-Tab — фокус · Space/←→ — переключить", STYLE_HINT)),
        Line::from(Span::styled("Enter — проверить и старт · Esc — отмена", STYLE_HINT)),
    ];
    rows.truncate(inner.height as usize);
    rows
}

/// Строка текстового поля: подпись + значение (горизонтальный скролл до
/// курсора); сфокусированное поле — реверс.
fn text_row(label: &str, field: &TextFieldView, focused: bool, width: usize) -> Line<'static> {
    let (visible, _) = visible_value(field, width);
    let style = if focused {
        Style::new().add_modifier(Modifier::REVERSED)
    } else {
        Style::new()
    };
    Line::from(vec![
        Span::styled(format!("{label:<LABEL_WIDTH$}"), STYLE_DIM),
        Span::styled(visible, style),
    ])
}

/// Переключатель: `[x]/[ ]` + подпись состояния; сфокусированный — жёлтый.
fn toggle_row(label: &str, on: bool, on_text: &str, off_text: &str, focused: bool) -> Line<'static> {
    let mark = if on { "x" } else { " " };
    let (mark_style, text) = if focused {
        (STYLE_ACTIVE, if on { on_text } else { off_text })
    } else {
        (STYLE_DIM, if on { on_text } else { off_text })
    };
    Line::from(vec![
        Span::styled(format!("{label:<LABEL_WIDTH$}"), STYLE_DIM),
        Span::styled(format!("[{mark}] "), mark_style),
        Span::raw(text.to_string()),
    ])
}

/// Переключатель модели: выбранная — `[sw]`/`[6s]` жёлтым, другая — dim.
fn model_row(view: &PopupView) -> Line<'static> {
    let segment = |label: &'static str, active: bool| {
        if active {
            Span::styled(format!("[{label}]"), STYLE_ACTIVE)
        } else {
            Span::styled(format!(" {label} "), STYLE_DIM)
        }
    };
    Line::from(vec![
        Span::styled(format!("{:<LABEL_WIDTH$}", "Model:"), STYLE_DIM),
        Span::raw("  "),
        segment("sw", view.model == Model::Sw),
        Span::raw("  "),
        segment("6s", view.model == Model::SixS),
    ])
}

/// Строка статуса валидации: после успешного листинга — факт (сколько .wav
/// найдено); ноль — жёлтый варнинг (старт не блокируется, но путь под вопросом).
fn status_row(validation: &ValidationStatus) -> Line<'static> {
    match validation {
        ValidationStatus::Idle => Line::default(),
        ValidationStatus::Checking => {
            Line::from(Span::styled("проверка директорий…", STYLE_INFO))
        }
        ValidationStatus::Ok { wav_count } if *wav_count > 0 => Line::from(Span::styled(
            format!("✓ директория доступна · .wav: {wav_count}"),
            STYLE_OK,
        )),
        ValidationStatus::Ok { .. } => Line::from(Span::styled(
            "✓ директория доступна · .wav: 0 · .wav не найдены — проверь путь",
            STYLE_WARN,
        )),
        ValidationStatus::Failed(message) => {
            Line::from(Span::styled(format!("✗ {message}"), Style::new().fg(Color::Red)))
        }
    }
}

/// Видимое значение и колонка курсора: окно скроллится, курсор всегда в кадре.
fn visible_value(field: &TextFieldView, width: usize) -> (String, usize) {
    let width = width.max(1);
    let offset = field.cursor.saturating_sub(width - 1);
    let visible: String = field.value.chars().skip(offset).take(width).collect();
    let cursor_col = (field.cursor - offset).min(width - 1);
    (visible, cursor_col)
}

/// Позиция терминального курсора для сфокусированного текстового поля.
fn cursor_position(inner: Rect, view: &PopupView) -> Option<Position> {
    if !view.focus.is_text() || inner.width as usize <= LABEL_WIDTH {
        return None;
    }
    let (field, row) = match view.focus {
        Focus::Host => (&view.host, 0_u16),
        Focus::Share => (&view.share, 1),
        Focus::RemoteDir => (&view.remote_dir, 2),
        Focus::LocalDir => (&view.local_dir, 3),
        Focus::Model | Focus::Dedup | Focus::LocalMode => return None,
    };
    let value_width = inner.width as usize - LABEL_WIDTH;
    let (_, cursor_col) = visible_value(field, value_width);
    let x = inner.x + LABEL_WIDTH as u16 + cursor_col as u16;
    let y = inner.y + row;
    (x < inner.x + inner.width && y < inner.y + inner.height).then_some(Position { x, y })
}

/// Клавиша-переключатель элемента (для подсказок и тестов фокуса).
pub fn is_toggle_key(code: KeyCode) -> bool {
    matches!(code, KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    use crate::tui::app::ValidationStatus;
    use crate::tui::widgets::{self, FooterView, HeaderView, ScreenView};

    /// Рендер попапа в TestBackend: текст буфера + терминал для проверок ячеек.
    fn render_popup_frame(width: u16, height: u16, view: &PopupView) -> Terminal<TestBackend> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| render_popup(frame, frame.area(), view)).unwrap();
        terminal
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal.backend().buffer().content().iter().map(|cell| cell.symbol()).collect()
    }

    fn field(value: &str) -> TextFieldView {
        TextFieldView { value: value.into(), cursor: value.chars().count() }
    }

    /// Попап с заполненными полями (фокус — Remote dir).
    fn view() -> PopupView {
        PopupView {
            host: field("nas.local"),
            share: field("media"),
            remote_dir: field("/recordings"),
            local_dir: field("/tmp/work"),
            model: Model::Sw,
            dedup: true,
            local_mode: false,
            focus: Focus::RemoteDir,
            validation: ValidationStatus::Idle,
            hint: None,
        }
    }

    // --- геометрия ---

    #[test]
    fn popup_area_is_centered_and_capped() {
        let area = popup_area(Rect::new(0, 0, 100, 30));
        assert_eq!(area, Rect::new(16, 8, POPUP_WIDTH, POPUP_HEIGHT));

        let small = popup_area(Rect::new(0, 0, 40, 10));
        assert_eq!((small.width, small.height), (40, 10), "не больше экрана");
    }

    // --- golden-рендеры ---

    #[test]
    fn popup_renders_title_labels_and_values() {
        let terminal = render_popup_frame(100, 30, &view());
        let text = buffer_text(&terminal);
        assert!(text.contains("Запуск прогона"));
        assert!(text.contains("Host:"));
        assert!(text.contains("Share:"));
        assert!(text.contains("Remote dir:"));
        assert!(text.contains("Local dir:"));
        assert!(text.contains("nas.local"));
        assert!(text.contains("/recordings"));
        assert!(text.contains("/tmp/work"));
        assert!(text.contains("Enter — проверить и старт"));
        assert!(text.contains("Esc — отмена"));
    }

    #[test]
    fn model_row_marks_selected_segment() {
        let mut six = view();
        six.model = Model::SixS;
        let text = buffer_text(&render_popup_frame(100, 30, &six));
        assert!(text.contains("[6s]"), "выбранная модель в скобках");
        assert!(text.contains(" sw "), "невыбранная — без скобок");
        assert!(!text.contains("[sw]"));

        let text = buffer_text(&render_popup_frame(100, 30, &view()));
        assert!(text.contains("[sw]"));
        assert!(!text.contains("[6s]"));
    }

    #[test]
    fn toggles_reflect_state_with_mark_and_label() {
        let mut local = view();
        local.local_mode = true;
        local.dedup = false;
        let text = buffer_text(&render_popup_frame(100, 30, &local));
        assert!(text.contains("[x] локальная папка"));
        assert!(text.contains("[ ] выкл"), "dedup выключен");

        let text = buffer_text(&render_popup_frame(100, 30, &view()));
        assert!(text.contains("[x] вкл"));
        assert!(text.contains("[ ] smb-шара"));
    }

    #[test]
    fn status_row_shows_checking_ok_and_error() {
        let mut checking = view();
        checking.validation = ValidationStatus::Checking;
        assert!(buffer_text(&render_popup_frame(100, 30, &checking)).contains("проверка директорий…"));

        let mut failed = view();
        failed.validation = ValidationStatus::Failed("NT_STATUS_LOGON_FAILURE: logon failure".into());
        let text = buffer_text(&render_popup_frame(100, 30, &failed));
        assert!(text.contains("✗ NT_STATUS_LOGON_FAILURE"));

        let mut ok = view();
        ok.validation = ValidationStatus::Ok { wav_count: 1 };
        assert!(buffer_text(&render_popup_frame(100, 30, &ok)).contains("директория доступна"));

        // Idle — строка статуса пуста (подсказки на месте).
        let text = buffer_text(&render_popup_frame(100, 30, &view()));
        assert!(!text.contains("✗") && !text.contains("проверка"));
    }

    /// Успешная валидация показывает факт: сколько .wav нашлось в директории.
    #[test]
    fn status_row_reports_wav_count_after_validation() {
        let mut ok = view();
        ok.validation = ValidationStatus::Ok { wav_count: 47 };
        let text = buffer_text(&render_popup_frame(100, 30, &ok));
        assert!(text.contains("✓ директория доступна · .wav: 47"), "{text}");
    }

    /// Ноль .wav — жёлтый варнинг до запуска: путь, скорее всего, неверный.
    #[test]
    fn status_row_warns_when_no_wav_found() {
        let mut empty = view();
        empty.validation = ValidationStatus::Ok { wav_count: 0 };
        let text = buffer_text(&render_popup_frame(100, 30, &empty));
        assert!(text.contains(".wav: 0"), "{text}");
        assert!(text.contains(".wav не найдены — проверь путь"), "{text}");
    }

    /// Распознанный SMB/GVFS-paste виден статус-строкой (на месте пустой
    /// строки-разделителя, подсказки клавиш не уезжают).
    #[test]
    fn hint_line_reports_recognized_smb_path() {
        let mut hinted = view();
        hinted.hint = Some("Распознан SMB/GVFS путь — host/share/папка заполнены".into());
        let text = buffer_text(&render_popup_frame(100, 30, &hinted));
        assert!(text.contains("Распознан SMB/GVFS путь"), "{text}");
        assert!(text.contains("Enter — проверить и старт"), "{text}");

        let text = buffer_text(&render_popup_frame(100, 30, &view()));
        assert!(!text.contains("Распознан"), "без hint строки нет");
    }

    #[test]
    fn focused_field_is_reversed_and_unfocused_are_not() {
        let terminal = render_popup_frame(100, 30, &view());
        // Remote dir — 4-я строка контента: inner.y = 9 (попап с y=8) + 2.
        let remote_value_x = (16 + LABEL_WIDTH + 1) as u16;
        let remote_cell = terminal.backend().buffer()[(remote_value_x, 11)].clone();
        assert!(remote_cell.modifier.contains(Modifier::REVERSED), "фокус подсвечен");
        let host_cell = terminal.backend().buffer()[(remote_value_x, 9)].clone();
        assert!(!host_cell.modifier.contains(Modifier::REVERSED), "не-фокус без реверса");
    }

    #[test]
    fn terminal_cursor_sits_inside_focused_text_field() {
        use ratatui::backend::Backend;

        let mut terminal = render_popup_frame(100, 30, &view());
        let position = terminal.backend_mut().get_cursor_position().unwrap();
        // inner.x = 17, значение с колонки 17+12, курсор в конце "/recordings" (+11).
        assert_eq!(position, Position { x: 40, y: 11 });

        // Фокус на переключателе — курсор не рисуется.
        let mut toggles = view();
        toggles.focus = Focus::Dedup;
        let mut terminal = render_popup_frame(100, 30, &toggles);
        assert_eq!(terminal.backend_mut().get_cursor_position().unwrap(), Position::ORIGIN);
    }

    #[test]
    fn long_value_scrolls_so_cursor_stays_visible() {
        // Уникальные голова/хвост: повторяющаяся середина не годится для
        // проверки «начала за левым краем» (паттерн самоподобен).
        let long = format!("голова-уникальная/{}", "/very/long/".repeat(10)) + "хвост-уникальный.wav";
        let mut scrolled = view();
        scrolled.remote_dir = field(&long);
        let terminal = render_popup_frame(100, 30, &scrolled);
        let text = buffer_text(&terminal);
        // Хвост пути виден, уникальное начало — за левым краем окна.
        assert!(text.contains("хвост-уникальный.wav"));
        assert!(!text.contains("голова-уникальная"));
    }

    #[test]
    fn popup_overlays_screen_keeping_surrounding_content() {
        let screen = ScreenView {
            header: HeaderView::default(),
            tab: widgets::Tab::Files,
            files: Vec::new(),
            selected: None,
            details: None,
            footer: FooterView { summary: String::new() },
            ..Default::default()
        };
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal
            .draw(|frame| {
                widgets::render_screen(frame, frame.area(), &screen);
                render_popup(frame, frame.area(), &view());
            })
            .unwrap();
        let text = buffer_text(&terminal);
        assert!(text.contains("Host:"), "попап поверх экрана");
        assert!(text.contains("[1]Files"), "экран за пределами попапа жив");
        assert!(text.contains("[?]help"), "footer виден");
    }

    #[test]
    fn degenerate_areas_do_not_panic() {
        let small = view();
        render_popup_frame(10, 4, &small);
        render_popup_frame(0, 0, &small);
    }

    // --- чистые хелперы ---

    #[test]
    fn visible_value_keeps_cursor_inside_window() {
        let (visible, cursor) = visible_value(&field("abcdef"), 4);
        assert_eq!(visible, "def", "окно показывает хвост до курсора");
        assert_eq!(cursor, 3, "курсор в последней колонке окна");

        let (_, cursor) = visible_value(&TextFieldView { value: "abcdef".into(), cursor: 1 }, 4);
        assert_eq!(cursor, 1, "курсор в начале — окно от начала");
    }

    #[test]
    fn cursor_position_outside_text_fields_is_none() {
        let mut toggles = view();
        toggles.focus = Focus::Model;
        let area = Rect::new(0, 0, 100, 30);
        assert_eq!(cursor_position(popup_area(area), &toggles), None);
    }

    #[test]
    fn toggle_keys_cover_space_and_arrows() {
        assert!(is_toggle_key(KeyCode::Char(' ')));
        assert!(is_toggle_key(KeyCode::Left));
        assert!(is_toggle_key(KeyCode::Right));
        assert!(!is_toggle_key(KeyCode::Char('a')));
    }
}
