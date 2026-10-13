//! TUI-бинарник `audio_split_tui`: каркас — init/restore, event loop, отрисовка.
//!
//! Скелет (subtask 01): терминал через `ratatui::init`/`restore`
//! (panic-safe: ratatui ставит panic hook, восстанавливающий терминал),
//! event loop на `tokio::select` (crossterm-события по mpsc + tick),
//! автомат `Idle → Config → Running → Finished` без бизнес-логики.

pub mod app;
pub mod event;

use std::io;
use std::time::Duration;

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc;

pub use app::{App, State};
use event::{spawn_input_thread, AppEvent};

/// Минимально поддерживаемый размер терминала.
pub const MIN_WIDTH: u16 = 80;
pub const MIN_HEIGHT: u16 = 24;
/// Период tick-событий event loop.
const TICK_MILLIS: u64 = 250;

/// Точка входа TUI: `init → event loop → restore`.
///
/// Panic-safe: `ratatui::init` ставит panic hook, восстанавливающий
/// терминал до паники; `restore` выполняется при любом обычном выходе
/// (quit, ошибка event loop).
pub async fn run() -> io::Result<()> {
    let terminal = ratatui::init();
    let result = event_loop(terminal).await;
    ratatui::restore();
    result
}

/// Основной цикл: отрисовать кадр → ждать событие (ключ/тик) → обновить App.
///
/// Выход: `app.should_quit()` после обработки события либо закрытие
/// канала ввода (поток-читатель умер).
async fn event_loop(mut terminal: DefaultTerminal) -> io::Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    spawn_input_thread(tx);
    let mut tick = tokio::time::interval(Duration::from_millis(TICK_MILLIS));
    let mut app = App::new();

    loop {
        terminal.draw(|frame| draw(frame, &app))?;
        tokio::select! {
            maybe_event = rx.recv() => match maybe_event {
                Some(event) => app.handle(&event),
                None => break, // читатель закрыл канал — завершаемся чисто
            },
            _ = tick.tick() => app.handle(&AppEvent::Tick),
        }
        if app.should_quit() {
            break;
        }
    }
    Ok(())
}

/// Выбор кадра: слишком маленький терминал → заглушка, иначе основной UI.
fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if is_too_small(area) {
        draw_too_small(frame, area);
    } else {
        draw_main(frame, app, area);
    }
}

fn is_too_small(area: Rect) -> bool {
    area.width < MIN_WIDTH || area.height < MIN_HEIGHT
}

/// Основной кадр: рамка, текущий экран автомата, подсказки клавиш.
fn draw_main(frame: &mut Frame, app: &App, area: Rect) {
    let text = vec![
        Line::from(format!("Экран: {:?}", app.state())),
        Line::from("Enter: настроить · q: выход"),
    ];
    let block = Block::bordered().title("audio_split_tui");
    frame.render_widget(Paragraph::new(text).block(block), area);
}

/// Заглушка при недостаточном размере (нужно `MIN_WIDTH`×`MIN_HEIGHT`).
fn draw_too_small(frame: &mut Frame, area: Rect) {
    let message = vec![
        Line::from("Слишком маленький терминал"),
        Line::from(format!(
            "нужно {MIN_WIDTH}×{MIN_HEIGHT}, сейчас {}×{}",
            area.width, area.height
        )),
    ];
    let paragraph = Paragraph::new(message)
        .style(Style::new().yellow())
        .alignment(ratatui::layout::Alignment::Center);
    frame.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Рендер одного кадра в TestBackend, возвращает весь текст буфера.
    fn render(width: u16, height: u16, app: &App) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn testbackend_smoke_renders_idle_frame() {
        let text = render(100, 30, &App::new());
        assert!(text.contains("audio_split_tui"));
        assert!(text.contains("Idle"));
    }

    #[test]
    fn min_size_still_renders_main_ui() {
        let text = render(MIN_WIDTH, MIN_HEIGHT, &App::new());
        assert!(text.contains("audio_split_tui"));
        assert!(!text.contains("маленький"));
    }

    #[test]
    fn too_small_terminal_shows_message_without_panic() {
        let text = render(40, 12, &App::new());
        assert!(text.contains("маленький"));
        assert!(text.contains("40×12"));
    }

    #[test]
    fn running_state_is_rendered() {
        let mut app = App::new();
        app.handle(&AppEvent::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::empty(),
        )));
        app.handle(&AppEvent::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::empty(),
        )));
        let text = render(100, 30, &app);
        assert!(text.contains("Running"));
    }
}
