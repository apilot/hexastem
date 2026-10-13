//! События TUI: `AppEvent` для event loop + поток-читатель crossterm.

use crossterm::event::{read, Event, KeyEvent, KeyEventKind};
use tokio::sync::mpsc::UnboundedSender;

/// Событие, обрабатываемое event loop'ом приложения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppEvent {
    /// Нажатие клавиши (только `KeyEventKind::Press`, см. `From<Event>`).
    Key(KeyEvent),
    /// Изменение размера терминала.
    Resize { width: u16, height: u16 },
    /// Регулярный тик (перерисовка, будущие поллеры).
    Tick,
    /// Прочие события crossterm (mouse/paste/focus) — в скелете отбрасываются.
    Ignored,
}

impl From<Event> for AppEvent {
    fn from(event: Event) -> Self {
        match event {
            // Windows шлёт Release/Repeat; реагируем только на Press.
            Event::Key(key) if key.kind == KeyEventKind::Press => AppEvent::Key(key),
            Event::Resize(width, height) => AppEvent::Resize { width, height },
            _ => AppEvent::Ignored,
        }
    }
}

/// Запускает поток-читатель: блокирующий `Event::read` → `AppEvent` в канал.
///
/// crossterm 0.29 не даёт асинхронного чтения без feature `event-stream`,
/// поэтому каноничный паттерн — выделенный поток + mpsc; event loop
/// забирает события через `tokio::select`. Поток умирает вместе с
/// процессом: `main` вернулся → `process::exit` гасит все потоки.
pub fn spawn_input_thread(sender: UnboundedSender<AppEvent>) {
    std::thread::spawn(move || {
        while let Ok(event) = read() {
            let event = AppEvent::from(event);
            // Ignored не шлём: каждое сообщение будит select и вызывает перерисовку.
            if event != AppEvent::Ignored && sender.send(event).is_err() {
                break; // приёмник закрыт — event loop закончился
            }
        }
        // Err от read — источник событий умер: завершаем поток.
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn press_key_event_maps_to_key() {
        let key = KeyEvent::new(KeyCode::Enter, KeyModifiers::empty());
        let mapped = AppEvent::from(Event::Key(key));
        assert_eq!(mapped, AppEvent::Key(key));
    }

    #[test]
    fn release_key_event_is_filtered_out() {
        let key = KeyEvent::new_with_kind(
            KeyCode::Char('x'),
            KeyModifiers::empty(),
            KeyEventKind::Release,
        );
        assert_eq!(AppEvent::from(Event::Key(key)), AppEvent::Ignored);
    }

    #[test]
    fn resize_event_carries_dimensions() {
        let mapped = AppEvent::from(Event::Resize(100, 30));
        assert_eq!(mapped, AppEvent::Resize { width: 100, height: 30 });
    }

    #[test]
    fn mouse_event_is_ignored() {
        let mouse = crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::Down(
                crossterm::event::MouseButton::Left,
            ),
            column: 1,
            row: 1,
            modifiers: KeyModifiers::empty(),
        };
        assert_eq!(AppEvent::from(Event::Mouse(mouse)), AppEvent::Ignored);
    }
}
