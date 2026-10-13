//! Состояние TUI: конечный автомат `Idle → Config → Running → Finished`.
//!
//! Скелет: переходы только по клавишам, без бизнес-логики пайплайна.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};

use crate::tui::event::AppEvent;

/// Экраны приложения.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum State {
    /// Стартовый экран.
    #[default]
    Idle,
    /// Настройка прогона (позже — попап с полями).
    Config,
    /// Идёт прогон пайплайна (позже — runner-интеграция).
    Running,
    /// Прогон завершён (позже — сводка ok/fail/dup).
    Finished,
}

/// Состояние приложения: текущий экран + флаг запроса выхода.
#[derive(Debug, Default)]
pub struct App {
    state: State,
    quit_requested: bool,
}

impl App {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn should_quit(&self) -> bool {
        self.quit_requested
    }

    /// Единственная точка изменения состояния: чистый переход по событию.
    pub fn handle(&mut self, event: &AppEvent) {
        if let AppEvent::Key(key) = *event {
            self.on_key(key);
        }
        // Resize/Tick/Ignored на автомат не влияют (перерисовка — в event loop).
    }

    fn on_key(&mut self, key: KeyEvent) {
        // Release/Repeat не переключают экраны (важно для Windows-раскладок).
        if key.kind != KeyEventKind::Press {
            return;
        }
        match self.state {
            State::Idle => match key.code {
                KeyCode::Enter => self.state = State::Config,
                KeyCode::Char('q') | KeyCode::Esc => self.quit_requested = true,
                _ => {}
            },
            State::Config => match key.code {
                // 'q' в Config занято под ввод полей — выход только через Esc.
                KeyCode::Enter => self.state = State::Running,
                KeyCode::Esc => self.state = State::Idle,
                _ => {}
            },
            State::Running => {
                // Esc имитирует завершение прогона; 'q' игнорируется (позже — quit-confirm).
                if key.code == KeyCode::Esc {
                    self.state = State::Finished;
                }
            }
            State::Finished => match key.code {
                KeyCode::Enter => self.state = State::Idle,
                KeyCode::Char('q') | KeyCode::Esc => self.quit_requested = true,
                _ => {}
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode) -> AppEvent {
        AppEvent::Key(KeyEvent::new(code, crossterm::event::KeyModifiers::empty()))
    }

    #[test]
    fn new_app_starts_idle_without_quit() {
        let app = App::new();
        assert_eq!(app.state(), State::Idle);
        assert!(!app.should_quit());
    }

    #[test]
    fn idle_enter_opens_config() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        assert_eq!(app.state(), State::Config);
    }

    #[test]
    fn idle_q_requests_quit() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Char('q')));
        assert!(app.should_quit());
        assert_eq!(app.state(), State::Idle);
    }

    #[test]
    fn idle_esc_requests_quit() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Esc));
        assert!(app.should_quit());
    }

    #[test]
    fn idle_other_keys_do_nothing() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Char('x')));
        app.handle(&press(KeyCode::Tab));
        assert_eq!(app.state(), State::Idle);
        assert!(!app.should_quit());
    }

    #[test]
    fn config_enter_starts_running() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        assert_eq!(app.state(), State::Running);
    }

    #[test]
    fn config_esc_returns_to_idle() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Esc));
        assert_eq!(app.state(), State::Idle);
    }

    #[test]
    fn config_q_does_not_quit_reserved_for_input() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Char('q')));
        assert!(!app.should_quit());
        assert_eq!(app.state(), State::Config);
    }

    #[test]
    fn running_esc_finishes() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Esc));
        assert_eq!(app.state(), State::Finished);
    }

    #[test]
    fn running_q_is_ignored_quit_confirm_comes_later() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Char('q')));
        assert!(!app.should_quit());
        assert_eq!(app.state(), State::Running);
    }

    #[test]
    fn finished_enter_starts_new_cycle() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Esc));
        app.handle(&press(KeyCode::Enter));
        assert_eq!(app.state(), State::Idle);
    }

    #[test]
    fn finished_q_requests_quit() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Esc));
        app.handle(&press(KeyCode::Char('q')));
        assert!(app.should_quit());
    }

    #[test]
    fn tick_and_resize_do_not_change_state() {
        let mut app = App::new();
        app.handle(&AppEvent::Tick);
        app.handle(&AppEvent::Resize {
            width: 100,
            height: 30,
        });
        app.handle(&AppEvent::Ignored);
        assert_eq!(app.state(), State::Idle);
    }

    #[test]
    fn non_press_key_events_are_ignored() {
        let release = AppEvent::Key(KeyEvent::new_with_kind(
            KeyCode::Enter,
            crossterm::event::KeyModifiers::empty(),
            KeyEventKind::Release,
        ));
        let mut app = App::new();
        app.handle(&release);
        assert_eq!(app.state(), State::Idle);
    }
}
