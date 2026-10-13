//! События TUI: `AppEvent` для event loop + поток-читатель crossterm.
//!
//! Помимо ввода crossterm канал несёт события раннера пайплайна
//! (см. `runner.rs`): смена стадии/прогресс по файлу, файл done/failed/dup,
//! батч-итог и готовая огибающая. Payload-типы ([`Stage`], [`PlanSummary`],
//! [`BatchOutcome`]) — UI-модель: чистые данные без рантайм-состояния.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{read, Event, KeyEvent, KeyEventKind};
use tokio::sync::mpsc::UnboundedSender;

use crate::dedup::DedupSummary;
use crate::pipeline::{BatchResult, FileResult};

/// Стадия файла в UI-модели. Переходы детерминированы пайплайном:
/// Queued → Download → Separate → Upload → Rename → (терминальный статус).
/// Отдельной стадии scan нет — план приходит одним событием [`AppEvent::RunStarted`];
/// cleanup сливается с успешным rename (файл получает `Ok`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Stage {
    /// В плане, обработка ещё не начата.
    #[default]
    Queued,
    /// Скачивание с шары.
    Download,
    /// Разделение движком (процент — из `EngineEvent::Progress`).
    Separate,
    /// Заливка стемов (после успешного разделения).
    Upload,
    /// Переименование исходника в `CONVERTED_*`.
    Rename,
}

/// Снимок плана батча для засева таблицы файлов (по именам на шаре).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PlanSummary {
    /// Файлы к обработке.
    pub pending: Vec<String>,
    /// Пропущены: уже обработаны ранее (`CONVERTED_*`).
    pub skipped_converted: Vec<String>,
    /// Пропущены: ранее признаны дублями (`DUPLICATE_*`).
    pub skipped_duplicate: Vec<String>,
    /// Пропущены: resume — все стемы уже на NAS.
    pub skipped_done: Vec<String>,
}

/// Итог батча для UI: снимок [`BatchResult`] без рантайм-полей (`Instant`),
/// поэтому `PartialEq`/`Clone` — пригоден для событий и состояния App.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchOutcome {
    /// Всего обработано (включая снятые дубли).
    pub total: usize,
    /// Успешно.
    pub ok: usize,
    /// С ошибкой.
    pub failed: usize,
    /// Пофайловые результаты (авторитетная сверка статусов UI).
    pub results: Vec<FileResult>,
    /// Итог дедупликации (пар дубль → хранитель) — для таба Dedup.
    pub dedup: DedupSummary,
    /// Длительность прогона.
    pub elapsed: Duration,
}

impl From<&BatchResult> for BatchOutcome {
    fn from(result: &BatchResult) -> Self {
        Self {
            total: result.total,
            ok: result.ok,
            failed: result.failed,
            results: result.results.clone(),
            dedup: result.dedup.clone(),
            elapsed: result.elapsed,
        }
    }
}

/// Событие, обрабатываемое event loop'ом приложения.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    /// Нажатие клавиши (только `KeyEventKind::Press`, см. `From<Event>`).
    Key(KeyEvent),
    /// Изменение размера терминала.
    Resize { width: u16, height: u16 },
    /// Регулярный тик (перерисовка, поллеры).
    Tick,
    /// Прочие события crossterm (mouse/paste/focus) — отбрасываются.
    Ignored,
    /// Runner собрал план прогона: засев таблицы файлов, Running.
    RunStarted { plan: PlanSummary },
    /// Runner не смог начать прогон (скан/сеть/отмена до старта).
    RunFailed { error: String },
    /// Смена стадии обработки файла.
    FileStage { name: String, stage: Stage },
    /// Прогресс разделения файла (pct 0..100, из `EngineEvent::Progress`).
    FileProgress { name: String, pct: f64 },
    /// Прогресс загрузки модели (глобальный, стадия `load` драйвера).
    ModelProgress { pct: f64 },
    /// Файл полностью готов: стемы залиты, исходник переименован.
    FileDone { name: String },
    /// Файл не обработан (download/engine/upload/rename — любая стадия).
    FileFailed { name: String, error: String },
    /// Файл — дубль: снят с обработки, будет обработан хранитель.
    FileDuplicate { name: String },
    /// Прогон завершён: авторитетная сверка статусов и переход в Finished.
    BatchFinished { outcome: BatchOutcome },
    /// Огибающая файла посчитана (после download, ровно 1 раз на файл);
    /// при терминальном статусе файла-адресата отбрасывается.
    Envelope { path: PathBuf, peaks: Arc<[f32]> },
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
    use crate::pipeline::FileStatus;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::path::PathBuf;

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

    /// Фикстура BatchResult (поля pub) для проверки снимка.
    fn batch_result_fixture() -> BatchResult {
        let results = vec![
            FileResult {
                name: "a.wav".to_string(),
                status: FileStatus::Ok,
                error: None,
                elapsed: Duration::from_secs(1),
            },
            FileResult {
                name: "b.wav".to_string(),
                status: FileStatus::Duplicate { kept: "a.wav".to_string() },
                error: None,
                elapsed: Duration::ZERO,
            },
        ];
        crate::pipeline::summarize(results, std::time::Instant::now())
    }

    #[test]
    fn batch_outcome_snapshots_result_without_runtime_fields() {
        let result = batch_result_fixture();

        let outcome = BatchOutcome::from(&result);

        assert_eq!(outcome.total, 2);
        assert_eq!(outcome.ok, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.results.len(), 2);
        assert_eq!(outcome.dedup, DedupSummary::default());
        assert_eq!(outcome, BatchOutcome::from(&result), "снимок детерминирован");
    }

    #[test]
    fn runner_events_are_cloneable_and_comparable() {
        let event = AppEvent::Envelope {
            path: PathBuf::from("/tmp/a.wav"),
            peaks: Arc::from(vec![0.5_f32, 0.25]),
        };
        let copy = event.clone();

        assert_eq!(event, copy);
        assert!(matches!(
            copy,
            AppEvent::Envelope { ref path, ref peaks } if path == &PathBuf::from("/tmp/a.wav") && peaks.len() == 2
        ));
    }

    #[test]
    fn plan_summary_defaults_to_empty_lists() {
        let plan = PlanSummary::default();

        assert!(plan.pending.is_empty());
        assert!(plan.skipped_converted.is_empty());
        assert!(plan.skipped_duplicate.is_empty());
        assert!(plan.skipped_done.is_empty());
    }

    #[test]
    fn stage_defaults_to_queued() {
        assert_eq!(Stage::default(), Stage::Queued);
    }
}
