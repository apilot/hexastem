//! События TUI: `AppEvent` для event loop + поток-читатель crossterm.
//!
//! Помимо ввода crossterm канал несёт события раннера пайплайна
//! (см. `runner.rs`): смена стадии/прогресс по файлу, файл done/failed/dup,
//! батч-итог и готовая огибающая. Payload-типы ([`Stage`], [`PlanSummary`],
//! [`BatchOutcome`]) — UI-модель: чистые данные без рантайм-состояния.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{read, Event, KeyEvent, KeyEventKind, MouseEvent};
use tokio::sync::mpsc::UnboundedSender;

use crate::dedup::DedupSummary;
use crate::envelope::StereoEnvelope;
use crate::pipeline::{BatchResult, FileResult};
use crate::tui::gpu::GpuEvent;

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

impl PlanSummary {
    /// План без единого файла: в директории не нашлось .wav (проверь remote-dir).
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
            && self.skipped_converted.is_empty()
            && self.skipped_duplicate.is_empty()
            && self.skipped_done.is_empty()
    }
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

/// Подсказка при пустом плане/итоге: «отработало», не сделав ничего —
/// причина почти всегда в remote-dir. Одна формулировка для Log и футера.
pub const EMPTY_PLAN_HINT: &str = "0 .wav в директории — проверь remote-dir";

/// Спецификация старта из попапа: значения полей на момент Enter.
///
/// Не путать с [`crate::tui::runner::RunnerOptions`] (параметры раннера):
/// `RunSpec` — то, что ввёл пользователь, и это же множество уходит в
/// валидацию и persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSpec {
    /// Хост SMB-сервера (в `local_mode` не используется).
    pub host: String,
    /// Имя шары (в `local_mode` не используется).
    pub share: String,
    /// Исходная директория на шаре/диске.
    pub remote_dir: PathBuf,
    /// Локальная рабочая директория.
    pub local_dir: PathBuf,
    /// Дедупликация включена.
    pub dedup: bool,
    /// Локальная папка вместо SMB: бэкенд — [`crate::local_fs::LocalFs`].
    pub local_mode: bool,
}

/// Событие, обрабатываемое event loop'ом приложения.
#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    /// Нажатие клавиши (только `KeyEventKind::Press`, см. `From<Event>`).
    Key(KeyEvent),
    /// Событие мыши (crossterm): маршрутизацией по зонам кадра занимается
    /// event loop (`mod.rs::handle_mouse`), `App::handle` его игнорирует.
    Mouse(MouseEvent),
    /// Текст вставлен из буфера (bracketed paste: drag&drop пути из ФМ).
    Paste { text: String },
    /// Изменение размера терминала.
    Resize { width: u16, height: u16 },
    /// Регулярный тик (перерисовка, поллеры).
    Tick,
    /// Прочие события crossterm (focus) — отбрасываются.
    Ignored,
    /// Снимок GPU-поллера (маппинг в `GpuView` шапки).
    Gpu(GpuEvent),
    /// Валидация полей попапа завершена (spawn_blocking): `None` — все
    /// проверки пройдены (`wav_count` — найдено .wav листингом), спека ждёт
    /// подтверждения старта повторным Enter.
    ValidationFinished { spec: RunSpec, error: Option<String>, wav_count: usize },
    /// Runner собрал план прогона: засев таблицы файлов, Running.
    RunStarted { plan: PlanSummary },
    /// Runner не смог начать прогон (скан/сеть/отмена до старта).
    RunFailed { error: String },
    /// Смена стадии обработки файла.
    FileStage { name: String, stage: Stage },
    /// Прогресс по файлу (pct 0..100): стадия `Separate` из `EngineEvent::Progress`,
    /// `Download` — от поллера размеров (фактический локальный объём).
    FileProgress { name: String, stage: Stage, pct: f64, estimated: bool },
    /// Драйвер сообщил вычислительное устройство (после загрузки модели):
    /// шапка показывает «Считает: <name>» (GPU или `cpu`).
    Device { name: String },
    /// Прогресс загрузки модели (глобальный, стадия `load` драйвера).
    ModelProgress { pct: f64 },
    /// Драйвер сообщил список стемов файла (`EngineEvent::FileDone`):
    /// живой счётчик x/6 в колонке STEMS — ещё до upload/rename.
    /// Терминальный исход не меняет.
    FileStems { name: String, count: u8 },
    /// Файл полностью готов: стемы залиты, исходник переименован.
    FileDone { name: String },
    /// Файл не обработан (download/engine/upload/rename — любая стадия).
    FileFailed { name: String, error: String },
    /// Файл — дубль: снят с обработки, будет обработан хранитель.
    FileDuplicate { name: String },
    /// Движку известны дубли текущего батча (после download, до separate):
    /// строки-файлы помечаются дублями сразу — без «мёртвой» стадии download.
    DuplicatesKnown { names: Vec<String> },
    /// Строка stderr движка (`[engine] …`): в TUI идёт только в таб Log,
    /// терминал не засоряет.
    EngineLog { line: String },
    /// Прогон завершён: авторитетная сверка статусов и переход в Finished.
    BatchFinished { outcome: BatchOutcome },
    /// Огибающая файла посчитана (после download, ровно 1 раз на файл);
    /// при терминальном статусе файла-адресата отбрасывается.
    Envelope { path: PathBuf, envelope: Arc<StereoEnvelope> },
}

impl From<Event> for AppEvent {
    fn from(event: Event) -> Self {
        match event {
            // Windows шлёт Release/Repeat; реагируем только на Press.
            Event::Key(key) if key.kind == KeyEventKind::Press => AppEvent::Key(key),
            Event::Mouse(mouse) => AppEvent::Mouse(mouse),
            Event::Paste(text) => AppEvent::Paste { text },
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
    fn paste_event_carries_text_and_is_not_dropped() {
        let mapped = AppEvent::from(Event::Paste("/mnt/nas/записи".into()));

        assert_eq!(mapped, AppEvent::Paste { text: "/mnt/nas/записи".into() });
        assert_ne!(mapped, AppEvent::Ignored, "paste обязан доходить до event loop");
    }

    #[test]
    fn run_spec_equality_covers_all_fields() {
        let spec = RunSpec {
            host: "nas.local".into(),
            share: "media".into(),
            remote_dir: PathBuf::from("/rec"),
            local_dir: PathBuf::from("/tmp/w"),
            dedup: false,
            local_mode: true,
        };

        assert_eq!(spec, spec.clone());
        assert!(matches!(&spec, RunSpec { local_mode: true, dedup: false, .. }));
    }

    #[test]
    fn mouse_event_is_forwarded_to_event_loop() {
        let mouse = crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::Up(crossterm::event::MouseButton::Left),
            column: 1,
            row: 1,
            modifiers: KeyModifiers::empty(),
        };
        // Мышь не отбрасывается: маршрутизация по зонам — в mod.rs::handle_mouse.
        assert_eq!(AppEvent::from(Event::Mouse(mouse)), AppEvent::Mouse(mouse));
    }

    #[test]
    fn focus_events_are_still_ignored() {
        assert_eq!(AppEvent::from(Event::FocusGained), AppEvent::Ignored);
        assert_eq!(AppEvent::from(Event::FocusLost), AppEvent::Ignored);
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
            envelope: Arc::new(StereoEnvelope {
                left: vec![0.5],
                right: vec![0.25],
            }),
        };
        let copy = event.clone();

        assert_eq!(event, copy);
        assert!(matches!(
            copy,
            AppEvent::Envelope { ref path, ref envelope }
                if path == &PathBuf::from("/tmp/a.wav")
                    && envelope.left.len() == 1
                    && envelope.right[0] == 0.25
        ));
    }

    #[test]
    fn device_event_carries_engine_name() {
        let event = AppEvent::Device { name: "NVIDIA GeForce RTX 4090".into() };
        let copy = event.clone();
        assert_eq!(event, copy);
        assert!(matches!(copy, AppEvent::Device { ref name } if name.contains("RTX")));
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
