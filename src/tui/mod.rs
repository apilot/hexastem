//! TUI-бинарник `audio_split_tui`: event loop, интеграция раннера и GPU,
//! композиция кадра (экран + попап старта).
//!
//! Потоки данных:
//! - ввод crossterm → поток-читатель → канал `AppEvent` (клавиши/paste);
//! - раннер (`runner::spawn`) и валидация попапа (spawn_blocking) → второй
//!   канал `AppEvent`;
//! - GPU-поллер (`gpu::spawn_poller`) → канал `GpuEvent` → `AppEvent::Gpu`.
//!
//! Все три источника мультиплексируются `tokio::select!` вместе с tick.
//!
//! Жизненный цикл прогона: Enter в попапе → локальные проверки → тяжёлая
//! валидация (листинг шары через реальный бэкенд) → [`runner::spawn`] с
//! `SmbClient`/`EngineRunner` (или `LocalFs` в local mode) + GPU-поллер;
//! `BatchFinished` → State::Finished; retry'и из экрана Finished живут, пока
//! пользователь не уйдёт с него (shutdown) или не выйдет из TUI — итог серии
//! ([`runner::exit_code`]) становится кодом завершения процесса.
//!
//! Bracketed paste и захват мыши: `Enable*` при старте, `Disable*` — при
//! любом выходе, включая панику (panic hook цепочкой поверх хука ratatui).

pub mod app;
pub mod event;
pub mod gpu;
pub mod log;
pub mod persist;
pub mod popup;
pub mod runner;
pub mod widgets;

use std::io::{self};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::ExecutableCommand;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::widgets::Paragraph;
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::task::JoinHandle;

use crate::config::Model;
use crate::local_fs::LocalFs;
use crate::pipeline::{EngineOps, EngineRunner, SmbOps};
use crate::smb::SmbClient;
use crate::tui::event::{spawn_input_thread, AppEvent, BatchOutcome, RunSpec, Stage};
use crate::tui::gpu::GpuEvent;
use crate::tui::persist::TuiSettings;
use crate::tui::runner::{RunnerHandle, RunnerOptions};
use crate::tui::widgets::{CurrentFileView, FileBadge, FileRowView, FooterView, HeaderView, ScreenView};

use app::{App, FileOutcome, FileView, State};

/// Минимально поддерживаемый размер терминала.
pub const MIN_WIDTH: u16 = 80;
pub const MIN_HEIGHT: u16 = 24;
/// Период tick-событий event loop.
const TICK_MILLIS: u64 = 250;
/// Формат стемов и расширение для resume-плана (как у CLI по умолчанию).
const OUTPUT_FORMAT: &str = "flac";
/// Колонки waveform-огибающей (сжимается под ширину Details при рендере).
const ENVELOPE_COLUMNS: usize = 48;
/// Кооперативная пауза при включённой паузе (применяется на границе файлов).
const PAUSE_WHEN_PAUSED: Duration = Duration::from_secs(30);

/// Точка входа TUI: `init → event loop → restore`, возвращает код выхода
/// последнего прогона (`runner::exit_code`; 0 — прогонов не было).
///
/// Panic-safe: `ratatui::init` ставит panic hook, восстанавливающий
/// терминал; поверх него цепочкой ставится hook, гасящий bracketed paste.
pub async fn run() -> io::Result<i32> {
    let terminal = ratatui::init();
    wrap_panic_hook_with_paste_restore();
    let paste_enabled = enable_bracketed_paste().is_ok();
    let mouse_enabled = enable_mouse_capture().is_ok();
    let result = event_loop(terminal).await;
    ratatui::restore();
    if paste_enabled {
        let _ = disable_bracketed_paste();
    }
    if mouse_enabled {
        let _ = disable_mouse_capture();
    }
    result
}

/// Основной цикл: отрисовать кадр → ждать событие (ввод/внутренние/GPU/tick)
/// → обновить App → синхронизировать флаги управления с раннером.
///
/// Выход: quit по клавише либо закрытие канала ввода (читатель умер).
async fn event_loop(mut terminal: DefaultTerminal) -> io::Result<i32> {
    let (tx_input, mut rx_input) = mpsc::unbounded_channel::<AppEvent>();
    spawn_input_thread(tx_input);
    let (tx_events, mut rx_events) = mpsc::unbounded_channel::<AppEvent>();
    let (tx_gpu, mut rx_gpu) = mpsc::unbounded_channel::<GpuEvent>();
    let mut tick = tokio::time::interval(Duration::from_millis(TICK_MILLIS));
    let mut app = App::new();
    let mut run: Option<ActiveRun> = None;
    let mut popup_open = false;
    let mut quit = false;

    while !quit {
        terminal.draw(|frame| draw(frame, &app))?;
        tokio::select! {
            maybe_event = rx_input.recv() => match maybe_event {
                // Мышь маршрутизируется по зонам кадра: App геометрии не знает.
                Some(AppEvent::Mouse(mouse)) => {
                    handle_mouse(&mut app, mouse, terminal.size()?.into())
                }
                Some(event) => app.handle(&event),
                None => quit = true, // читатель умер — завершаемся чисто
            },
            maybe_event = rx_events.recv() => if let Some(event) = maybe_event {
                on_internal_event(&mut app, event, &tx_events, &tx_gpu, &mut run);
            },
            maybe_gpu = rx_gpu.recv() => if let Some(event) = maybe_gpu {
                app.handle(&AppEvent::Gpu(event));
            },
            _ = tick.tick() => app.handle(&AppEvent::Tick),
        }
        // Попап открыт только что перешедшему в Config: преффилл из пресета,
        // плюс путь из paste вне попапа (Remote dir + фокус на нём).
        if app.state() == State::Config && !popup_open {
            popup_open = true;
            app.open_config(persist::load());
            if let Some(dir) = app.take_paste_prefill() {
                app.prefill_remote_dir(&dir);
            }
        }
        if app.state() != State::Config {
            popup_open = false;
        }
        // Уход с экрана Finished (в Idle/Config) гасит серию retry'ев
        // прежнего прогона; в Running run — сам активный прогон.
        if matches!(app.state(), State::Idle | State::Config) {
            if let Some(active) = run.take() {
                discard_run(active);
            }
        }
        sync_run_controls(&mut app, run.as_ref());
        // Старт-страж: запрос старта учитывается только вне прогона
        // (App::on_key уже не пускает 's'/Enter в Running — двойная защита).
        if app.state() == State::Config {
            if let Some(spec) = app.take_start_request() {
                spawn_validation(spec, tx_events.clone());
            }
        }
        quit |= app.should_quit();
    }
    Ok(finish_run(run.take()).await)
}

/// Активный прогон: ручка раннера + хэндл GPU-поллера.
struct ActiveRun {
    handle: RunnerHandle,
    gpu: JoinHandle<()>,
}

/// Обработка внутреннего события: в App, затем — запуск прогона на
/// успешной валидации (App уже перешёл в Running).
fn on_internal_event(
    app: &mut App,
    event: AppEvent,
    tx_events: &UnboundedSender<AppEvent>,
    tx_gpu: &UnboundedSender<GpuEvent>,
    run: &mut Option<ActiveRun>,
) {
    apply_event(app, &event);
    let AppEvent::ValidationFinished { spec, error: None } = &event else {
        return;
    };
    if app.state() != State::Running {
        return; // попап закрыли, пока шла проверка — старт отменён
    }
    let _ = persist::save(&settings_of_spec(spec));
    *run = Some(start_run(spec.clone(), tx_events.clone(), tx_gpu.clone()));
}

/// Единая точка применения события (путь внутреннего канала event loop'а):
/// обновление App + журнал. Публично для интеграционных тестов — мок-прогон
/// гонит события через ту же пару вызовов, что и живой цикл.
pub fn apply_event(app: &mut App, event: &AppEvent) {
    app.handle(event);
    log_event(app, event);
}

/// Единая точка журналирования: события пайплайна (стадии/ошибки/итог/дедуп)
/// дописываются в кольцевой буфер таба Log. GPU-сэмплы и покадровый прогресс
/// не журналируем (шум).
fn log_event(app: &mut App, event: &AppEvent) {
    let time = log::format_hhmmss(log::now_unix_secs());
    for message in event_log_messages(event) {
        app.log_mut().push(time.clone(), message);
    }
}

/// Сообщения журнала для события (пусто — событие не журналируется).
fn event_log_messages(event: &AppEvent) -> Vec<String> {
    match event {
        AppEvent::RunStarted { plan } => vec![format!(
            "прогон начат: {} файлов к обработке",
            plan.pending.len()
        )],
        AppEvent::RunFailed { error } => {
            vec![format!("ОШИБКА: прогон не состоялся: {error}")]
        }
        AppEvent::ValidationFinished { error: Some(error), .. } => {
            vec![format!("ОШИБКА: валидация: {error}")]
        }
        AppEvent::FileStage { name, stage } => vec![format!("{name}: {}", stage_label(*stage))],
        AppEvent::FileDone { name } => vec![format!("{name}: готово")],
        AppEvent::FileFailed { name, error } => vec![format!("ОШИБКА: {name}: {error}")],
        AppEvent::FileDuplicate { name } => vec![format!("{name}: дубль (снят с обработки)")],
        AppEvent::DuplicatesKnown { names } => {
            vec![format!("дубли распознаны до разделения: {}", names.join(", "))]
        }
        // Строки `[engine]` (прогресс/сводка драйвера) — как есть в таб Log.
        AppEvent::EngineLog { line } => vec![line.clone()],
        AppEvent::BatchFinished { outcome } => batch_log_messages(outcome),
        _ => Vec::new(),
    }
}

/// Итог батча + решения дедупа с причинами (текст причин — widgets::keeper_reason_label).
fn batch_log_messages(outcome: &BatchOutcome) -> Vec<String> {
    let mut messages = vec![format!(
        "прогон завершён: {} ок, {} ошибок, {} дублей за {}",
        outcome.ok,
        outcome.failed,
        outcome.dedup.pairs.len(),
        widgets::format_mmss(outcome.elapsed.as_secs_f64())
    )];
    for pair in &outcome.dedup.pairs {
        messages.push(format!(
            "дубль {} → хранитель {}: {}",
            pair.duplicate,
            pair.kept,
            widgets::keeper_reason_label(&pair.reason)
        ));
    }
    if !outcome.dedup.unparseable.is_empty() {
        messages.push(format!(
            "! не разобраны как WAV: {}",
            outcome.dedup.unparseable.join(", ")
        ));
    }
    messages
}

/// Управление активным прогоном по флагам App (cancel/pause/retry).
fn sync_run_controls(app: &mut App, run: Option<&ActiveRun>) {
    let Some(active) = run else {
        return;
    };
    if app.cancel_requested() {
        active.handle.cancel();
    }
    // Форс-отмена: убить subprocess движка немедленно (текущий файл без
    // rename — resume); раннер завершится ошибкой, finish не зависнет.
    if app.take_force_kill() {
        active.handle.force_kill();
    }
    active.handle.set_pause(app.paused().then_some(PAUSE_WHEN_PAUSED));
    for name in app.take_retry_requests() {
        active.handle.request_retry(name);
    }
}

/// Запустить серию: бэкенды по спеке (LocalFs / SmbClient + EngineRunner),
/// раннер и GPU-поллер. Прежний прогон к этому моменту погашен.
fn start_run(spec: RunSpec, tx_events: UnboundedSender<AppEvent>, tx_gpu: UnboundedSender<GpuEvent>) -> ActiveRun {
    let smb: Arc<dyn SmbOps + Send + Sync> = if spec.local_mode {
        Arc::new(LocalFs)
    } else {
        Arc::new(SmbClient::new(spec.host.clone(), spec.share.clone(), None))
    };
    let tx_engine_log = tx_events.clone();
    let engine: Arc<dyn EngineOps + Send + Sync> = Arc::new(
        EngineRunner::new(&venv_path(), &driver_script(), spec.model.checkpoint(), OUTPUT_FORMAT)
            // Строки stderr драйвера — в общий канал событий: только таб Log,
            // терминал TUI не засоряется (A: STDERR-ЛИК).
            .on_stderr(Box::new(move |line: &str| {
                let _ = tx_engine_log.send(AppEvent::EngineLog { line: line.to_string() });
            })),
    );
    let deps = runner::RunnerDeps { smb, engine };
    let opts = RunnerOptions {
        remote_dir: spec.remote_dir.clone(),
        local_dir: spec.local_dir.clone(),
        stem_ext: OUTPUT_FORMAT.to_string(),
        dedup: spec.dedup,
        pause_between_files: None,
        envelope_columns: ENVELOPE_COLUMNS,
    };
    let handle = runner::spawn(deps, opts, tx_events);
    let gpu = gpu::spawn_poller(tx_gpu);
    ActiveRun { handle, gpu }
}

/// Погасить серию без ожидания итога (уход с экрана Finished, новый старт).
fn discard_run(active: ActiveRun) {
    active.handle.cancel();
    active.handle.shutdown();
    active.gpu.abort();
}

/// Итог серии при выходе: shutdown → объединённый [`crate::pipeline::BatchResult`]
/// → exit-code. Ошибка (прогон не состоялся) — код 1.
async fn finish_run(run: Option<ActiveRun>) -> i32 {
    let Some(active) = run else {
        return 0;
    };
    active.handle.shutdown();
    active.gpu.abort();
    match active.handle.finish().await {
        Ok(result) => runner::exit_code(&result),
        Err(_) => 1,
    }
}

/// Тяжёлая валидация полей попапа в spawn_blocking: локальная директория
/// существует/создаётся, удалённая листится реальным бэкендом (ошибка
/// NT_STATUS возвращается текстом — видна в попапе).
fn spawn_validation(spec: RunSpec, tx: UnboundedSender<AppEvent>) {
    tokio::spawn(async move {
        let error = tokio::task::spawn_blocking({
            let spec = spec.clone();
            move || validate_spec(&spec)
        })
        .await
        .unwrap_or_else(|err| Some(format!("validation task crashed: {err}")));
        let _ = tx.send(AppEvent::ValidationFinished { spec, error });
    });
}

/// Проверки спеки: `None` — всё в порядке, `Some(текст)` — ошибка для попапа.
/// Публично для интеграционных тестов (та же валидация, что перед стартом).
pub fn validate_spec(spec: &RunSpec) -> Option<String> {
    if spec.remote_dir.as_os_str().is_empty() {
        return Some("исходная директория не указана".to_string());
    }
    if !spec.local_mode {
        if spec.host.trim().is_empty() {
            return Some("хост SMB-сервера не указан".to_string());
        }
        if spec.share.trim().is_empty() {
            return Some("имя шары не указано".to_string());
        }
    }
    if !spec.local_dir.exists() {
        if let Err(err) = std::fs::create_dir_all(&spec.local_dir) {
            return Some(format!("локальная директория: {err}"));
        }
    }
    let listing = if spec.local_mode {
        LocalFs.list(&spec.remote_dir)
    } else {
        SmbClient::new(&spec.host, &spec.share, None).list(&spec.remote_dir)
    };
    listing.err().map(|err| err.to_string())
}

/// Пресет из спеки для записи в `tui.toml`.
fn settings_of_spec(spec: &RunSpec) -> TuiSettings {
    TuiSettings {
        host: spec.host.clone(),
        share: spec.share.clone(),
        remote_dir: spec.remote_dir.to_string_lossy().into_owned(),
        local_dir: spec.local_dir.to_string_lossy().into_owned(),
        model: persist::model_label(spec.model).to_string(),
        dedup: spec.dedup,
        local_mode: spec.local_mode,
    }
}

/// venv с audio-separator (как дефолт CLI).
fn venv_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv")
}

/// Python-драйвер относительно крейта (путь фиксируется при сборке).
fn driver_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/separator_driver.py")
}

// --- bracketed paste и захват мыши ------------------------------------------------

fn enable_bracketed_paste() -> io::Result<()> {
    io::stdout().execute(EnableBracketedPaste).map(|_| ())
}

fn disable_bracketed_paste() -> io::Result<()> {
    io::stdout().execute(DisableBracketedPaste).map(|_| ())
}

fn enable_mouse_capture() -> io::Result<()> {
    io::stdout().execute(EnableMouseCapture).map(|_| ())
}

fn disable_mouse_capture() -> io::Result<()> {
    io::stdout().execute(DisableMouseCapture).map(|_| ())
}

/// Panic hook цепочкой: сначала гасим paste и захват мыши, затем прежний
/// hook (ratatui восстанавливает raw mode/альтернативный экран).
fn wrap_panic_hook_with_paste_restore() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_bracketed_paste();
        let _ = disable_mouse_capture();
        previous(info);
    }));
}

// --- маршрутизация мыши ------------------------------------------------------------

/// Клик мышью (только `Up(Left)`): таб → переключение таба, строка таблицы
/// Files → выбор файла. Модальные состояния (попап, confirm, мелкий экран)
/// клики не получают — зоны перекрыты модалом.
pub fn handle_mouse(app: &mut App, mouse: MouseEvent, area: Rect) {
    if mouse.kind != MouseEventKind::Up(MouseButton::Left)
        || is_too_small(area)
        || app.state() == State::Config
        || app.confirm_open()
    {
        return;
    }
    let zones = widgets::compute_zones(area);
    if covers(zones.tabs, mouse.row) {
        if let Some(tab) = widgets::tab_at(mouse.column, zones.tabs) {
            app.set_tab(tab);
        }
        return;
    }
    if covers(zones.content, mouse.row) && app.active_tab() == widgets::Tab::Files {
        let files_area = widgets::files_table_area(zones.content, false);
        if let Some(row) = widgets::file_row_at(mouse.row, files_area) {
            app.select_row(row);
        }
    }
}

/// Строка `y` попадает в прямоугольник по вертикали.
fn covers(rect: Rect, y: u16) -> bool {
    rect.height > 0 && y >= rect.y && y < rect.y + rect.height
}

// --- композиция кадра ------------------------------------------------------------

/// Выбор кадра: слишком маленький терминал → заглушка, иначе основной UI
/// (+ попап поверх в состоянии Config). Публично для интеграционных тестов
/// (TestBackend рендерит тот же кадр, что и живой event loop).
pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if is_too_small(area) {
        draw_too_small(frame, area);
        return;
    }
    widgets::render_screen(frame, area, &screen_view(app));
    if app.state() == State::Config {
        if let Some(popup) = app.popup_view() {
            popup::render_popup(frame, area, &popup);
        }
    }
    // Confirm выхода — поверх всего (и попапа невозможен: Running ≠ Config).
    if app.confirm_open() {
        widgets::render_confirm(frame, area);
    }
}

fn is_too_small(area: Rect) -> bool {
    area.width < MIN_WIDTH || area.height < MIN_HEIGHT
}

/// Взгляд на кадр из состояния App: шапка из конфига прогона, активный таб
/// (Files/Dedup/Log/Help), таблица, журнал, сводка в footer.
fn screen_view(app: &App) -> ScreenView {
    let (dedup_pairs, dedup_unparseable) = app
        .outcome()
        .map(|outcome| (outcome.dedup.pairs.clone(), outcome.dedup.unparseable.clone()))
        .unwrap_or_default();
    let log_filter_input = app
        .log_filter_input()
        .map(|field| (field.as_str().to_string(), field.cursor()));
    let tick = app.tick_count();
    ScreenView {
        header: header_view(app),
        tab: app.active_tab(),
        files: app.files().iter().map(|view| file_row_view(view, tick)).collect(),
        selected: (!app.files().is_empty()).then_some(app.selected_index()),
        // Details-панель — отдельный этап (таб Files, полный режим).
        details: None,
        footer: FooterView { summary: footer_summary(app) },
        dedup_pairs,
        dedup_unparseable,
        log: app.log().clone(),
        log_filter_input,
        banner: cancel_banner(app),
        current: current_file_view(app),
    }
}

/// Баннер мягкой отмены: живёт, пока движок доигрывает текущий файл.
fn cancel_banner(app: &App) -> Option<String> {
    app.cancelling().then(|| "Прерывание… текущий файл доигрывается (форс: Enter ещё раз)".to_string())
}

/// Индикатор текущего файла: первый файл без терминального исхода в Running.
fn current_file_view(app: &App) -> Option<CurrentFileView> {
    if app.state() != State::Running {
        return None;
    }
    let view = app.files().iter().find(|file| file.outcome.is_none())?;
    Some(CurrentFileView {
        name: view.name.clone(),
        stage: stage_label(view.stage).to_string(),
        pct: view.pct,
        phase: app.tick_count(),
        stage_elapsed: view.stage_started.map_or(0.0, |started| started.elapsed().as_secs_f64()),
    })
}

/// Шапка: подключение/дир/модель из открытого конфига, GPU-слот из поллера,
/// индикатор загрузки модели — глобальный.
fn header_view(app: &App) -> HeaderView {
    let model_pct = app.model_pct();
    let Some(cfg) = app.config() else {
        return HeaderView { gpu: app.gpu().cloned(), model_pct, ..HeaderView::default() };
    };
    HeaderView {
        host: cfg.host.as_str().to_string(),
        share: cfg.share.as_str().to_string(),
        remote_dir: cfg.remote_dir.as_str().to_string(),
        model: model_label(cfg.model).to_string(),
        gpu: app.gpu().cloned(),
        model_pct,
    }
}

/// Метка модели как в CLI (`sw` | `6s`).
fn model_label(model: Model) -> &'static str {
    persist::model_label(model)
}

/// Строка таблицы из модели прогона: бейдж и стадия по исходу (терминальные
/// строки без «мёртвой» стадии), таймер стадии, индeterminate-бар.
fn file_row_view(view: &FileView, tick: u64) -> FileRowView {
    let terminal = view.outcome.is_some();
    let stage = view
        .outcome
        .as_ref()
        .map_or_else(|| stage_label(view.stage), stage_label_of_outcome);
    let stage_elapsed = if terminal {
        None
    } else {
        view.stage_started.map(|started| started.elapsed().as_secs_f64())
    };
    // Стадии без процента (download/upload/rename) — анимированный бар.
    let moving = matches!(view.stage, Stage::Download | Stage::Upload | Stage::Rename);
    let indeterminate = (!terminal && view.pct.is_none() && moving).then_some(tick);
    FileRowView {
        badge: file_badge(view),
        name: view.name.clone(),
        stage: stage.to_string(),
        pct: view.pct.unwrap_or(0.0),
        stems_done: 0,
        stage_elapsed,
        indeterminate,
    }
}

/// Бейдж: терминальный исход приоритетен, иначе RUN.
fn file_badge(view: &FileView) -> FileBadge {
    match &view.outcome {
        Some(FileOutcome::Ok) => FileBadge::Ok,
        Some(FileOutcome::Failed { .. }) => FileBadge::Fail,
        Some(FileOutcome::Duplicate { .. }) => FileBadge::Dup,
        Some(FileOutcome::Skipped { .. }) => FileBadge::Skip,
        None => FileBadge::Run,
    }
}

/// Метка стадии (совпадает с терминологией пайплайна).
fn stage_label(stage: Stage) -> &'static str {
    match stage {
        Stage::Queued => "queued",
        Stage::Download => "download",
        Stage::Separate => "separate",
        Stage::Upload => "upload",
        Stage::Rename => "rename",
    }
}

/// Метка стадии по терминальному исходу: строки не «зависают» на последней
/// стадии (дубль больше не выглядит как «download 0%»).
fn stage_label_of_outcome(outcome: &FileOutcome) -> &'static str {
    match outcome {
        FileOutcome::Ok => "готово",
        FileOutcome::Failed { .. } => "ошибка",
        FileOutcome::Duplicate { .. } => "дубль",
        FileOutcome::Skipped { .. } => "скип",
    }
}

/// Сводка footer: глобальная полоса прогона — файлы done/total, счётчики
/// исходов, общий elapsed и ETA (среднее по завершённым × остаток;
/// до первого завершённого — «--»). После завершения ETA скрыт,
/// при батчевой ошибке прогона — её текст.
fn footer_summary(app: &App) -> String {
    if let Some(error) = app.run_error() {
        return format!("ошибка: {error}");
    }
    let mut counts = [0_usize; 4]; // ok, fail, dup, skip
    for file in app.files() {
        let slot = match &file.outcome {
            Some(FileOutcome::Ok) => Some(0),
            Some(FileOutcome::Failed { .. }) => Some(1),
            Some(FileOutcome::Duplicate { .. }) => Some(2),
            Some(FileOutcome::Skipped { .. }) => Some(3),
            None => None,
        };
        if let Some(slot) = slot {
            counts[slot] += 1;
        }
    }
    let [ok, fail, dup, skip] = counts;
    let done = ok + fail + dup + skip;
    let total = app.files().len();
    // Idle без истории: пустая сводка — место в footer отдано подсказкам.
    if total == 0 && app.outcome().is_none() {
        return String::new();
    }
    let mut summary = format!("файлы {done}/{total} · OK {ok} · FAIL {fail} · DUP {dup} · SKIP {skip}");
    match app.outcome() {
        Some(outcome) => {
            // Завершённый прогон: авторитетная длительность, ETA не нужен.
            summary.push_str(&format!(" · {}", widgets::format_mmss(outcome.elapsed.as_secs_f64())));
        }
        None => {
            let elapsed = app.run_started_at().map_or(0.0, |started| started.elapsed().as_secs_f64());
            let eta = widgets::eta_secs(elapsed, done, total)
                .map_or_else(|| "--".to_string(), widgets::format_mmss);
            summary.push_str(&format!(" · {} · ETA {eta}", widgets::format_mmss(elapsed)));
        }
    }
    summary
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
    use crate::tui::event::PlanSummary;
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

    fn enter() -> AppEvent {
        AppEvent::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::empty(),
        ))
    }

    /// Нажатие клавиши без модификаторов.
    fn key(code: crossterm::event::KeyCode) -> AppEvent {
        AppEvent::Key(crossterm::event::KeyEvent::new(
            code,
            crossterm::event::KeyModifiers::empty(),
        ))
    }

    /// Событие через ту же пару вызовов, что и в event loop
    /// (on_internal_event): обновить App + дописать журнал.
    fn feed(app: &mut App, event: AppEvent) {
        app.handle(&event);
        log_event(app, &event);
    }

    /// Запущенный прогон (Running, план из plan_fixture).
    fn running_app() -> App {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_fixture() });
        app
    }

    /// Итог с одной парой дедупликации (для таба Dedup и журнала).
    fn outcome_with_dedup() -> BatchOutcome {
        BatchOutcome {
            total: 2,
            ok: 1,
            failed: 0,
            results: Vec::new(),
            dedup: crate::dedup::DedupSummary {
                pairs: vec![crate::dedup::PairReport {
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
                    reason: crate::dedup::KeeperReason::Louder { margin_db: 11.1 },
                }],
                unparseable: vec!["broken.wav".into()],
            },
            elapsed: std::time::Duration::from_secs(65),
        }
    }

    fn plan_fixture() -> PlanSummary {
        PlanSummary {
            pending: ["a.wav".to_string()].into_iter().collect(),
            ..PlanSummary::default()
        }
    }

    // --- багфикс-волна: живой цикл отмены, инфографика, журнал движка ---

    #[test]
    fn mouse_clicks_work_while_cancelling() {
        let mut app = running_app();
        app.handle(&key(crossterm::event::KeyCode::Esc));
        app.handle(&key(crossterm::event::KeyCode::Enter));
        assert!(app.cancelling());

        handle_mouse(&mut app, click_up(30, 3), SCREEN);

        assert_eq!(app.active_tab(), widgets::Tab::Dedup, "клики живут в Cancelling");
    }

    #[test]
    fn cancelling_screen_shows_banner() {
        let mut app = running_app();
        app.handle(&key(crossterm::event::KeyCode::Esc));
        app.handle(&key(crossterm::event::KeyCode::Enter));

        let text = render(100, 30, &app);

        assert!(text.contains("Прерывание…"), "баннер мягкой отмены");
    }

    #[test]
    fn running_screen_shows_current_file_indicator() {
        let mut app = running_app();
        app.handle(&AppEvent::FileStage { name: "a.wav".into(), stage: Stage::Download });
        app.handle(&AppEvent::Tick);

        let text = render(100, 30, &app);

        assert!(text.contains("▶"), "спиннер текущего файла");
        assert!(text.contains("a.wav"));
        assert!(text.contains("download · "), "тикающий таймер стадии");
        assert!(text.contains("··"), "download без процента — индeterminate-бар");
    }

    #[test]
    fn footer_shows_global_progress_with_eta_placeholder() {
        let app = running_app();

        let text = render(100, 30, &app);

        assert!(text.contains("файлы 0/1"), "глобальный счётчик");
        assert!(text.contains("ETA --"), "до первого завершённого ETA нет");
    }

    #[test]
    fn footer_eta_computed_from_completed_files() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_with_three_files() });
        feed(&mut app, AppEvent::FileDone { name: "a.wav".into() });
        // run_started_at только что — elapsed ≈ 0, но ETA уже не «--».
        let text = render(100, 30, &app);
        assert!(text.contains("файлы 1/3"));
        assert!(!text.contains("ETA --"), "после первого файла ETA есть");
    }

    #[test]
    fn engine_log_lines_land_in_log_tab_and_leave_state_alone() {
        let mut app = App::new();

        apply_event(&mut app, &AppEvent::EngineLog { line: "[engine] tmp/a.wav: load 3.0%".into() });

        assert!(app.log().visible().iter().any(|entry| entry.message.contains("[engine] tmp/a.wav: load 3.0%")));
        assert_eq!(app.state(), State::Idle, "строки движка не меняют состояние");
    }

    #[test]
    fn run_started_replaces_rows_of_previous_series() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_with_three_files() });
        feed(&mut app, AppEvent::BatchFinished { outcome: outcome_with_dedup() });

        feed(&mut app, AppEvent::RunStarted { plan: plan_fixture() });

        let names: Vec<&str> = app.files().iter().map(|file| file.name.as_str()).collect();
        assert_eq!(names, ["a.wav"], "новая серия заменяет строки, а не дописывает");
    }

    #[test]
    fn duplicate_rows_are_seeded_from_prefix_dups_without_dead_stage() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_with_three_files() });

        // seeded из skipped_duplicate: см. app::run_started_seeds_files —
        // здесь проверяем рендер: DUP-строка без стадии download.
        let text = render(100, 30, &app);
        assert!(!text.contains("download  0%"), "мёртвых «download 0%» нет");
    }

    // --- рендеры ------------------------------------------------------------

    #[test]
    fn idle_screen_renders_frame_with_empty_state() {
        let text = render(100, 30, &App::new());
        assert!(text.contains("audio_split"));
        assert!(text.contains("нет файлов — Esc → настройка"));
        assert!(text.contains("[s]tart"));
    }

    #[test]
    fn min_size_still_renders_main_ui() {
        let text = render(MIN_WIDTH, MIN_HEIGHT, &App::new());
        assert!(text.contains("audio_split"));
        assert!(!text.contains("маленький"));
    }

    #[test]
    fn too_small_terminal_shows_message_without_panic() {
        let text = render(40, 12, &App::new());
        assert!(text.contains("маленький"));
        assert!(text.contains("40×12"));
    }

    #[test]
    fn config_state_renders_popup_over_screen() {
        let mut app = App::new();
        app.handle(&enter()); // Idle → Config (попап откроет event loop)
        app.open_config(persist::load());

        let text = render(100, 30, &app);
        assert!(text.contains("Запуск прогона"));
        assert!(text.contains("Host:"));
        assert!(text.contains("[1]Files"), "экран за попапом жив");
    }

    #[test]
    fn running_state_renders_table_with_run_badge() {
        let mut app = App::new();
        app.handle(&AppEvent::RunStarted { plan: plan_fixture() });
        let text = render(100, 30, &app);
        assert!(text.contains("FILES"));
        assert!(text.contains("RUN"));
        assert!(text.contains("a.wav"));
    }

    #[test]
    fn gpu_snapshot_renders_in_header_and_na_without_it() {
        let mut app = App::new();
        let text = render(100, 30, &app);
        assert!(text.contains("GPU: n/a"));

        app.handle(&AppEvent::Gpu(GpuEvent::Info(Some(crate::tui::gpu::GpuInfo {
            name: "RTX".into(),
            util_percent: Some(10),
            mem_used_mib: 1024,
            mem_total_mib: 8192,
        }))));
        let text = render(100, 30, &app);
        assert!(text.contains("1.0/8.0GiB"));
        assert!(!text.contains("n/a"));
    }

    // --- валидация спеки (реальный диск, без сети) ----------------------------

    #[test]
    fn validate_local_spec_with_existing_dirs_is_ok() {
        let base = std::env::temp_dir().join(format!("audio_split_mod_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let spec = RunSpec {
            host: String::new(),
            share: String::new(),
            remote_dir: base.clone(),
            local_dir: base.join("work"),
            model: Model::Sw,
            dedup: true,
            local_mode: true,
        };

        let error = validate_spec(&spec);

        assert_eq!(error, None);
        assert!(spec.local_dir.exists(), "локальная директория создана при проверке");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn validate_reports_missing_remote_directory() {
        let base = std::env::temp_dir().join(format!("audio_split_mod_missing_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let spec = RunSpec {
            host: String::new(),
            share: String::new(),
            remote_dir: base.join("nope"),
            local_dir: base.clone(),
            model: Model::Sw,
            dedup: true,
            local_mode: true,
        };

        let error = validate_spec(&spec);

        assert!(error.is_some_and(|message| message.contains("no such file") || message.contains("No such file")));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn validate_smb_without_host_fails_before_spawn() {
        let spec = RunSpec {
            host: String::new(),
            share: "media".into(),
            remote_dir: "/rec".into(),
            local_dir: std::env::temp_dir(),
            model: Model::Sw,
            dedup: true,
            local_mode: false,
        };

        let error = validate_spec(&spec);

        assert!(error.is_some(), "пустой хост — ошибка до запуска smbclient");
        assert!(error.unwrap().contains("хост"));
    }

    // --- жизненный цикл прогона (реальный раннер на LocalFs) --------------------

    /// Движок-заглушка: пайплайн с пустым планом его не вызывает.
    #[allow(dead_code)]
    struct IdleEngine;

    impl EngineOps for IdleEngine {
        fn separate_batch(
            &self,
            _files: &[std::path::PathBuf],
            _output_dir: &std::path::Path,
            _on_event: &mut dyn FnMut(crate::pipeline::EngineEvent),
        ) -> Result<(), crate::error::AppError> {
            Err(crate::error::AppError::Pipeline("движок не должен зваться".into()))
        }
    }

    #[tokio::test]
    async fn local_run_completes_with_zero_files_and_exit_code_zero() {
        let base = std::env::temp_dir().join(format!("audio_split_mod_run_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let (tx_events, mut rx_events) = mpsc::unbounded_channel();
        let (tx_gpu, _rx_gpu) = mpsc::unbounded_channel();
        let spec = RunSpec {
            host: String::new(),
            share: String::new(),
            remote_dir: base.clone(),
            local_dir: base.join("work"),
            model: Model::Sw,
            dedup: true,
            local_mode: true,
        };

        let active = start_run(spec, tx_events, tx_gpu);
        let event = tokio::time::timeout(Duration::from_secs(30), rx_events.recv())
            .await
            .expect("раннер завис")
            .expect("канал закрыт");
        let code = finish_run(Some(active)).await;
        let _ = std::fs::remove_dir_all(&base);

        assert!(matches!(event, AppEvent::RunStarted { ref plan } if plan.pending.is_empty()));
        assert_eq!(code, 0, "пустой прогон без отказов — код 0");
    }

    #[test]
    fn settings_and_model_labels_roundtrip_through_spec() {
        let spec = RunSpec {
            host: "nas".into(),
            share: "media".into(),
            remote_dir: "/rec".into(),
            local_dir: "/tmp/w".into(),
            model: Model::SixS,
            dedup: false,
            local_mode: true,
        };

        let settings = settings_of_spec(&spec);

        assert_eq!(settings.model, "6s");
        assert_eq!(settings.model(), Model::SixS);
        assert!(!settings.dedup);
        assert!(settings.local_mode);
        assert_eq!(settings.remote_dir, "/rec");
    }

    #[test]
    fn finish_run_without_active_run_is_zero() {
        // Синхронная часть: без прогона код 0 (обёртка в tokio для await).
        let code = tokio::runtime::Runtime::new().unwrap().block_on(finish_run(None));
        assert_eq!(code, 0);
    }

    // --- табы / журнал / confirm (seq 05) ---------------------------------------

    #[test]
    fn confirm_dialog_renders_over_running_screen() {
        let mut app = running_app();
        app.handle(&key(crossterm::event::KeyCode::Char('q')));
        assert!(app.confirm_open());

        let text = render(100, 30, &app);

        assert!(text.contains("Подтверждение"), "модал поверх экрана");
        assert!(text.contains("Прервать прогон и выйти?"));
        assert!(text.contains("границе файлов"), "предупреждение о границе отмены");
        assert!(text.contains("Esc"), "путь отказа виден текстом");
    }

    #[test]
    fn dedup_tab_renders_pairs_from_last_outcome() {
        let mut app = running_app();
        app.handle(&AppEvent::BatchFinished { outcome: outcome_with_dedup() });
        app.handle(&key(crossterm::event::KeyCode::Char('2')));

        let text = render(100, 30, &app);

        assert!(text.contains("DUP"), "заголовок таблицы");
        assert!(text.contains("unknown 3.wav"), "дубль");
        assert!(text.contains("unknown 2.wav"), "хранитель");
        assert!(text.contains("0.99/1.00/0.99"), "corr по окнам");
        assert!(text.contains("+11.1"), "gain");
        assert!(text.contains("28.2/23.0"), "crest");
        assert!(text.contains("громче на 11.1 дБ"), "причина KeeperReason");
        assert!(text.contains("не разобраны как WAV: broken.wav"), "примечание");
    }

    #[test]
    fn dedup_tab_without_outcome_shows_empty_state() {
        let mut app = App::new();
        app.handle(&key(crossterm::event::KeyCode::Char('2')));

        let text = render(100, 30, &app);

        assert!(text.contains("дублей не найдено"));
    }

    #[test]
    fn log_tab_renders_pipeline_events() {
        let mut app = running_app();
        feed(
            &mut app,
            AppEvent::FileFailed { name: "a.wav".into(), error: "engine boom".into() },
        );
        app.handle(&key(crossterm::event::KeyCode::Char('3')));

        let text = render(100, 30, &app);

        assert!(text.contains("прогон начат: 1 файлов"), "старт прогона");
        assert!(text.contains("ОШИБКА: a.wav: engine boom"), "ошибка файла");
        assert!(text.contains("f — фильтр"), "подсказки журнала");
    }

    #[test]
    fn log_tab_batch_summary_includes_dedup_reasons() {
        let mut app = running_app();
        feed(&mut app, AppEvent::BatchFinished { outcome: outcome_with_dedup() });
        app.handle(&key(crossterm::event::KeyCode::Char('3')));

        let text = render(100, 30, &app);

        assert!(text.contains("прогон завершён: 1 ок, 0 ошибок, 1 дублей"));
        assert!(text.contains("дубль unknown 3.wav → хранитель unknown 2.wav: громче на 11.1 дБ"));
    }

    #[test]
    fn log_tab_empty_state() {
        let mut app = App::new();
        app.handle(&key(crossterm::event::KeyCode::Char('3')));

        let text = render(100, 30, &app);

        assert!(text.contains("журнал пуст"));
    }

    #[test]
    fn log_tab_filter_input_renders_cursor() {
        let mut app = App::new();
        app.handle(&key(crossterm::event::KeyCode::Char('3')));
        app.handle(&key(crossterm::event::KeyCode::Char('f')));
        app.handle(&key(crossterm::event::KeyCode::Char('x')));

        let text = render(100, 30, &app);

        assert!(text.contains("фильтр:"), "строка ввода");
        assert!(text.contains("x▌"), "курсор после текста");
        assert!(text.contains("Enter — применить"));
    }

    #[test]
    fn help_tab_overlays_keybinds_over_files() {
        let mut app = App::new();
        app.handle(&key(crossterm::event::KeyCode::Char('4')));

        let text = render(100, 30, &app);

        assert!(text.contains("Справка"));
        assert!(text.contains("[1–4]/Tab"), "навигация табов");
        assert!(text.contains("PgUp/PgDn"), "прокрутка журнала");
        assert!(text.contains("зарезервировано"), "m/o помечены");
        // Рамка таблицы Files видна в левом поле попапа — фон жив под оверлеем.
        assert!(text.contains("FILES"), "фон (таблица Files) жив под оверлеем");
    }

    #[test]
    fn event_log_messages_map_pipeline_events() {
        let mut app = App::new();
        log_event(&mut app, &AppEvent::RunStarted { plan: plan_fixture() });
        log_event(
            &mut app,
            &AppEvent::FileStage { name: "a.wav".into(), stage: Stage::Separate },
        );
        log_event(&mut app, &AppEvent::FileDone { name: "a.wav".into() });
        log_event(&mut app, &AppEvent::FileDuplicate { name: "z.wav".into() });
        log_event(&mut app, &AppEvent::BatchFinished { outcome: outcome_with_dedup() });

        let messages: Vec<String> = app.log().visible().iter().map(|e| e.message.clone()).collect();
        assert_eq!(messages.len(), 4 + 3, "старт + стадия + done + dup + сводка + пара + unparseable");
        assert!(messages[0].contains("прогон начат: 1 файлов"));
        assert!(messages[1].contains("a.wav: separate"));
        assert!(messages[2].contains("a.wav: готово"));
        assert!(messages[3].contains("z.wav: дубль"));
        assert!(messages[4].starts_with("прогон завершён"));
    }

    #[test]
    fn event_log_skips_gpu_progress_and_resize() {
        let mut app = App::new();
        log_event(&mut app, &AppEvent::ModelProgress { pct: 42.0 });
        log_event(&mut app, &AppEvent::Resize { width: 100, height: 30 });
        log_event(&mut app, &AppEvent::Tick);

        assert!(app.log().is_empty(), "шум не журналируем");
    }

    // --- мышь (seq 07): клики по зонам кадра ------------------------------------

    const SCREEN: ratatui::layout::Rect = ratatui::layout::Rect::new(0, 0, 100, 30);

    /// Клик «отпущена левая кнопка» в координате кадра.
    fn click_up(column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind: MouseEventKind::Up(crossterm::event::MouseButton::Left),
            column,
            row,
            modifiers: crossterm::event::KeyModifiers::empty(),
        }
    }

    /// Клик «нажата левая кнопка» (не триггерит действия).
    fn click_down(column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind: MouseEventKind::Down(crossterm::event::MouseButton::Left),
            column,
            row,
            modifiers: crossterm::event::KeyModifiers::empty(),
        }
    }

    fn plan_with_three_files() -> PlanSummary {
        PlanSummary {
            pending: ["a.wav", "b.wav", "c.wav"]
                .map(str::to_string)
                .into_iter()
                .collect(),
            ..PlanSummary::default()
        }
    }

    #[test]
    fn mouse_click_on_tab_bar_switches_tab() {
        let mut app = running_app();

        handle_mouse(&mut app, click_up(30, 3), SCREEN); // зона Dedup (25..49)

        assert_eq!(app.active_tab(), widgets::Tab::Dedup);
    }

    #[test]
    fn mouse_click_on_file_row_selects_it() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_with_three_files() });

        // Контент с y=5 (статус-строка); строка таблицы 0 — с y=7 (рамка+заголовок).
        handle_mouse(&mut app, click_up(10, 8), SCREEN);

        assert_eq!(app.selected_index(), 1, "клик по второй строке таблицы");
        assert_eq!(app.selected_file().map(|file| file.name.as_str()), Some("b.wav"));
    }

    #[test]
    fn mouse_click_beyond_last_row_clamps_to_last_file() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_with_three_files() });

        handle_mouse(&mut app, click_up(10, 20), SCREEN); // пустая зона таблицы

        assert_eq!(app.selected_index(), 2, "курсор не выходит за последнюю строку");
    }

    #[test]
    fn mouse_press_without_release_is_ignored() {
        let mut app = running_app();

        handle_mouse(&mut app, click_down(30, 3), SCREEN);

        assert_eq!(app.active_tab(), widgets::Tab::Files, "действие — только на Up");
    }

    #[test]
    fn mouse_click_outside_zones_is_noop() {
        let mut app = running_app();

        handle_mouse(&mut app, click_up(10, 29), SCREEN); // footer

        assert_eq!(app.active_tab(), widgets::Tab::Files);
        assert_eq!(app.selected_index(), 0);
    }

    #[test]
    fn mouse_click_is_ignored_while_popup_open() {
        let mut app = App::new();
        app.handle(&enter()); // Idle → Config
        app.open_config(persist::load());

        handle_mouse(&mut app, click_up(30, 3), SCREEN);

        assert_eq!(app.active_tab(), widgets::Tab::Files, "попап модален");
    }

    #[test]
    fn mouse_click_is_ignored_while_confirm_open() {
        let mut app = running_app();
        app.handle(&key(crossterm::event::KeyCode::Char('q')));
        assert!(app.confirm_open());

        handle_mouse(&mut app, click_up(10, 7), SCREEN);

        assert_eq!(app.selected_index(), 0, "confirm модален");
    }

    #[test]
    fn mouse_click_on_too_small_screen_is_ignored() {
        let mut app = running_app();

        handle_mouse(&mut app, click_up(30, 3), ratatui::layout::Rect::new(0, 0, 40, 12));

        assert_eq!(app.active_tab(), widgets::Tab::Files);
    }
}
