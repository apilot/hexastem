//! TUI-бинарник `hexastem_tui`: event loop, интеграция раннера и GPU,
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
//! валидация (листинг шары через реальный бэкенд, с фактом «сколько .wav»)
//! → повторный Enter подтверждает старт → [`runner::spawn`] с
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
pub mod i18n;
pub mod log;
pub mod pathmode;
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
use crate::tui::event::{
    spawn_input_thread, AppEvent, BatchOutcome, RunSpec, Stage,
};
use crate::tui::gpu::GpuEvent;
use crate::tui::i18n::{tr, Lang, StrTable};
use crate::tui::persist::TuiSettings;
use crate::tui::runner::{RunnerHandle, RunnerOptions};
use crate::tui::widgets::{
    CurrentFileView, FileBadge, FileRowView, FooterView, HeaderView, ScreenView, WaveformView,
};

use app::{App, FileOutcome, FileView, State};

/// Минимально поддерживаемый размер терминала.
pub const MIN_WIDTH: u16 = 80;
pub const MIN_HEIGHT: u16 = 24;
/// Период tick-событий event loop.
const TICK_MILLIS: u64 = 250;
/// Формат стемов и расширение для resume-плана (как у CLI по умолчанию).
const OUTPUT_FORMAT: &str = "flac";
/// Колонки waveform-огибающей: брайль-карта рисует 2 пиксельные колонки на
/// колонку терминала (160 px при ширине 80) — берём с запасом на широкие экраны.
const ENVELOPE_COLUMNS: usize = 200;
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
    // Язык интерфейса — из пресета (tui.toml); отсутствие файла = дефолт Ru.
    app.set_lang(persist::load().lang());
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
                apply_event(&mut app, &event);
            },
            maybe_gpu = rx_gpu.recv() => if let Some(event) = maybe_gpu {
                app.handle(&AppEvent::Gpu(event));
            },
            _ = tick.tick() => app.handle(&AppEvent::Tick),
        }
        // Попап открыт только что перешедшему в Config: преффилл из пресета,
        // плюс путь из paste вне попапа (SMB/GVFS → поля, иначе Remote dir).
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
        // Переключение языка (`l`): мгновенная перерисовка происходит сама
        // (кадр на каждой итерации), выбор сохраняем в пресет.
        if app.take_lang_dirty() {
            let mut settings = persist::load();
            settings.lang = app.lang().label().to_string();
            let _ = persist::save(&settings);
        }
        // Старт-страж: запрос старта учитывается только вне прогона
        // (App::on_key уже не пускает 's'/Enter в Running — двойная защита).
        if app.state() == State::Config {
            if let Some(spec) = app.take_start_request() {
                spawn_validation(spec, app.lang(), tx_events.clone());
            }
        }
        // Подтверждённый старт (повторный Enter после успешной валидации):
        // спека уже проверена — сохранить пресет и запустить прогон.
        if let Some(spec) = app.take_confirmed_start() {
            let _ = persist::save(&settings_of_spec(&spec, app.lang()));
            run = Some(start_run(spec, tx_events.clone(), tx_gpu.clone()));
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
    for message in event_log_messages(event, tr(app.lang())) {
        app.log_mut().push(time.clone(), message);
    }
}

/// Сообщения журнала для события (пусто — событие не журналируется).
/// Текст — на языке интерфейса в момент события (исторические записи
/// не переписываются задним числом при переключении `l`).
fn event_log_messages(event: &AppEvent, table: &StrTable) -> Vec<String> {
    match event {
        AppEvent::RunStarted { plan } => vec![(table.run_started)(plan.pending.len())],
        AppEvent::RunFailed { error } => vec![(table.run_failed)(error)],
        AppEvent::ValidationFinished { error: Some(error), .. } => {
            vec![format!("{} валидация: {error}", table.error_log_prefix)]
        }
        AppEvent::FileStage { name, stage } => {
            vec![format!("{name}: {}", stage_label(*stage, table))]
        }
        AppEvent::FileStems { .. } => Vec::new(),
        AppEvent::FileDone { name } => vec![(table.file_done_log)(name)],
        AppEvent::FileFailed { name, error } => {
            vec![format!("{} {name}: {error}", table.error_log_prefix)]
        }
        AppEvent::FileDuplicate { name } => vec![(table.file_duplicate_log)(name)],
        AppEvent::DuplicatesKnown { names } => {
            vec![(table.dups_known_log)(&names.join(", "))]
        }
        // Строки `[engine]` (прогресс/сводка драйвера) — как есть в таб Log.
        AppEvent::EngineLog { line } => vec![line.clone()],
        AppEvent::BatchFinished { outcome } => batch_log_messages(outcome, table),
        _ => Vec::new(),
    }
}

/// Итог батча + решения дедупа с причинами (текст причин — widgets::keeper_reason_label).
fn batch_log_messages(outcome: &BatchOutcome, table: &StrTable) -> Vec<String> {
    let elapsed = widgets::format_mmss(outcome.elapsed.as_secs_f64());
    let mut messages = vec![(table.batch_summary_log)(outcome.ok, outcome.failed, outcome.dedup.pairs.len(), &elapsed)];
    // Пустой прогон не должен выглядеть как тихий успех: явная подсказка.
    if outcome.total == 0 {
        messages.push(table.empty_plan_hint.to_string());
    }
    for pair in &outcome.dedup.pairs {
        let reason = widgets::keeper_reason_label(&pair.reason, table);
        messages.push((table.dup_pair_log)(&pair.duplicate, &pair.kept, &reason));
    }
    if !outcome.dedup.unparseable.is_empty() {
        let names = outcome.dedup.unparseable.join(", ");
        messages.push((table.unparseable_note)(&names));
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
            // Стартовая RTF-оценка модели для estimированного separate-прогресса
            // драйвера (уточняется им по бою между файлами).
            .with_rtf_estimate(spec.model.rtf_estimate())
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
/// NT_STATUS возвращается текстом — видна в попапе); при успехе считается
/// факт — сколько .wav в директории (нуль виден в попапе до старта).
fn spawn_validation(spec: RunSpec, lang: Lang, tx: UnboundedSender<AppEvent>) {
    tokio::spawn(async move {
        let outcome = tokio::task::spawn_blocking({
            let spec = spec.clone();
            move || validate_spec(&spec, lang)
        })
        .await
        .unwrap_or_else(|err| Err(format!("validation task crashed: {err}")));
        let (error, wav_count) = match outcome {
            Ok(wav_count) => (None, wav_count),
            Err(message) => (Some(message), 0),
        };
        let _ = tx.send(AppEvent::ValidationFinished { spec, error, wav_count });
    });
}

/// Проверки спеки: `Ok(N)` — всё в порядке, N — .wav найдено листингом;
/// `Err(текст)` — ошибка для попапа (на языке `lang`). Публично для
/// интеграционных тестов (та же валидация, что перед стартом).
pub fn validate_spec(spec: &RunSpec, lang: Lang) -> Result<usize, String> {
    let table = tr(lang);
    if spec.remote_dir.as_os_str().is_empty() {
        return Err(table.err_empty_remote.to_string());
    }
    if !spec.local_mode {
        if spec.host.trim().is_empty() {
            return Err(table.err_empty_host.to_string());
        }
        if spec.share.trim().is_empty() {
            return Err(table.err_empty_share.to_string());
        }
    }
    if !spec.local_dir.exists() {
        if let Err(err) = std::fs::create_dir_all(&spec.local_dir) {
            return Err((table.err_local_dir_io)(&err.to_string()));
        }
    }
    let listing = if spec.local_mode {
        LocalFs.list(&spec.remote_dir)
    } else {
        SmbClient::new(&spec.host, &spec.share, None).list(&spec.remote_dir)
    };
    let entries = listing.map_err(|err| err.to_string())?;
    Ok(entries.iter().filter(|entry| entry.is_wav()).count())
}

/// Пресет из спеки для записи в `tui.toml` (язык — текущий, не сбрасывается).
fn settings_of_spec(spec: &RunSpec, lang: Lang) -> TuiSettings {
    TuiSettings {
        host: spec.host.clone(),
        share: spec.share.clone(),
        remote_dir: spec.remote_dir.to_string_lossy().into_owned(),
        local_dir: spec.local_dir.to_string_lossy().into_owned(),
        model: persist::model_label(spec.model).to_string(),
        dedup: spec.dedup,
        local_mode: spec.local_mode,
        lang: lang.label().to_string(),
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
    // Высота статус-зоны та же, что у рендера: крупная карта с текущим файлом.
    let status_height =
        widgets::status_zone_height(current_file_view(app, tr(app.lang())).is_some(), area.height);
    let zones = widgets::compute_zones(area, status_height);
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
    let table = tr(app.lang());
    if is_too_small(area) {
        draw_too_small(frame, area, table);
        return;
    }
    widgets::render_screen(frame, area, &screen_view(app), table);
    if app.state() == State::Config {
        if let Some(popup) = app.popup_view() {
            popup::render_popup(frame, area, &popup, table);
        }
    }
    // Confirm выхода — поверх всего (и попапа невозможен: Running ≠ Config).
    if app.confirm_open() {
        widgets::render_confirm(frame, area, table);
    }
}

fn is_too_small(area: Rect) -> bool {
    area.width < MIN_WIDTH || area.height < MIN_HEIGHT
}

/// Взгляд на кадр из состояния App: шапка из конфига прогона, активный таб
/// (Files/Dedup/Log/Help), таблица, журнал, сводка в footer.
fn screen_view(app: &App) -> ScreenView {
    let table = tr(app.lang());
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
        files: app.files().iter().map(|view| file_row_view(view, tick, table)).collect(),
        selected: (!app.files().is_empty()).then_some(app.selected_index()),
        // Details-панель — отдельный этап (таб Files, полный режим).
        details: None,
        footer: FooterView { summary: footer_summary(app, table) },
        dedup_pairs,
        dedup_unparseable,
        log: app.log().clone(),
        log_filter_input,
        banner: cancel_banner(app, table),
        current: current_file_view(app, table),
    }
}

/// Баннер мягкой отмены: живёт, пока движок доигрывает текущий файл.
fn cancel_banner(app: &App, table: &StrTable) -> Option<String> {
    app.cancelling().then(|| table.cancel_banner.to_string())
}

/// Индикатор текущего файла: первый файл без терминального исхода в Running.
/// Наличие индикатора определяет и высоту статус-зоны (рендер и мышь).
fn current_file_view(app: &App, table: &StrTable) -> Option<CurrentFileView> {
    if app.state() != State::Running {
        return None;
    }
    let view = app.files().iter().find(|file| file.outcome.is_none())?;
    Some(CurrentFileView {
        name: view.name.clone(),
        stage: stage_label(view.stage, table).to_string(),
        pct: view.pct,
        estimated: view.estimated,
        phase: app.tick_count(),
        stage_elapsed: view.stage_started.map_or(0.0, |started| started.elapsed().as_secs_f64()),
        // Огибающая из App по ссылке (Arc-клон, без копирования каналов).
        waveform: WaveformView {
            pct: view.pct.unwrap_or(0.0),
            duration_secs: None,
            estimated: view.estimated,
            envelope: view.envelope.clone().unwrap_or_default(),
        },
    })
}

/// Шапка: подключение/дир/модель из открытого конфига, GPU-слот из поллера,
/// индикатор загрузки модели — глобальный.
fn header_view(app: &App) -> HeaderView {
    let model_pct = app.model_pct();
    let compute_device = app.gpu_device_name().map(str::to_string);
    let Some(cfg) = app.config() else {
        return HeaderView {
            gpu: app.gpu().cloned(),
            compute_device,
            model_pct,
            ..HeaderView::default()
        };
    };
    HeaderView {
        host: cfg.host.as_str().to_string(),
        share: cfg.share.as_str().to_string(),
        remote_dir: cfg.remote_dir.as_str().to_string(),
        model: model_label(cfg.model).to_string(),
        gpu: app.gpu().cloned(),
        compute_device,
        model_pct,
    }
}

/// Метка модели как в CLI (`sw` | `6s`).
fn model_label(model: Model) -> &'static str {
    persist::model_label(model)
}

/// Строка таблицы из модели прогона: бейдж и стадия по исходу (терминальные
/// строки без «мёртвой» стадии), таймер стадии, индeterminate-бар.
fn file_row_view(view: &FileView, tick: u64, table: &StrTable) -> FileRowView {
    let terminal = view.outcome.is_some();
    let stage = view
        .outcome
        .as_ref()
        .map_or_else(|| stage_label(view.stage, table), |outcome| stage_label_of_outcome(outcome, table));
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
        estimated: view.estimated,
        stems_done: view.stems_done,
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
fn stage_label(stage: Stage, table: &StrTable) -> &'static str {
    match stage {
        Stage::Queued => table.stage_queued,
        Stage::Download => table.stage_download,
        Stage::Separate => table.stage_separate,
        Stage::Upload => table.stage_upload,
        Stage::Rename => table.stage_rename,
    }
}

/// Метка стадии по терминальному исходу: строки не «зависают» на последней
/// стадии (дубль больше не выглядит как «download 0%»).
fn stage_label_of_outcome(outcome: &FileOutcome, table: &StrTable) -> &'static str {
    match outcome {
        FileOutcome::Ok => table.stage_done,
        FileOutcome::Failed { .. } => table.stage_failed,
        FileOutcome::Duplicate { .. } => table.stage_duplicate,
        FileOutcome::Skipped { .. } => table.stage_skipped,
    }
}

/// Сводка footer: глобальная полоса прогона — файлы done/total, счётчики
/// исходов, общий elapsed и ETA (среднее по завершённым × остаток;
/// до первого завершённого — «--»). После завершения ETA скрыт,
/// при батчевой ошибке прогона — её текст.
fn footer_summary(app: &App, table: &StrTable) -> String {
    if let Some(error) = app.run_error() {
        return (table.footer_error)(error);
    }
    // Пустой итог вместо незаметного «файлы 0/0» — явная подсказка.
    if app.outcome().is_some_and(|outcome| outcome.total == 0) {
        return table.empty_plan_hint.to_string();
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
    let mut summary = format!(
        "{} · {} {ok} · {} {fail} · {} {dup} · {} {skip}",
        widgets::files_progress(table, done, total),
        table.badge_ok,
        table.badge_fail,
        table.badge_dup,
        table.badge_skip,
    );
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
fn draw_too_small(frame: &mut Frame, area: Rect, table: &StrTable) {
    let message = vec![
        Line::from(table.too_small_title),
        Line::from((table.too_small_size)(MIN_WIDTH, MIN_HEIGHT, area.width, area.height)),
    ];
    let paragraph = Paragraph::new(message)
        .style(Style::new().yellow())
        .alignment(ratatui::layout::Alignment::Center);
    frame.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::app::{Focus, ValidationStatus};
    use crate::tui::event::{BatchOutcome, PlanSummary};
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
        assert!(text.contains("загрузка · "), "тикающий таймер стадии");
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
        assert!(text.contains("hexastem"));
        assert!(text.contains("нет файлов — Esc → настройка"));
        assert!(text.contains("[s]тарт"));
    }

    #[test]
    fn min_size_still_renders_main_ui() {
        let text = render(MIN_WIDTH, MIN_HEIGHT, &App::new());
        assert!(text.contains("hexastem"));
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
        assert!(text.contains("Хост:"));
        assert!(text.contains("[1]Файлы"), "экран за попапом жив");
    }

    #[test]
    fn running_state_renders_table_with_run_badge() {
        let mut app = App::new();
        app.handle(&AppEvent::RunStarted { plan: plan_fixture() });
        let text = render(100, 30, &app);
        assert!(text.contains("ФАЙЛЫ"));
        assert!(text.contains("ПРЦ"));
        assert!(text.contains("a.wav"));
    }

    #[test]
    fn stems_column_is_honest_live_and_dash_before_data() {
        let mut app = running_app();

        let text = render(100, 30, &app);
        assert!(text.contains("—"), "счётчик неизвестен — «—», не «0/6»: {text}");
        assert!(!text.contains("0/6"));

        feed(&mut app, AppEvent::FileStems { name: "a.wav".into(), count: 6 });
        let text = render(100, 30, &app);
        assert!(text.contains("6/6"), "живой счётчик по событию драйвера: {text}");
    }

    #[test]
    fn key_screens_render_in_both_locales() {
        let mut app = running_app();
        feed(&mut app, AppEvent::FileStage { name: "a.wav".into(), stage: Stage::Download });

        // ru (дефолт): шапка-приложение, табы, таблица, футер.
        let text = render(100, 30, &app);
        assert!(text.contains("hexastem"), "шапка: {text}");
        assert!(text.contains("[1]Файлы"), "табы ru: {text}");
        assert!(text.contains("СТАДИЯ"), "заголовок таблицы ru: {text}");
        assert!(text.contains("файлы 0/1"), "футер ru: {text}");
        // Подсказки клавиш — в широком кадре (со сводкой не влезают в 100).
        let wide = render(160, 30, &app);
        assert!(wide.contains("[s]тарт"), "подсказки клавиш ru: {wide}");
        assert!(text.contains("загрузка"), "стадия ru: {text}");

        // Переключение `l` — те же экраны на en, состояние живо.
        app.handle(&key(crossterm::event::KeyCode::Char('l')));
        let text = render(100, 30, &app);
        assert!(text.contains("[1]Files"), "табы en: {text}");
        assert!(text.contains("STAGE"), "заголовок таблицы en: {text}");
        assert!(text.contains("files 0/1"), "футер en: {text}");
        let wide = render(160, 30, &app);
        assert!(wide.contains("[s]tart"), "подсказки клавиш en: {wide}");
        assert!(text.contains("download"), "стадия en: {text}");
        assert_eq!(app.state(), State::Running, "язык не сбрасывает прогон");
        assert!(!text.contains("ФАЙЛЫ"), "ru-строки ушли: {text}");

        // Попап на en (язык задаётся до открытия — в попапе `l` модален).
        let mut config = App::new();
        config.set_lang(Lang::En);
        config.handle(&enter());
        config.open_config(persist::load());
        let text = render(100, 30, &config);
        assert!(text.contains("Run setup"), "попап en: {text}");
        assert!(text.contains("Remote dir:"), "подписи полей en: {text}");

        // Тот же попап на ru.
        let mut config_ru = App::new();
        config_ru.handle(&enter());
        config_ru.open_config(persist::load());
        let text = render(100, 30, &config_ru);
        assert!(text.contains("Запуск прогона"), "попап ru: {text}");
        assert!(text.contains("Исходная папка:"), "подписи полей ru: {text}");
    }

    #[test]
    fn confirm_dialog_renders_in_both_locales() {
        // ru: confirm по q/Esc в Running.
        let mut ru = running_app();
        ru.handle(&key(crossterm::event::KeyCode::Char('q')));
        let text = render(100, 30, &ru);
        assert!(text.contains("Подтверждение"), "заголовок ru: {text}");
        assert!(text.contains("Прервать прогон и выйти?"), "вопрос ru: {text}");
        assert!(text.contains("границе файлов"), "мягкая отмена ru: {text}");

        // en: переключение до открытия модала.
        let mut en = running_app();
        en.handle(&key(crossterm::event::KeyCode::Char('l')));
        en.handle(&key(crossterm::event::KeyCode::Char('q')));
        let text = render(100, 30, &en);
        assert!(text.contains("Confirm"), "заголовок en: {text}");
        assert!(text.contains("Interrupt the run and quit?"), "вопрос en: {text}");
        assert!(text.contains("file boundaries"), "мягкая отмена en: {text}");
        assert!(!text.contains("Подтверждение"), "ru-строки ушли: {text}");
    }

    #[test]
    fn log_messages_render_in_both_locales() {
        let mut app = App::new();
        app.set_lang(Lang::En);
        feed(&mut app, AppEvent::RunStarted { plan: plan_fixture() });
        feed(&mut app, AppEvent::FileDone { name: "a.wav".into() });
        feed(&mut app, AppEvent::FileDuplicate { name: "z.wav".into() });
        feed(&mut app, AppEvent::BatchFinished { outcome: outcome_with_dedup() });

        app.handle(&key(crossterm::event::KeyCode::Char('3')));
        let text = render(100, 30, &app);
        assert!(text.contains("run started: 1 files"), "журнал en: {text}");
        assert!(text.contains("a.wav: done"), "журнал en: {text}");
        assert!(text.contains("z.wav: duplicate (skipped)"), "журнал en: {text}");
        assert!(text.contains("run finished"), "сводка en: {text}");
        assert!(text.contains("louder by 11.1 dB"), "причина дедупа en: {text}");
        assert!(text.contains("unparseable as WAV: broken.wav"), "примечание en: {text}");
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

    /// Событие `device` драйвера обновляет шапку («Считает: …»); до события —
    /// прежний рендер (слота нет, GPU-виджет «n/a»).
    #[test]
    fn device_event_shows_computing_in_header() {
        let mut app = running_app();
        let text = render(100, 30, &app);
        assert!(!text.contains("Считает:"), "до события device слота нет: {text}");
        assert!(text.contains("GPU: n/a"), "прежний рендер шапки: {text}");

        feed(&mut app, AppEvent::Device { name: "RTX 4090".into() });
        let text = render(100, 30, &app);
        assert!(text.contains("Считает: RTX 4090"), "событие device обновляет шапку: {text}");
    }

    // --- валидация спеки (реальный диск, без сети) ----------------------------

    #[test]
    fn validate_local_spec_with_existing_dirs_is_ok() {
        let base = std::env::temp_dir().join(format!("hexastem_mod_{}", std::process::id()));
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

        let wav_count = validate_spec(&spec, Lang::Ru).expect("локальная валидация проходит");

        assert_eq!(wav_count, 0, "пустая директория — ноль .wav");
        assert!(spec.local_dir.exists(), "локальная директория создана при проверке");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn validate_counts_wav_files_in_local_dir() {
        let base = std::env::temp_dir().join(format!("hexastem_mod_wav_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("a.wav"), b"x").unwrap();
        std::fs::write(base.join("B.WAV"), b"x").unwrap();
        std::fs::write(base.join("notes.txt"), b"x").unwrap();
        let spec = RunSpec {
            host: String::new(),
            share: String::new(),
            remote_dir: base.clone(),
            local_dir: base.join("work"),
            model: Model::Sw,
            dedup: true,
            local_mode: true,
        };

        let wav_count = validate_spec(&spec, Lang::Ru).expect("листинг проходит");

        assert_eq!(wav_count, 2, "регистр расширения не важен, не-wav не считается");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn validate_reports_missing_remote_directory() {
        let base = std::env::temp_dir().join(format!("hexastem_mod_missing_{}", std::process::id()));
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

        let error = validate_spec(&spec, Lang::Ru).expect_err("нет директории — ошибка");

        assert!(error.contains("no such file") || error.contains("No such file"));
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

        let error = validate_spec(&spec, Lang::Ru).expect_err("пустой хост — ошибка до запуска smbclient");

        assert!(error.contains("хост"));
    }

    // --- SMB/GVFS-paste → валидация с фактом → подтверждение старта -------------

    /// Боевой сценарий: paste `smb://` в попапе заполняет host/share/remote_dir
    /// и выключает local mode; валидация показывает факт, старт — вторым Enter.
    #[test]
    fn smb_paste_then_validate_then_confirm_starts_run() {
        let mut app = App::new();
        app.handle(&enter());
        app.open_config(TuiSettings { local_mode: true, dedup: true, ..TuiSettings::default() });

        app.handle(&AppEvent::Paste { text: "smb://nas/media/records".into() });
        let cfg = app.config().expect("попап открыт");
        assert_eq!(cfg.host.as_str(), "nas");
        assert_eq!(cfg.share.as_str(), "media");
        assert_eq!(cfg.remote_dir.as_str(), "records");
        assert!(!cfg.local_mode, "SMB-путь выключает local mode");
        assert_eq!(cfg.focus, Focus::RemoteDir);
        assert!(cfg.hint.is_some(), "статус-строка распознанного пути");

        app.handle(&key(crossterm::event::KeyCode::Tab)); // RemoteDir → LocalDir
        app.handle(&AppEvent::Paste { text: "/tmp/work".into() });

        // Enter №1 — запрос валидации; факт приходит из spawn_validation.
        app.handle(&enter());
        let spec = app.take_start_request().expect("валидация запрошена");
        app.handle(&AppEvent::ValidationFinished { spec: spec.clone(), error: None, wav_count: 3 });

        assert_eq!(app.state(), State::Config, "валидация показывает факт, но не стартует");
        assert_eq!(app.config().unwrap().validation, ValidationStatus::Ok { wav_count: 3 });

        // Enter №2 — подтверждение старта проверенной спеки.
        app.handle(&enter());
        assert_eq!(app.state(), State::Running);
        assert_eq!(app.take_confirmed_start().as_ref(), Some(&spec));
    }

    /// Правка поля после успешной валидации устаревает факт: следующий Enter
    /// перепроверяет, подтверждённого старта нет.
    #[test]
    fn editing_after_validation_ok_requires_revalidation() {
        let mut app = App::new();
        app.handle(&enter());
        app.open_config(TuiSettings { local_mode: true, ..TuiSettings::default() });
        app.handle(&AppEvent::Paste { text: "smb://nas/media/rec".into() });
        app.handle(&key(crossterm::event::KeyCode::Tab));
        app.handle(&AppEvent::Paste { text: "/tmp/work".into() });
        app.handle(&enter());
        let spec = app.take_start_request().expect("валидация запрошена");
        app.handle(&AppEvent::ValidationFinished { spec, error: None, wav_count: 2 });

        app.handle(&key(crossterm::event::KeyCode::Backspace));

        assert_eq!(app.config().unwrap().validation, ValidationStatus::Idle, "факт устарел");
        app.handle(&enter());
        assert!(app.take_start_request().is_some(), "перепроверка после правки");
        assert!(app.take_confirmed_start().is_none());
    }

    /// GVFS-paste вне попапа: event loop преффиллит попап — поля SMB заполнены.
    #[test]
    fn global_paste_prefill_applies_smb_hint() {
        let mut app = App::new();
        let gvfs = "/run/user/1000/gvfs/smb-share:server=192.0.2.10,share=public/from_share/ZOOM0003";
        app.handle(&AppEvent::Paste { text: gvfs.to_string() });
        assert_eq!(app.state(), State::Config, "paste пути открывает попап");
        let dir = app.take_paste_prefill().expect("путь в преффилле");

        // Как живой event loop: открыть попап и применить преффилл.
        app.open_config(TuiSettings::default());
        app.prefill_remote_dir(&dir);

        let cfg = app.config().expect("попап открыт");
        assert_eq!(cfg.host.as_str(), "192.0.2.10");
        assert_eq!(cfg.share.as_str(), "public");
        assert_eq!(cfg.remote_dir.as_str(), "from_share/ZOOM0003");
        assert!(!cfg.local_mode, "GVFS-путь переключает в SMB-режим");
        assert!(cfg.hint.is_some());
    }

    /// Пустой план и пустой итог — явная подсказка в футере и в табе Log.
    #[test]
    fn empty_plan_and_outcome_warn_about_directory() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: PlanSummary::default() });
        feed(
            &mut app,
            AppEvent::BatchFinished {
                outcome: BatchOutcome {
                    total: 0,
                    ok: 0,
                    failed: 0,
                    results: Vec::new(),
                    dedup: Default::default(),
                    elapsed: std::time::Duration::ZERO,
                },
            },
        );

        let text = render(100, 30, &app);
        assert!(text.contains("0 .wav в директории — проверь remote-dir"), "footer итога: {text}");

        app.handle(&key(crossterm::event::KeyCode::Char('3')));
        let text = render(100, 30, &app);
        assert!(text.contains("0 .wav в директории — проверь remote-dir"), "таб Log: {text}");
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
        let base = std::env::temp_dir().join(format!("hexastem_mod_run_{}", std::process::id()));
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

        let settings = settings_of_spec(&spec, Lang::Ru);

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

        assert!(text.contains("ДУБЛ"), "заголовок таблицы");
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
        assert!(text.contains("ФАЙЛЫ"), "фон (таблица Files) жив под оверлеем");
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
        assert!(messages[1].contains("a.wav: разделение"));
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

        // Контент с y=16 (крупная статус-зона 12 рядов на 30-строчном экране);
        // строка таблицы 0 — с y=18 (рамка+заголовок), клик по y=19 — вторая.
        handle_mouse(&mut app, click_up(10, 19), SCREEN);

        assert_eq!(app.selected_index(), 1, "клик по второй строке таблицы");
        assert_eq!(app.selected_file().map(|file| file.name.as_str()), Some("b.wav"));
    }

    #[test]
    fn mouse_click_beyond_last_row_clamps_to_last_file() {
        let mut app = App::new();
        feed(&mut app, AppEvent::RunStarted { plan: plan_with_three_files() });

        handle_mouse(&mut app, click_up(10, 25), SCREEN); // пустая зона таблицы

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
