//! Runner: мост между TUI и пайплайном lib (`Pipeline`).
//!
//! `SmbOps`/`EngineOps` — синхронные трейты, `EngineOps::separate_batch`
//! принимает БЛОКИРУЮЩИЙ `on_event`, поэтому пайплайн исполняется в
//! `tokio::task::spawn_blocking`. События движка и стадии SMB пробрасываются
//! в канал [`AppEvent`] декораторами [`TeeSmb`]/[`TeeEngine`] (обёртки над
//! инжектируемыми операциями — lib остаётся без знаний о TUI):
//!
//! - `TeeSmb`: list/get/create_dir/put/rename → стадии Queued → Download →
//!   Separate → Upload → Rename; после успешного download запускает фоновый
//!   расчёт огибающей (1 раз на файл, `envelope::wav_stereo_envelope`);
//! - `TeeEngine`: `EngineEvent` → Progress(pct)/ModelProgress/FileFailed;
//!   на входе батча вычисляет дубли (скачано, но не в worklist) → DUP;
//!   после доработки файла пайплайном — кооперативная пауза (граница файлов).
//! - `download_poller_loop`: фоновая задача — фактический download-прогресс
//!   по размерам файлов (smbclient не отдаёт прогресс передачи).
//!
//! Управление — кооперативное, через [`RunnerHandle`]: cancel/пауза/retry на
//! атомарках и очереди, проверки — на границах файлов и стадий.
//!
//! # Ограничения (принято, lib не правится)
//!
//! - **Отмена внутри separate невозможна**: батч — один вызов `separate_batch`,
//!   движок прерывается только мягко (флаг отмены между чанками). Cancel,
//!   запрошенный во время батча, останавливает заливку/rename завершающихся
//!   файлов (они получают Failed, исходники на NAS нетронуты — инвариант
//!   rename; resume подхватит при следующем прогоне), но текущий GPU-батч
//!   отрабатывает до границы чанка.
//! - **Scan-стадия не имеет отдельного события**: план приходит одним
//!   [`AppEvent::RunStarted`] после скана; стадия scan видима как Running
//!   без строк таблицы.
//! - **Процент separate — оценка движка** (`est: true`), download-прогресс —
//!   фактический (поллер размера локального файла против размера на шаре),
//!   upload/rename — indeterminate (размеров операций у UI нет).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

use crate::envelope::wav_stereo_envelope;
use crate::error::AppError;
use crate::pipeline::{
    plan as plan_batch, summarize_with_dedup, BatchPlan, BatchResult, DUPLICATE_PREFIX, EngineEvent,
    EngineOps, Pipeline, SmbOps,
};
use crate::smb::SmbEntry;
use crate::tui::event::{AppEvent, BatchOutcome, PlanSummary, Stage};

/// Период опроса очереди retry между батчами.
const RETRY_POLL: Duration = Duration::from_millis(100);
/// Порция паузы между проверками cancel.
const PAUSE_SLICE_MS: u64 = 50;
/// Текст ошибки кооперативной отмены.
const CANCELLED: &str = "cancelled by user";

/// Скачанные файлы текущего прогона: локальный путь → имя на шаре
/// (общая память tee-декораторов для вывода дублей на входе батча).
type SharedDownloads = Arc<Mutex<BTreeMap<String, String>>>;

/// Зависимости раннера: инжектируемые операции SMB и движка (продакшн —
/// `SmbClient`/`NativeEngine`, тесты — моки `mocks`).
pub struct RunnerDeps {
    /// Операции SMB (list/get/put/rename).
    pub smb: Arc<dyn SmbOps + Send + Sync>,
    /// Движок разделения (separate_batch с блокирующим on_event).
    pub engine: Arc<dyn EngineOps + Send + Sync>,
}

/// Параметры прогона (из попапа запуска/пресета).
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    /// Исходная директория на шаре.
    pub remote_dir: PathBuf,
    /// Локальная рабочая директория.
    pub local_dir: PathBuf,
    /// Расширение стемов без точки (для resume-плана).
    pub stem_ext: String,
    /// Дедупликация включена (флаг `--no-dedup` выключает).
    pub dedup: bool,
    /// Пауза между файлами; применяется на границе файлов, меняется на лету
    /// через [`RunnerHandle::set_pause`].
    pub pause_between_files: Option<Duration>,
    /// Колонки waveform-огибающей (0 — не считать).
    pub envelope_columns: usize,
}

/// Общее управление прогоном: cancel/pause/retry/shutdown.
#[derive(Default)]
struct Ctrl {
    cancel: AtomicBool,
    shutdown: AtomicBool,
    pause_ms: AtomicU64,
    retries: Mutex<Vec<String>>,
}

impl Ctrl {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn take_retry(&self) -> Option<String> {
        self.retries.lock().expect("retries lock").pop()
    }
}

/// Ручка раннера для event loop'а: управление прогоном + итог.
pub struct RunnerHandle {
    ctrl: Arc<Ctrl>,
    task: JoinHandle<Result<BatchResult, AppError>>,
    /// Движок прогона: адресат форс-kill'а.
    engine: Arc<dyn EngineOps + Send + Sync>,
}

impl RunnerHandle {
    /// Кооперативная отмена: проверки срабатывают на границах файлов/стадий
    /// (см. ограничения модуля — текущий батч не прерывается жёстко).
    pub fn cancel(&self) {
        self.ctrl.cancel.store(true, Ordering::Relaxed);
    }

    /// Форс-отмена: немедленно убить движок (текущий файл
    /// останется без rename — resume подберёт его в следующем прогоне).
    /// Ожидание движка завершается ошибкой, раннер доходит до выхода сам.
    pub fn force_kill(&self) {
        self.engine.kill_current_engine();
    }

    /// Серия завершена для раннера: больше не принимать retry; итог — в `finish`.
    pub fn shutdown(&self) {
        self.ctrl.shutdown.store(true, Ordering::Relaxed);
    }

    /// Пауза между файлами (None — выключить); применяется со следующей границы.
    pub fn set_pause(&self, pause: Option<Duration>) {
        self.ctrl.pause_ms.store(pause_millis(pause), Ordering::Relaxed);
    }

    /// Запросить повтор упавшего файла; исполнится после текущего батча
    /// (мини-пайплайн из одного pending). Отмена снимает очередь.
    pub fn request_retry(&self, name: impl Into<String>) {
        self.ctrl.retries.lock().expect("retries lock").push(name.into());
    }

    /// Итог серии: объединённый [`BatchResult`] (exit-code по нему, см.
    /// [`exit_code`]). Err — прогон не состоялся (скан/сеть/отмена до старта).
    pub async fn finish(self) -> Result<BatchResult, AppError> {
        self.task.await.map_err(|err| AppError::Pipeline(format!("runner task crashed: {err}")))?
    }
}

/// Запустить серию прогонов (только в контексте tokio-рантайма).
///
/// Серия: полный прогон (скан → план → execute), затем — до shutdown/cancel —
/// retry'и упавших файлов по одному. События — в `events`, итог — через
/// [`RunnerHandle::finish`].
pub fn spawn(deps: RunnerDeps, opts: RunnerOptions, events: UnboundedSender<AppEvent>) -> RunnerHandle {
    let ctrl = Arc::new(Ctrl {
        cancel: AtomicBool::new(false),
        shutdown: AtomicBool::new(false),
        pause_ms: AtomicU64::new(pause_millis(opts.pause_between_files)),
        retries: Mutex::new(Vec::new()),
    });
    let engine = Arc::clone(&deps.engine);
    let task = tokio::spawn(runner_loop(deps, opts, events, ctrl.clone()));
    RunnerHandle { ctrl, task, engine }
}

/// Exit-code прогона по [`BatchResult`]: 0 — нет отказов (дубли не влияют),
/// 1 — есть failed. Как у CLI.
pub fn exit_code(result: &BatchResult) -> i32 {
    i32::from(result.failed > 0)
}

async fn runner_loop(
    deps: RunnerDeps,
    opts: RunnerOptions,
    events: UnboundedSender<AppEvent>,
    ctrl: Arc<Ctrl>,
) -> Result<BatchResult, AppError> {
    let runtime = tokio::runtime::Handle::current();
    let (plan, mut merged) = match run_once(&deps, &opts, &events, &ctrl, &runtime, None).await {
        Ok(pair) => pair,
        Err(err) => {
            let _ = events.send(AppEvent::RunFailed { error: err.to_string() });
            return Err(err);
        }
    };
    // Серия retry'ей: по одному файлу, строго между батчами; cancel/shutdown
    // прерывает серию (очередь откладывается до следующего прогона).
    while !ctrl.shutdown.load(Ordering::Relaxed) && !ctrl.cancelled() {
        let Some(name) = ctrl.take_retry() else {
            tokio::time::sleep(RETRY_POLL).await;
            continue;
        };
        let Some(entry) = plan.pending.iter().find(|entry| entry.name == name) else {
            continue; // не-pending/неизвестное имя — не отказ, игнорируем
        };
        let retry_plan = BatchPlan { pending: vec![entry.clone()], ..BatchPlan::default() };
        if let Ok((_, result)) = run_once(&deps, &opts, &events, &ctrl, &runtime, Some(retry_plan)).await {
            merge_run(&mut merged, result);
        }
    }
    Ok(merged)
}

/// Один прогон: скан+план (если план не задан) → `Pipeline::execute` в
/// `spawn_blocking`. Scan (list+план) и execute идут через tee-декораторы.
async fn run_once(
    deps: &RunnerDeps,
    opts: &RunnerOptions,
    events: &UnboundedSender<AppEvent>,
    ctrl: &Arc<Ctrl>,
    runtime: &tokio::runtime::Handle,
    initial_plan: Option<BatchPlan>,
) -> Result<(BatchPlan, BatchResult), AppError> {
    let is_initial = initial_plan.is_none();
    let downloaded: SharedDownloads = Arc::new(Mutex::new(BTreeMap::new()));
    let (smb, engine) = (deps.smb.clone(), deps.engine.clone());
    let (opts, events, ctrl, runtime) = (opts.clone(), events.clone(), ctrl.clone(), runtime.clone());
    let task = tokio::task::spawn_blocking(move || {
        let smb_tee = TeeSmb {
            inner: &*smb,
            remote_dir: opts.remote_dir.clone(),
            envelope_columns: opts.envelope_columns,
            events: events.clone(),
            ctrl: ctrl.clone(),
            runtime: runtime.clone(),
            envelope_paths: RefCell::default(),
            downloaded: downloaded.clone(),
        };
        let plan = match initial_plan {
            Some(plan) => plan,
            None => {
                let entries = smb_tee.list(&opts.remote_dir)?;
                plan_batch(&entries, &smb_tee, &opts.remote_dir, &opts.stem_ext)?
            }
        };
        if is_initial {
            // Retry-прогон таблицу не пересевает: строки уже в UI.
            let _ = events.send(AppEvent::RunStarted {
                plan: PlanSummary {
                    pending: plan.pending.iter().map(|entry| entry.name.clone()).collect(),
                    skipped_converted: plan.skipped_converted.clone(),
                    skipped_duplicate: plan.skipped_duplicate.clone(),
                    skipped_done: plan.skipped_done.clone(),
                },
            });
        }
        let engine_tee =
            TeeEngine { inner: &*engine, events: events.clone(), ctrl: ctrl.clone(), downloaded: downloaded.clone() };
        // Download-прогресс: размеры pending-файлов с шары (SmbEntry.size) —
        // поллер сверяет их с ростом локальных файлов (см. download_poller_loop).
        let remote_sizes: RemoteSizes = plan
            .pending
            .iter()
            .filter(|entry| entry.size > 0)
            .map(|entry| (entry.name.clone(), entry.size))
            .collect();
        let poller = runtime.spawn(download_poller_loop(
            remote_sizes,
            opts.local_dir.clone(),
            downloaded.clone(),
            events.clone(),
            DOWNLOAD_POLL,
        ));
        let pipeline = Pipeline::new(
            &smb_tee,
            &engine_tee,
            opts.remote_dir.clone(),
            opts.local_dir.clone(),
            &opts.stem_ext,
        )
        .with_dedup(opts.dedup);
        let result = pipeline.execute(&plan);
        poller.abort(); // поллер живёт только пока идёт батч
        let result = result?;
        let _ = events.send(AppEvent::BatchFinished { outcome: BatchOutcome::from(&result) });
        Ok((plan, result))
    });
    task.await.map_err(|err| AppError::Pipeline(format!("runner blocking task failed: {err}")))?
}

/// Интервал поллера download-прогресса.
const DOWNLOAD_POLL: Duration = Duration::from_secs(1);

/// Размеры pending-файлов на шаре: имя → байты (из `SmbEntry.size`).
type RemoteSizes = BTreeMap<String, u64>;

/// Фактический download-прогресс: локальный размер / удалённый. `None` —
/// удалённый размер неизвестен или локального файла ещё нет.
fn download_pct(local: &Path, remote_size: u64) -> Option<f64> {
    if remote_size == 0 {
        return None;
    }
    let len = std::fs::metadata(local).ok()?.len();
    Some(((len as f64 / remote_size as f64) * 100.0).min(100.0))
}

/// Фоновый поллер download-прогресса: раз в `period` для каждого pending-файла
/// сверяет локальный размер с размером на шаре и шлёт фактический
/// [`AppEvent::FileProgress`] (Download) — smbclient прогресс передачи не даёт.
/// Событие — только при изменении целого процента (без спама одинаковыми).
/// Гасится `abort`'ом после батча (см. `run_once`).
async fn download_poller_loop(
    sizes: RemoteSizes,
    local_dir: PathBuf,
    downloaded: SharedDownloads,
    events: UnboundedSender<AppEvent>,
    period: Duration,
) {
    let mut last_sent: BTreeMap<String, i64> = BTreeMap::new();
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        for (name, remote) in &sizes {
            let local = local_dir.join(name);
            let done = downloaded
                .lock()
                .map(|map| map.contains_key(&local.to_string_lossy().into_owned()))
                .unwrap_or(false);
            if done {
                continue; // скачан и передан пайплайну — прогресс не нужен
            }
            let Some(pct) = download_pct(&local, *remote) else { continue };
            let whole = pct as i64;
            if last_sent.get(name).is_some_and(|sent| whole - *sent < 1) {
                continue; // шлём только изменения ≥1%
            }
            last_sent.insert(name.clone(), whole);
            let _ = events.send(AppEvent::FileProgress {
                name: name.clone(),
                stage: Stage::Download,
                pct,
                estimated: false, // поллер показывает фактический объём
            });
        }
    }
}

/// Слияние retry-результата в общий итог: статусы замещаются по имени файла,
/// счётчики пересчитываются (суммаризация lib), dedup/старт — от первого прогона.
fn merge_run(merged: &mut BatchResult, retry: BatchResult) {
    let mut results = std::mem::take(&mut merged.results);
    for retry_result in retry.results {
        match results.iter_mut().find(|result| result.name == retry_result.name) {
            Some(slot) => *slot = retry_result,
            None => results.push(retry_result),
        }
    }
    *merged = summarize_with_dedup(results, merged.dedup.clone(), merged.started);
}

/// Миллисекунды паузы для атомарки (None — 0).
fn pause_millis(pause: Option<Duration>) -> u64 {
    pause.map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)).unwrap_or(0)
}

/// Кооперативная пауза на границе файлов: спит порциями, cancel снимает
/// остаток. Значение читается каждый раз — `set_pause` действует со
/// следующей границы.
fn wait_pause(ctrl: &Ctrl) {
    let mut remaining = ctrl.pause_ms.load(Ordering::Relaxed);
    while remaining > 0 && !ctrl.cancelled() {
        let step = remaining.min(PAUSE_SLICE_MS);
        std::thread::sleep(Duration::from_millis(step));
        remaining -= step;
    }
}

/// Последний компонент пути как строка (иначе — вход как есть).
fn file_name_of(path: &str) -> &str {
    Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path)
}

// --- tee-декораторы: pайплайн видит SmbOps/EngineOps, TUI — события ---------

/// Декоратор SMB: стадии и исходы операций → `AppEvent`, кооперативный
/// cancel до каждой операции, фоновая огибающая после успешного download.
struct TeeSmb<'a> {
    inner: &'a dyn SmbOps,
    remote_dir: PathBuf,
    envelope_columns: usize,
    events: UnboundedSender<AppEvent>,
    ctrl: Arc<Ctrl>,
    runtime: tokio::runtime::Handle,
    /// Guard «огибающая 1 раз на файл» (download в прогоне единственный).
    envelope_paths: RefCell<HashSet<PathBuf>>,
    downloaded: SharedDownloads,
}

impl TeeSmb<'_> {
    fn report(&self, event: AppEvent) {
        let _ = self.events.send(event);
    }

    fn cancel_err(&self, operation: &'static str, path: &Path) -> AppError {
        AppError::Smb { operation, path: path.to_path_buf(), message: CANCELLED.to_string() }
    }

    fn file_of(&self, path: &Path) -> String {
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned())
    }

    /// Имя файла-владельца операции над стем-директорией: директория зовётся
    /// по базе имени (`pipeline::stem_dir_name`: `a.wav` → `a`), в таблице
    /// файлов имя полное — резолвим по скачанным файлам прогона.
    fn resolve_owner(&self, path: &Path) -> String {
        let base = self.file_of(path);
        let downloaded = self.downloaded.lock().expect("downloads lock");
        downloaded
            .values()
            .find(|name| Path::new(name).file_stem().and_then(|n| n.to_str()) == Some(base.as_str()))
            .cloned()
            .unwrap_or(base)
    }

    /// Общий каркас операции с событиями: cancel-проверка → стадия →
    /// операция → FileFailed при ошибке.
    fn forward_op(
        &self,
        operation: &'static str,
        error_path: &Path,
        owner: String,
        stage: Stage,
        op: impl FnOnce() -> Result<(), AppError>,
    ) -> Result<(), AppError> {
        if self.ctrl.cancelled() {
            let err = self.cancel_err(operation, error_path);
            self.report(AppEvent::FileFailed { name: owner, error: err.to_string() });
            return Err(err);
        }
        self.report(AppEvent::FileStage { name: owner.clone(), stage });
        let result = op();
        if let Err(err) = &result {
            self.report(AppEvent::FileFailed { name: owner, error: err.to_string() });
        }
        result
    }

    /// Огибающая файла — фоновая задача (не блокирует пайплайн): чтение
    /// стриминговое, тяжёлое; при cancel/ошибке (файл мог быть удалён
    /// пайплайном) результат отбрасывается — UI рисует прогресс-бар.
    fn spawn_envelope(&self, local: &Path) {
        if self.envelope_columns == 0
            || !self.envelope_paths.borrow_mut().insert(local.to_path_buf())
        {
            return;
        }
        let (path, columns, events, ctrl) =
            (local.to_path_buf(), self.envelope_columns, self.events.clone(), self.ctrl.clone());
        self.runtime.spawn_blocking(move || match wav_stereo_envelope(&path, columns) {
            Ok(envelope) if !ctrl.cancelled() => {
                let _ = events.send(AppEvent::Envelope { path, envelope: Arc::new(envelope) });
            }
            _ => {}
        });
    }
}

impl SmbOps for TeeSmb<'_> {
    fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        // Скан корня — первая граница отмены; листинги стем-директорий
        // (resume-проверка плана) проходят без событий и проверок.
        if dir == self.remote_dir && self.ctrl.cancelled() {
            return Err(self.cancel_err("list", dir));
        }
        self.inner.list(dir)
    }

    fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
        let name = self.file_of(remote);
        if self.ctrl.cancelled() {
            let err = self.cancel_err("get", remote);
            self.report(AppEvent::FileFailed { name, error: err.to_string() });
            return Err(err);
        }
        self.report(AppEvent::FileStage { name: name.clone(), stage: Stage::Download });
        let result = self.inner.get(remote, local);
        match &result {
            Ok(()) => {
                self.downloaded
                    .lock()
                    .expect("downloads lock")
                    .insert(local.to_string_lossy().into_owned(), name);
                self.spawn_envelope(local);
            }
            Err(err) => self.report(AppEvent::FileFailed { name, error: err.to_string() }),
        }
        result
    }

    fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
        // Владелец стема — директория `<basename>/` в удалённом пути.
        let owner = self.resolve_owner(remote.parent().unwrap_or(remote));
        self.forward_op("put", remote, owner, Stage::Upload, || self.inner.put(local, remote))
    }

    fn create_dir(&self, dir: &Path) -> Result<(), AppError> {
        let owner = self.resolve_owner(dir);
        self.forward_op("mkdir", dir, owner, Stage::Upload, || self.inner.create_dir(dir))
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
        let name = self.file_of(from);
        // Rename дубля (`DUPLICATE_*`) статуса не меняет: DUP выставлен
        // на входе батча; терминальный Ok — только у обычного файла.
        let is_duplicate = to
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|name| name.starts_with(DUPLICATE_PREFIX));
        if self.ctrl.cancelled() {
            let err = self.cancel_err("rename", from);
            self.report(AppEvent::FileFailed { name, error: err.to_string() });
            return Err(err);
        }
        self.report(AppEvent::FileStage { name: name.clone(), stage: Stage::Rename });
        let result = self.inner.rename(from, to);
        match &result {
            Ok(()) if !is_duplicate => self.report(AppEvent::FileDone { name }),
            Ok(()) => {}
            Err(err) => self.report(AppEvent::FileFailed { name, error: err.to_string() }),
        }
        result
    }

    fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
        self.inner.remote_exists(path)
    }
}

/// Декоратор движка: `EngineEvent` → события UI, дубли на входе батча,
/// пауза на границе файлов (после доработки файла пайплайном).
struct TeeEngine<'a> {
    inner: &'a dyn EngineOps,
    events: UnboundedSender<AppEvent>,
    ctrl: Arc<Ctrl>,
    downloaded: SharedDownloads,
}

impl TeeEngine<'_> {
    /// Файлы, скачанные, но не попавшие в worklist движка — дубли,
    /// снятые стадией дедупликации (единственная причина расхождения).
    fn report_duplicates(&self, worklist: &[PathBuf]) {
        let in_engine: HashSet<String> =
            worklist.iter().map(|path| path.to_string_lossy().into_owned()).collect();
        let mut dup_names = Vec::new();
        {
            let downloaded = self.downloaded.lock().expect("downloads lock");
            for (local, name) in downloaded.iter() {
                if !in_engine.contains(local) {
                    let _ = self.events.send(AppEvent::FileDuplicate { name: name.clone() });
                    dup_names.push(name.clone());
                }
            }
        }
        // Пачкой, после пер-файловых событий: UI помечает строки дублями
        // сразу — до старта separate (без «мёртвой» стадии download).
        if !dup_names.is_empty() {
            let _ = self.events.send(AppEvent::DuplicatesKnown { names: dup_names });
        }
    }
}

impl EngineOps for TeeEngine<'_> {
    fn separate_batch(
        &self,
        files: &[PathBuf],
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(), AppError> {
        if self.ctrl.cancelled() {
            // Батч не стартует: все его файлы получают отказ с текстом отмены
            // (пайплайн проставит их Failed и сам — события для UI).
            for file in files {
                let path = file.to_string_lossy();
                let _ = self.events.send(AppEvent::FileFailed {
                    name: file_name_of(&path).to_string(),
                    error: CANCELLED.to_string(),
                });
            }
            return Err(AppError::Engine {
                file: files.first().cloned().unwrap_or_default(),
                exit_code: None,
                stderr_tail: CANCELLED.to_string(),
            });
        }
        self.report_duplicates(files);
        let events = &self.events;
        let ctrl = &self.ctrl;
        let mut tee = |event: EngineEvent| {
            // Граница файла определяется до передачи события пайплайну
            // (доработка файла — upload/rename — тоже входит в «файл»).
            let boundary = matches!(event, EngineEvent::FileDone { .. } | EngineEvent::FileError { .. });
            match &event {
                EngineEvent::Progress { file, stage, pct, estimated } => {
                    let name = file_name_of(file).to_string();
                    let event = if stage == "separate" {
                        AppEvent::FileProgress {
                            name,
                            stage: Stage::Separate,
                            pct: *pct,
                            estimated: *estimated,
                        }
                    } else {
                        AppEvent::ModelProgress { pct: *pct } // load и пр. — глобально
                    };
                    let _ = events.send(event);
                }
                EngineEvent::FileDone { file, stems, .. } => {
                    let name = file_name_of(file).to_string();
                    let _ = events.send(AppEvent::FileProgress {
                        name: name.clone(),
                        stage: Stage::Separate,
                        pct: 100.0,
                        estimated: false,
                    });
                    // Список стемов движка не теряем: живой счётчик STEMS
                    // честен уже на upload/rename (раньше здесь был `..`).
                    let _ = events.send(AppEvent::FileStems {
                        name,
                        count: stems.len().min(u8::MAX as usize) as u8,
                    });
                }
                EngineEvent::FileError { file, error } => {
                    let _ = events.send(AppEvent::FileFailed {
                        name: file_name_of(file).to_string(),
                        error: error.clone(),
                    });
                }
                // Имя вычислительного устройства — событие шапки TUI.
                EngineEvent::Device { name } => {
                    let _ = events.send(AppEvent::Device { name: name.clone() });
                }
                // Драйверная сводка: итог придёт авторитетным BatchResult.
                EngineEvent::BatchDone { .. } => {}
            }
            on_event(event); // пайплайн: upload → rename → cleanup файла
            if boundary {
                wait_pause(ctrl); // кооперативная пауза на границе файлов
            }
        };
        self.inner.separate_batch(files, output_dir, &mut tee)
    }
}

#[cfg(test)]
#[path = "mocks.rs"]
mod mocks;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedup::wav_fix;
    use crate::pipeline::FileStatus;
    use crate::tui::app::{App, State};
    use mocks::{done_event, error_event, progress_event, MockEngine, MockSmb};
    use std::time::Instant;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

    const EXT: &str = "flac";
    const RUN_TIMEOUT: Duration = Duration::from_secs(30);

    /// Уникальная локальная рабочая директория теста.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hexastem_runner_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn opts(local_dir: &Path, envelope_columns: usize) -> RunnerOptions {
        RunnerOptions {
            remote_dir: PathBuf::from("/nas"),
            local_dir: local_dir.to_path_buf(),
            stem_ext: EXT.to_string(),
            dedup: true,
            pause_between_files: None,
            envelope_columns,
        }
    }

    /// Инфраструктура теста: моки под Arc (доступны для проверок после прогона).
    struct Rig {
        smb: Arc<MockSmb>,
        engine: Arc<MockEngine>,
        opts: RunnerOptions,
    }

    impl Rig {
        fn deps(&self) -> RunnerDeps {
            RunnerDeps { smb: self.smb.clone(), engine: self.engine.clone() }
        }
    }

    /// Сценарий движка: загрузка модели + per-file progress/done + сводка.
    fn engine_events(local: &Path, names: &[&str]) -> Vec<EngineEvent> {
        let mut events = vec![EngineEvent::Progress {
            file: local.join(names[0]).to_string_lossy().into_owned(),
            stage: "load".to_string(),
            pct: 50.0,
            estimated: false,
        }];
        for (i, name) in names.iter().enumerate() {
            events.push(progress_event(local, name, 40.0 + i as f64 * 10.0));
            events.push(done_event(local, name, EXT));
        }
        events.push(EngineEvent::BatchDone { ok: names.len() as u32, failed: 0 });
        events
    }

    /// Собирать события до `stops` терминальных (BatchFinished/RunFailed).
    async fn collect(rx: &mut UnboundedReceiver<AppEvent>, stops: usize) -> Vec<AppEvent> {
        let mut events = Vec::new();
        let mut seen = 0;
        while seen < stops {
            let event = tokio::time::timeout(RUN_TIMEOUT, rx.recv())
                .await
                .expect("runner hung")
                .expect("events channel closed");
            if matches!(event, AppEvent::BatchFinished { .. } | AppEvent::RunFailed { .. }) {
                seen += 1;
            }
            events.push(event);
        }
        events
    }

    /// Полный прогон: события до первого итога + хвост фоновых (огибающие)
    /// + объединённый результат.
    async fn run_full(rig: &Rig) -> (Vec<AppEvent>, Result<BatchResult, AppError>) {
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);
        let mut events = collect(&mut rx, 1).await;
        handle.shutdown();
        let result = handle.finish().await;
        // Фоновые задачи огибающих могут дописать события после итога.
        while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(150), rx.recv()).await {
            events.push(event);
        }
        (events, result)
    }

    fn has_stage(events: &[AppEvent], name: &str, stage: Stage) -> bool {
        events.iter().any(|event| {
            matches!(event, AppEvent::FileStage { name: n, stage: s } if n == name && *s == stage)
        })
    }

    fn has_progress(events: &[AppEvent], name: &str, pct: f64) -> bool {
        events.iter().any(|event| {
            matches!(event, AppEvent::FileProgress { name: n, pct: p, .. } if n == name && (*p - pct).abs() < 1e-9)
        })
    }

    fn has_failed(events: &[AppEvent], name: &str, needle: &str) -> bool {
        events.iter().any(|event| {
            matches!(event, AppEvent::FileFailed { name: n, error } if n == name && error.contains(needle))
        })
    }

    fn result_status<'a>(result: &'a BatchResult, name: &str) -> &'a FileStatus {
        &result.results.iter().find(|r| r.name == name).expect("file in result").status
    }

    // --- download-прогресс (поллер фактического размера) ----------------------

    #[test]
    fn download_pct_measures_local_against_remote_size() {
        let dir = temp_dir("dl_pct");
        let local = dir.join("a.wav");
        assert_eq!(download_pct(&local, 100), None, "файла ещё нет → нет процента");
        std::fs::write(&local, vec![0u8; 50]).unwrap();
        assert_eq!(download_pct(&local, 100), Some(50.0));
        assert_eq!(download_pct(&local, 25), Some(100.0), "перелив клампится в 100");
        assert_eq!(download_pct(&local, 0), None, "нулевой удалённый размер → нет процента");
    }

    #[tokio::test]
    async fn download_poller_sends_rising_pct_while_file_grows() {
        let dir = temp_dir("dl_poller_grow");
        let local = dir.join("a.wav");
        std::fs::write(&local, vec![0u8; 40]).unwrap();
        let sizes: RemoteSizes = [("a.wav".to_string(), 100)].into_iter().collect();
        let downloaded: SharedDownloads = Arc::new(Mutex::new(BTreeMap::new()));
        let (tx, mut rx) = unbounded_channel();
        let poller = tokio::spawn(download_poller_loop(
            sizes,
            dir.clone(),
            downloaded,
            tx,
            Duration::from_millis(20),
        ));
        tokio::time::sleep(Duration::from_millis(60)).await;
        std::fs::write(&local, vec![0u8; 90]).unwrap();
        tokio::time::sleep(Duration::from_millis(60)).await;
        poller.abort();

        let mut pcts = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let AppEvent::FileProgress { name, stage: Stage::Download, pct, estimated } = event {
                assert_eq!(name, "a.wav");
                assert!(!estimated, "download-прогресс — фактический");
                pcts.push(pct);
            }
        }
        assert_eq!(pcts.first(), Some(&40.0), "первый тик: 40/100, события: {pcts:?}");
        assert!(pcts.contains(&90.0), "рост файла виден: {pcts:?}");
        assert!(pcts.windows(2).all(|w| w[0] < w[1]), "монотонно и без спама: {pcts:?}");
    }

    #[tokio::test]
    async fn download_poller_skips_missing_files_and_downloaded() {
        let dir = temp_dir("dl_poller_skip");
        let done_local = dir.join("done.wav");
        std::fs::write(&done_local, b"x").unwrap();
        let downloaded: SharedDownloads = Arc::new(Mutex::new(BTreeMap::new()));
        downloaded
            .lock()
            .expect("downloads lock")
            .insert(done_local.to_string_lossy().into_owned(), "done.wav".to_string());
        let sizes: RemoteSizes = [
            ("missing.wav".to_string(), 100),
            ("done.wav".to_string(), 100),
        ]
        .into_iter()
        .collect();
        let (tx, mut rx) = unbounded_channel();
        let poller = tokio::spawn(download_poller_loop(
            sizes,
            dir.clone(),
            downloaded,
            tx,
            Duration::from_millis(20),
        ));
        tokio::time::sleep(Duration::from_millis(70)).await;
        poller.abort();
        assert!(rx.try_recv().is_err(), "отсутствующий и скачанный файлы молчат");
    }

    // --- полный мок-прогон ---------------------------------------------------

    #[tokio::test]
    async fn full_run_maps_pipeline_to_events_and_app() {
        let dir = temp_dir("full");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav", "b.wav"])),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav", "b.wav"]))),
            opts: opts(&dir, 0),
        };

        let (events, result) = run_full(&rig).await;

        let result = result.expect("batch must succeed");
        assert_eq!((result.total, result.ok, result.failed), (2, 2, 0));
        assert_eq!(exit_code(&result), 0);
        assert!(result.results.iter().all(|r| r.status == FileStatus::Ok));

        let AppEvent::RunStarted { plan } = &events[0] else {
            panic!("первое событие — план прогона, получено {:?}", events[0]);
        };
        assert_eq!(plan.pending, ["a.wav", "b.wav"]);
        assert!(has_stage(&events, "a.wav", Stage::Download));
        assert!(has_stage(&events, "a.wav", Stage::Upload));
        assert!(has_stage(&events, "a.wav", Stage::Rename));
        assert!(has_progress(&events, "a.wav", 40.0), "pct separate из EngineEvent");
        assert!(has_progress(&events, "b.wav", 50.0));
        assert!(events.iter().any(|e| matches!(e, AppEvent::FileDone { name } if name == "b.wav")));
        assert!(matches!(&events[events.len() - 1], AppEvent::BatchFinished { outcome } if outcome.ok == 2));
        // rename исходников после Ok — инвариант пайплайна.
        let calls = rig.smb.calls();
        assert!(calls.iter().any(|c| c.contains("rename /nas/a.wav -> /nas/CONVERTED_a.wav")));

        // События доходят до App: автомат в Finished со сводкой.
        let mut app = App::new();
        for event in &events {
            app.handle(event);
        }
        assert_eq!(app.state(), State::Finished);
        assert_eq!(app.outcome().map(|outcome| outcome.ok), Some(2));
    }

    #[tokio::test]
    async fn engine_done_reports_stem_count_to_ui() {
        let dir = temp_dir("stems");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav"]))),
            opts: opts(&dir, 0),
        };

        let (events, result) = run_full(&rig).await;

        result.expect("батч успешен");
        assert!(
            events.iter().any(|event| matches!(
                event,
                AppEvent::FileStems { name, count: 6 } if name == "a.wav"
            )),
            "список стемов движка не теряется в маппинге: {events:?}"
        );
        // Живой счётчик доезжает до App раньше терминального Ok.
        let mut app = App::new();
        for event in &events {
            app.handle(event);
        }
        assert_eq!(app.files()[0].stems_done, Some(6), "колонка STEMS честна в Finished");
    }

    /// Событие `device` проходит сквозь tee в канал UI (шапка TUI).
    #[tokio::test]
    async fn device_event_is_forwarded_to_ui_channel() {
        let dir = temp_dir("device");
        let mut engine = engine_events(&dir, &["a.wav"]);
        engine.insert(1, EngineEvent::Device { name: "NVIDIA GeForce RTX 4090".into() });
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(MockEngine::new(engine)),
            opts: opts(&dir, 0),
        };

        let (events, result) = run_full(&rig).await;

        result.expect("батч успешен");
        assert!(
            events.iter().any(|event| matches!(
                event,
                AppEvent::Device { name } if name == "NVIDIA GeForce RTX 4090"
            )),
            "device доходит до UI: {events:?}"
        );
        // Шапка App обновляется тем же потоком событий прогона.
        let mut app = App::new();
        for event in &events {
            app.handle(event);
        }
        assert_eq!(app.gpu_device_name(), Some("NVIDIA GeForce RTX 4090"));
    }

    #[tokio::test]
    async fn download_failure_fails_file_and_batch_continues() {
        let dir = temp_dir("dlfail");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav", "b.wav"]).failing_get("/nas/b.wav")),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav"]))),
            opts: opts(&dir, 0),
        };

        let (events, result) = run_full(&rig).await;

        let result = result.expect("батч с отказом файла — не батчевая ошибка");
        assert_eq!((result.ok, result.failed), (1, 1));
        assert_eq!(exit_code(&result), 1);
        assert_eq!(*result_status(&result, "b.wav"), FileStatus::Failed);
        assert!(has_failed(&events, "b.wav", "forced failure"));
        assert_eq!(rig.engine.called_batches(), vec![vec![dir.join("a.wav")]], "движку ушёл только успешный");
    }

    #[tokio::test]
    async fn duplicate_pair_is_announced_and_renamed_after_keeper() {
        let dir = temp_dir("dup");
        // Байт-в-байт одинаковые тела: гарантированный кластер дедупликации.
        let wav = wav_fix::encode_wav(8_000, 16, 1, &wav_fix::burst_signal(8_000, 2));
        let rig = Rig {
            smb: Arc::new(
                MockSmb::new()
                    .with_dir("/nas", &["keeper.wav", "copy.wav"])
                    .with_body("/nas/keeper.wav", wav.clone())
                    .with_body("/nas/copy.wav", wav),
            ),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["keeper.wav"]))),
            opts: opts(&dir, 0),
        };

        let (events, result) = run_full(&rig).await;

        let result = result.expect("batch ok");
        assert_eq!((result.total, result.ok, result.failed), (2, 1, 0));
        assert_eq!(exit_code(&result), 0, "дубли не влияют на exit-code");
        assert_eq!(
            *result_status(&result, "copy.wav"),
            FileStatus::Duplicate { kept: "keeper.wav".to_string() }
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, AppEvent::FileDuplicate { name } if name == "copy.wav")));
        assert!(!has_stage(&events, "copy.wav", Stage::Separate), "дубль движку не уходил");
        assert!(rig
            .smb
            .calls()
            .iter()
            .any(|call| call.contains("rename /nas/copy.wav -> /nas/DUPLICATE_copy.wav")));
    }

    #[tokio::test]
    async fn duplicates_known_announced_after_per_file_events_before_separate() {
        let dir = temp_dir("dup_known");
        let wav = wav_fix::encode_wav(8_000, 16, 1, &wav_fix::burst_signal(8_000, 2));
        let rig = Rig {
            smb: Arc::new(
                MockSmb::new()
                    .with_dir("/nas", &["keeper.wav", "copy.wav"])
                    .with_body("/nas/keeper.wav", wav.clone())
                    .with_body("/nas/copy.wav", wav),
            ),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["keeper.wav"]))),
            opts: opts(&dir, 0),
        };

        let (events, _) = run_full(&rig).await;

        let dup_pos = events
            .iter()
            .position(|event| matches!(event, AppEvent::FileDuplicate { name } if name == "copy.wav"))
            .expect("пер-файловое событие дубля");
        let known_pos = events
            .iter()
            .position(|event| matches!(event, AppEvent::DuplicatesKnown { names } if names.as_slice() == ["copy.wav"]))
            .expect("пачка дублей после пер-файловых");
        assert!(known_pos > dup_pos, "пачка идёт после пер-файловых событий");
        // Дубли известны ДО первого separate-прогресса: строки не показывают
        // «мёртвый» download 0%.
        if let Some(separate_pos) = events
            .iter()
            .position(|event| matches!(event, AppEvent::FileProgress { .. }))
        {
            assert!(known_pos < separate_pos, "дубли распознаны до separate");
        }
    }

    #[tokio::test]
    async fn force_kill_unblocks_engine_and_finish_resolves_quickly() {
        let dir = temp_dir("force_kill");
        // Движок «висит» 30 c: без форс-kill ожидание не прервать.
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(
                MockEngine::new(engine_events(&dir, &["a.wav"])).with_delay_before(Duration::from_secs(30)),
            ),
            opts: opts(&dir, 0),
        };
        let (tx, _rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);

        // Ждём старта батча движка (окно для kill), затем дёргаем форс —
        // как App.request_force_kill → sync_run_controls → handle.force_kill.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !rig.engine.started() && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(rig.engine.started(), "движок не стартовал — тест некорректен");

        let started = std::time::Instant::now();
        // Как sync_run_controls: cancel_requested → cancel(), take_force_kill → force_kill().
        handle.cancel();
        handle.force_kill();
        let result = tokio::time::timeout(Duration::from_secs(5), handle.finish())
            .await
            .expect("finish завис после форс-kill");

        assert!(rig.engine.kill_flag().load(std::sync::atomic::Ordering::Relaxed), "kill-флаг не взведён");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "форс-kill обязан завершить ожидание быстро"
        );
        let result = result.expect("раннер завершился с итогом батча (ошибка движка = failed-файл)");
        assert_eq!(result.failed, 1);
        let error = result.results[0].error.clone().unwrap_or_default();
        assert!(error.contains("engine killed"), "неожиданная ошибка: {error}");
    }

    #[tokio::test]
    async fn engine_error_fails_file_with_exit_code() {
        let dir = temp_dir("engerr");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(MockEngine::new(vec![error_event(&dir, "a.wav", "CUDA OOM")])),
            opts: opts(&dir, 0),
        };

        let (events, result) = run_full(&rig).await;

        let result = result.expect("батч завершён, файл отказал");
        assert_eq!((result.ok, result.failed), (0, 1));
        assert_eq!(exit_code(&result), 1);
        assert!(has_failed(&events, "a.wav", "CUDA OOM"));
    }

    // --- пауза -----------------------------------------------------------------

    #[tokio::test]
    async fn pause_between_files_delayes_completion() {
        let dir = temp_dir("pause");
        let mut options = opts(&dir, 0);
        options.pause_between_files = Some(Duration::from_millis(300));
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav", "b.wav"])),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav", "b.wav"]))),
            opts: options,
        };

        let started = Instant::now();
        let (events, result) = run_full(&rig).await;

        let result = result.expect("batch ok");
        assert_eq!(result.ok, 2);
        assert!(
            started.elapsed() >= Duration::from_millis(280),
            "пауза применена на границах файлов: {:?}",
            started.elapsed()
        );
        assert!(has_stage(&events, "b.wav", Stage::Rename));
    }

    // --- отмена ------------------------------------------------------------------

    #[tokio::test]
    async fn cancel_before_start_fails_scan() {
        let dir = temp_dir("cancel0");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav"]))),
            opts: opts(&dir, 0),
        };
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);

        handle.cancel();
        let events = collect(&mut rx, 1).await;
        handle.shutdown();
        let result = handle.finish().await;

        assert!(matches!(&events[events.len() - 1], AppEvent::RunFailed { error } if error.contains("cancelled")));
        assert!(result.is_err(), "прогона не было — итога нет");
    }

    #[tokio::test]
    async fn cancel_during_download_stops_remaining_files() {
        let dir = temp_dir("canceldl");
        let rig = Rig {
            smb: Arc::new(
                MockSmb::new()
                    .with_dir("/nas", &["a.wav", "b.wav"])
                    .with_get_delay(Duration::from_millis(150)),
            ),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav"]))),
            opts: opts(&dir, 0),
        };
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);

        // Ждём старта download первого файла и отменяем в его окне (150 мс).
        let mut events = Vec::new();
        loop {
            let event = tokio::time::timeout(RUN_TIMEOUT, rx.recv()).await.expect("hung").expect("closed");
            let started_first =
                matches!(&event, AppEvent::FileStage { name, stage: Stage::Download } if name == "a.wav");
            events.push(event);
            if started_first {
                break;
            }
        }
        handle.cancel();
        events.extend(collect(&mut rx, 1).await);
        handle.shutdown();
        let result = handle.finish().await.expect("батч завершается с частичным отказом");

        // Семантика отмены «стоп новых стадий»: b не начал download, a скачан,
        // но батч движка стартует после отмены — оба файла Failed, исходники
        // на NAS нетронуты, resume подхватит.
        assert_eq!((result.ok, result.failed), (0, 2));
        assert!(has_failed(&events, "b.wav", "cancelled"), "оставшийся download отменён");
        assert!(has_failed(&events, "a.wav", "cancelled"), "батч движка не стартует после отмены");
        assert_eq!(*result_status(&result, "a.wav"), FileStatus::Failed);
        assert!(!rig.smb.calls().iter().any(|call| call.starts_with("rename")));
    }

    #[tokio::test]
    async fn cancel_during_engine_fails_file_at_upload_boundary() {
        let dir = temp_dir("canceleng");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(
                MockEngine::new(engine_events(&dir, &["a.wav"])).with_delay_before(Duration::from_millis(400)),
            ),
            opts: opts(&dir, 0),
        };
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);

        tokio::time::sleep(Duration::from_millis(50)).await; //Separate уже идёт
        handle.cancel();
        let mut events = collect(&mut rx, 1).await;
        handle.shutdown();
        let result = handle.finish().await.expect("батч завершён");
        while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await {
            events.push(event);
        }

        assert_eq!((result.ok, result.failed), (0, 1));
        assert!(has_failed(&events, "a.wav", "cancelled"), "заливка отменена на границе стадии");
        assert!(
            !rig.smb.calls().iter().any(|call| call.starts_with("rename")),
            "инвариант: исходник не переименован, resume подхватит"
        );
        assert!(!has_stage(&events, "a.wav", Stage::Rename));
    }

    // --- retry ---------------------------------------------------------------------

    #[tokio::test]
    async fn retry_failed_file_reruns_only_it_and_merges() {
        let dir = temp_dir("retry");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav", "b.wav"])),
            engine: Arc::new(
                MockEngine::new(engine_events(&dir, &["a.wav", "b.wav"])).failing_file_once("b.wav"),
            ),
            opts: opts(&dir, 0),
        };
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);

        // Два итога: первый прогон (b упал) и retry (b успешен).
        let mut events = Vec::new();
        let mut batches = 0;
        while batches < 2 {
            let event = tokio::time::timeout(RUN_TIMEOUT, rx.recv()).await.expect("hung").expect("closed");
            if matches!(&event, AppEvent::BatchFinished { .. }) {
                batches += 1;
                if batches == 1 {
                    handle.request_retry("b.wav");
                }
            }
            events.push(event);
        }
        handle.shutdown();
        let result = handle.finish().await.expect("итог серии");
        while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await {
            events.push(event);
        }

        assert_eq!((result.ok, result.failed), (2, 0), "retry замещает отказ");
        assert_eq!(exit_code(&result), 0);
        assert_eq!(
            events.iter().filter(|event| matches!(event, AppEvent::RunStarted { .. })).count(),
            1,
            "retry не пересевает таблицу"
        );
        assert_eq!(
            rig.engine.called_batches().last().map(Vec::as_slice),
            Some(&[dir.join("b.wav")][..]),
            "повторно идёт только упавший файл"
        );
        assert!(events
            .iter()
            .any(|event| matches!(event, AppEvent::FileDone { name } if name == "b.wav")));
    }

    #[tokio::test]
    async fn retry_of_unknown_file_is_ignored() {
        let dir = temp_dir("retry_unknown");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav"]))),
            opts: opts(&dir, 0),
        };
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);
        handle.request_retry("nope.wav");

        let events = collect(&mut rx, 1).await;
        handle.shutdown();
        let result = handle.finish().await.expect("итог");

        assert_eq!(result.ok, 1);
        assert_eq!(
            events.iter().filter(|event| matches!(event, AppEvent::BatchFinished { .. })).count(),
            1,
            "повторного прогона нет"
        );
    }

    #[tokio::test]
    async fn cancel_skips_queued_retries() {
        let dir = temp_dir("cancelretry");
        let rig = Rig {
            smb: Arc::new(MockSmb::new().with_dir("/nas", &["a.wav"])),
            engine: Arc::new(MockEngine::new(vec![error_event(&dir, "a.wav", "boom")])),
            opts: opts(&dir, 0),
        };
        let (tx, mut rx) = unbounded_channel();
        let handle = spawn(rig.deps(), rig.opts.clone(), tx);

        let events = collect(&mut rx, 1).await;
        handle.cancel();
        handle.request_retry("a.wav");
        handle.shutdown();
        let result = handle.finish().await.expect("итог");

        assert_eq!(result.failed, 1, "cancel снимает очередь retry");
        assert_eq!(
            events.iter().filter(|event| matches!(event, AppEvent::BatchFinished { .. })).count(),
            1
        );
    }

    // --- огибающая --------------------------------------------------------------------

    #[tokio::test]
    async fn envelope_computed_exactly_once_after_download() {
        let dir = temp_dir("envelope");
        let rate = 8_000_u32;
        let samples: Vec<f32> = (0..rate * 2)
            .map(|i| 0.6 * (i as f64 / f64::from(rate) * 220.0).sin() as f32)
            .collect();
        let rig = Rig {
            smb: Arc::new(
                MockSmb::new()
                    .with_dir("/nas", &["a.wav"])
                    .with_body("/nas/a.wav", wav_fix::encode_wav(rate, 16, 1, &samples)),
            ),
            engine: Arc::new(MockEngine::new(engine_events(&dir, &["a.wav"]))),
            opts: opts(&dir, 8),
        };

        let (events, result) = run_full(&rig).await;

        result.expect("batch ok");
        let envelopes: Vec<&AppEvent> =
            events.iter().filter(|event| matches!(event, AppEvent::Envelope { .. })).collect();
        assert_eq!(envelopes.len(), 1, "огибающая считается ровно 1 раз на файл");
        let Some(AppEvent::Envelope { path, envelope }) = envelopes.first() else {
            panic!("checked above");
        };
        assert!(path.to_string_lossy().ends_with("a.wav"));
        assert_eq!(envelope.left.len(), 8, "число колонок == columns");
        assert_eq!(envelope.left, envelope.right, "моно-файл — оба канала одинаковы");
        assert!(
            envelope.left.iter().all(|p| *p > 0.55 && *p <= 0.65),
            "пик колонки ≈ амплитуде: {:?}",
            envelope.left
        );
    }
}
