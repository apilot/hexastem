//! Оркестратор батча: план (скан/фильтр/resume) и цикл per-file.
//!
//! Сценарий [`Pipeline::run`]:
//! 1. листинг SMB-директории → [`plan`]: только `*.wav`, скип `CONVERTED_*`
//!    и `DUPLICATE_*`, resume-скип (все 6 стемов уже на NAS = файл обработан);
//! 2. download всех pending в локальную рабочую директорию (отказ файла —
//!    в отчёт, батч продолжается);
//! 3. дедупликация (см. `dedup`): файлы с одинаковыми (data_size, frames)
//!    попарно сравниваются; дубли снимаются с обработки, их rename на NAS —
//!    строго после `Ok` хранителя;
//! 4. ОДИН вызов движка на батч хранителей и уникальных («model hot», см.
//!    python/separator_driver.py): JSON-события драйвера читаются построчно;
//!    по `file_done` — upload стемов → rename `CONVERTED_*` → локальный
//!    cleanup;
//! 5. инвариант: rename ТОЛЬКО после успешной заливки всех 6 стемов — любой
//!    сбой раньше оставляет исходник на NAS нетронутым;
//! 6. итог — [`BatchResult`] (ok/failed/duplicate по каждому файлу).
//!
//! DI: SMB и движок инжектятся трейтами [`SmbOps`]/[`EngineOps`] — тесты идут
//! на моках, без сети и GPU. Таймаут/kill драйвера ТЗ не предусмотрен: отказы
//! изолирует сам драйвер (per-file retry), зависание процесса означает
//! зависание батча (принятый KISS-компромисс).

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{ChildStderr, ChildStdout, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::dedup::{self, DedupSummary, FileAnalysis, KeeperReason, PairReport, PairVerdict};
use crate::error::AppError;
use crate::smb::{SmbClient, SmbEntry};

/// Префикс переименованного (обработанного) исходника на NAS.
pub const CONVERTED_PREFIX: &str = "CONVERTED_";

/// Префикс переименованного дубля на NAS (хранитель обработан вместо него).
pub const DUPLICATE_PREFIX: &str = "DUPLICATE_";

/// Стемы пресета — должны совпадать с EXPECTED_STEMS драйвера
/// (python/separator_driver.py).
pub const EXPECTED_STEMS: [&str; 6] = ["vocals", "drums", "bass", "guitar", "piano", "other"];

/// Ожидимое число стемов на файл (инвариант rename).
pub const EXPECTED_STEM_COUNT: usize = EXPECTED_STEMS.len();

/// Хвост stderr движка в тексте ошибки (диагностика без засорения логов).
const STDERR_TAIL_CHARS: usize = 300;

// --- протокол драйвера ---------------------------------------------------------

/// Событие JSON-протокола драйвера: одна строка stdout = одно событие.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EngineEvent {
    /// Прогресс (загрузка модели / разделение файла).
    Progress {
        /// Локальный путь файла.
        file: String,
        /// Стадия: `load` | `separate`.
        stage: String,
        /// Процент 0..100.
        pct: f64,
    },
    /// Файл разделён: локальные пути стемов.
    FileDone {
        /// Локальный путь файла.
        file: String,
        /// Пути стемов на диске.
        stems: Vec<String>,
        /// Секунды разделения.
        seconds: f64,
    },
    /// Файл не разделён (после ретраев драйвера).
    FileError {
        /// Локальный путь файла.
        file: String,
        /// Текст последней ошибки.
        error: String,
    },
    /// Финальная сводка батча драйвера.
    BatchDone {
        /// Успешных файлов.
        ok: u32,
        /// Неудачных файлов.
        failed: u32,
    },
}

/// Разбор строки stdout драйвера; `None` — пустая/не протокольная строка.
/// Решающие сигналы — `file_done`/`file_error` и код выхода, поэтому мусорная
/// строка не фатальна (логи библиотеки идут в stderr и сюда не попадают).
pub fn parse_event(line: &str) -> Option<EngineEvent> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

// --- DI-границы ----------------------------------------------------------------

/// SMB-операции, нужные пайплайну (продакшн — [`SmbClient`], тесты — мок).
pub trait SmbOps {
    /// Листинг директории на шаре.
    fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError>;
    /// Скачивание удалённого файла локально.
    fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError>;
    /// Заливка локального файла на шару.
    fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError>;
    /// Создание директории на шаре (идемпотентно: уже существующая = Ok).
    /// Обязателен перед первой заливкой: `put` промежуточные директории
    /// не создаёт.
    fn create_dir(&self, dir: &Path) -> Result<(), AppError>;
    /// Переименование на шаре.
    fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError>;
    /// Существует ли путь на шаре.
    fn remote_exists(&self, path: &Path) -> Result<bool, AppError>;
}

impl SmbOps for SmbClient {
    fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        SmbClient::list(self, dir)
    }

    fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
        SmbClient::get(self, remote, local)
    }

    fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
        SmbClient::put(self, local, remote)
    }

    fn create_dir(&self, dir: &Path) -> Result<(), AppError> {
        SmbClient::create_dir(self, dir)
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
        SmbClient::rename(self, from, to)
    }

    fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
        SmbClient::remote_exists(self, path)
    }
}

/// Движок разделения (продакшн — [`EngineRunner`], тесты — мок).
pub trait EngineOps {
    /// Разделяет пакет локальных файлов одной загрузкой модели; события
    /// протокола отдаются в `on_event` по мере поступления.
    fn separate_batch(
        &self,
        files: &[PathBuf],
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(), AppError>;
}

/// Rust-адаптер python-драйвера: subprocess `{venv}/bin/python separator_driver.py`.
#[derive(Debug, Clone)]
pub struct EngineRunner {
    /// Интерпретатор venv (`{venv}/bin/python`).
    python: PathBuf,
    /// Путь к python/separator_driver.py.
    driver: PathBuf,
    /// Чекпойнт модели (см. `Model::checkpoint`).
    model_checkpoint: String,
    /// Формат стемов (напр. `flac`).
    output_format: String,
}

impl EngineRunner {
    /// Адаптер драйвера из путей venv и скрипта.
    pub fn new(venv_path: &Path, driver_script: &Path, model_checkpoint: &str, output_format: &str) -> Self {
        Self {
            python: venv_path.join("bin/python"),
            driver: driver_script.to_path_buf(),
            model_checkpoint: model_checkpoint.to_string(),
            output_format: output_format.to_string(),
        }
    }

    /// Команда запуска драйвера (отдельно — для юнит-теста аргументов).
    fn build_command(&self, files: &[PathBuf], output_dir: &Path) -> Command {
        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.driver)
            .arg("--model-file")
            .arg(&self.model_checkpoint)
            .arg("--output-dir")
            .arg(output_dir)
            .arg("--output-format")
            .arg(&self.output_format)
            .arg("--files")
            .args(files)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd
    }
}

impl EngineOps for EngineRunner {
    fn separate_batch(
        &self,
        files: &[PathBuf],
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(), AppError> {
        if files.is_empty() {
            return Ok(());
        }
        let mut child = self
            .build_command(files, output_dir)
            .spawn()
            .map_err(|err| AppError::Engine {
                file: files[0].clone(),
                exit_code: None,
                stderr_tail: format!("spawn failed: {err}"),
            })?;
        let stderr_tail = read_stderr(child.stderr.take());
        let saw_summary = relay_stdout(child.stdout.take(), &files[0], on_event)?;
        let status = child
            .wait()
            .map_err(|err| AppError::Io { path: self.python.clone(), source: err })?;
        evaluate_exit(status, saw_summary, stderr_tail, &files[0])
    }
}

/// Читает stderr драйвера в фоне; возвращает хвост для текста ошибки
/// (отдельный поток — иначе переполненный пайп stderr заблокирует stdout).
fn read_stderr(stderr: Option<ChildStderr>) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut pipe) = stderr {
            let _ = pipe.read_to_string(&mut text); // диагностика, не критично
        }
        tail_chars(&text, STDERR_TAIL_CHARS)
    })
}

/// Потоковое чтение stdout построчно → события в колбэк; возвращает признак
/// того, что финальная сводка (`batch_done`) была получена.
fn relay_stdout(
    stdout: Option<ChildStdout>,
    file: &Path,
    on_event: &mut dyn FnMut(EngineEvent),
) -> Result<bool, AppError> {
    let stdout = stdout.ok_or_else(|| AppError::Pipeline("engine stdout was not captured".to_string()))?;
    let mut saw_summary = false;
    for line in BufReader::new(stdout).lines() {
        let line = line.map_err(|err| engine_io_error(file, err))?;
        if let Some(event) = parse_event(&line) {
            saw_summary |= matches!(event, EngineEvent::BatchDone { .. });
            on_event(event);
        }
    }
    Ok(saw_summary)
}

/// Код выхода: 0 либо полученная сводка — батч завершён (пер-файловые отказы
/// уже в событиях); иначе драйвер рухнул — батчевая ошибка.
fn evaluate_exit(
    status: ExitStatus,
    saw_summary: bool,
    stderr_tail: JoinHandle<String>,
    file: &Path,
) -> Result<(), AppError> {
    if status.success() || saw_summary {
        return Ok(());
    }
    Err(AppError::Engine {
        file: file.to_path_buf(),
        exit_code: status.code(),
        stderr_tail: stderr_tail.join().unwrap_or_default(),
    })
}

fn engine_io_error(file: &Path, err: std::io::Error) -> AppError {
    AppError::Engine {
        file: file.to_path_buf(),
        exit_code: None,
        stderr_tail: format!("stdout read failed: {err}"),
    }
}

/// Последние `max` символов без разрыва границ символов.
fn tail_chars(text: &str, max: usize) -> String {
    let skip = text.chars().count().saturating_sub(max);
    text.chars().skip(skip).collect()
}

// --- план батча ------------------------------------------------------------------

/// План батча: что обрабатывать и что пропущено (и почему).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchPlan {
    /// Файлы к обработке (download → separate → upload → rename).
    pub pending: Vec<SmbEntry>,
    /// Пропущены: уже обработаны ранее (префикс `CONVERTED_`).
    pub skipped_converted: Vec<String>,
    /// Пропущены: ранее признаны дублями (префикс `DUPLICATE_`).
    pub skipped_duplicate: Vec<String>,
    /// Пропущены: resume — все 6 стемов уже на NAS.
    pub skipped_done: Vec<String>,
}

/// План батча из листинга SMB-директории: только `*.wav`, скип `CONVERTED_*`
/// и `DUPLICATE_*`, скип уже обработанного (resume: у файла есть
/// `<basename>/` со всеми 6 стемами). Проверка resume идёт через
/// инжектируемый `smb` — тестируется на моке.
pub fn plan(entries: &[SmbEntry], smb: &dyn SmbOps, remote_dir: &Path, stem_ext: &str) -> Result<BatchPlan, AppError> {
    let mut batch = BatchPlan::default();
    for entry in entries.iter().filter(|e| e.is_wav()) {
        if entry.name.starts_with(CONVERTED_PREFIX) {
            batch.skipped_converted.push(entry.name.clone());
        } else if entry.name.starts_with(DUPLICATE_PREFIX) {
            batch.skipped_duplicate.push(entry.name.clone());
        } else if resume_done(smb, remote_dir, stem_dir_name(&entry.name), stem_ext)? {
            batch.skipped_done.push(entry.name.clone());
        } else {
            batch.pending.push(entry.clone());
        }
    }
    Ok(batch)
}

/// Resume: на NAS есть директория `<basename>/` со всеми 6 стемами.
/// Отсутствие директории — нормальный «не обработан», а не ошибка.
fn resume_done(smb: &dyn SmbOps, remote_dir: &Path, basename: &str, stem_ext: &str) -> Result<bool, AppError> {
    let stems_dir = remote_dir.join(basename);
    if !smb.remote_exists(&stems_dir)? {
        return Ok(false);
    }
    Ok(has_all_stems(&smb.list(&stems_dir)?, stem_ext))
}

/// Все 6 ожидаемых стемов `<stem>.<ext>` есть в листинге (регистр не важен).
fn has_all_stems(entries: &[SmbEntry], stem_ext: &str) -> bool {
    EXPECTED_STEMS.iter().all(|stem| {
        let expected = format!("{stem}.{stem_ext}");
        entries.iter().any(|e| !e.is_dir && e.name.eq_ignore_ascii_case(&expected))
    })
}

// --- результаты ------------------------------------------------------------------

/// Исход одной единицы работы (файла).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    /// Полный цикл прошёл: стемы на NAS, исходник переименован.
    Ok,
    /// Отказ (зафиксирован в отчёте, батч продолжился).
    Failed,
    /// Дубль-запись: снят с обработки, вместо него обработан хранитель
    /// `kept`. На exit code не влияет; rename `DUPLICATE_*` — после `Ok`
    /// хранителя.
    Duplicate {
        /// Имя файла-хранителя.
        kept: String,
    },
}

/// Результат обработки одного файла.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileResult {
    /// Имя файла на шаре.
    pub name: String,
    /// Статус.
    pub status: FileStatus,
    /// Текст ошибки (для `Failed`).
    pub error: Option<String>,
    /// Потраченное время (стадия, контролируемая пайплайном).
    pub elapsed: Duration,
}

impl FileResult {
    fn failed(name: impl Into<String>, error: impl Into<String>, elapsed: Duration) -> Self {
        Self { name: name.into(), status: FileStatus::Failed, error: Some(error.into()), elapsed }
    }
}

/// Итог батча: счётчики + пофайловые результаты + тайминг + дедупликация.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Всего обработано (без скипнутых; включая снятые дубли).
    pub total: usize,
    /// Успешно.
    pub ok: usize,
    /// С ошибкой.
    pub failed: usize,
    /// Пофайловые результаты.
    pub results: Vec<FileResult>,
    /// Итог стадии дедупликации (для отчёта).
    pub dedup: DedupSummary,
    /// Момент старта батча.
    pub started: Instant,
    /// Общая длительность.
    pub elapsed: Duration,
}

/// Чистая агрегация результатов (`started` — момент старта батча).
pub fn summarize(results: Vec<FileResult>, started: Instant) -> BatchResult {
    summarize_with_dedup(results, DedupSummary::default(), started)
}

/// Агрегация с итогом дедупликации. Дубли — не отказ: `failed` считает
/// только [`FileStatus::Failed`], exit code дедупликация не меняет.
pub fn summarize_with_dedup(results: Vec<FileResult>, dedup: DedupSummary, started: Instant) -> BatchResult {
    let ok = results.iter().filter(|r| r.status == FileStatus::Ok).count();
    let failed = results.iter().filter(|r| r.status == FileStatus::Failed).count();
    BatchResult {
        total: results.len(),
        ok,
        failed,
        results,
        dedup,
        started,
        elapsed: started.elapsed(),
    }
}

// --- пайплайн --------------------------------------------------------------------

/// Пайплайн батча. SMB и движок инжектятся как трейт-объекты — без
/// глобального состояния; тесты подставляют моки.
pub struct Pipeline<'a> {
    /// Операции SMB.
    pub smb: &'a dyn SmbOps,
    /// Движок разделения.
    pub engine: &'a dyn EngineOps,
    /// Исходная директория на шаре.
    pub remote_dir: PathBuf,
    /// Локальная рабочая директория (загрузка исходников и стемов).
    pub local_dir: PathBuf,
    /// Расширение стемов без точки (напр. `flac`) — для resume-проверки.
    pub stem_ext: String,
    /// Дедупликация дублей включена (отключается флагом `--no-dedup`).
    pub dedup: bool,
}

impl<'a> Pipeline<'a> {
    /// Собранный пайплайн (дедупликация включена по умолчанию).
    pub fn new(
        smb: &'a dyn SmbOps,
        engine: &'a dyn EngineOps,
        remote_dir: PathBuf,
        local_dir: PathBuf,
        stem_ext: &str,
    ) -> Self {
        Self { smb, engine, remote_dir, local_dir, stem_ext: stem_ext.to_string(), dedup: true }
    }

    /// Включить/выключить дедупликацию (флаг `--no-dedup` выключает).
    pub fn with_dedup(mut self, enabled: bool) -> Self {
        self.dedup = enabled;
        self
    }

    /// Полный прогон: план → download → separate → finalize → агрегат.
    /// Отказ одного файла не останавливает батч (копится в результатах).
    pub fn run(&self) -> Result<BatchResult, AppError> {
        let entries = self.smb.list(&self.remote_dir)?;
        let plan = plan(&entries, self.smb, &self.remote_dir, &self.stem_ext)?;
        self.execute(&plan)
    }

    /// Исполнение ГОТОВОГО плана (download → dedup → separate → rename дублей
    /// → агрегат). Отделено от [`Pipeline::run`], чтобы интеграция (main)
    /// построила план сама и положила скипы в отчёт без повторного скана.
    pub fn execute(&self, plan: &BatchPlan) -> Result<BatchResult, AppError> {
        let started = Instant::now();
        let (downloaded, mut results) = self.download_all(&plan.pending);
        let pass = self.dedup_pass(&downloaded, &mut results);
        self.separate_and_finalize(&pass.worklist, &mut results);
        let pairs = self.finalize_duplicates(&pass, &results);
        Ok(summarize_with_dedup(
            results,
            DedupSummary { pairs, unparseable: pass.unparseable },
            started,
        ))
    }

    /// Скачивает pending-файлы; отказ скачивания — результат `Failed`,
    /// файл исключается из батча движка.
    fn download_all(&self, pending: &[SmbEntry]) -> (Vec<Downloaded>, Vec<FileResult>) {
        let mut downloaded = Vec::new();
        let mut failed = Vec::new();
        for entry in pending {
            let started = Instant::now();
            let local = self.local_dir.join(&entry.name);
            match self.smb.get(&self.remote_dir.join(&entry.name), &local) {
                Ok(()) => downloaded.push(Downloaded { name: entry.name.clone(), local }),
                Err(err) => failed.push(FileResult::failed(entry.name.clone(), err.to_string(), started.elapsed())),
            }
        }
        (downloaded, failed)
    }

    /// Один вызов движка на батч + пост-обработка по событиям; файлы без
    /// события завершения (крах драйвера) помечаются отказом.
    fn separate_and_finalize(&self, downloaded: &[Downloaded], results: &mut Vec<FileResult>) {
        if downloaded.is_empty() {
            return;
        }
        let files: Vec<PathBuf> = downloaded.iter().map(|d| d.local.clone()).collect();
        let mut done: HashMap<String, FileResult> = HashMap::new();
        let engine_error = self.run_engine(&files, &mut done);
        for entry in downloaded {
            match done.remove(&entry.key()) {
                Some(result) => results.push(result),
                None => {
                    let error = engine_error
                        .clone()
                        .unwrap_or_else(|| "engine produced no completion event".to_string());
                    results.push(FileResult::failed(entry.name.clone(), error, Duration::ZERO));
                }
            }
        }
    }

    /// Прогон движка с разбором событий; возвращает текст батчевой ошибки
    /// (spawn/крах), если движок отчитался ею.
    fn run_engine(&self, files: &[PathBuf], done: &mut HashMap<String, FileResult>) -> Option<String> {
        let mut on_event = |event: EngineEvent| match event {
            EngineEvent::Progress { file, stage, pct } => eprintln!("[engine] {file}: {stage} {pct:.1}%"),
            EngineEvent::BatchDone { ok, failed } => eprintln!("[engine] batch done: ok={ok} failed={failed}"),
            EngineEvent::FileDone { file, stems, .. } => {
                let result = self.finalize_separated(&file, &stems);
                done.insert(file, result);
            }
            EngineEvent::FileError { file, error } => {
                let name = file_name_of(&file).to_string();
                done.insert(file, FileResult::failed(name, error, Duration::ZERO));
            }
        };
        self.engine.separate_batch(files, &self.local_dir, &mut on_event).err().map(|err| err.to_string())
    }

    /// Пост-обработка разделённого файла: upload → rename → cleanup.
    fn finalize_separated(&self, local_file: &str, stems: &[String]) -> FileResult {
        let started = Instant::now();
        let name = file_name_of(local_file);
        match self.upload_rename_cleanup(name, local_file, stems) {
            Ok(()) => FileResult {
                name: name.to_string(),
                status: FileStatus::Ok,
                error: None,
                elapsed: started.elapsed(),
            },
            Err(error) => FileResult::failed(name, error, started.elapsed()),
        }
    }

    /// Заливка всех стемов → rename `CONVERTED_*` → локальный cleanup.
    /// Инвариант: rename ТОЛЬКО после успешной заливки всех 6 стемов —
    /// любой сбой раньше оставляет исходник на NAS нетронутым.
    fn upload_rename_cleanup(&self, name: &str, local_file: &str, stems: &[String]) -> Result<(), String> {
        let stem_dir = self.remote_dir.join(stem_dir_name(name));
        let uploads = stem_uploads(stems, &stem_dir)?;
        // smbclient `put` не создаёт промежуточных директорий: без этого на
        // реальном NAS каждая заливка падает с NT_STATUS_OBJECT_PATH_NOT_FOUND.
        // Идемпотентно, строго до первой заливки; сбой → файл Failed, стеммы
        // не заливаются, исходник не переименовывается.
        self.smb
            .create_dir(&stem_dir)
            .map_err(|err| format!("create dir `{}`: {err}", stem_dir.display()))?;
        for (local, remote) in uploads {
            self.smb.put(&local, &remote).map_err(|err| format!("upload `{}`: {err}", remote.display()))?;
        }
        let converted = self.remote_dir.join(converted_name(name));
        self.smb
            .rename(&self.remote_dir.join(name), &converted)
            .map_err(|err| format!("rename `{name}`: {err}"))?;
        self.cleanup_local(name, local_file);
        Ok(())
    }

    /// Best-effort очистка: после успешной заливки и rename остатки лишь
    /// занимают диск и не могут испортить уже завершённый файл.
    fn cleanup_local(&self, name: &str, local_file: &str) {
        let _ = std::fs::remove_file(local_file);
        // Локальная раскладка — за драйвером (сырой stem, без trim):
        // удаляем ровно то, что создал он.
        let _ = std::fs::remove_dir_all(self.local_dir.join(stem_of(name)));
    }
}

/// Успешно скачанный файл: имя на шаре + локальный путь.
#[derive(Clone)]
struct Downloaded {
    /// Имя на шаре (напр. `song.wav`).
    name: String,
    /// Локальный путь.
    local: PathBuf,
}

impl Downloaded {
    /// Ключ сопоставления с событиями драйвера (строка локального пути).
    fn key(&self) -> String {
        self.local.to_string_lossy().into_owned()
    }
}

// --- дедупликация ------------------------------------------------------------------

/// Дубль, ждущий переименования за своим хранителем.
struct PendingDup {
    /// Индекс дубля в `downloaded`.
    duplicate_idx: usize,
    /// Имя дубля на шаре.
    duplicate: String,
    /// Имя хранителя на шаре.
    kept: String,
    /// Метрики сравнения пары (хранитель, дубль) — для отчёта.
    verdict: PairVerdict,
    /// Причина выбора хранителя — для отчёта/TUI.
    reason: KeeperReason,
}

/// Исход стадии дедупликации внутри батча.
struct DedupPass {
    /// Файлы к движку (хранители + уникальные), в порядке скачивания.
    worklist: Vec<Downloaded>,
    /// Дубли: статус уже выставлен, rename — после `Ok` хранителя.
    pending: Vec<PendingDup>,
    /// Не разобранные как PCM WAV (считаны уникальными) — примечание в отчёт.
    unparseable: Vec<String>,
}

impl DedupPass {
    /// Стадия выключена/нечего сравнивать: всё идёт движку как есть.
    fn passthrough(downloaded: &[Downloaded]) -> Self {
        Self { worklist: downloaded.to_vec(), pending: Vec::new(), unparseable: Vec::new() }
    }
}

impl<'a> Pipeline<'a> {
    /// Стадия дедупликации: анализ скачанных файлов, группировка по
    /// (data_size, frames), попарные сравнения в группах ≥ 2. Дубли сразу
    /// получают статус `Duplicate` и локальный cleanup; их rename на NAS —
    /// позже, строго после `Ok` хранителя ([`Pipeline::finalize_duplicates`]).
    fn dedup_pass(&self, downloaded: &[Downloaded], results: &mut Vec<FileResult>) -> DedupPass {
        if !self.dedup || downloaded.len() < 2 {
            return DedupPass::passthrough(downloaded);
        }
        let (analyses, unparseable) = analyze_downloaded(downloaded);
        let (pending, dup_indexes) = resolve_duplicates(&analyses, downloaded);
        for dup in &pending {
            let local = downloaded[dup.duplicate_idx].local.clone();
            results.push(FileResult {
                name: dup.duplicate.clone(),
                status: FileStatus::Duplicate { kept: dup.kept.clone() },
                error: None,
                elapsed: Duration::ZERO,
            });
            // Локальная копия дубля больше не нужна: решение о NAS —
            // только rename после `Ok` хранителя, файл не удаляется никогда.
            let _ = std::fs::remove_file(local);
        }
        let worklist = downloaded
            .iter()
            .enumerate()
            .filter(|(i, _)| !dup_indexes.contains(i))
            .map(|(_, d)| d.clone())
            .collect();
        DedupPass { worklist, pending, unparseable }
    }

    /// Rename дубля `DUPLICATE_<имя>` на NAS строго после `Ok` его хранителя;
    /// иначе дубль не трогается (никогда не удаляется), пояснение — в отчёте.
    fn finalize_duplicates(&self, pass: &DedupPass, results: &[FileResult]) -> Vec<PairReport> {
        pass.pending
            .iter()
            .map(|dup| {
                let keeper_ok = results.iter().any(|r| r.name == dup.kept && r.status == FileStatus::Ok);
                let (renamed, rename_error) = if keeper_ok {
                    match self.rename_on_nas(&dup.duplicate, &format!("{DUPLICATE_PREFIX}{}", dup.duplicate)) {
                        Ok(()) => (true, None),
                        Err(err) => (false, Some(err.to_string())),
                    }
                } else {
                    (false, None)
                };
                PairReport {
                    duplicate: dup.duplicate.clone(),
                    kept: dup.kept.clone(),
                    corr: dup.verdict.corr,
                    gain_db: dup.verdict.gain_db,
                    crest_kept_db: dup.verdict.crest_a_db,
                    crest_dup_db: dup.verdict.crest_b_db,
                    kept_clipping: dup.verdict.clip_a.is_clipping(),
                    dup_clipping: dup.verdict.clip_b.is_clipping(),
                    renamed,
                    rename_error,
                    reason: dup.reason.clone(),
                }
            })
            .collect()
    }

    /// Переименование на шаре (имена внутри `remote_dir`).
    fn rename_on_nas(&self, from_name: &str, to_name: &str) -> Result<(), AppError> {
        self.smb.rename(&self.remote_dir.join(from_name), &self.remote_dir.join(to_name))
    }
}

/// Анализ скачанных файлов: `Some` — анализ для дедупа, `None` — не PCM WAV
/// (ошибки I/O в том числе): файл считается уникальным, имя — в примечания.
fn analyze_downloaded(downloaded: &[Downloaded]) -> (Vec<Option<FileAnalysis>>, Vec<String>) {
    let mut analyses = Vec::with_capacity(downloaded.len());
    let mut unparseable = Vec::new();
    for entry in downloaded {
        let parsed = dedup::analyze_file(&entry.local).ok().flatten();
        if parsed.is_none() {
            unparseable.push(entry.name.clone());
        }
        analyses.push(parsed);
    }
    (analyses, unparseable)
}

/// Группировка по (data_size, frames) → разбор групп ≥ 2 через
/// `dedup::resolve_group`. Возвращает дубли (для статусов/ренейма)
/// и множество их индексов в `downloaded`.
fn resolve_duplicates(
    analyses: &[Option<FileAnalysis>],
    downloaded: &[Downloaded],
) -> (Vec<PendingDup>, HashSet<usize>) {
    let mut groups: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
    for (i, analysis) in analyses.iter().enumerate() {
        if let Some(a) = analysis {
            groups.entry((a.info.data_size, a.info.frames)).or_default().push(i);
        }
    }
    // Детерминизм отчёта: группы в порядке первого вхождения.
    let mut ordered: Vec<&Vec<usize>> = groups.values().filter(|g| g.len() >= 2).collect();
    ordered.sort_by_key(|g| g[0]);
    let (mut pending, mut dup_indexes) = (Vec::new(), HashSet::new());
    for group in ordered {
        let group_analyses: Vec<FileAnalysis> =
            group.iter().filter_map(|&i| analyses[i].clone()).collect();
        let resolution = dedup::resolve_group(&group_analyses);
        for matched in resolution.duplicates {
            let (dup_i, kept_i) = (group[matched.duplicate], group[matched.kept]);
            dup_indexes.insert(dup_i);
            pending.push(PendingDup {
                duplicate_idx: dup_i,
                duplicate: downloaded[dup_i].name.clone(),
                kept: downloaded[kept_i].name.clone(),
                verdict: matched.verdict,
                reason: matched.reason,
            });
        }
    }
    (pending, dup_indexes)
}

/// Пары (локальный стем → удалённый `<basename>/<имя>`); число стемов —
/// инвариант: отклонение означает отказ ДО любых заливок.
fn stem_uploads(stems: &[String], remote_stem_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    if stems.len() != EXPECTED_STEM_COUNT {
        return Err(format!("engine returned {} stems, expected {EXPECTED_STEM_COUNT}", stems.len()));
    }
    stems
        .iter()
        .map(|stem| {
            let file_name =
                Path::new(stem).file_name().ok_or_else(|| format!("stem path has no file name: `{stem}`"))?;
            Ok((PathBuf::from(stem), remote_stem_dir.join(file_name)))
        })
        .collect()
}

/// `song.wav` → `song` (имя без расширения; имя без точки — как есть).
fn stem_of(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((base, _)) => base,
        None => name,
    }
}

/// Имя стем-директории на NAS: [`stem_of`] без краевых пробелов
/// («name .WAV» → «name») — хвостовой пробел неудобен в файловых менеджерах.
/// Единая точка для upload-директории и resume-проверки: имена консистентны
/// по построению (DRY). Локальная директория стемов — не здесь: драйвер
/// раскладывает стемы по сырому stem, см. [`Pipeline::cleanup_local`].
fn stem_dir_name(name: &str) -> &str {
    stem_of(name).trim()
}

/// Последний компонент пути как строка (иначе — вход как есть).
fn file_name_of(path: &str) -> &str {
    Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path)
}

/// `song.wav` → `CONVERTED_song.wav`.
fn converted_name(name: &str) -> String {
    format!("{CONVERTED_PREFIX}{name}")
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;

    // --- моки ---------------------------------------------------------------

    /// Захваченный вызов SMB (для проверок порядка операций).
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum SmbCall {
        Get(String, String),
        Put(String, String),
        CreateDir(String),
        Rename(String, String),
        List(String),
        Exists(String),
    }

    /// SMB-мок: листинги/существование из фикстуры, отказы — по путям.
    /// `bodies` — содержимое get (по умолчанию не-WAV байты; для дедуп-сценариев
    /// кладутся настоящие WAV-фикстуры).
    struct MockSmb {
        calls: Mutex<Vec<SmbCall>>,
        dirs: HashMap<PathBuf, Vec<SmbEntry>>,
        exists: HashSet<PathBuf>,
        fail_get: HashSet<PathBuf>,
        fail_put: HashSet<PathBuf>,
        fail_create_dir: HashSet<PathBuf>,
        fail_rename: bool,
        bodies: HashMap<PathBuf, Vec<u8>>,
    }

    impl MockSmb {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                dirs: HashMap::new(),
                exists: HashSet::new(),
                fail_get: HashSet::new(),
                fail_put: HashSet::new(),
                fail_create_dir: HashSet::new(),
                fail_rename: false,
                bodies: HashMap::new(),
            }
        }

        fn with_dir(mut self, dir: &str, names: &[&str]) -> Self {
            self.dirs
                .insert(PathBuf::from(dir), names.iter().map(|n| file_entry(n)).collect());
            self
        }

        /// Содержимое, отдаваемое get для удалённого пути.
        fn with_body(mut self, remote: &str, bytes: Vec<u8>) -> Self {
            self.bodies.insert(PathBuf::from(remote), bytes);
            self
        }

        fn with_exists(mut self, path: &str) -> Self {
            self.exists.insert(PathBuf::from(path));
            self
        }

        fn failing_get(mut self, remote: &str) -> Self {
            self.fail_get.insert(PathBuf::from(remote));
            self
        }

        fn failing_put(mut self, remote: &str) -> Self {
            self.fail_put.insert(PathBuf::from(remote));
            self
        }

        fn failing_create_dir(mut self, dir: &str) -> Self {
            self.fail_create_dir.insert(PathBuf::from(dir));
            self
        }

        fn failing_rename(mut self) -> Self {
            self.fail_rename = true;
            self
        }

        fn calls(&self) -> Vec<SmbCall> {
            self.calls.lock().expect("lock").clone()
        }
    }

    impl SmbOps for MockSmb {
        fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
            self.calls.lock().expect("lock").push(SmbCall::List(dir.display().to_string()));
            self.dirs.get(dir).cloned().ok_or_else(|| AppError::Smb {
                operation: "list",
                path: dir.to_path_buf(),
                message: "not found in fixture".to_string(),
            })
        }

        fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
            self.calls.lock().expect("lock").push(SmbCall::Get(remote.display().to_string(), local.display().to_string()));
            if self.fail_get.contains(remote) {
                return Err(AppError::Smb {
                    operation: "get",
                    path: remote.to_path_buf(),
                    message: "forced failure".to_string(),
                });
            }
            if let Some(parent) = local.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            let bytes = self
                .bodies
                .get(remote)
                .cloned()
                .unwrap_or_else(|| b"wav-bytes".to_vec());
            fs::write(local, bytes).unwrap();
            Ok(())
        }

        fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
            self.calls.lock().expect("lock").push(SmbCall::Put(local.display().to_string(), remote.display().to_string()));
            if self.fail_put.contains(remote) {
                return Err(AppError::Smb {
                    operation: "put",
                    path: remote.to_path_buf(),
                    message: "forced failure".to_string(),
                });
            }
            Ok(())
        }

        fn create_dir(&self, dir: &Path) -> Result<(), AppError> {
            self.calls.lock().expect("lock").push(SmbCall::CreateDir(dir.display().to_string()));
            if self.fail_create_dir.contains(dir) {
                return Err(AppError::Smb {
                    operation: "mkdir",
                    path: dir.to_path_buf(),
                    message: "forced failure".to_string(),
                });
            }
            Ok(())
        }

        fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
            self.calls.lock().expect("lock").push(SmbCall::Rename(from.display().to_string(), to.display().to_string()));
            if self.fail_rename {
                return Err(AppError::Smb {
                    operation: "rename",
                    path: to.to_path_buf(),
                    message: "forced failure".to_string(),
                });
            }
            Ok(())
        }

        fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
            self.calls.lock().expect("lock").push(SmbCall::Exists(path.display().to_string()));
            Ok(self.exists.contains(path))
        }
    }

    /// Движок-мок: проигрывает сценарные события; для `FileDone` создаёт
    /// файлы стемов на диске (как реальный драйвер) — ради честного cleanup.
    struct MockEngine {
        calls: Mutex<Vec<(Vec<PathBuf>, PathBuf)>>,
        events: Vec<EngineEvent>,
        fail: bool,
    }

    impl MockEngine {
        fn new(events: Vec<EngineEvent>) -> Self {
            Self { calls: Mutex::new(Vec::new()), events, fail: false }
        }

        fn failing() -> Self {
            let mut engine = Self::new(Vec::new());
            engine.fail = true;
            engine
        }

        fn called_files(&self) -> Vec<PathBuf> {
            self.calls.lock().expect("lock").iter().flat_map(|(files, _)| files.clone()).collect()
        }
    }

    impl EngineOps for MockEngine {
        fn separate_batch(
            &self,
            files: &[PathBuf],
            output_dir: &Path,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<(), AppError> {
            self.calls
                .lock()
                .expect("lock")
                .push((files.to_vec(), output_dir.to_path_buf()));
            if self.fail {
                return Err(AppError::Engine {
                    file: files.first().cloned().unwrap_or_default(),
                    exit_code: None,
                    stderr_tail: "spawn failed: forced".to_string(),
                });
            }
            for event in &self.events {
                if let EngineEvent::FileDone { stems, .. } = event {
                    for stem in stems {
                        if let Some(parent) = Path::new(stem).parent() {
                            fs::create_dir_all(parent).unwrap();
                        }
                        fs::write(stem, b"flac-bytes").unwrap();
                    }
                }
                on_event(event.clone());
            }
            Ok(())
        }
    }

    // --- фикстуры -------------------------------------------------------------

    /// Уникальная локальная рабочая директория теста (параллельные тесты не
    /// пересекаются; повторный запуск стартует с чистой).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("audio_split_pipeline_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    const EXT: &str = "flac";

    /// Локальные пути стемов файла (порядок EXPECTED_STEMS).
    fn stem_paths(local_dir: &Path, base: &str) -> Vec<String> {
        EXPECTED_STEMS
            .iter()
            .map(|s| local_dir.join(base).join(format!("{s}.{EXT}")).to_string_lossy().into_owned())
            .collect()
    }

    /// Событие успешного разделения файла (как у реального драйвера).
    fn done_event(local_dir: &Path, name: &str) -> EngineEvent {
        EngineEvent::FileDone {
            file: local_dir.join(name).to_string_lossy().into_owned(),
            stems: stem_paths(local_dir, stem_of(name)),
            seconds: 1.0,
        }
    }

    /// Событие отказа файла.
    fn error_event(local_dir: &Path, name: &str, message: &str) -> EngineEvent {
        EngineEvent::FileError {
            file: local_dir.join(name).to_string_lossy().into_owned(),
            error: message.to_string(),
        }
    }

    fn pipeline<'a>(smb: &'a MockSmb, engine: &'a MockEngine, remote: &str, local: &Path) -> Pipeline<'a> {
        Pipeline::new(smb, engine, PathBuf::from(remote), local.to_path_buf(), EXT)
    }

    fn file_entry(name: &str) -> SmbEntry {
        SmbEntry { name: name.to_string(), is_dir: false, size: 1 }
    }

    fn wav_entry(name: &str) -> SmbEntry {
        file_entry(name)
    }

    /// Пары (local, remote) всех заливок из журнала.
    fn puts(calls: &[SmbCall]) -> Vec<(String, String)> {
        calls
            .iter()
            .filter_map(|c| match c {
                SmbCall::Put(local, remote) => Some((local.clone(), remote.clone())),
                _ => None,
            })
            .collect()
    }

    /// Все переименования из журнала.
    fn renames(calls: &[SmbCall]) -> Vec<(String, String)> {
        calls
            .iter()
            .filter_map(|c| match c {
                SmbCall::Rename(from, to) => Some((from.clone(), to.clone())),
                _ => None,
            })
            .collect()
    }

    // --- план -------------------------------------------------------------------

    #[test]
    fn plan_skips_converted_and_non_wav() {
        let smb = MockSmb::new();
        let entries = vec![
            wav_entry("a.wav"),
            wav_entry("CONVERTED_b.wav"),
            file_entry("notes.txt"),
            SmbEntry { name: "nested".to_string(), is_dir: true, size: 0 },
        ];

        let batch = plan(&entries, &smb, Path::new("/remote"), EXT).unwrap();

        assert_eq!(batch.pending.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["a.wav"]);
        assert_eq!(batch.skipped_converted, ["CONVERTED_b.wav"]);
        assert!(batch.skipped_done.is_empty());
        // Из сетевых запросов — только resume-проверка кандидата (remote_exists).
        assert_eq!(smb.calls(), vec![SmbCall::Exists("/remote/a".to_string())]);
    }

    #[test]
    fn plan_resume_skips_only_complete_stem_sets() {
        let smb = MockSmb::new()
            .with_exists("/remote/done")
            .with_dir("/remote/done", &["vocals.flac", "drums.flac", "bass.flac", "guitar.flac", "piano.flac", "other.flac"])
            .with_exists("/remote/half")
            .with_dir("/remote/half", &["vocals.flac", "drums.flac", "bass.flac", "guitar.flac", "piano.flac"]);
        let entries = vec![wav_entry("done.wav"), wav_entry("half.wav"), wav_entry("todo.wav")];

        let batch = plan(&entries, &smb, Path::new("/remote"), EXT).unwrap();

        assert_eq!(batch.skipped_done, ["done.wav"], "6 of 6 stems = done");
        assert_eq!(
            batch.pending.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            ["half.wav", "todo.wav"],
            "partial stems and missing dir = pending"
        );
    }

    #[test]
    fn plan_resume_hit_uses_trimmed_stem_dir_name() {
        // Боевой кейс «...unnamed place .WAV» (пробел перед расширением):
        // resume-проверка обязана найти существующую директорию по имени
        // без хвостового пробела — тому же, под которым заливаются стемы.
        let smb = MockSmb::new()
            .with_exists("/remote/name")
            .with_dir(
                "/remote/name",
                &["vocals.flac", "drums.flac", "bass.flac", "guitar.flac", "piano.flac", "other.flac"],
            );
        let entries = vec![wav_entry("name .WAV")];

        let batch = plan(&entries, &smb, Path::new("/remote"), EXT).unwrap();

        assert_eq!(batch.skipped_done, ["name .WAV"], "trim-имя нашло готовую директорию");
        assert_eq!(
            smb.calls(),
            vec![
                SmbCall::Exists("/remote/name".to_string()),
                SmbCall::List("/remote/name".to_string()),
            ],
            "resume-проверка (exists + листинг стемов) идёт по trim-имени"
        );
    }

    #[test]
    fn has_all_stems_matches_case_insensitively() {
        let entries = vec![
            file_entry("VOCALS.FLAC"),
            file_entry("drums.Flac"),
            file_entry("bass.flac"),
            file_entry("guitar.flac"),
            file_entry("piano.flac"),
            file_entry("other.flac"),
        ];
        assert!(has_all_stems(&entries, "flac"));
        assert!(!has_all_stems(&entries[..5], "flac"), "5 of 6 is not done");
    }

    // --- протокол драйвера ---------------------------------------------------------

    #[test]
    fn parse_event_reads_all_driver_events() {
        let cases = [
            (
                r#"{"event":"progress","file":"a.wav","stage":"separate","pct":42.5}"#,
                EngineEvent::Progress { file: "a.wav".to_string(), stage: "separate".to_string(), pct: 42.5 },
            ),
            (
                r#"{"event":"file_done","file":"/tmp/a.wav","stems":["/tmp/a/vocals.flac"],"seconds":1.25}"#,
                EngineEvent::FileDone {
                    file: "/tmp/a.wav".to_string(),
                    stems: vec!["/tmp/a/vocals.flac".to_string()],
                    seconds: 1.25,
                },
            ),
            (
                r#"{"event":"file_error","file":"a.wav","error":"boom"}"#,
                EngineEvent::FileError { file: "a.wav".to_string(), error: "boom".to_string() },
            ),
            (
                r#"{"event":"batch_done","ok":2,"failed":1}"#,
                EngineEvent::BatchDone { ok: 2, failed: 1 },
            ),
        ];
        for (line, expected) in cases {
            assert_eq!(parse_event(line), Some(expected), "line: {line}");
        }
    }

    #[test]
    fn parse_event_ignores_blank_and_non_protocol_lines() {
        assert_eq!(parse_event(""), None);
        assert_eq!(parse_event("   "), None);
        assert_eq!(parse_event("not json at all"), None);
    }

    // --- адаптер движка --------------------------------------------------------------

    #[test]
    fn engine_runner_builds_driver_command() {
        let runner = EngineRunner::new(Path::new("/venv"), Path::new("/app/driver.py"), "BS-Roformer-SW.ckpt", "flac");
        let cmd = runner.build_command(&[PathBuf::from("/tmp/a.wav")], Path::new("/tmp/work"));

        assert_eq!(cmd.get_program().to_string_lossy(), "/venv/bin/python");
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(
            args,
            [
                "/app/driver.py",
                "--model-file",
                "BS-Roformer-SW.ckpt",
                "--output-dir",
                "/tmp/work",
                "--output-format",
                "flac",
                "--files",
                "/tmp/a.wav",
            ]
        );
    }

    // --- цикл per-file -------------------------------------------------------------

    #[test]
    fn run_happy_path_orders_get_put_rename_and_cleans_up() {
        let local = temp_dir("happy");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav"]);
        let engine = MockEngine::new(vec![done_event(&local, "a.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.total, result.ok, result.failed), (1, 1, 0));
        assert_eq!(engine.called_files(), vec![local.join("a.wav")]);
        let calls = smb.calls();
        assert_eq!(calls[0], SmbCall::List("/remote".to_string()), "run starts with the directory listing");
        let gets: Vec<_> = calls
            .iter()
            .filter(|c| matches!(c, SmbCall::Get(..)))
            .cloned()
            .collect();
        assert_eq!(
            gets,
            vec![SmbCall::Get("/remote/a.wav".to_string(), local.join("a.wav").display().to_string())]
        );
        // 6 заливок в `<basename>/` на шаре.
        let uploads = puts(&calls);
        assert_eq!(uploads.len(), 6);
        assert!(uploads.iter().all(|(_, remote)| remote.starts_with("/remote/a/")), "{uploads:?}");
        // Rename ровно один, последняя операция, целевое имя CONVERTED_*.
        assert_eq!(
            calls.last(),
            Some(&SmbCall::Rename("/remote/a.wav".to_string(), "/remote/CONVERTED_a.wav".to_string()))
        );
        assert_eq!(renames(&calls).len(), 1);
        // Локальный cleanup: исходник и стемы удалены.
        assert!(!local.join("a.wav").exists());
        assert!(!local.join("a").exists());
    }

    #[test]
    fn trailing_space_name_uploads_to_trimmed_stem_dir_and_keeps_original_rename() {
        // «name .WAV» (пробел перед расширением): стем-директория на NAS —
        // trim-имя; rename исходника и локальный cleanup — по исходным именам.
        let local = temp_dir("trailing_space");
        let smb = MockSmb::new().with_dir("/remote", &["name .WAV"]);
        // Драйвер кладёт стемы в сырой stem («name ») — как done_event ниже.
        let engine = MockEngine::new(vec![done_event(&local, "name .WAV")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.ok, result.failed), (1, 0));
        let calls = smb.calls();
        assert!(
            calls.contains(&SmbCall::CreateDir("/remote/name".to_string())),
            "stems dir created by trim-name: {calls:?}"
        );
        let uploads = puts(&calls);
        assert_eq!(uploads.len(), 6);
        assert!(
            uploads.iter().all(|(_, remote)| remote.starts_with("/remote/name/")),
            "uploads go to the trimmed dir: {uploads:?}"
        );
        // Исходник переименовывается по своему полному имени — без trim.
        assert_eq!(
            renames(&calls),
            vec![("/remote/name .WAV".to_string(), "/remote/CONVERTED_name .WAV".to_string())]
        );
        // Локальный cleanup — по сырой раскладке драйвера («name »).
        assert!(!local.join("name .WAV").exists(), "source cleaned up");
        assert!(!local.join("name ").exists(), "local stems dir (raw stem) cleaned up");
    }

    #[test]
    fn failed_upload_prevents_rename_and_batch_continues() {
        let local = temp_dir("put_fail");
        let smb = MockSmb::new()
            .with_dir("/remote", &["a.wav", "b.wav"])
            .failing_put("/remote/a/other.flac");
        let engine = MockEngine::new(vec![done_event(&local, "a.wav"), done_event(&local, "b.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.ok, result.failed), (1, 1), "isolation: a failed, b succeeded");
        let failed = result.results.iter().find(|r| r.name == "a.wav").unwrap();
        assert_eq!(failed.status, FileStatus::Failed);
        assert!(failed.error.as_deref().unwrap_or_default().contains("upload"));
        // Инвариант: rename выполнен только для b — у a заливка неполная.
        assert_eq!(
            renames(&smb.calls()),
            vec![("/remote/b.wav".to_string(), "/remote/CONVERTED_b.wav".to_string())]
        );
    }

    #[test]
    fn create_dir_for_stems_dir_runs_before_first_put() {
        let local = temp_dir("mkdir_order");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav"]);
        let engine = MockEngine::new(vec![done_event(&local, "a.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!(result.ok, 1);
        let calls = smb.calls();
        let first_put =
            calls.iter().position(|c| matches!(c, SmbCall::Put(..))).expect("stems uploaded");
        assert!(first_put >= 1, "listing must precede uploads; calls: {calls:?}");
        assert_eq!(
            calls.get(first_put - 1),
            Some(&SmbCall::CreateDir("/remote/a".to_string())),
            "stems dir must be created immediately before the first put: {calls:?}"
        );
    }

    #[test]
    fn failed_create_dir_fails_file_without_any_put_or_rename() {
        let local = temp_dir("mkdir_fail");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav"]).failing_create_dir("/remote/a");
        let engine = MockEngine::new(vec![done_event(&local, "a.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!(result.failed, 1);
        let failed = result.results.iter().find(|r| r.name == "a.wav").expect("result");
        assert_eq!(failed.status, FileStatus::Failed);
        assert!(
            failed.error.as_deref().unwrap_or_default().contains("create dir"),
            "{failed:?}"
        );
        let calls = smb.calls();
        assert!(
            calls.iter().all(|c| !matches!(c, SmbCall::Put(..) | SmbCall::Rename(..))),
            "invariant: no uploads and no rename when create_dir fails: {calls:?}"
        );
        assert!(renames(&calls).is_empty(), "source must not be renamed to CONVERTED_");
    }

    #[test]
    fn failed_create_dir_isolates_file_and_batch_continues() {
        let local = temp_dir("mkdir_isolation");
        let smb = MockSmb::new()
            .with_dir("/remote", &["a.wav", "b.wav"])
            .failing_create_dir("/remote/a");
        let engine = MockEngine::new(vec![done_event(&local, "a.wav"), done_event(&local, "b.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.ok, result.failed), (1, 1), "isolation: a failed, b succeeded");
        assert_eq!(
            renames(&smb.calls()),
            vec![("/remote/b.wav".to_string(), "/remote/CONVERTED_b.wav".to_string())]
        );
    }

    #[test]
    fn wrong_stem_count_fails_file_without_any_upload() {
        let local = temp_dir("stem_count");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav"]);
        let short = EngineEvent::FileDone {
            file: local.join("a.wav").to_string_lossy().into_owned(),
            stems: vec![
                local.join("a").join("vocals.flac").to_string_lossy().into_owned(),
                local.join("a").join("drums.flac").to_string_lossy().into_owned(),
            ],
            seconds: 1.0,
        };

        let result = pipeline(&smb, &MockEngine::new(vec![short]), "/remote", &local).run().unwrap();

        assert_eq!(result.failed, 1);
        assert!(result.results[0].error.as_deref().unwrap_or_default().contains("expected 6"));
        assert!(
            smb.calls().iter().all(|c| !matches!(c, SmbCall::Put(..) | SmbCall::Rename(..))),
            "invariant: no uploads and no rename for an incomplete stem set"
        );
    }

    #[test]
    fn failed_rename_marks_file_failed_but_stems_are_uploaded() {
        let local = temp_dir("rename_fail");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav"]).failing_rename();
        let engine = MockEngine::new(vec![done_event(&local, "a.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!(result.failed, 1);
        assert_eq!(puts(&smb.calls()).len(), 6, "all stems uploaded before the rename attempt");
        assert!(result.results[0].error.as_deref().unwrap_or_default().contains("rename"));
    }

    #[test]
    fn engine_file_error_does_not_stop_batch() {
        let local = temp_dir("file_error");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav", "b.wav"]);
        let engine = MockEngine::new(vec![error_event(&local, "a.wav", "CUDA OOM"), done_event(&local, "b.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.ok, result.failed), (1, 1));
        let failed = result.results.iter().find(|r| r.name == "a.wav").unwrap();
        assert_eq!(failed.error.as_deref(), Some("CUDA OOM"));
        assert_eq!(renames(&smb.calls()).len(), 1, "only b was renamed");
    }

    #[test]
    fn failed_download_is_reported_and_excluded_from_engine() {
        let local = temp_dir("get_fail");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav", "b.wav"]).failing_get("/remote/a.wav");
        let engine = MockEngine::new(vec![done_event(&local, "b.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.ok, result.failed), (1, 1));
        assert_eq!(engine.called_files(), vec![local.join("b.wav")], "failed download must not reach the engine");
        assert_eq!(
            result.results.iter().find(|r| r.name == "a.wav").map(|r| &r.status),
            Some(&FileStatus::Failed)
        );
    }

    #[test]
    fn engine_crash_fails_remaining_files_without_rename() {
        let local = temp_dir("crash");
        let smb = MockSmb::new().with_dir("/remote", &["a.wav"]);
        let engine = MockEngine::failing();

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.ok, result.failed), (0, 1));
        assert!(result.results[0].error.as_deref().unwrap_or_default().contains("spawn failed"));
        assert!(
            smb.calls().iter().all(|c| !matches!(c, SmbCall::Put(..) | SmbCall::Rename(..))),
            "nothing is uploaded or renamed after an engine crash"
        );
    }

    #[test]
    fn run_with_nothing_pending_spawns_no_engine() {
        let local = temp_dir("empty");
        let smb = MockSmb::new().with_dir("/remote", &["notes.txt"]);
        let engine = MockEngine::new(vec![]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!((result.total, result.ok, result.failed), (0, 0, 0));
        assert!(engine.called_files().is_empty());
    }

    // --- агрегация и имена ------------------------------------------------------------

    #[test]
    fn summarize_aggregates_counts() {
        let results = vec![
            FileResult { name: "a".to_string(), status: FileStatus::Ok, error: None, elapsed: Duration::ZERO },
            FileResult::failed("b", "boom", Duration::ZERO),
            FileResult::failed("c", "boom", Duration::ZERO),
        ];

        let summary = summarize(results, Instant::now());

        assert_eq!((summary.total, summary.ok, summary.failed), (3, 1, 2));
        assert_eq!(summary.results.len(), 3);
    }

    #[test]
    fn stem_and_converted_names_derive_from_entry() {
        assert_eq!(stem_of("song.wav"), "song");
        assert_eq!(stem_of("noext"), "noext");
        assert_eq!(converted_name("song.wav"), "CONVERTED_song.wav");
        assert_eq!(file_name_of("/tmp/work/song.wav"), "song.wav");
    }

    #[test]
    fn stem_dir_name_trims_edge_spaces_keeps_inner() {
        // Краевые пробелы срезаются, внутренние остаются.
        assert_eq!(stem_dir_name("name .WAV"), "name");
        assert_eq!(stem_dir_name("  padded.wav"), "padded");
        assert_eq!(stem_dir_name("a b.wav"), "a b");
        assert_eq!(stem_dir_name("name.wav"), "name");
        assert_eq!(stem_dir_name("noext"), "noext");
    }

    // --- дедупликация ------------------------------------------------------------

    use crate::dedup::wav_fix;

    /// Частота дедуп-фикстур: компактные файлы при полной методике сравнения.
    const DEDUP_RATE: u32 = 8_000;

    /// WAV-байты «чистой» записи и её гейн-копии с лимитером (+16 дБ).
    fn duplicate_pair_bytes() -> (Vec<u8>, Vec<u8>) {
        let clean = wav_fix::burst_signal(DEDUP_RATE, 10);
        let loud = wav_fix::gain_limited(&clean, 6.31, 0.96);
        (
            wav_fix::encode_wav(DEDUP_RATE, 16, 1, &clean),
            wav_fix::encode_wav(DEDUP_RATE, 16, 1, &loud),
        )
    }

    /// SMB-мок с парой дублей на шаре.
    fn smb_with_pair(local_tag: &str) -> (PathBuf, MockSmb) {
        let local = temp_dir(local_tag);
        let (clean_bytes, loud_bytes) = duplicate_pair_bytes();
        let smb = MockSmb::new()
            .with_dir("/remote", &["clean.wav", "loud.wav"])
            .with_body("/remote/clean.wav", clean_bytes)
            .with_body("/remote/loud.wav", loud_bytes);
        (local, smb)
    }

    #[test]
    fn dedup_pair_engine_runs_keeper_only_and_renames_duplicate_after_ok() {
        let (local, smb) = smb_with_pair("dedup_pair");
        let engine = MockEngine::new(vec![done_event(&local, "clean.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        // Движок — ровно один вызов и только с хранителем (более динамичным).
        assert_eq!(engine.calls.lock().expect("lock").len(), 1, "one batch call");
        assert_eq!(engine.called_files(), vec![local.join("clean.wav")]);
        // После Ok хранителя на NAS есть CONVERTED_<keeper> и DUPLICATE_<dup>.
        assert_eq!(
            renames(&smb.calls()),
            vec![
                ("/remote/clean.wav".to_string(), "/remote/CONVERTED_clean.wav".to_string()),
                ("/remote/loud.wav".to_string(), "/remote/DUPLICATE_loud.wav".to_string()),
            ]
        );
        // Статусы: хранитель Ok, дубль Duplicate{kept}; дубль — не отказ.
        let dup = result.results.iter().find(|r| r.name == "loud.wav").expect("dup result");
        assert_eq!(dup.status, FileStatus::Duplicate { kept: "clean.wav".to_string() });
        assert_eq!((result.ok, result.failed), (1, 0));
        // Заливок ровно 6 — только стемы хранителя.
        assert_eq!(puts(&smb.calls()).len(), 6);
        // Локальная копия дубля убрана сразу после снятия с обработки.
        assert!(!local.join("loud.wav").exists());
    }

    #[test]
    fn dedup_keeper_failure_leaves_duplicate_untouched() {
        let (local, smb) = smb_with_pair("dedup_fail");
        let engine = MockEngine::new(vec![error_event(&local, "clean.wav", "CUDA OOM")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        let keeper = result.results.iter().find(|r| r.name == "clean.wav").expect("keeper result");
        assert_eq!(keeper.status, FileStatus::Failed);
        let dup = result.results.iter().find(|r| r.name == "loud.wav").expect("dup result");
        assert_eq!(dup.status, FileStatus::Duplicate { kept: "clean.wav".to_string() });
        // Хранитель не дошёл до Ok → дубль не переименован (и не удалён на NAS).
        assert!(
            renames(&smb.calls()).is_empty(),
            "no renames at all when the keeper failed"
        );
    }

    #[test]
    fn no_dedup_processes_both_files_without_duplicate_renames() {
        let (local, smb) = smb_with_pair("no_dedup");
        let engine =
            MockEngine::new(vec![done_event(&local, "clean.wav"), done_event(&local, "loud.wav")]);

        let result = Pipeline::new(&smb, &engine, PathBuf::from("/remote"), local.clone(), EXT)
            .with_dedup(false)
            .run()
            .unwrap();

        assert_eq!(engine.called_files(), vec![local.join("clean.wav"), local.join("loud.wav")]);
        assert_eq!((result.ok, result.failed), (2, 0));
        assert!(result.dedup.pairs.is_empty());
        assert!(!renames(&smb.calls()).iter().any(|(_, to)| to.contains("DUPLICATE_")));
    }

    #[test]
    fn unparseable_downloads_are_reported_and_processed_as_unique() {
        let local = temp_dir("dedup_junk");
        // Дефолтный мок отдаёт b"wav-bytes" — не PCM WAV.
        let smb = MockSmb::new().with_dir("/remote", &["a.wav", "b.wav"]);
        let engine = MockEngine::new(vec![done_event(&local, "a.wav"), done_event(&local, "b.wav")]);

        let result = pipeline(&smb, &engine, "/remote", &local).run().unwrap();

        assert_eq!(result.dedup.unparseable, ["a.wav", "b.wav"], "примечание в отчёт");
        assert!(result.dedup.pairs.is_empty());
        assert_eq!((result.ok, result.failed), (2, 0), "нечитаемые считаются уникальными");
    }

    #[test]
    fn plan_skips_duplicate_prefix_without_resume_probe() {
        let smb = MockSmb::new();
        let entries = vec![wav_entry("DUPLICATE_x.wav"), wav_entry("a.wav")];

        let batch = plan(&entries, &smb, Path::new("/remote"), EXT).unwrap();

        assert_eq!(batch.skipped_duplicate, ["DUPLICATE_x.wav"]);
        assert_eq!(batch.pending.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(), ["a.wav"]);
        assert_eq!(
            smb.calls(),
            vec![SmbCall::Exists("/remote/a".to_string())],
            "resume-проверка только для кандидата"
        );
    }

    #[test]
    fn summarize_counts_duplicate_separately_from_failed() {
        let results = vec![
            FileResult { name: "a".into(), status: FileStatus::Ok, error: None, elapsed: Duration::ZERO },
            FileResult {
                name: "b".into(),
                status: FileStatus::Duplicate { kept: "a".into() },
                error: None,
                elapsed: Duration::ZERO,
            },
        ];

        let summary = summarize(results, Instant::now());

        assert_eq!((summary.total, summary.ok, summary.failed), (2, 1, 0), "дубль — не отказ");
    }
}
