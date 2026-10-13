//! Тестовые моки `SmbOps`/`EngineOps` для раннера (test-only; подключается
//! через `#[cfg(test)] #[path = "mocks.rs"] mod mocks;` в runner.rs).
//!
//! По образцу моков pipeline.rs: SMB — листинги/содержимое из фикстур,
//! отказы по путям, опциональная задержка get (окно для cancel-теста);
//! движок — проигрывает сценарные события (создавая файлы стемов, как
//! реальный драйвер), опциональная задержка перед событиями.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::error::AppError;
use crate::pipeline::{EngineEvent, EngineOps, EXPECTED_STEMS, SmbOps};
use crate::smb::SmbEntry;

/// SMB-мок: фикстуры директорий/содержимого, отказы по путям.
pub(crate) struct MockSmb {
    dirs: HashMap<PathBuf, Vec<SmbEntry>>,
    exists: HashSet<PathBuf>,
    fail_get: HashSet<PathBuf>,
    bodies: HashMap<PathBuf, Vec<u8>>,
    /// Задержка каждого `get` (эмуляция медленной сети; окно для cancel).
    get_delay: Option<Duration>,
    /// Журнал операций "op path" (для проверок порядка/наличия).
    calls: Mutex<Vec<String>>,
}

impl MockSmb {
    pub(crate) fn new() -> Self {
        Self {
            dirs: HashMap::new(),
            exists: HashSet::new(),
            fail_get: HashSet::new(),
            bodies: HashMap::new(),
            get_delay: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn with_dir(mut self, dir: &str, names: &[&str]) -> Self {
        self.dirs.insert(
            PathBuf::from(dir),
            names.iter().map(|n| SmbEntry { name: n.to_string(), is_dir: false, size: 1 }).collect(),
        );
        self
    }

    /// Содержимое, отдаваемое `get` (по умолчанию не-WAV байты).
    pub(crate) fn with_body(mut self, remote: &str, bytes: Vec<u8>) -> Self {
        self.bodies.insert(PathBuf::from(remote), bytes);
        self
    }

    pub(crate) fn failing_get(mut self, remote: &str) -> Self {
        self.fail_get.insert(PathBuf::from(remote));
        self
    }

    pub(crate) fn with_get_delay(mut self, delay: Duration) -> Self {
        self.get_delay = Some(delay);
        self
    }

    pub(crate) fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("lock").clone()
    }
}

impl SmbOps for MockSmb {
    fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        self.calls.lock().expect("lock").push(format!("list {}", dir.display()));
        self.dirs.get(dir).cloned().ok_or_else(|| AppError::Smb {
            operation: "list",
            path: dir.to_path_buf(),
            message: "not found in fixture".to_string(),
        })
    }

    fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
        self.calls.lock().expect("lock").push(format!("get {}", remote.display()));
        if let Some(delay) = self.get_delay {
            std::thread::sleep(delay);
        }
        if self.fail_get.contains(remote) {
            return Err(AppError::Smb {
                operation: "get",
                path: remote.to_path_buf(),
                message: "forced failure".to_string(),
            });
        }
        if let Some(parent) = local.parent() {
            fs::create_dir_all(parent).expect("create local dir");
        }
        let bytes =
            self.bodies.get(remote).cloned().unwrap_or_else(|| b"wav-bytes".to_vec());
        fs::write(local, bytes).expect("write local file");
        Ok(())
    }

    fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
        self.calls.lock().expect("lock").push(format!("put {} -> {}", local.display(), remote.display()));
        Ok(())
    }

    fn create_dir(&self, dir: &Path) -> Result<(), AppError> {
        self.calls.lock().expect("lock").push(format!("mkdir {}", dir.display()));
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
        self.calls.lock().expect("lock").push(format!("rename {} -> {}", from.display(), to.display()));
        Ok(())
    }

    fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
        Ok(self.exists.contains(path))
    }
}

/// Движок-мок: проигрывает сценарные события; для `FileDone` создаёт файлы
/// стемов на диске (как реальный драйвер). `delay_before` — пауза до
/// проигрывания (окно для cancel-теста на границе upload).
pub(crate) struct MockEngine {
    events: Vec<EngineEvent>,
    fail: bool,
    delay_before: Option<Duration>,
    /// Файлы, отказ которых эмулируется ОДИН раз (первый вызов), для retry.
    fail_file_once: Mutex<HashSet<String>>,
    calls: Mutex<Vec<Vec<PathBuf>>>,
}

impl MockEngine {
    pub(crate) fn new(events: Vec<EngineEvent>) -> Self {
        Self {
            events,
            fail: false,
            delay_before: None,
            fail_file_once: Mutex::new(HashSet::new()),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn with_delay_before(mut self, delay: Duration) -> Self {
        self.delay_before = Some(delay);
        self
    }

    /// Отказать файлу при ПЕРВОМ вызове, на повторном (retry) — успех.
    pub(crate) fn failing_file_once(mut self, file_name: &str) -> Self {
        self.fail_file_once = Mutex::new(HashSet::from([file_name.to_string()]));
        self
    }

    pub(crate) fn called_batches(&self) -> Vec<Vec<PathBuf>> {
        self.calls.lock().expect("lock").clone()
    }
}

impl EngineOps for MockEngine {
    fn separate_batch(
        &self,
        files: &[PathBuf],
        output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(), AppError> {
        self.calls.lock().expect("lock").push(files.to_vec());
        if self.fail {
            return Err(AppError::Engine {
                file: files.first().cloned().unwrap_or_default(),
                exit_code: None,
                stderr_tail: "spawn failed: forced".to_string(),
            });
        }
        if let Some(delay) = self.delay_before {
            std::thread::sleep(delay);
        }
        for event in &self.events {
            if let EngineEvent::FileDone { file, stems, .. } = event {
                // Отказ файла ровно один раз: первый прогон падает, retry — нет.
                if let Some(name) = Path::new(file).file_name().and_then(|n| n.to_str()) {
                    if self.fail_file_once.lock().expect("lock").remove(name) {
                        on_event(EngineEvent::FileError {
                            file: file.clone(),
                            error: "engine boom (first attempt)".to_string(),
                        });
                        return Ok(());
                    }
                }
                for stem in stems {
                    if let Some(parent) = Path::new(stem).parent() {
                        fs::create_dir_all(parent).expect("create stem dir");
                    }
                    fs::write(stem, b"flac-bytes").expect("write stem");
                }
            }
            on_event(event.clone());
        }
        let _ = output_dir;
        Ok(())
    }
}

/// Локальные пути стемов файла (порядок EXPECTED_STEMS) — как у драйвера.
pub(crate) fn stem_paths(local_dir: &Path, base: &str, ext: &str) -> Vec<String> {
    EXPECTED_STEMS
        .iter()
        .map(|s| local_dir.join(base).join(format!("{s}.{ext}")).to_string_lossy().into_owned())
        .collect()
}

/// Событие успешного разделения файла.
pub(crate) fn done_event(local_dir: &Path, name: &str, ext: &str) -> EngineEvent {
    EngineEvent::FileDone {
        file: local_dir.join(name).to_string_lossy().into_owned(),
        stems: stem_paths(local_dir, Path::new(name).file_stem().and_then(|n| n.to_str()).unwrap_or(name), ext),
        seconds: 1.0,
    }
}

/// Событие отказа файла.
pub(crate) fn error_event(local_dir: &Path, name: &str, message: &str) -> EngineEvent {
    EngineEvent::FileError {
        file: local_dir.join(name).to_string_lossy().into_owned(),
        error: message.to_string(),
    }
}

/// Прогресс разделения файла.
pub(crate) fn progress_event(local_dir: &Path, name: &str, pct: f64) -> EngineEvent {
    EngineEvent::Progress {
        file: local_dir.join(name).to_string_lossy().into_owned(),
        stage: "separate".to_string(),
        pct,
    }
}
