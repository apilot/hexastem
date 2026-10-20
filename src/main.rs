//! CLI-вход: только сборка компонентов из [`Config`] и порядок шагов прогона;
//! логика — в библиотеке (pipeline/report/local_fs/smb). `main` = parse + exit,
//! вся сборка — в шаговых функциях (`run` → `assemble` → шаги библиотеки).

use std::path::PathBuf;
use std::time::SystemTime;

use clap::Parser;

use hexastem::pipeline::{plan, BatchPlan, BatchResult, EngineOps, Pipeline, SmbOps};
use hexastem::report::{self, Skipped};
use hexastem::smb::SmbClient;
use hexastem::Config;

use hexastem::engine::native::NativeEngine;
use hexastem::error::AppError;
use hexastem::local_fs::LocalFs;

fn main() {
    let config = Config::parse();
    std::process::exit(run(&config));
}

/// Прогон целиком: бэкенды из конфига → полный цикл → код возврата
/// (0/1 по отказам батча). Инфраструктурная ошибка (скан/конфиг/отчёт) —
/// в stderr и код 1; отдельные файлы отказом прогон не роняют.
fn run(config: &Config) -> i32 {
    let smb = storage_backend(config);
    let engine = match native_engine(config) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("hexastem: {err}");
            return 1;
        }
    };
    match assemble(config, smb.as_ref(), engine.as_ref()) {
        Ok(outcome) => report::exit_code(&outcome.batch),
        Err(err) => {
            eprintln!("hexastem: {err}");
            1
        }
    }
}

/// Бэкенд хранилища: `--local` — локальная папка (тот же трейт [`SmbOps`]),
/// иначе smbclient поверх шары (креды — файл/env, не код).
fn storage_backend(config: &Config) -> Box<dyn SmbOps> {
    if config.local {
        Box::new(LocalFs)
    } else {
        Box::new(SmbClient::new(
            config.host.clone(),
            config.share.clone(),
            config.smb_creds_file.clone(),
        ))
    }
}

/// Движок разделения — единственный, нативный (ONNX-модель из конфига;
/// dylib ONNX Runtime — через ORT_DYLIB_PATH). Пишет только FLAC:
/// другой output_format — ошибка конфигурации.
fn native_engine(config: &Config) -> Result<Box<dyn EngineOps>, AppError> {
    if !config.output_format.eq_ignore_ascii_case("flac") {
        return Err(AppError::Config(format!(
            "native engine writes FLAC stems only, --output-format `{}` is not supported",
            config.output_format
        )));
    }
    Ok(Box::new(NativeEngine::new(&config.onnx_model)))
}

/// Итог прогона: батч, скипы плана и путь отчёта на хранилище
/// (скипы и путь читают проверки e2e, потому `allow(dead_code)`).
#[allow(dead_code)]
struct RunOutcome {
    batch: BatchResult,
    skipped: Skipped,
    report_remote: PathBuf,
}

/// Полный цикл на инжектируемых бэкендах — точка сборки и тестируемости:
/// план (скан/скипы) → исполнение → markdown → консоль → публикация отчёта.
fn assemble(config: &Config, smb: &dyn SmbOps, engine: &dyn EngineOps) -> Result<RunOutcome, AppError> {
    let batch_plan = build_plan(config, smb)?;
    let pipeline = Pipeline::new(smb, engine, config.remote_dir.clone(), config.local_dir.clone(), &config.output_format)
        .with_dedup(!config.no_dedup);
    let batch = pipeline.execute(&batch_plan)?;
    let skipped = Skipped {
        converted: batch_plan.skipped_converted.len(),
        resume: batch_plan.skipped_done.len(),
        duplicates: batch_plan.skipped_duplicate.len(),
    };
    let now = SystemTime::now();
    let markdown = report::generate_markdown(&batch, &skipped, config, now);
    println!("{}", report::console_summary(&batch, &skipped));
    let report_remote = report::publish(smb, &config.local_dir, &config.remote_dir, &report::report_name(now), &markdown)?;
    Ok(RunOutcome { batch, skipped, report_remote })
}

/// Скан директории хранилища и план батча (скип `CONVERTED_*` и resume).
fn build_plan(config: &Config, smb: &dyn SmbOps) -> Result<BatchPlan, AppError> {
    let entries = smb.list(&config.remote_dir)?;
    plan(&entries, smb, &config.remote_dir, &config.output_format)
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::Path;
    use std::sync::Mutex;

    use hexastem::pipeline::{EngineEvent, EXPECTED_STEMS};
    use hexastem::smb::SmbEntry;
    use hexastem::Config;

    use super::*;

    /// Байты-заглушка «исходника» в моке хранилища.
    const WAV_BYTES: &[u8] = b"wav-bytes";

    // --- моки e2e ------------------------------------------------------------

    /// Хранилище-мок «на HashMap»: путь → байты (файлы) + множество
    /// директорий. Листинг — дети запрошенной директории; rename двигает
    /// запись и отказывает при существующей цели (как реальный smbclient).
    struct InMemorySmb {
        files: Mutex<HashMap<PathBuf, Vec<u8>>>,
        dirs: Mutex<HashSet<PathBuf>>,
    }

    impl Default for InMemorySmb {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InMemorySmb {
        fn new() -> Self {
            Self { files: Mutex::new(HashMap::new()), dirs: Mutex::new(HashSet::new()) }
        }

        /// Сидирует файлы-заглушки в директории «шары».
        fn seed(&self, dir: &Path, names: &[&str]) {
            let mut files = self.files.lock().unwrap();
            for name in names {
                files.insert(dir.join(name), WAV_BYTES.to_vec());
            }
        }

        /// Прямая вставка файла (сидирование предыстории, минуя диск).
        fn force_put(&self, path: &Path, bytes: &[u8]) {
            let mut files = self.files.lock().unwrap();
            let mut dirs = self.dirs.lock().unwrap();
            register_parents(&mut dirs, path);
            files.insert(path.to_path_buf(), bytes.to_vec());
        }

        fn has_file(&self, path: &Path) -> bool {
            self.files.lock().unwrap().contains_key(path)
        }
    }

    /// Регистрирует всех предков пути как директории мока.
    fn register_parents(dirs: &mut HashSet<PathBuf>, path: &Path) {
        let mut parent = path.parent();
        while let Some(dir) = parent {
            dirs.insert(dir.to_path_buf());
            parent = dir.parent();
        }
    }

    impl SmbOps for InMemorySmb {
        fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
            let files = self.files.lock().unwrap();
            let dirs = self.dirs.lock().unwrap();
            let mut entries: Vec<SmbEntry> = files
                .iter()
                .filter(|(path, _)| path.parent() == Some(dir))
                .map(|(path, bytes)| SmbEntry {
                    name: path.file_name().unwrap().to_string_lossy().into_owned(),
                    is_dir: false,
                    size: bytes.len() as u64,
                })
                .chain(dirs.iter().filter(|d| d.parent() == Some(dir)).map(|d| SmbEntry {
                    name: d.file_name().unwrap().to_string_lossy().into_owned(),
                    is_dir: true,
                    size: 0,
                }))
                .collect();
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(entries)
        }

        fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
            let bytes = self.files.lock().unwrap().get(remote).cloned().ok_or_else(|| AppError::Smb {
                operation: "get",
                path: remote.to_path_buf(),
                message: "not found in mock".to_string(),
            })?;
            if let Some(parent) = local.parent() {
                fs::create_dir_all(parent).map_err(|source| AppError::Io { path: parent.to_path_buf(), source })?;
            }
            fs::write(local, bytes).map_err(|source| AppError::Io { path: local.to_path_buf(), source })
        }

        fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
            let bytes = fs::read(local).map_err(|source| AppError::Io { path: local.to_path_buf(), source })?;
            let mut files = self.files.lock().unwrap();
            let mut dirs = self.dirs.lock().unwrap();
            register_parents(&mut dirs, remote);
            files.insert(remote.to_path_buf(), bytes);
            Ok(())
        }

        fn create_dir(&self, dir: &Path) -> Result<(), AppError> {
            self.dirs.lock().unwrap().insert(dir.to_path_buf());
            Ok(())
        }

        fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
            let mut files = self.files.lock().unwrap();
            let mut dirs = self.dirs.lock().unwrap();
            if files.contains_key(to) || dirs.contains(to) {
                return Err(AppError::Smb {
                    operation: "rename",
                    path: to.to_path_buf(),
                    message: "target already exists".to_string(),
                });
            }
            if let Some(bytes) = files.remove(from) {
                files.insert(to.to_path_buf(), bytes);
                return Ok(());
            }
            if dirs.remove(from) {
                dirs.insert(to.to_path_buf());
                return Ok(());
            }
            Err(AppError::Smb {
                operation: "rename",
                path: from.to_path_buf(),
                message: "not found in mock".to_string(),
            })
        }

        fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
            Ok(self.files.lock().unwrap().contains_key(path) || self.dirs.lock().unwrap().contains(path))
        }
    }

    /// Движок-мок: на каждый файл — 6 фейковых стемов на диске и `file_done`
    /// (как реальный движок); файл, чей stem совпал с `fail_stem`, получает
    /// `file_error` без стемов. Завершает `batch_done` со счётчиками.
    struct StubEngine {
        fail_stem: Option<String>,
        called: Mutex<Vec<PathBuf>>,
    }

    impl StubEngine {
        fn ok() -> Self {
            Self { fail_stem: None, called: Mutex::new(Vec::new()) }
        }

        fn failing_on(stem: &str) -> Self {
            Self { fail_stem: Some(stem.to_string()), called: Mutex::new(Vec::new()) }
        }
    }

    impl EngineOps for StubEngine {
        fn separate_batch(
            &self,
            files: &[PathBuf],
            output_dir: &Path,
            on_event: &mut dyn FnMut(EngineEvent),
        ) -> Result<(), AppError> {
            self.called.lock().unwrap().extend_from_slice(files);
            let (mut ok, mut failed) = (0u32, 0u32);
            for file in files {
                let stem = file.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                if self.fail_stem.as_deref() == Some(stem.as_str()) {
                    failed += 1;
                    on_event(EngineEvent::FileError {
                        file: file.to_string_lossy().into_owned(),
                        error: "stub: corrupted wav".to_string(),
                    });
                    continue;
                }
                let stems = stub_stem_paths(output_dir, &stem);
                for path in &stems {
                    fs::create_dir_all(Path::new(path).parent().unwrap()).unwrap();
                    fs::write(path, b"flac-bytes").unwrap();
                }
                ok += 1;
                on_event(EngineEvent::FileDone {
                    file: file.to_string_lossy().into_owned(),
                    stems,
                    seconds: 0.0,
                });
            }
            on_event(EngineEvent::BatchDone { ok, failed });
            Ok(())
        }
    }

    /// Локальные пути стемов файла (порядок EXPECTED_STEMS) — как у движка.
    fn stub_stem_paths(output_dir: &Path, base: &str) -> Vec<String> {
        EXPECTED_STEMS
            .iter()
            .map(|stem| output_dir.join(base).join(format!("{stem}.flac")).to_string_lossy().into_owned())
            .collect()
    }

    // --- фикстуры ------------------------------------------------------------

    /// Пара уникальных директорий теста: remote-«шара» и local-рабочая.
    fn temp_pair(tag: &str) -> (PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!("hexastem_main_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let (remote, local) = (base.join("remote"), base.join("local"));
        fs::create_dir_all(&remote).unwrap();
        fs::create_dir_all(&local).unwrap();
        (remote, local)
    }

    /// Конфиг локального прогона (бэкенды инжектятся, host/share — заглушки).
    fn test_config(local_dir: &Path, remote_dir: &Path) -> Config {
        Config {
            host: "local".to_string(),
            share: "local".to_string(),
            remote_dir: remote_dir.to_path_buf(),
            local_dir: local_dir.to_path_buf(),
            output_format: "flac".to_string(),
            onnx_model: PathBuf::from("/unused/onnx/model.onnx"),
            smb_creds_file: None,
            local: true,
            no_dedup: false,
        }
    }

    // --- e2e: полный цикл ----------------------------------------------------

    #[test]
    fn e2e_full_cycle_processes_batch_and_publishes_report() {
        let (remote, local) = temp_pair("full");
        let smb = InMemorySmb::new();
        smb.seed(&remote, &["a.wav", "b.wav", "notes.txt"]);
        let config = test_config(&local, &remote);

        let outcome = assemble(&config, &smb, &StubEngine::ok()).expect("assemble must succeed");

        assert_eq!((outcome.batch.total, outcome.batch.ok, outcome.batch.failed), (2, 2, 0));
        assert_eq!(report::exit_code(&outcome.batch), 0);
        // CONVERTED_* проставлены, исходники ушли с «шары».
        assert!(smb.has_file(&remote.join("CONVERTED_a.wav")));
        assert!(smb.has_file(&remote.join("CONVERTED_b.wav")));
        assert!(!smb.has_file(&remote.join("a.wav")));
        assert!(!smb.has_file(&remote.join("b.wav")));
        // По 6 стемов на «удалённой» стороне в `<basename>/`.
        for base in ["a", "b"] {
            for stem in EXPECTED_STEMS {
                assert!(smb.has_file(&remote.join(base).join(format!("{stem}.flac"))), "{base}/{stem}");
            }
        }
        // Локальный cleanup: остался только отчёт.
        assert!(!local.join("a.wav").exists());
        assert!(!local.join("a").exists());
        let local_report = local.join(outcome.report_remote.file_name().unwrap());
        let markdown = fs::read_to_string(&local_report).unwrap();
        assert!(markdown.contains("| Успешно | 2 |"), "{markdown}");
        assert!(markdown.contains("`a.wav`") && markdown.contains("`b.wav`"), "{markdown}");
    }

    #[test]
    fn e2e_second_run_skips_converted_and_never_calls_engine() {
        let (remote, local) = temp_pair("resume_converted");
        let smb = InMemorySmb::new();
        smb.seed(&remote, &["a.wav"]);
        let config = test_config(&local, &remote);
        assemble(&config, &smb, &StubEngine::ok()).unwrap();

        // «Свежий» движок: любой вызов означал бы повторную обработку.
        let engine = StubEngine::failing_on("a");
        let outcome = assemble(&config, &smb, &engine).unwrap();

        assert!(engine.called.lock().unwrap().is_empty(), "processed files must not reach the engine");
        assert_eq!((outcome.batch.total, outcome.batch.ok, outcome.batch.failed), (0, 0, 0));
        assert_eq!(outcome.skipped.converted, 1, "renamed source is skipped as CONVERTED_*");
        assert_eq!(outcome.skipped.resume, 0);
        assert_eq!(report::exit_code(&outcome.batch), 0);
    }

    #[test]
    fn e2e_resume_skips_file_with_complete_stems_before_rename() {
        // Предыстория: стемы залиты, rename не успел (крах между шагами).
        let (remote, local) = temp_pair("resume_stems");
        let smb = InMemorySmb::new();
        smb.seed(&remote, &["a.wav"]);
        for stem in EXPECTED_STEMS {
            smb.force_put(&remote.join("a").join(format!("{stem}.flac")), b"flac-bytes");
        }

        let engine = StubEngine::failing_on("a");
        let outcome = assemble(&test_config(&local, &remote), &smb, &engine).unwrap();

        assert!(engine.called.lock().unwrap().is_empty(), "complete stem set on the share = done");
        assert_eq!(outcome.skipped.resume, 1);
        assert!(smb.has_file(&remote.join("a.wav")), "source stays untouched");
    }

    #[test]
    fn e2e_engine_failure_isolates_file_without_rename() {
        let (remote, local) = temp_pair("failure");
        let smb = InMemorySmb::new();
        smb.seed(&remote, &["bad.wav", "good.wav"]);
        let config = test_config(&local, &remote);

        let outcome = assemble(&config, &smb, &StubEngine::failing_on("bad")).unwrap();

        assert_eq!((outcome.batch.ok, outcome.batch.failed), (1, 1), "isolation: bad failed, good ok");
        assert_eq!(report::exit_code(&outcome.batch), 1);
        // Rename для failed НЕ вызван: исходник на месте, CONVERTED_ и стемов нет.
        assert!(smb.has_file(&remote.join("bad.wav")));
        assert!(!smb.has_file(&remote.join("CONVERTED_bad.wav")));
        for stem in EXPECTED_STEMS {
            assert!(!smb.has_file(&remote.join("bad").join(format!("{stem}.flac"))));
        }
        // good прошёл весь цикл.
        assert!(smb.has_file(&remote.join("CONVERTED_good.wav")));
        // Отказ попал в отчёт.
        let markdown = fs::read_to_string(local.join(outcome.report_remote.file_name().unwrap())).unwrap();
        assert!(markdown.contains("stub: corrupted wav"), "{markdown}");
    }

    /// Локальная папка (`--local`, [`LocalFs`]) с копией тест-трека:
    /// полный цикл на реальном диске + resume повторным прогоном.
    #[test]
    fn e2e_local_backend_full_cycle_and_resume_on_real_folder() {
        let (remote, local) = temp_pair("local_fs");
        fs::write(remote.join("track.wav"), WAV_BYTES).unwrap(); // копия тест-трека
        let config = test_config(&local, &remote);

        let first = assemble(&config, &LocalFs, &StubEngine::ok()).unwrap();

        assert_eq!((first.batch.ok, first.batch.failed), (1, 0));
        assert!(remote.join("CONVERTED_track.wav").exists());
        assert!(!remote.join("track.wav").exists());
        for stem in EXPECTED_STEMS {
            assert!(remote.join("track").join(format!("{stem}.flac")).exists());
        }
        assert!(first.report_remote.exists(), "report published into the local folder");

        // Повторный прогон: всё скипнуто, движок не звался, код 0.
        let engine = StubEngine::failing_on("track");
        let second = assemble(&config, &LocalFs, &engine).unwrap();
        assert!(engine.called.lock().unwrap().is_empty());
        assert_eq!((second.batch.total, second.skipped.converted), (0, 1));
        assert_eq!(report::exit_code(&second.batch), 0);
    }
}
