//! E2E-сценарии TUI через моки (subtask tui-audio-split-07).
//!
//! Публичный API крейта, без сети и GPU. Попап (paste/валидация) → прогон
//! через реальный `runner::spawn` на `MockSmb`/`MockEngine` (файлы ok, fail
//! и дубль) → сводка Finished → табы Dedup/Log → resume-прогон, retry,
//! cancel через quit-confirm, GPU-шапка из мок-вывода nvidia-smi, мышь.
//! Рендер — тот же `tui::draw` на `TestBackend`, что и у живого event loop.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use audio_split::error::AppError;
use audio_split::pipeline::{EngineEvent, EngineOps, FileStatus, SmbOps};
use audio_split::smb::SmbEntry;
use audio_split::tui::app::{App, State};
use audio_split::tui::event::{AppEvent, RunSpec};
use audio_split::tui::gpu::{GpuEvent, parse_nvidia_smi_csv};
use audio_split::tui::persist::TuiSettings;
use audio_split::tui::runner::{RunnerDeps, RunnerOptions, exit_code, spawn};
use audio_split::tui::{apply_event, draw, handle_mouse, validate_spec};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const EXT: &str = "flac";
const RUN_TIMEOUT: Duration = Duration::from_secs(30);
const SCREEN: ratatui::layout::Rect = ratatui::layout::Rect::new(0, 0, 100, 30);
/// Мок-вывод `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.
const NVIDIA_SMI_SAMPLE: &str = "NVIDIA GeForce RTX 4090, 63, 2100, 24564\n";

// --- утилиты теста -----------------------------------------------------------

/// Нажатие клавиши без модификаторов (как читатель crossterm: Press).
fn key(code: KeyCode) -> AppEvent {
    AppEvent::Key(KeyEvent::new(code, KeyModifiers::empty()))
}

/// Клик «левая кнопка отпущена» в координате кадра.
fn click_up(column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind: MouseEventKind::Up(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::empty(),
    }
}

/// Рендер кадра приложения в строку (тот же `draw`, что в event loop).
fn render(app: &App) -> String {
    let mut terminal = Terminal::new(TestBackend::new(SCREEN.width, SCREEN.height)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal.backend().buffer().content().iter().map(|cell| cell.symbol()).collect()
}

/// Уникальная локальная рабочая директория теста.
fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("audio_split_e2e_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// Раннер-опции как у TUI (Envelope выключен колонками 0 — кроме сценария с ним).
fn opts(local_dir: &Path, remote_dir: &Path) -> RunnerOptions {
    RunnerOptions {
        remote_dir: remote_dir.to_path_buf(),
        local_dir: local_dir.to_path_buf(),
        stem_ext: EXT.to_string(),
        dedup: true,
        pause_between_files: None,
        envelope_columns: 0,
    }
}

/// Применять события раннера к App до `stops` итогов (BatchFinished/RunFailed).
async fn drive(app: &mut App, rx: &mut UnboundedReceiver<AppEvent>, stops: usize) {
    let mut seen = 0;
    while seen < stops {
        let event = tokio::time::timeout(RUN_TIMEOUT, rx.recv())
            .await
            .expect("раннер завис")
            .expect("канал закрыт");
        if matches!(event, AppEvent::BatchFinished { .. } | AppEvent::RunFailed { .. }) {
            seen += 1;
        }
        apply_event(app, &event);
    }
}

/// Хвост фоновых событий (огибающие) после shutdown — тоже в App.
async fn drain(app: &mut App, rx: &mut UnboundedReceiver<AppEvent>) {
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_millis(150), rx.recv()).await {
        apply_event(app, &event);
    }
}

/// Старт прогона из попапа, как живой пользователь: paste путей (с хвостовым
/// `\n` — sanitize_paste должен его снять) → Enter → спека на валидацию.
fn start_from_popup(app: &mut App, remote: &Path, local: &Path) -> RunSpec {
    app.handle(&key(KeyCode::Enter)); // Idle → Config
    assert_eq!(app.state(), State::Config);
    app.open_config(TuiSettings { local_mode: true, dedup: true, ..TuiSettings::default() });

    app.handle(&key(KeyCode::Tab)); // Host → Share
    app.handle(&key(KeyCode::Tab)); // Share → RemoteDir
    let remote_text = remote.display().to_string();
    app.handle(&AppEvent::Paste { text: format!("{remote_text}\n") });
    app.handle(&key(KeyCode::Tab)); // RemoteDir → LocalDir
    let local_text = local.display().to_string();
    app.handle(&AppEvent::Paste { text: local_text });

    app.handle(&key(KeyCode::Enter)); // быстрые проверки → запрос старта
    let spec = app.take_start_request().expect("Enter в попапе даёт спеку");
    assert_eq!(spec.remote_dir, remote, "paste очищен от перевода строки");
    assert_eq!(spec.local_dir, local);
    assert!(spec.local_mode);
    assert!(spec.dedup);
    spec
}

// --- WAV-фикстуры (RIFF PCM16, как wav_fix в lib, но для внешнего теста) ------

/// Энкодер PCM16-моно WAV в байты.
fn encode_wav(rate: u32, samples: &[f32]) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_size as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1_u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes()); // byte_rate
    bytes.extend_from_slice(&2_u16.to_le_bytes()); // block_align
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&((sample.clamp(-1.0, 1.0) * 32_767.0) as i16).to_le_bytes());
    }
    bytes
}

/// Детерминированная «запись»: тональные вспышки. Огибающая (гейт/амплитуда)
/// задаётся параметрами: у разных файлов она должна быть РАЗНОЙ, иначе
/// дедупликация склеит их в один кластер (она сравнивает громкость, не спектр).
fn wav_bytes(gate_period: f64, gate_width: f64, amp: f64, freqs: &[f64]) -> Vec<u8> {
    let rate = 8_000_u32;
    let samples: Vec<f32> = (0..u64::from(rate) * 2)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            let gate = if t % gate_period < gate_width { 1.0 } else { 0.001 };
            let tone: f64 = freqs
                .iter()
                .enumerate()
                .map(|(k, freq)| 0.7 / (k + 1) as f64 * (t * freq).sin())
                .sum();
            (gate * amp * tone) as f32
        })
        .collect();
    encode_wav(rate, &samples)
}

// --- моки SmbOps/EngineOps (самодостаточные, как src/tui/mocks.rs) -----------

/// SMB-мок: фикстуры директорий/содержимого, отказы get, задержка get.
struct MockSmb {
    dirs: HashMap<PathBuf, Vec<SmbEntry>>,
    exists: HashSet<PathBuf>,
    fail_get: HashSet<PathBuf>,
    bodies: HashMap<PathBuf, Vec<u8>>,
    get_delay: Option<Duration>,
}

impl MockSmb {
    fn new() -> Self {
        Self {
            dirs: HashMap::new(),
            exists: HashSet::new(),
            fail_get: HashSet::new(),
            bodies: HashMap::new(),
            get_delay: None,
        }
    }

    fn with_dir(mut self, dir: &Path, names: &[&str]) -> Self {
        self.dirs.insert(
            dir.to_path_buf(),
            names.iter().map(|n| SmbEntry { name: n.to_string(), is_dir: false, size: 1 }).collect(),
        );
        self
    }

    fn with_dir_entries(mut self, dir: &Path, entries: Vec<SmbEntry>) -> Self {
        self.dirs.insert(dir.to_path_buf(), entries);
        self
    }

    fn with_existing_dir(mut self, dir: &Path) -> Self {
        self.exists.insert(dir.to_path_buf());
        self
    }

    fn with_body(mut self, remote: &Path, bytes: Vec<u8>) -> Self {
        self.bodies.insert(remote.to_path_buf(), bytes);
        self
    }

    fn failing_get(mut self, remote: &Path) -> Self {
        self.fail_get.insert(remote.to_path_buf());
        self
    }

    fn with_get_delay(mut self, delay: Duration) -> Self {
        self.get_delay = Some(delay);
        self
    }

    fn entry(name: &str) -> SmbEntry {
        SmbEntry { name: name.to_string(), is_dir: false, size: 1 }
    }
}

impl SmbOps for MockSmb {
    fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        self.dirs.get(dir).cloned().ok_or_else(|| AppError::Smb {
            operation: "list",
            path: dir.to_path_buf(),
            message: "not found in fixture".to_string(),
        })
    }

    fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
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
            std::fs::create_dir_all(parent).expect("create local dir");
        }
        let bytes = self.bodies.get(remote).cloned().unwrap_or_else(|| b"wav-bytes".to_vec());
        std::fs::write(local, bytes).expect("write local file");
        Ok(())
    }

    fn put(&self, _local: &Path, _remote: &Path) -> Result<(), AppError> {
        Ok(())
    }

    fn create_dir(&self, _dir: &Path) -> Result<(), AppError> {
        Ok(())
    }

    fn rename(&self, _from: &Path, _to: &Path) -> Result<(), AppError> {
        Ok(())
    }

    fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
        Ok(self.exists.contains(path))
    }
}

/// Движок-мок: проигрывает сценарные события (создавая стемы на диске,
/// как реальный драйвер); `fail_file_once` — отказ файла ровно один раз.
struct MockEngine {
    events: Vec<EngineEvent>,
    fail_file_once: Mutex<HashSet<String>>,
    batches: Mutex<Vec<Vec<PathBuf>>>,
}

impl MockEngine {
    fn new(events: Vec<EngineEvent>) -> Self {
        Self {
            events,
            fail_file_once: Mutex::new(HashSet::new()),
            batches: Mutex::new(Vec::new()),
        }
    }

    /// Отказать файлу при ПЕРВОМ вызове, на повторном (retry) — успех.
    fn failing_file_once(self, name: &str) -> Self {
        self.fail_file_once.lock().expect("lock").insert(name.to_string());
        self
    }

    fn called_batches(&self) -> Vec<Vec<PathBuf>> {
        self.batches.lock().expect("lock").clone()
    }
}

impl EngineOps for MockEngine {
    fn separate_batch(
        &self,
        files: &[PathBuf],
        _output_dir: &Path,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<(), AppError> {
        self.batches.lock().expect("lock").push(files.to_vec());
        for event in &self.events {
            if let EngineEvent::FileDone { file, stems, .. } = event {
                let name = Path::new(file).file_name().and_then(|n| n.to_str()).unwrap_or(file);
                if self.fail_file_once.lock().expect("lock").remove(name) {
                    on_event(EngineEvent::FileError {
                        file: file.clone(),
                        error: "engine boom (first attempt)".to_string(),
                    });
                    return Ok(());
                }
                for stem in stems {
                    if let Some(parent) = Path::new(stem).parent() {
                        std::fs::create_dir_all(parent).expect("create stem dir");
                    }
                    std::fs::write(stem, b"flac-bytes").expect("write stem");
                }
            }
            on_event(event.clone());
        }
        Ok(())
    }
}

/// Локальные пути стемов файла (6 штук, порядок EXPECTED_STEMS).
fn stem_paths(local_dir: &Path, base: &str) -> Vec<String> {
    audio_split::pipeline::EXPECTED_STEMS
        .iter()
        .map(|stem| local_dir.join(base).join(format!("{stem}.{EXT}")).to_string_lossy().into_owned())
        .collect()
}

/// Сценарий движка: загрузка модели + per-file progress/done + сводка.
fn engine_events(local: &Path, names: &[&str]) -> Vec<EngineEvent> {
    let mut events = vec![EngineEvent::Progress {
        file: local.join(names[0]).to_string_lossy().into_owned(),
        stage: "load".to_string(),
        pct: 50.0,
    }];
    for (i, name) in names.iter().enumerate() {
        events.push(EngineEvent::Progress {
            file: local.join(name).to_string_lossy().into_owned(),
            stage: "separate".to_string(),
            pct: 40.0 + i as f64 * 10.0,
        });
        let file = local.join(name).to_string_lossy().into_owned();
        let base = Path::new(name).file_stem().and_then(|n| n.to_str()).unwrap_or(name);
        events.push(EngineEvent::FileDone {
            file,
            stems: stem_paths(local, base),
            seconds: 1.0,
        });
    }
    events.push(EngineEvent::BatchDone { ok: names.len() as u32, failed: 0 });
    events
}

fn result_status<'a>(
    result: &'a audio_split::pipeline::BatchResult,
    name: &str,
) -> &'a FileStatus {
    &result.results.iter().find(|r| r.name == name).expect("файл в итоге").status
}

// --- сценарий 1: полный прогон из попапа -------------------------------------

/// Попап (paste/валидация) → прогон (ok + fail + dup) → сводка Finished →
/// табы Dedup/Log → GPU-шапка из мок-вывода nvidia-smi → exit-code.
#[tokio::test]
async fn e2e_popup_to_finished_with_summary_dedup_log_and_gpu() {
    let base = temp_dir("main");
    let work = base.join("work");
    let remote = PathBuf::from("/nas");
    let wav_a = wav_bytes(2.0, 0.25, 0.7, &[220.0, 331.0]);
    let wav_b = wav_bytes(3.0, 0.4, 0.6, &[442.0]);
    let wav_keeper = wav_bytes(5.0, 0.3, 0.65, &[517.0, 733.0]);
    let smb = MockSmb::new()
        .with_dir(&remote, &["a.wav", "b.wav", "keeper.wav", "copy.wav", "fail.wav"])
        .with_body(&remote.join("a.wav"), wav_a)
        .with_body(&remote.join("b.wav"), wav_b)
        .with_body(&remote.join("keeper.wav"), wav_keeper.clone())
        .with_body(&remote.join("copy.wav"), wav_keeper)
        .failing_get(&remote.join("fail.wav"));
    let engine = MockEngine::new(engine_events(&work, &["a.wav", "b.wav", "keeper.wav"]));

    // Попап: paste путей → спека → та же валидация, что в event loop.
    let mut app = App::new();
    let spec = start_from_popup(&mut app, &base, &work);
    assert_eq!(validate_spec(&spec), None, "валидация реальных директорий проходит");
    apply_event(&mut app, &AppEvent::ValidationFinished { spec: spec.clone(), error: None });
    assert_eq!(app.state(), State::Running);

    // GPU-поллер (мок-вывод nvidia-smi) → шапка.
    assert!(render(&app).contains("GPU: n/a"), "без данных — заглушка n/a");
    apply_event(
        &mut app,
        &AppEvent::Gpu(GpuEvent::Info(parse_nvidia_smi_csv(NVIDIA_SMI_SAMPLE))),
    );

    // Прогон через реальный раннер на моках.
    let (tx, mut rx) = unbounded_channel();
    let handle = spawn(RunnerDeps { smb: std::sync::Arc::new(smb), engine: std::sync::Arc::new(engine) }, opts(&work, &remote), tx);
    drive(&mut app, &mut rx, 1).await;
    handle.shutdown();
    let result = handle.finish().await.expect("итог прогона");
    drain(&mut app, &mut rx).await;

    // Авторитетный итог: a/b/keeper ok, copy dup (kept=keeper), fail.wav failed.
    assert_eq!((result.ok, result.failed), (3, 1));
    assert_eq!(exit_code(&result), 1, "есть отказ — код 1");
    assert_eq!(
        result_status(&result, "copy.wav"),
        &FileStatus::Duplicate { kept: "keeper.wav".to_string() }
    );
    assert_eq!(result_status(&result, "fail.wav"), &FileStatus::Failed);
    assert_eq!(result.dedup.pairs.len(), 1, "одна пара дедупликации");

    // UI-модель: Finished, счётчики и статусы строк.
    assert_eq!(app.state(), State::Finished);
    let outcome = app.outcome().expect("итог в App");
    assert_eq!((outcome.ok, outcome.failed), (3, 1));

    // Кадр: сводка footer + GPU-шапка из мок-вывода (шапка показывает
    // бар + VRAM; имя в кадр не выводится — см. widgets::gpu_spans).
    let text = render(&app);
    assert!(text.contains("OK 3 · FAIL 1 · DUP 1 · SKIP 0"), "сводка: {text}");
    assert!(text.contains("2.1/24.0GiB"), "VRAM из nvidia-smi в шапке");
    assert!(!text.contains("GPU: n/a"), "заглушка сменилась данными поллера");

    // Таб Dedup: решения (дубль → хранитель).
    app.handle(&key(KeyCode::Char('2')));
    let text = render(&app);
    assert!(text.contains("copy.wav"), "дубль в таблице Dedup");
    assert!(text.contains("keeper.wav"), "хранитель в таблице Dedup");

    // Таб Log: сводка прогона + пара дедупликации (хвост кольцевого буфера).
    // Ранний отказ fail.wav вытеснен вверх записями стадий — проверяем буфер.
    app.handle(&key(KeyCode::Char('3')));
    let text = render(&app);
    assert!(text.contains("прогон завершён: 3 ок, 1 ошибок"), "{text}");
    assert!(text.contains("дубль copy.wav → хранитель keeper.wav"));
    let messages: Vec<String> =
        app.log().visible().iter().map(|entry| entry.message.clone()).collect();
    assert!(messages.iter().any(|message| {
        message.starts_with("ОШИБКА: fail.wav:") && message.contains("forced failure")
    }), "отказ файла в буфере журнала: {messages:?}");
    assert!(text.contains("дубль copy.wav → хранитель keeper.wav"));

    let _ = std::fs::remove_dir_all(&base);
}

// --- сценарий 2: retry после искусственного фейла -----------------------------

/// Прогон с отказом b.wav (мок-движок) → клик мышью по строке → `r` →
/// retry-прогон заменяет отказ на Ok, итог серии без отказов (код 0).
#[tokio::test]
async fn e2e_retry_after_engine_failure_via_ui_makes_batch_clean() {
    let base = temp_dir("retry");
    let work = base.join("work");
    let remote = PathBuf::from("/nas");
    let smb = MockSmb::new().with_dir(&remote, &["a.wav", "b.wav"]);
    let engine =
        MockEngine::new(engine_events(&work, &["a.wav", "b.wav"])).failing_file_once("b.wav");

    let mut app = App::new();
    let (tx, mut rx) = unbounded_channel();
    let handle = spawn(
        RunnerDeps { smb: std::sync::Arc::new(smb), engine: std::sync::Arc::new(engine) },
        opts(&work, &remote),
        tx,
    );
    drive(&mut app, &mut rx, 1).await;

    // Первый итог: b.wav отказал (exit-код 1), таблица в Finished.
    assert_eq!(app.state(), State::Finished);
    assert_eq!(app.outcome().map(|outcome| outcome.failed), Some(1));

    // Мышью выбираем строку b.wav (вторая строка таблицы) и жмём `r`.
    handle_mouse(&mut app, click_up(10, 7), SCREEN);
    assert_eq!(app.selected_file().map(|file| file.name.as_str()), Some("b.wav"));
    app.handle(&key(KeyCode::Char('r')));
    let retries = app.take_retry_requests();
    assert_eq!(retries, vec!["b.wav".to_string()], "retry запрошен из UI");
    for name in retries {
        handle.request_retry(name);
    }

    drive(&mut app, &mut rx, 1).await; // итог retry-прогона
    handle.shutdown();
    let result = handle.finish().await.expect("итог серии");
    drain(&mut app, &mut rx).await;

    assert_eq!((result.ok, result.failed), (2, 0), "retry замещает отказ (итог серии)");
    assert_eq!(exit_code(&result), 0);
    assert_eq!(result_status(&result, "b.wav"), &FileStatus::Ok);
    // Итог в App — ПОСЛЕДНИЙ батч (только b), а не объединённая серия.
    assert_eq!(app.outcome().map(|outcome| outcome.ok), Some(1));
    let text = render(&app);
    assert!(text.contains("OK 2 · FAIL 0 · DUP 0 · SKIP 0"), "{text}");

    let _ = std::fs::remove_dir_all(&base);
}

// --- сценарий 3: cancel через quit-confirm ------------------------------------

/// `q` во время прогона → confirm → Enter = кооперативная отмена + выход:
/// оба файла Failed («cancelled»), исходники не переименованы, код 1.
#[tokio::test]
async fn e2e_cancel_confirmed_through_quit_dialog_fails_run() {
    let base = temp_dir("cancel");
    let work = base.join("work");
    let remote = PathBuf::from("/nas");
    let smb = MockSmb::new()
        .with_dir(&remote, &["a.wav", "b.wav"])
        .with_get_delay(Duration::from_millis(150));
    let engine = MockEngine::new(engine_events(&work, &["a.wav"]));

    let mut app = App::new();
    let (tx, mut rx) = unbounded_channel();
    let handle = spawn(
        RunnerDeps { smb: std::sync::Arc::new(smb), engine: std::sync::Arc::new(engine) },
        opts(&work, &remote),
        tx,
    );

    // Ждём download первого файла (окно задержки get) и открываем quit-confirm.
    loop {
        let event = tokio::time::timeout(RUN_TIMEOUT, rx.recv()).await.expect("завис").expect("закрыт");
        let started_download =
            matches!(&event, AppEvent::FileStage { name, stage: audio_split::tui::event::Stage::Download } if name == "a.wav");
        apply_event(&mut app, &event);
        if started_download {
            break;
        }
    }
    app.handle(&key(KeyCode::Char('q')));
    assert!(app.confirm_open(), "q в Running открывает подтверждение");
    app.handle(&key(KeyCode::Enter)); // подтвердить: cancel + quit
    assert!(app.cancel_requested() && app.should_quit());

    // Event loop синхронизирует флаги с раннером — повторяем это здесь.
    handle.cancel();
    drive(&mut app, &mut rx, 1).await;
    handle.shutdown();
    let result = handle.finish().await.expect("батч завершён с отказами");
    drain(&mut app, &mut rx).await;

    assert_eq!((result.ok, result.failed), (0, 2));
    assert_eq!(exit_code(&result), 1);
    assert!(result
        .results
        .iter()
        .all(|r| r.error.as_deref().is_some_and(|error| error.contains("cancelled"))));
    assert_eq!(app.state(), State::Finished);
    let text = render(&app);
    assert!(text.contains("OK 0 · FAIL 2 · DUP 0 · SKIP 0"), "{text}");

    let _ = std::fs::remove_dir_all(&base);
}

// --- сценарий 4: повторный прогон = resume-skip -------------------------------

/// Второй прогон на «том же» шаре: CONVERTED_*/DUPLICATE_*/файл с полным
/// набором стемов уходят в SKIP и движку не передаются, обрабатывается
/// только новый файл.
#[tokio::test]
async fn e2e_second_run_skips_converted_duplicate_and_fully_stemmed() {
    let base = temp_dir("resume");
    let work = base.join("work");
    let remote = PathBuf::from("/nas");

    // Прогон 1: обычный файл обработан (на «шаре» появится CONVERTED_ok.wav).
    let mut app = App::new();
    let smb1 = MockSmb::new().with_dir(&remote, &["ok.wav"]);
    let engine1 = MockEngine::new(engine_events(&work, &["ok.wav"]));
    let (tx, mut rx) = unbounded_channel();
    let handle = spawn(
        RunnerDeps { smb: std::sync::Arc::new(smb1), engine: std::sync::Arc::new(engine1) },
        opts(&work, &remote),
        tx,
    );
    drive(&mut app, &mut rx, 1).await;
    handle.shutdown();
    let first = handle.finish().await.expect("итог первого прогона");
    drain(&mut app, &mut rx).await;
    assert_eq!(first.ok, 1);

    // Состояние шары после прогонов: переименованный, дубль, файл со всеми
    // 6 стемами на NAS (resume-done) и новый файл.
    let stems_dir = remote.join("resume");
    let stems: Vec<SmbEntry> = audio_split::pipeline::EXPECTED_STEMS
        .iter()
        .map(|stem| MockSmb::entry(&format!("{stem}.{EXT}")))
        .collect();
    let smb2 = MockSmb::new()
        .with_dir(&remote, &["CONVERTED_ok.wav", "DUPLICATE_d.wav", "resume.wav", "fresh.wav"])
        .with_dir_entries(&stems_dir, stems)
        .with_existing_dir(&stems_dir);
    let engine2 = MockEngine::new(engine_events(&work, &["fresh.wav"]));
    let (tx, mut rx) = unbounded_channel();
    let engine2 = std::sync::Arc::new(engine2);
    let handle = spawn(
        RunnerDeps { smb: std::sync::Arc::new(smb2), engine: engine2.clone() },
        opts(&work, &remote),
        tx,
    );

    // План второго прогона: только fresh.wav в работе, остальные — SKIP.
    let mut plan = None;
    while plan.is_none() {
        let event = tokio::time::timeout(RUN_TIMEOUT, rx.recv()).await.expect("завис").expect("закрыт");
        if let AppEvent::RunStarted { plan: summary } = &event {
            plan = Some(summary.clone());
        }
        apply_event(&mut app, &event);
    }
    let plan = plan.expect("RunStarted получен");
    assert_eq!(plan.pending, vec!["fresh.wav".to_string()]);
    assert_eq!(plan.skipped_converted, vec!["CONVERTED_ok.wav".to_string()]);
    assert_eq!(plan.skipped_duplicate, vec!["DUPLICATE_d.wav".to_string()]);
    assert_eq!(plan.skipped_done, vec!["resume.wav".to_string()]);

    drive(&mut app, &mut rx, 1).await;
    handle.shutdown();
    let second = handle.finish().await.expect("итог второго прогона");
    drain(&mut app, &mut rx).await;

    // Не переобработано: движку ушёл только fresh.wav.
    assert_eq!(engine2.called_batches(), vec![vec![work.join("fresh.wav")]]);
    assert_eq!(second.ok, 1);
    assert_eq!(exit_code(&second), 0);

    // Таблица пересеяна: fresh в работе/готов, три SKIP-строки.
    assert_eq!(app.files().len(), 4);
    assert_eq!(app.state(), State::Finished);
    let text = render(&app);
    assert!(text.contains("OK 1 · FAIL 0 · DUP 0 · SKIP 3"), "{text}");
    assert!(text.contains("SKIP"));
    assert!(text.contains("fresh.wav"));
    assert!(text.contains("CONVERTED_ok.wav"));

    let _ = std::fs::remove_dir_all(&base);
}

// --- сценарий 5: валидация попапа (негативные пути) ---------------------------

/// Валидация отклоняет несуществующую исходную директорию: TUI остаётся
/// в попапе с сообщением, прогон не стартует.
#[tokio::test]
async fn e2e_validation_rejects_missing_remote_dir_and_stays_in_popup() {
    let base = temp_dir("validate");
    let work = base.join("work");
    let mut app = App::new();

    let spec = start_from_popup(&mut app, &base.join("nope"), &work);
    let error = validate_spec(&spec).expect("несуществующая директория — ошибка");
    assert!(error.contains("No such file") || error.contains("no such file"), "{error}");

    apply_event(&mut app, &AppEvent::ValidationFinished { spec, error: Some(error) });
    assert_eq!(app.state(), State::Config, "прогон не стартовал");
    let text = render(&app);
    assert!(text.contains("Запуск прогона"), "попап открыт");
    assert!(text.contains('✗'), "маркер ошибки валидации виден в попапе: {text}");

    let _ = std::fs::remove_dir_all(&base);
}

/// Быстрые проверки попапа: без хоста в SMB-режиме спека не формируется.
#[test]
fn e2e_popup_fast_validation_requires_host_in_smb_mode() {
    let mut app = App::new();
    app.handle(&key(KeyCode::Enter));
    app.open_config(TuiSettings::default()); // local_mode = false, хост пуст

    app.handle(&key(KeyCode::Tab)); // Host → Share
    app.handle(&key(KeyCode::Tab)); // Share → RemoteDir
    app.handle(&AppEvent::Paste { text: "/nas/records".to_string() });
    app.handle(&key(KeyCode::Tab)); // RemoteDir → LocalDir
    app.handle(&AppEvent::Paste { text: "/tmp/work".to_string() });
    app.handle(&key(KeyCode::Enter));

    assert!(app.take_start_request().is_none(), "пустой хост — старт не запрошен");
    assert_eq!(app.state(), State::Config);
    let text = render(&app);
    assert!(text.contains("укажите хост SMB-сервера"), "{text}");
}
