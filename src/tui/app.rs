//! Состояние TUI: конечный автомат `Idle → Config → Running → Finished`
//! плюс модель прогона (таблица файлов, стадии/прогресс, огибающие, итог).
//!
//! Все переходы — чистая реакция на [`AppEvent`] (ввод crossterm + события
//! раннера); бизнес-логики пайплайна здесь нет. Управляющие решения
//! (cancel/pause/retry) фиксируются флагами — event loop снимает их и
//! передаёт раннеру (см. `runner::RunnerHandle`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::config::Model;
use crate::pipeline::FileStatus;
use crate::tui::event::{AppEvent, BatchOutcome, PlanSummary, RunSpec, Stage};
use crate::tui::gpu::{GpuEvent, GpuInfo};
use crate::tui::persist::TuiSettings;
use crate::tui::popup::{PopupView, TextFieldView};
use crate::tui::widgets::GpuView;

/// Статус валидации полей попапа запуска.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationStatus {
    /// Проверка ещё не запускалась.
    Idle,
    /// Идёт тяжёлая проверка (листинг шары в spawn_blocking).
    Checking,
    /// Проверки пройдены (прогон запускается).
    Ok,
    /// Ошибка (текст показывается в попапе).
    Failed(String),
}

/// Экраны приложения.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum State {
    /// Стартовый экран.
    #[default]
    Idle,
    /// Настройка прогона (позже — попап с полями).
    Config,
    /// Идёт прогон пайплайна.
    Running,
    /// Прогон завершён (сводка ok/fail/dup; возможны retry'и).
    Finished,
}

/// Почему файл пропущен без обработки (из плана батча).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Уже обработан ранее (`CONVERTED_*`).
    Converted,
    /// Ранее признан дублем (`DUPLICATE_*`).
    DuplicatePrefix,
    /// Resume: все стемы уже на NAS.
    Done,
}

/// Терминальный исход файла в UI. `Duplicate.kept == ""` — хранитель ещё
/// не известен (событие до итога); после `BatchFinished` заполняется.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOutcome {
    /// Полный цикл прошёл.
    Ok,
    /// Отказ на одной из стадий.
    Failed { error: String },
    /// Дубль: обработан хранитель `kept`.
    Duplicate { kept: String },
    /// Пропущен планом (см. [`SkipReason`]).
    Skipped { reason: SkipReason },
}

/// Строка таблицы файлов: имя, стадия, прогресс, огибающая, исход.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FileView {
    /// Имя файла на шаре.
    pub name: String,
    /// Текущая стадия.
    pub stage: Stage,
    /// Процент separate (только на стадии Separate).
    pub pct: Option<f64>,
    /// Полная огибающая пиков (после download; ровно 1 раз на файл).
    pub peaks: Option<Arc<[f32]>>,
    /// Терминальный исход (Ok/Failed/Duplicate/Skipped).
    pub outcome: Option<FileOutcome>,
}

/// Текстовое поле ввода с курсором (индекс в СИМВОЛАХ — unicode-safe).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextField {
    value: String,
    cursor: usize,
}

impl TextField {
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        let cursor = value.chars().count();
        Self { value, cursor }
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Позиция курсора в символах (для рендера и тестов).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn insert(&mut self, c: char) {
        let byte = byte_index(&self.value, self.cursor);
        self.value.insert(byte, c);
        self.cursor += 1;
    }

    /// Вставка фрагмента (bracketed paste): символы добавляются по одному,
    /// границы строки не ломаются.
    pub fn insert_str(&mut self, text: &str) {
        for c in text.chars() {
            self.insert(c);
        }
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let byte = byte_index(&self.value, self.cursor - 1);
        self.value.remove(byte);
        self.cursor -= 1;
    }

    pub fn delete(&mut self) {
        if self.cursor < self.value.chars().count() {
            let byte = byte_index(&self.value, self.cursor);
            self.value.remove(byte);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.value.chars().count());
    }

    pub fn to_start(&mut self) {
        self.cursor = 0;
    }

    pub fn to_end(&mut self) {
        self.cursor = self.value.chars().count();
    }
}

/// Байтовый индекс символа с номером `char_index` (для `String::insert`).
fn byte_index(value: &str, char_index: usize) -> usize {
    value.char_indices().nth(char_index).map_or(value.len(), |(byte, _)| byte)
}

/// Фокус попапа запуска: четыре текстовых поля и три переключателя.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Focus {
    #[default]
    Host,
    Share,
    RemoteDir,
    LocalDir,
    Model,
    Dedup,
    LocalMode,
}

/// Порядок обхода фокуса Tab/Shift-Tab (кольцевой).
const FOCUS_ORDER: [Focus; 7] = [
    Focus::Host,
    Focus::Share,
    Focus::RemoteDir,
    Focus::LocalDir,
    Focus::Model,
    Focus::Dedup,
    Focus::LocalMode,
];

impl Focus {
    pub fn next(self) -> Self {
        let index = FOCUS_ORDER.iter().position(|f| *f == self).unwrap_or_default();
        FOCUS_ORDER[(index + 1) % FOCUS_ORDER.len()]
    }

    pub fn prev(self) -> Self {
        let index = FOCUS_ORDER.iter().position(|f| *f == self).unwrap_or_default();
        FOCUS_ORDER[(index + FOCUS_ORDER.len() - 1) % FOCUS_ORDER.len()]
    }

    /// Активный элемент — текстовое поле (принимает ввод/вставку).
    pub fn is_text(self) -> bool {
        matches!(self, Self::Host | Self::Share | Self::RemoteDir | Self::LocalDir)
    }
}

/// Состояние попапа запуска: поля ввода, переключатели, фокус, валидация.
#[derive(Debug, Clone)]
pub struct ConfigState {
    pub host: TextField,
    pub share: TextField,
    pub remote_dir: TextField,
    pub local_dir: TextField,
    pub model: Model,
    pub dedup: bool,
    pub local_mode: bool,
    pub focus: Focus,
    pub validation: ValidationStatus,
}

impl ConfigState {
    /// Преффилл из пресета (`~/.config/audio_split/tui.toml`).
    pub fn from_preset(preset: &TuiSettings) -> Self {
        Self {
            host: TextField::new(preset.host.clone()),
            share: TextField::new(preset.share.clone()),
            remote_dir: TextField::new(preset.remote_dir.clone()),
            local_dir: TextField::new(preset.local_dir.clone()),
            model: preset.model(),
            dedup: preset.dedup,
            local_mode: preset.local_mode,
            focus: Focus::default(),
            validation: ValidationStatus::Idle,
        }
    }

    pub fn focused_field_mut(&mut self) -> Option<&mut TextField> {
        match self.focus {
            Focus::Host => Some(&mut self.host),
            Focus::Share => Some(&mut self.share),
            Focus::RemoteDir => Some(&mut self.remote_dir),
            Focus::LocalDir => Some(&mut self.local_dir),
            Focus::Model | Focus::Dedup | Focus::LocalMode => None,
        }
    }

    pub fn focused_field(&self) -> Option<&TextField> {
        match self.focus {
            Focus::Host => Some(&self.host),
            Focus::Share => Some(&self.share),
            Focus::RemoteDir => Some(&self.remote_dir),
            Focus::LocalDir => Some(&self.local_dir),
            Focus::Model | Focus::Dedup | Focus::LocalMode => None,
        }
    }

    /// Быстрые проверки до тяжёлой валидации (spawn_blocking с сетью):
    /// директории указаны, в SMB-режиме заданы хост и шара.
    pub fn local_error(&self) -> Option<String> {
        if self.remote_dir.as_str().trim().is_empty() {
            return Some("укажите исходную директорию".to_string());
        }
        if self.local_dir.as_str().trim().is_empty() {
            return Some("укажите локальную директорию".to_string());
        }
        if !self.local_mode {
            if self.host.as_str().trim().is_empty() {
                return Some("укажите хост SMB-сервера".to_string());
            }
            if self.share.as_str().trim().is_empty() {
                return Some("укажите имя шары".to_string());
            }
        }
        None
    }

    /// Снимок значений полей для валидации/старта.
    pub fn to_spec(&self) -> RunSpec {
        RunSpec {
            host: self.host.as_str().trim().to_string(),
            share: self.share.as_str().trim().to_string(),
            remote_dir: PathBuf::from(self.remote_dir.as_str().trim()),
            local_dir: PathBuf::from(self.local_dir.as_str().trim()),
            model: self.model,
            dedup: self.dedup,
            local_mode: self.local_mode,
        }
    }

    /// Клавиатура попапа: навигация фокуса, редактирование, переключатели.
    fn on_key(&mut self, key: KeyEvent) {
        if key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) {
            return; // системные комбинации в поля не вводим
        }
        match key.code {
            KeyCode::Tab => {
                self.focus = if key.modifiers.contains(KeyModifiers::SHIFT) {
                    self.focus.prev()
                } else {
                    self.focus.next()
                };
            }
            KeyCode::Backspace => {
                if let Some(field) = self.focused_field_mut() {
                    field.backspace();
                }
            }
            KeyCode::Delete => {
                if let Some(field) = self.focused_field_mut() {
                    field.delete();
                }
            }
            KeyCode::Left => self.on_horizontal(false),
            KeyCode::Right => self.on_horizontal(true),
            KeyCode::Home => {
                if let Some(field) = self.focused_field_mut() {
                    field.to_start();
                }
            }
            KeyCode::End => {
                if let Some(field) = self.focused_field_mut() {
                    field.to_end();
                }
            }
            KeyCode::Char(c) => self.on_char(c),
            _ => {}
        }
    }

    /// Left/Right: в текстовом поле — курсор, на модели — переключение.
    fn on_horizontal(&mut self, forward: bool) {
        if self.focus == Focus::Model {
            self.toggle_model();
            return;
        }
        if let Some(field) = self.focused_field_mut() {
            if forward {
                field.right();
            } else {
                field.left();
            }
        }
    }

    fn on_char(&mut self, c: char) {
        match self.focus {
            Focus::Model if c == ' ' => self.toggle_model(),
            Focus::Dedup if c == ' ' => self.dedup = !self.dedup,
            Focus::LocalMode if c == ' ' => self.local_mode = !self.local_mode,
            _ => {
                if let Some(field) = self.focused_field_mut() {
                    field.insert(c);
                }
            }
        }
    }

    fn toggle_model(&mut self) {
        self.model = match self.model {
            Model::Sw => Model::SixS,
            Model::SixS => Model::Sw,
        };
    }
}

/// Состояние приложения: экран, модель прогона, управляющие флаги.
#[derive(Debug, Default)]
pub struct App {
    state: State,
    quit_requested: bool,
    /// Запрошена отмена прогона (кооперативная — на границах стадий).
    cancel_requested: bool,
    /// Пауза между файлами (переключается в Running).
    paused: bool,
    /// Прогресс загрузки модели (глобальный, из стадии `load` драйвера).
    model_pct: Option<f64>,
    /// Таблица файлов в порядке плана.
    files: Vec<FileView>,
    /// Выбранная строка таблицы.
    selected: usize,
    /// Итог последнего завершённого прогона.
    outcome: Option<BatchOutcome>,
    /// Текст ошибки, если прогон не состоялся (скан/сеть/отмена до старта).
    run_error: Option<String>,
    /// Запрошенные retry'и (drain'ится event loop'ом в RunnerHandle).
    retry_requests: Vec<String>,
    /// Попап запуска: `Some` пока открыты настройки (State::Config).
    config: Option<ConfigState>,
    /// Запрос на старт прогона (drain'ится event loop'ом → валидация).
    pending_start: Option<RunSpec>,
    /// Снимок GPU-поллера для шапки.
    gpu: Option<GpuView>,
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

    pub fn cancel_requested(&self) -> bool {
        self.cancel_requested
    }

    pub fn paused(&self) -> bool {
        self.paused
    }

    pub fn model_pct(&self) -> Option<f64> {
        self.model_pct
    }

    pub fn files(&self) -> &[FileView] {
        &self.files
    }

    pub fn selected_index(&self) -> usize {
        self.selected
    }

    pub fn selected_file(&self) -> Option<&FileView> {
        self.files.get(self.selected)
    }

    pub fn outcome(&self) -> Option<&BatchOutcome> {
        self.outcome.as_ref()
    }

    pub fn run_error(&self) -> Option<&str> {
        self.run_error.as_deref()
    }

    /// Забрать накопленные запросы retry (очередь очищается).
    pub fn take_retry_requests(&mut self) -> Vec<String> {
        std::mem::take(&mut self.retry_requests)
    }

    /// Попап запуска, пока открыты настройки.
    pub fn config(&self) -> Option<&ConfigState> {
        self.config.as_ref()
    }

    /// Открыть попап с преффиллом из пресета (вызывает event loop).
    pub fn open_config(&mut self, preset: TuiSettings) {
        self.config = Some(ConfigState::from_preset(&preset));
    }

    /// Вид попапа для рендера (`None` — настройки не открыты).
    pub fn popup_view(&self) -> Option<PopupView> {
        let cfg = self.config.as_ref()?;
        let field = |f: &TextField| TextFieldView { value: f.as_str().to_string(), cursor: f.cursor() };
        Some(PopupView {
            host: field(&cfg.host),
            share: field(&cfg.share),
            remote_dir: field(&cfg.remote_dir),
            local_dir: field(&cfg.local_dir),
            model: cfg.model,
            dedup: cfg.dedup,
            local_mode: cfg.local_mode,
            focus: cfg.focus,
            validation: cfg.validation.clone(),
        })
    }

    /// Запрос на старт прогона (очищается: валидация — в event loop).
    pub fn take_start_request(&mut self) -> Option<RunSpec> {
        std::mem::take(&mut self.pending_start)
    }

    /// Снимок GPU для шапки (`None` — «GPU: n/a»).
    pub fn gpu(&self) -> Option<&GpuView> {
        self.gpu.as_ref()
    }

    /// Единственная точка изменения состояния: чистый переход по событию.
    pub fn handle(&mut self, event: &AppEvent) {
        match event {
            AppEvent::Key(key) => self.on_key(*key),
            AppEvent::Paste { text } => self.on_paste(text),
            AppEvent::Gpu(event) => self.on_gpu(event.clone()),
            AppEvent::ValidationFinished { spec, error } => {
                self.on_validation_finished(spec.clone(), error.clone());
            }
            AppEvent::RunStarted { plan } => self.on_run_started(plan),
            AppEvent::RunFailed { error } => {
                self.run_error = Some(error.clone());
                self.state = State::Finished;
            }
            AppEvent::FileStage { name, stage } => self.update_file(name, |file| {
                file.stage = *stage;
                // Вне separate процент не определён (приближение не храним).
                if *stage != Stage::Separate {
                    file.pct = None;
                }
            }),
            AppEvent::FileProgress { name, pct } => self.update_file(name, |file| {
                file.stage = Stage::Separate;
                file.pct = Some(*pct);
            }),
            AppEvent::ModelProgress { pct } => self.model_pct = Some(*pct),
            AppEvent::FileDone { name } => self.update_file(name, |file| {
                file.outcome = Some(FileOutcome::Ok);
                file.pct = Some(100.0);
            }),
            AppEvent::FileFailed { name, error } => self.update_file(name, |file| {
                file.outcome = Some(FileOutcome::Failed { error: error.clone() });
            }),
            AppEvent::FileDuplicate { name } => self.update_file(name, |file| {
                file.outcome = Some(FileOutcome::Duplicate { kept: String::new() });
            }),
            AppEvent::BatchFinished { outcome } => self.on_batch_finished(outcome),
            AppEvent::Envelope { path, peaks } => self.on_envelope(path, peaks.clone()),
            // Resize/Tick/Ignored на автомат не влияют (перерисовка — в event loop).
            AppEvent::Resize { .. } | AppEvent::Tick | AppEvent::Ignored => {}
        }
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
            State::Config => self.on_key_config(key),
            State::Running => match key.code {
                // Esc — запрос кооперативной отмены (не выход): состояние
                // меняет только BatchFinished/RunFailed от раннера.
                KeyCode::Esc => self.cancel_requested = true,
                KeyCode::Char('p') => self.paused = !self.paused,
                KeyCode::Char('r') => self.request_retry_selected(),
                KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
                KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
                _ => {}
            },
            State::Finished => match key.code {
                KeyCode::Enter => self.state = State::Idle,
                KeyCode::Char('q') | KeyCode::Esc => self.quit_requested = true,
                KeyCode::Char('r') => self.request_retry_selected(),
                KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
                KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
                _ => {}
            },
        }
    }

    /// Засев таблицы из плана; сброс состояния прошлого прогона.
    fn on_run_started(&mut self, plan: &PlanSummary) {        self.files = plan
            .pending
            .iter()
            .map(|name| FileView { name: name.clone(), ..FileView::default() })
            .chain(plan.skipped_converted.iter().map(|name| skipped_view(name, SkipReason::Converted)))
            .chain(plan.skipped_duplicate.iter().map(|name| skipped_view(name, SkipReason::DuplicatePrefix)))
            .chain(plan.skipped_done.iter().map(|name| skipped_view(name, SkipReason::Done)))
            .collect();
        self.selected = 0;
        self.state = State::Running;
        self.cancel_requested = false;
        self.paused = false;
        self.model_pct = None;
        self.outcome = None;
        self.run_error = None;
        self.retry_requests.clear();
    }

    /// Авторитетная сверка статусов из [`BatchResult`] (в т.ч. kept дублей)
    /// и переход в Finished. При нескольких прогонах (retry) — итог последнего.
    fn on_batch_finished(&mut self, outcome: &BatchOutcome) {
        for result in &outcome.results {
            self.update_file(&result.name, |file| {
                file.outcome = Some(match &result.status {
                    FileStatus::Ok => FileOutcome::Ok,
                    FileStatus::Failed => FileOutcome::Failed {
                        error: result.error.clone().unwrap_or_default(),
                    },
                    FileStatus::Duplicate { kept } => FileOutcome::Duplicate { kept: kept.clone() },
                });
            });
        }
        self.outcome = Some(outcome.clone());
        self.state = State::Finished;
    }

    /// Огибающая принимается только для известного файла без терминального
    /// исхода (отмена при FAIL/SKIP/DUP — данные уже не нужны виджету).
    fn on_envelope(&mut self, path: &Path, peaks: Arc<[f32]>) {
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            return;
        };
        self.update_file(name, |file| {
            if file.outcome.is_none() {
                file.peaks = Some(peaks.clone());
            }
        });
    }

    fn update_file(&mut self, name: &str, update: impl FnOnce(&mut FileView)) {
        if let Some(file) = self.files.iter_mut().find(|file| file.name == name) {
            update(file);
        }
    }

    /// Retry доступен только выбранному упавшему файлу.
    fn request_retry_selected(&mut self) {
        if let Some(file) = self.files.get(self.selected) {
            if matches!(file.outcome, Some(FileOutcome::Failed { .. })) {
                self.retry_requests.push(file.name.clone());
            }
        }
    }

    fn move_selection(&mut self, delta: isize) {
        let len = self.files.len() as isize;
        if len == 0 {
            return;
        }
        self.selected = (self.selected as isize + delta).clamp(0, len - 1) as usize;
    }

    /// Клавиатура попапа запуска: Enter — старт, Esc — отмена, остальное —
    /// редактирование активного элемента.
    fn on_key_config(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.config = None;
                self.pending_start = None;
                self.state = State::Idle;
            }
            KeyCode::Enter => self.begin_start(),
            _ => {
                if let Some(cfg) = self.config.as_mut() {
                    cfg.on_key(key);
                }
            }
        }
    }

    /// Enter в попапе: быстрые проверки → запрос старта + Checking.
    /// Повторный Enter во время проверки игнорируется.
    fn begin_start(&mut self) {
        let Some(cfg) = self.config.as_ref() else {
            return;
        };
        if cfg.validation == ValidationStatus::Checking {
            return;
        }
        let (validation, pending) = match cfg.local_error() {
            Some(message) => (ValidationStatus::Failed(message), None),
            None => (ValidationStatus::Checking, Some(cfg.to_spec())),
        };
        if let Some(cfg) = self.config.as_mut() {
            cfg.validation = validation;
        }
        self.pending_start = pending;
    }

    /// Вставка из буфера (bracketed paste): в активное текстовое поле,
    /// вне попапа и на переключателях — игнорируется.
    fn on_paste(&mut self, text: &str) {
        if self.state != State::Config {
            return;
        }
        let Some(cfg) = self.config.as_mut() else {
            return;
        };
        if let Some(field) = cfg.focused_field_mut() {
            field.insert_str(&sanitize_paste(text));
        }
    }

    /// Снимок GPU-поллера: `None` — GPU недоступна («GPU: n/a» в шапке).
    fn on_gpu(&mut self, event: GpuEvent) {
        let GpuEvent::Info(info) = event;
        self.gpu = info.as_ref().map(gpu_view);
    }

    /// Итог валидации: ok → Running (конфиг остаётся для шапки), ошибка →
    /// остаёмся в попапе с сообщением. Пока шла проверка, попап могли
    /// закрыть — устаревший результат игнорируется.
    fn on_validation_finished(&mut self, _spec: RunSpec, error: Option<String>) {
        self.pending_start = None;
        if self.state != State::Config {
            return;
        }
        let Some(cfg) = self.config.as_mut() else {
            return;
        };
        match error {
            Some(message) => cfg.validation = ValidationStatus::Failed(message),
            None => {
                cfg.validation = ValidationStatus::Ok;
                self.state = State::Running;
            }
        }
    }
}

/// Строка пропущенного файла (терминальный исход сразу).
fn skipped_view(name: &str, reason: SkipReason) -> FileView {
    FileView {
        name: name.to_string(),
        outcome: Some(FileOutcome::Skipped { reason }),
        ..FileView::default()
    }
}

/// Paste из ФМ может нести перевод строки — в путь он не нужен.
fn sanitize_paste(text: &str) -> String {
    text.chars().filter(|c| *c != '\n' && *c != '\r').collect()
}

/// Снимок nvidia-smi → view шапки (MiB → GiB, `[N/A]`-утилизация → 0).
fn gpu_view(info: &GpuInfo) -> GpuView {
    GpuView {
        name: info.name.clone(),
        used_gib: info.mem_used_mib as f32 / 1024.0,
        total_gib: info.mem_total_mib as f32 / 1024.0,
        util_pct: info.util_percent.map_or(0.0, f64::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::FileResult;
    use crate::tui::event::BatchOutcome;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::path::PathBuf;
    use std::time::Duration;

    fn press(code: KeyCode) -> AppEvent {
        AppEvent::Key(KeyEvent::new(code, KeyModifiers::empty()))
    }

    fn start_run(app: &mut App) {
        app.handle(&AppEvent::RunStarted {
            plan: PlanSummary {
                pending: ["a.wav", "b.wav"].iter().map(|s| s.to_string()).collect(),
                skipped_converted: vec!["CONVERTED_old.wav".to_string()],
                skipped_duplicate: vec!["DUPLICATE_z.wav".to_string()],
                skipped_done: vec!["done.wav".to_string()],
            },
        });
    }

    fn file_result(name: &str, status: FileStatus) -> FileResult {
        FileResult { name: name.to_string(), status, error: None, elapsed: Duration::from_secs(1) }
    }

    fn outcome_fixture() -> BatchOutcome {
        BatchOutcome {
            total: 2,
            ok: 1,
            failed: 1,
            results: vec![
                file_result("a.wav", FileStatus::Ok),
                FileResult {
                    name: "b.wav".to_string(),
                    status: FileStatus::Failed,
                    error: Some("engine boom".to_string()),
                    elapsed: Duration::ZERO,
                },
            ],
            dedup: Default::default(),
            elapsed: Duration::from_secs(5),
        }
    }

    // --- скелет: переходы экранов -------------------------------------------

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
    fn config_enter_without_open_popup_stays_in_config() {
        // Попап открывает event loop (преффилл из пресета); без него Enter
        // в Config — no-op, из настроек не выходим.
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        assert_eq!(app.state(), State::Config);
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
    fn running_q_is_ignored_quit_confirm_comes_later() {
        let mut app = App::new();
        start_run(&mut app);
        app.handle(&press(KeyCode::Char('q')));
        assert!(!app.should_quit());
        assert_eq!(app.state(), State::Running);
    }

    #[test]
    fn finished_enter_starts_new_cycle() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        app.handle(&AppEvent::BatchFinished { outcome: outcome_fixture() });
        app.handle(&press(KeyCode::Enter));
        assert_eq!(app.state(), State::Idle);
    }

    #[test]
    fn finished_q_requests_quit() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Enter));
        app.handle(&press(KeyCode::Enter));
        app.handle(&AppEvent::BatchFinished { outcome: outcome_fixture() });
        app.handle(&press(KeyCode::Char('q')));
        assert!(app.should_quit());
    }

    #[test]
    fn tick_and_resize_do_not_change_state() {
        let mut app = App::new();
        app.handle(&AppEvent::Tick);
        app.handle(&AppEvent::Resize { width: 100, height: 30 });
        app.handle(&AppEvent::Ignored);
        assert_eq!(app.state(), State::Idle);
    }

    #[test]
    fn non_press_key_events_are_ignored() {
        let release = AppEvent::Key(KeyEvent::new_with_kind(
            KeyCode::Enter,
            KeyModifiers::empty(),
            KeyEventKind::Release,
        ));
        let mut app = App::new();
        app.handle(&release);
        assert_eq!(app.state(), State::Idle);
    }

    // --- план прогона --------------------------------------------------------

    #[test]
    fn run_started_seeds_files_and_enters_running() {
        let mut app = App::new();
        start_run(&mut app);

        assert_eq!(app.state(), State::Running);
        let names: Vec<&str> = app.files().iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["a.wav", "b.wav", "CONVERTED_old.wav", "DUPLICATE_z.wav", "done.wav"]);
        assert_eq!(app.files()[0].stage, Stage::Queued);
        assert_eq!(app.files()[0].outcome, None);
        assert_eq!(app.files()[2].outcome, Some(FileOutcome::Skipped { reason: SkipReason::Converted }));
        assert_eq!(app.files()[3].outcome, Some(FileOutcome::Skipped { reason: SkipReason::DuplicatePrefix }));
        assert_eq!(app.files()[4].outcome, Some(FileOutcome::Skipped { reason: SkipReason::Done }));
    }

    #[test]
    fn run_started_resets_previous_run_state() {
        let mut app = App::new();
        start_run(&mut app);
        app.handle(&press(KeyCode::Esc));
        app.handle(&press(KeyCode::Char('p')));
        app.handle(&AppEvent::BatchFinished { outcome: outcome_fixture() });

        start_run(&mut app);

        assert!(!app.cancel_requested());
        assert!(!app.paused());
        assert_eq!(app.outcome(), None);
        assert_eq!(app.run_error(), None);
        assert_eq!(app.model_pct(), None);
    }

    // --- стадии и прогресс ---------------------------------------------------

    #[test]
    fn file_stage_transitions_and_pct_resets_outside_separate() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::FileStage { name: "a.wav".into(), stage: Stage::Download });
        assert_eq!(app.files()[0].stage, Stage::Download);

        app.handle(&AppEvent::FileProgress { name: "a.wav".into(), pct: 40.0 });
        assert_eq!(app.files()[0].stage, Stage::Separate);
        assert_eq!(app.files()[0].pct, Some(40.0));

        app.handle(&AppEvent::FileStage { name: "a.wav".into(), stage: Stage::Upload });
        assert_eq!(app.files()[0].stage, Stage::Upload);
        assert_eq!(app.files()[0].pct, None, "вне separate процент сбрасывается");
    }

    #[test]
    fn model_progress_is_global() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::ModelProgress { pct: 75.0 });

        assert_eq!(app.model_pct(), Some(75.0));
        assert_eq!(app.files()[0].pct, None, "загрузка модели — не прогресс файла");
    }

    #[test]
    fn unknown_files_in_events_are_ignored() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::FileProgress { name: "ghost.wav".into(), pct: 50.0 });
        app.handle(&AppEvent::FileDone { name: "ghost.wav".into() });

        assert!(app.files().iter().all(|f| f.name != "ghost.wav"));
        assert_eq!(app.files()[0].outcome, None);
    }

    // --- терминальные исходы --------------------------------------------------

    #[test]
    fn file_done_marks_ok_with_full_progress() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::FileProgress { name: "a.wav".into(), pct: 80.0 });
        app.handle(&AppEvent::FileDone { name: "a.wav".into() });

        assert_eq!(app.files()[0].outcome, Some(FileOutcome::Ok));
        assert_eq!(app.files()[0].pct, Some(100.0));
    }

    #[test]
    fn file_failed_marks_terminal_with_error() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::FileFailed { name: "b.wav".into(), error: "disk full".into() });

        assert_eq!(
            app.files()[1].outcome,
            Some(FileOutcome::Failed { error: "disk full".to_string() })
        );
    }

    #[test]
    fn file_duplicate_marks_dup_before_keeper_known() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::FileDuplicate { name: "b.wav".into() });

        assert_eq!(app.files()[1].outcome, Some(FileOutcome::Duplicate { kept: String::new() }));
    }

    // --- огибающая -------------------------------------------------------------

    #[test]
    fn envelope_stored_for_running_file() {
        let mut app = App::new();
        start_run(&mut app);
        let peaks: Arc<[f32]> = Arc::from(vec![0.5, 0.25]);

        app.handle(&AppEvent::Envelope {
            path: PathBuf::from("/work/local/a.wav"),
            peaks: peaks.clone(),
        });

        assert_eq!(app.files()[0].peaks, Some(peaks));
    }

    #[test]
    fn envelope_discarded_for_terminal_file() {
        let mut app = App::new();
        start_run(&mut app);
        app.handle(&AppEvent::FileFailed { name: "a.wav".into(), error: "boom".into() });
        let peaks: Arc<[f32]> = Arc::from(vec![0.5]);

        app.handle(&AppEvent::Envelope { path: PathBuf::from("/work/a.wav"), peaks });

        assert_eq!(app.files()[0].peaks, None, "FAIL отменяет приём огибающей");
    }

    #[test]
    fn envelope_discarded_for_unknown_file() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::Envelope { path: PathBuf::from("/work/ghost.wav"), peaks: Arc::from(vec![0.5]) });

        assert!(app.files().iter().all(|f| f.peaks.is_none()));
    }

    // --- итог батча и ошибка прогона -------------------------------------------

    #[test]
    fn batch_finished_reconciles_statuses_and_finishes() {
        let mut app = App::new();
        start_run(&mut app);
        app.handle(&AppEvent::FileDuplicate { name: "b.wav".into() });

        app.handle(&AppEvent::BatchFinished { outcome: outcome_fixture() });

        assert_eq!(app.state(), State::Finished);
        assert_eq!(app.files()[0].outcome, Some(FileOutcome::Ok));
        assert_eq!(app.files()[1].outcome, Some(FileOutcome::Failed { error: "engine boom".into() }));
        assert_eq!(app.outcome().map(|o| o.ok), Some(1));
        assert_eq!(app.outcome().map(|o| o.failed), Some(1));
    }

    #[test]
    fn batch_finished_fills_keeper_of_duplicate() {
        let mut app = App::new();
        start_run(&mut app);
        let mut outcome = outcome_fixture();
        outcome.results[1] = file_result("b.wav", FileStatus::Duplicate { kept: "a.wav".to_string() });

        app.handle(&AppEvent::BatchFinished { outcome });

        assert_eq!(app.files()[1].outcome, Some(FileOutcome::Duplicate { kept: "a.wav".to_string() }));
    }

    #[test]
    fn run_failed_finishes_with_error() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&AppEvent::RunFailed { error: "share unreachable".into() });

        assert_eq!(app.state(), State::Finished);
        assert_eq!(app.run_error(), Some("share unreachable"));
    }

    // --- cancel / pause / retry / выбор -----------------------------------------

    #[test]
    fn running_esc_requests_cancel_and_stays_running() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&press(KeyCode::Esc));

        assert!(app.cancel_requested());
        assert_eq!(app.state(), State::Running, "отмена кооперативная: завершает раннер");
    }

    #[test]
    fn p_toggles_pause_in_running_only() {
        let mut app = App::new();
        start_run(&mut app);

        app.handle(&press(KeyCode::Char('p')));
        assert!(app.paused());
        app.handle(&press(KeyCode::Char('p')));
        assert!(!app.paused());

        let mut idle = App::new();
        idle.handle(&press(KeyCode::Char('p')));
        assert!(!idle.paused(), "вне Running пауза не переключается");
    }

    #[test]
    fn r_on_failed_selected_file_queues_retry() {
        let mut app = App::new();
        start_run(&mut app);
        app.handle(&AppEvent::FileFailed { name: "a.wav".into(), error: "boom".into() });
        // выбран 0-й файл (a.wav, Failed).

        app.handle(&press(KeyCode::Char('r')));

        assert_eq!(app.take_retry_requests(), vec!["a.wav".to_string()]);
        assert_eq!(app.take_retry_requests(), Vec::<String>::new(), "очередь drain'ится");
    }

    #[test]
    fn r_on_ok_or_skipped_file_does_nothing() {
        let mut app = App::new();
        start_run(&mut app);
        app.handle(&AppEvent::FileDone { name: "a.wav".into() });

        app.handle(&press(KeyCode::Char('r')));

        assert!(app.take_retry_requests().is_empty());
    }

    #[test]
    fn selection_moves_within_bounds() {
        let mut app = App::new();
        start_run(&mut app); // 5 строк

        app.handle(&press(KeyCode::Up));
        assert_eq!(app.selected_index(), 0, "верхняя граница");

        app.handle(&press(KeyCode::Down));
        app.handle(&press(KeyCode::Char('j')));
        assert_eq!(app.selected_index(), 2);

        for _ in 0..10 {
            app.handle(&press(KeyCode::Down));
        }
        assert_eq!(app.selected_index(), 4, "нижняя граница");
    }

    #[test]
    fn empty_table_selection_is_safe() {
        let mut app = App::new();
        app.handle(&press(KeyCode::Down));
        app.handle(&press(KeyCode::Char('r')));

        assert_eq!(app.selected_index(), 0);
        assert!(app.take_retry_requests().is_empty());
        assert_eq!(app.selected_file(), None);
    }
}
