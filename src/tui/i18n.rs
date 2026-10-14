//! I18N интерфейса TUI: переключатель `ru`/`en` без внешних крейтов (KISS).
//!
//! Каталог — [`StrTable`]: структуры полей-констант (строки и чистые
//! функции-форматтеры). Обе таблицы (`RU`/`EN`) — значения одного типа,
//! поэтому компилятор заставляет заполнить каждую строку в ОБОИХ языках:
//! забытое поле — ошибка сборки, а не дыра в локализации.
//!
//! Не переводится (решение владельца): логи движка (`[engine] …`), имена
//! файлов/путей, метки модели (`sw`/`6s`), числовые форматы (ETA, GiB).
//!
//! Переключение — клавиша `l` (см. `app.rs::on_key_global`), выбор
//! сохраняется в `~/.config/audio_split/tui.toml` (`persist.rs`).

/// Язык интерфейса. Дефолт — русский (решение владельца).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lang {
    #[default]
    Ru,
    En,
}

impl Lang {
    /// Переключиться на другой язык (клавиша `l`).
    pub fn toggle(self) -> Self {
        match self {
            Self::Ru => Self::En,
            Self::En => Self::Ru,
        }
    }

    /// Метка для persistence (`tui.toml`).
    pub fn label(self) -> &'static str {
        match self {
            Self::Ru => "ru",
            Self::En => "en",
        }
    }

    /// Язык из метки persistence; неизвестное значение → дефолт `Ru`.
    pub fn from_label(label: &str) -> Self {
        if label.eq_ignore_ascii_case(Self::En.label()) {
            Self::En
        } else {
            Self::Ru
        }
    }
}

/// Каталог всех пользовательских строк UI. Поля — константы или чистые
/// функции-форматтеры (для строк с подстановками).
#[derive(Debug, Clone, Copy)]
pub struct StrTable {
    // --- бейджи таблицы файлов (рендер фикс. ширины 4, см. widgets) ---
    pub badge_run: &'static str,
    pub badge_ok: &'static str,
    pub badge_fail: &'static str,
    pub badge_skip: &'static str,
    pub badge_dup: &'static str,

    // --- стадии (колонка STAGE / статус-строка) ---
    pub stage_queued: &'static str,
    pub stage_download: &'static str,
    pub stage_separate: &'static str,
    pub stage_upload: &'static str,
    pub stage_rename: &'static str,
    /// Терминальные подписи строк (строка не «зависает» на стадии).
    pub stage_done: &'static str,
    pub stage_failed: &'static str,
    pub stage_duplicate: &'static str,
    pub stage_skipped: &'static str,

    // --- табы ---
    pub tab_files: &'static str,
    pub tab_dedup: &'static str,
    pub tab_log: &'static str,
    pub tab_help: &'static str,

    // --- шапка ---
    pub model_loading: fn(f64) -> String,
    pub gpu_na: &'static str,
    /// Имя вычислительного устройства движка (событие `device` драйвера).
    pub computing_on: fn(&str) -> String,

    // --- футер ---
    pub keyhints: &'static str,
    pub files_word: &'static str,
    pub footer_error: fn(&str) -> String,

    // --- таблица файлов ---
    pub files_title: &'static str,
    pub no_files_hint: &'static str,
    pub th_st: &'static str,
    pub th_stage: &'static str,
    pub th_file: &'static str,
    pub th_progress: &'static str,
    pub th_stems: &'static str,
    pub stems_unknown: &'static str,

    // --- панель Details ---
    pub details_title: &'static str,
    pub details_stage: &'static str,
    pub details_source: &'static str,
    pub details_format: &'static str,
    pub details_events: &'static str,
    pub details_empty: &'static str,
    pub wf_processed: &'static str,
    pub wf_left: &'static str,

    // --- таб Dedup ---
    pub dedup_title: &'static str,
    pub dedup_empty: &'static str,
    pub th_dup: &'static str,
    pub th_kept: &'static str,
    pub th_corr: &'static str,
    pub th_gain: &'static str,
    pub th_crest: &'static str,
    pub th_clip: &'static str,
    pub th_ren: &'static str,
    pub th_reason: &'static str,
    pub unparseable_note: fn(&str) -> String,
    pub reason_sole: &'static str,
    pub reason_no_clip: &'static str,
    pub reason_dynamic: fn(f64) -> String,
    pub reason_louder: fn(f64) -> String,
    pub reason_all_clipped: fn(f64) -> String,

    // --- таб Log ---
    pub log_title: &'static str,
    pub log_empty: &'static str,
    pub log_no_match: &'static str,
    pub log_filter_label: &'static str,
    pub log_filter_active: fn(&str, usize, usize) -> String,
    pub log_filter_edit_hints: &'static str,
    pub log_filter_apply_hints: &'static str,
    pub log_hints: &'static str,
    pub log_scroll_back: fn(usize) -> String,

    // --- справка (клавиши — языконезависимы, описания переводятся) ---
    pub help_title: &'static str,
    pub help_tabs: &'static str,
    pub help_self: &'static str,
    pub help_start: &'static str,
    pub help_new_cycle: &'static str,
    pub help_pause: &'static str,
    pub help_retry: &'static str,
    pub help_clear_files: &'static str,
    pub help_select: &'static str,
    pub help_filter: &'static str,
    pub help_clear_log: &'static str,
    pub help_scroll: &'static str,
    pub help_confirm: &'static str,
    pub help_close: &'static str,
    pub help_quit: &'static str,
    pub help_reserved: &'static str,
    pub help_lang: &'static str,
    pub help_clip_legend: &'static str,

    // --- confirm выхода ---
    pub confirm_title: &'static str,
    pub confirm_question: &'static str,
    pub confirm_soft_head: &'static str,
    pub confirm_soft_body: &'static str,
    pub confirm_soft_action: &'static str,
    pub confirm_stay: &'static str,
    pub confirm_force_head: &'static str,
    pub confirm_force_body: &'static str,

    // --- попап запуска ---
    pub popup_title: &'static str,
    pub label_host: &'static str,
    pub label_share: &'static str,
    pub label_remote_dir: &'static str,
    pub label_local_dir: &'static str,
    pub label_model: &'static str,
    pub label_dedup: &'static str,
    pub label_local_mode: &'static str,
    pub toggle_on: &'static str,
    pub toggle_off: &'static str,
    pub local_mode_desc: &'static str,
    pub smb_mode_desc: &'static str,
    pub popup_focus_hints: &'static str,
    pub popup_action_hints: &'static str,
    pub checking_dirs: &'static str,
    pub validation_ok: fn(usize) -> String,
    pub validation_ok_zero: &'static str,
    pub smb_hint: &'static str,

    // --- ошибки полей попапа (быстрые проверки до тяжёлой валидации) ---
    pub err_empty_remote: &'static str,
    pub err_empty_local: &'static str,
    pub err_empty_host: &'static str,
    pub err_empty_share: &'static str,
    pub err_local_dir_io: fn(&str) -> String,

    // --- журнал событий (event loop / App) ---
    pub run_started: fn(usize) -> String,
    pub run_failed: fn(&str) -> String,
    pub error_log_prefix: &'static str,
    pub file_done_log: fn(&str) -> String,
    pub file_duplicate_log: fn(&str) -> String,
    pub dups_known_log: fn(&str) -> String,
    pub batch_summary_log: fn(usize, usize, usize, &str) -> String,
    pub dup_pair_log: fn(&str, &str, &str) -> String,
    pub empty_plan_hint: &'static str,
    pub cancel_banner: &'static str,
    pub start_guard_log: &'static str,
    pub force_kill_log: &'static str,
    pub paste_rejected_log: &'static str,

    // --- заглушка малого терминала ---
    pub too_small_title: &'static str,
    pub too_small_size: fn(u16, u16, u16, u16) -> String,
}

/// Русская таблица (дефолт). `static` — стабильный адрес (таблица одна,
/// переключение языка = выбор указателя; `const` инлайнится на местах).
static RU: StrTable = StrTable {
    badge_run: "ПРЦ",
    badge_ok: "ГОТ",
    badge_fail: "ОШБ",
    badge_skip: "ПРОП",
    badge_dup: "ДУБЛ",

    stage_queued: "в очереди",
    stage_download: "загрузка",
    stage_separate: "разделение",
    stage_upload: "заливка",
    stage_rename: "переименование",
    stage_done: "готово",
    stage_failed: "ошибка",
    stage_duplicate: "дубль",
    stage_skipped: "пропущено",

    tab_files: "[1]Файлы",
    tab_dedup: "[2]Дубли",
    tab_log: "[3]Журнал",
    tab_help: "[4]Справка",

    model_loading: |pct| format!("модель: {pct:.0}%"),
    gpu_na: "GPU: n/a",
    computing_on: |name| format!("Считает: {name}"),

    keyhints: "[s]тарт [p]ауза [r]етрай [m]одель [o]тчёт [?]справка [q]выход",
    files_word: "файлы",
    footer_error: |error| format!("ошибка: {error}"),

    files_title: " ФАЙЛЫ ",
    no_files_hint: "нет файлов — Esc → настройка",
    th_st: "СТ",
    th_stage: "СТАДИЯ",
    th_file: "ФАЙЛ",
    th_progress: "ПРОГРЕСС",
    th_stems: "СТЕМЫ",
    stems_unknown: "—",

    details_title: " DETAILS ",
    details_stage: "стадия:",
    details_source: "источник:",
    details_format: "формат:",
    details_events: "события:",
    details_empty: "нет выбранного файла",
    wf_processed: "обработано",
    wf_left: "осталось",

    dedup_title: " Дубли ",
    dedup_empty: "дублей не найдено",
    th_dup: "ДУБЛ",
    th_kept: "ХРАН",
    th_corr: "CORR",
    th_gain: "GAIN дБ",
    th_crest: "CREST дБ",
    th_clip: "CLIP",
    th_ren: "REN",
    th_reason: "ПРИЧИНА",
    unparseable_note: |names| format!("! не разобраны как WAV: {names}"),
    reason_sole: "единственный кандидат",
    reason_no_clip: "без клипа",
    reason_dynamic: |db| format!("динамичнее на {db:.1} дБ"),
    reason_louder: |db| format!("громче на {db:.1} дБ"),
    reason_all_clipped: |db| format!("макс. crest при клипе (+{db:.1} дБ)"),

    log_title: " Журнал ",
    log_empty: "журнал пуст — события прогона появятся здесь",
    log_no_match: "нет записей по фильтру",
    log_filter_label: "фильтр: ",
    log_filter_active: |filter, shown, total| {
        format!("фильтр: \"{filter}\" — {shown} из {total} · ")
    },
    log_filter_edit_hints: "f — изменить · c — очистить",
    log_filter_apply_hints: "  Enter — применить · Esc — отмена",
    log_hints: "f — фильтр · c — очистить · PgUp/PgDn — прокрутка",
    log_scroll_back: |lines| format!("  ↑{lines} от хвоста"),

    help_title: " Справка ",
    help_tabs: "табы: Файлы, Дубли, Журнал, Справка",
    help_self: "эта справка (повторно — назад)",
    help_start: "настройки прогона (Idle)",
    help_new_cycle: "новый цикл (Finished)",
    help_pause: "пауза между файлами (Running)",
    help_retry: "повтор выбранного упавшего файла",
    help_clear_files: "очистить список файлов (Idle/Finished)",
    help_select: "выбор файла в таблице",
    help_filter: "фильтр журнала по подстроке (Log)",
    help_clear_log: "очистить журнал (Log)",
    help_scroll: "прокрутка журнала (Log)",
    help_confirm: "подтвердить в диалогах и попапе",
    help_close: "закрыть попап/справку/диалог",
    help_quit: "выход; в Running — подтверждение отмены",
    help_reserved: "модель/отчёт — зарезервировано",
    help_lang: "язык интерфейса (ru/en)",
    help_clip_legend: " CLIP: K/D — клипует хранитель/дубль · REN: ✓ — переименован",

    confirm_title: " Подтверждение ",
    confirm_question: "Прервать прогон и выйти?",
    confirm_soft_head: "Мягкая отмена — на границе файлов:",
    confirm_soft_body: "текущий файл будет доигран раннером.",
    confirm_soft_action: " — мягко: доиграть текущий файл и выйти · ",
    confirm_stay: " — остаться",
    confirm_force_head: "Повторный Enter/q в баннере «Прерывание…» — форс: движок будет убит,",
    confirm_force_body: "текущий файл останется без rename (resume подберёт его сам).",

    popup_title: " Запуск прогона ",
    label_host: "Хост:",
    label_share: "Шара:",
    label_remote_dir: "Исходная папка:",
    label_local_dir: "Рабочая папка:",
    label_model: "Модель:",
    label_dedup: "Дедуп:",
    label_local_mode: "Локально:",
    toggle_on: "вкл",
    toggle_off: "выкл",
    local_mode_desc: "локальная папка",
    smb_mode_desc: "smb-шара",
    popup_focus_hints: "Tab/Shift-Tab — фокус · Space/←→ — переключить",
    popup_action_hints: "Enter — проверить и старт · Esc — отмена",
    checking_dirs: "проверка директорий…",
    validation_ok: |wav_count| format!("✓ директория доступна · .wav: {wav_count}"),
    validation_ok_zero: "✓ директория доступна · .wav: 0 · .wav не найдены — проверь путь",
    smb_hint: "Распознан SMB/GVFS путь — host/share/папка заполнены",

    err_empty_remote: "укажите исходную директорию",
    err_empty_local: "укажите локальную директорию",
    err_empty_host: "укажите хост SMB-сервера",
    err_empty_share: "укажите имя шары",
    err_local_dir_io: |error| format!("локальная директория: {error}"),

    run_started: |count| format!("прогон начат: {count} файлов к обработке"),
    run_failed: |error| format!("ОШИБКА: прогон не состоялся: {error}"),
    error_log_prefix: "ОШИБКА:",
    file_done_log: |name| format!("{name}: готово"),
    file_duplicate_log: |name| format!("{name}: дубль (снят с обработки)"),
    dups_known_log: |names| format!("дубли распознаны до разделения: {names}"),
    batch_summary_log: |ok, failed, dups, elapsed| {
        format!("прогон завершён: {ok} ок, {failed} ошибок, {dups} дублей за {elapsed}")
    },
    dup_pair_log: |duplicate, kept, reason| {
        format!("дубль {duplicate} → хранитель {kept}: {reason}")
    },
    empty_plan_hint: "0 .wav в директории — проверь remote-dir",
    cancel_banner: "Прерывание… текущий файл доигрывается (форс: Enter ещё раз)",
    start_guard_log: "прогон уже идёт — дождитесь завершения или прервите (q)",
    force_kill_log: "форс: движок остановлен, текущий файл останется без rename (resume)",
    paste_rejected_log: "paste отклонён: идёт прогон",

    too_small_title: "Слишком маленький терминал",
    too_small_size: |need_w, need_h, width, height| {
        format!("нужно {need_w}×{need_h}, сейчас {width}×{height}")
    },
};

/// Английская таблица (нынешние английские строки UI).
static EN: StrTable = StrTable {
    badge_run: "RUN",
    badge_ok: "OK",
    badge_fail: "FAIL",
    badge_skip: "SKIP",
    badge_dup: "DUP",

    stage_queued: "queued",
    stage_download: "download",
    stage_separate: "separate",
    stage_upload: "upload",
    stage_rename: "rename",
    stage_done: "done",
    stage_failed: "failed",
    stage_duplicate: "duplicate",
    stage_skipped: "skipped",

    tab_files: "[1]Files",
    tab_dedup: "[2]Dedup",
    tab_log: "[3]Log",
    tab_help: "[4]Help",

    model_loading: |pct| format!("model: {pct:.0}%"),
    gpu_na: "GPU: n/a",
    computing_on: |name| format!("Compute: {name}"),

    keyhints: "[s]tart [p]ause [r]etry [m]odel [o]pen report [?]help [q]uit",
    files_word: "files",
    footer_error: |error| format!("error: {error}"),

    files_title: " FILES ",
    no_files_hint: "no files — Esc → settings",
    th_st: "ST",
    th_stage: "STAGE",
    th_file: "FILE",
    th_progress: "PROGRESS",
    th_stems: "STEMS",
    stems_unknown: "—",

    details_title: " DETAILS ",
    details_stage: "stage:",
    details_source: "source:",
    details_format: "format:",
    details_events: "events:",
    details_empty: "no file selected",
    wf_processed: "processed",
    wf_left: "left",

    dedup_title: " Dedup ",
    dedup_empty: "no duplicates found",
    th_dup: "DUP",
    th_kept: "KEPT",
    th_corr: "CORR",
    th_gain: "GAIN dB",
    th_crest: "CREST dB",
    th_clip: "CLIP",
    th_ren: "REN",
    th_reason: "REASON",
    unparseable_note: |names| format!("! unparseable as WAV: {names}"),
    reason_sole: "sole candidate",
    reason_no_clip: "no clipping",
    reason_dynamic: |db| format!("more dynamic by {db:.1} dB"),
    reason_louder: |db| format!("louder by {db:.1} dB"),
    reason_all_clipped: |db| format!("max crest while clipping (+{db:.1} dB)"),

    log_title: " Log ",
    log_empty: "log is empty — run events will appear here",
    log_no_match: "no entries match the filter",
    log_filter_label: "filter: ",
    log_filter_active: |filter, shown, total| {
        format!("filter: \"{filter}\" — {shown} of {total} · ")
    },
    log_filter_edit_hints: "f — edit · c — clear",
    log_filter_apply_hints: "  Enter — apply · Esc — cancel",
    log_hints: "f — filter · c — clear · PgUp/PgDn — scroll",
    log_scroll_back: |lines| format!("  ↑{lines} from tail"),

    help_title: " Help ",
    help_tabs: "tabs: Files, Dedup, Log, Help",
    help_self: "this help (again — back)",
    help_start: "run settings (Idle)",
    help_new_cycle: "new cycle (Finished)",
    help_pause: "pause between files (Running)",
    help_retry: "retry selected failed file",
    help_clear_files: "clear file list (Idle/Finished)",
    help_select: "select file in the table",
    help_filter: "filter log by substring (Log)",
    help_clear_log: "clear log (Log)",
    help_scroll: "scroll log (Log)",
    help_confirm: "confirm in dialogs and popup",
    help_close: "close popup/help/dialog",
    help_quit: "quit; in Running — cancel confirmation",
    help_reserved: "model/report — reserved",
    help_lang: "interface language (ru/en)",
    help_clip_legend: " CLIP: K/D — keeper/duplicate clips · REN: ✓ — renamed",

    confirm_title: " Confirm ",
    confirm_question: "Interrupt the run and quit?",
    confirm_soft_head: "Soft cancel — at file boundaries:",
    confirm_soft_body: "the current file will be finished by the runner.",
    confirm_soft_action: " — soft: finish the current file and quit · ",
    confirm_stay: " — stay",
    confirm_force_head: "Second Enter/q on the «Interrupting…» banner — force: the engine will be killed,",
    confirm_force_body: "the current file stays without rename (resume will pick it up).",

    popup_title: " Run setup ",
    label_host: "Host:",
    label_share: "Share:",
    label_remote_dir: "Remote dir:",
    label_local_dir: "Local dir:",
    label_model: "Model:",
    label_dedup: "Dedup:",
    label_local_mode: "Local mode:",
    toggle_on: "on",
    toggle_off: "off",
    local_mode_desc: "local folder",
    smb_mode_desc: "smb share",
    popup_focus_hints: "Tab/Shift-Tab — focus · Space/←→ — toggle",
    popup_action_hints: "Enter — validate and start · Esc — cancel",
    checking_dirs: "checking directories…",
    validation_ok: |wav_count| format!("✓ directory available · .wav: {wav_count}"),
    validation_ok_zero: "✓ directory available · .wav: 0 · no .wav found — check the path",
    smb_hint: "SMB/GVFS path recognized — host/share/dir filled in",

    err_empty_remote: "enter the source directory",
    err_empty_local: "enter the local directory",
    err_empty_host: "enter the SMB server host",
    err_empty_share: "enter the share name",
    err_local_dir_io: |error| format!("local directory: {error}"),

    run_started: |count| format!("run started: {count} files to process"),
    run_failed: |error| format!("ERROR: run did not start: {error}"),
    error_log_prefix: "ERROR:",
    file_done_log: |name| format!("{name}: done"),
    file_duplicate_log: |name| format!("{name}: duplicate (skipped)"),
    dups_known_log: |names| format!("duplicates detected before separation: {names}"),
    batch_summary_log: |ok, failed, dups, elapsed| {
        format!("run finished: {ok} ok, {failed} failed, {dups} dups in {elapsed}")
    },
    dup_pair_log: |duplicate, kept, reason| {
        format!("duplicate {duplicate} → keeper {kept}: {reason}")
    },
    empty_plan_hint: "0 .wav files in the directory — check remote-dir",
    cancel_banner: "Interrupting… the current file is finishing (force: press Enter again)",
    start_guard_log: "a run is already in progress — wait for it to finish or cancel (q)",
    force_kill_log: "force: engine stopped, the current file will stay without rename (resume)",
    paste_rejected_log: "paste rejected: a run is in progress",

    too_small_title: "Terminal too small",
    too_small_size: |need_w, need_h, width, height| {
        format!("need {need_w}×{need_h}, have {width}×{height}")
    },
};

/// Таблица строк по языку (возвращается `&'static` — рендер копирует
/// только нужные строки; переключение языка — просто другой указатель).
pub fn tr(lang: Lang) -> &'static StrTable {
    match lang {
        Lang::Ru => &RU,
        Lang::En => &EN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_switches_both_ways() {
        assert_eq!(Lang::Ru.toggle(), Lang::En);
        assert_eq!(Lang::En.toggle(), Lang::Ru);
        assert_eq!(Lang::Ru.toggle().toggle(), Lang::Ru);
    }

    #[test]
    fn labels_roundtrip_and_fall_back_to_ru() {
        for lang in [Lang::Ru, Lang::En] {
            assert_eq!(Lang::from_label(lang.label()), lang);
        }
        assert_eq!(Lang::from_label("RU"), Lang::Ru);
        assert_eq!(Lang::from_label("bogus"), Lang::Ru, "неизвестное — дефолт");
        assert_eq!(Lang::default(), Lang::Ru);
    }

    #[test]
    fn badge_labels_are_fixed_width_four_in_both_locales() {
        // ФИКСИРОВАННАЯ ширина бейджа 4: русские метки — ровно 4 символа,
        // английские добиваются пробелом при рендере (`format!("{:<4}")`).
        for lang in [Lang::Ru, Lang::En] {
            let table = tr(lang);
            let labels = [
                table.badge_run,
                table.badge_ok,
                table.badge_fail,
                table.badge_skip,
                table.badge_dup,
            ];
            for label in labels {
                assert!(
                    label.chars().count() <= 4,
                    "{lang:?}: метка «{label}» шире 4"
                );
                let rendered = format!("{label:<4}");
                assert_eq!(rendered.chars().count(), 4, "{lang:?}: «{label}»");
            }
        }
        // Русские метки умещаются в 4 колонки (ПРОП/ДУБЛ — ровно 4);
        // более короткие (ПРЦ/ГОТ/ОШБ, RUN/OK/DUP) добиваются пробелом.
        assert_eq!(tr(Lang::Ru).badge_skip.chars().count(), 4);
        assert_eq!(tr(Lang::Ru).badge_dup.chars().count(), 4);
    }

    #[test]
    fn tables_are_distinct_and_complete_by_type() {
        // Полнота гарантируется типом (оба значения — StrTable); здесь
        // проверяем, что таблицы реально разные и тр отдает обе.
        assert!(std::ptr::eq(tr(Lang::Ru), &RU));
        assert!(std::ptr::eq(tr(Lang::En), &EN));
        assert!(!std::ptr::eq(tr(Lang::Ru), tr(Lang::En)));
    }

    #[test]
    fn computing_on_label_in_both_locales() {
        assert_eq!((tr(Lang::Ru).computing_on)("RTX 4090"), "Считает: RTX 4090");
        assert_eq!((tr(Lang::En).computing_on)("RTX 4090"), "Compute: RTX 4090");
    }
}
