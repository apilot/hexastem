//! Отчётность: markdown-отчёт на SMB + консольная сводка + exit code (0/1).
//!
//! Всё формирование — чистые функции (текст из [`BatchResult`]/[`Skipped`]/[`Config`],
//! без побочных эффектов); единственная точка с эффектами — [`publish`], который
//! пишет отчёт локально и заливает на SMB через инжектируемый [`SmbOps`]
//! (продакшн — [`SmbClient`](crate::smb::SmbClient), тесты — мок, сети нет).
//!
//! Время прогона рендерится в UTC без внешних крейтов (civil_from_days) —
//! детерминированно и тестируемо; имя отчёта: `split_report_YYYYMMDD-HHMMSS.md`.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::config::{Config, Model};
use crate::dedup::{DedupSummary, PairReport};
use crate::error::AppError;
use crate::pipeline::{BatchResult, EXPECTED_STEM_COUNT, FileResult, FileStatus, SmbOps, DUPLICATE_PREFIX};

// --- метки времени (UTC, без внешних крейтов) -------------------------------------

/// Секунды от эпохи; время до эпохи не используется — берём 0 (отчёт, не лог).
fn unix_seconds(run_time: SystemTime) -> i64 {
    run_time.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Дни от эпохи → (год, месяц, день) — алгоритм civil_from_days (H. Hinnant).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// (год, месяц, день, час, минута, секунда) UTC-момента.
fn utc_parts(run_time: SystemTime) -> (i64, u32, u32, u32, u32, u32) {
    let secs = unix_seconds(run_time);
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400) as u32;
    (y, mo, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Момент → `YYYY-MM-DD HH:MM:SS` (UTC) — для шапки отчёта.
pub fn format_utc(run_time: SystemTime) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(run_time);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// Момент → имя файла отчёта `split_report_YYYYMMDD-HHMMSS.md` (UTC).
pub fn report_name(run_time: SystemTime) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(run_time);
    format!("split_report_{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}.md")
}

/// Длительность → компактно: `12.5s` | `1m 05s` | `1h 02m 03s`.
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{:.1}s", d.as_secs_f64())
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m {:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}

// --- чистое формирование ------------------------------------------------------------

/// Файлы, пропущенные планом батча (не обрабатывались в этом прогоне).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Skipped {
    /// Уже обработаны ранее (префикс `CONVERTED_*`).
    pub converted: usize,
    /// Resume: все стемы уже на NAS.
    pub resume: usize,
    /// Ранее признаны дублями (префикс `DUPLICATE_*`).
    pub duplicates: usize,
}

/// Код возврата прогона: 0 — нет отказов (успех/скип), 1 — есть хотя бы один сбой.
pub fn exit_code(batch: &BatchResult) -> i32 {
    if batch.failed == 0 {
        0
    } else {
        1
    }
}

/// Markdown-отчёт прогона — ЧИСТАЯ функция: тот же вход → тот же текст.
/// Хвосты сообщений об ошибках уже усечены пайплайном (≤300 символов stderr).
pub fn generate_markdown(batch: &BatchResult, skipped: &Skipped, config: &Config, run_time: SystemTime) -> String {
    let mut md = markdown_header(config, run_time);
    md.push_str(&markdown_summary(batch, skipped));
    md.push_str(&markdown_files(batch));
    md.push_str(&markdown_dedup(&batch.dedup));
    md.push_str(&markdown_errors(batch));
    md
}

/// Человекочитаемая консольная сводка — ЧИСТАЯ функция (печать — на вызывающем).
pub fn console_summary(batch: &BatchResult, skipped: &Skipped) -> String {
    let mut out = format!(
        "audio_split: всего={} ok={} failed={} skip(resume)={} skip(converted)={} dup={} время={}\n",
        batch.total,
        batch.ok,
        batch.failed,
        skipped.resume,
        skipped.converted,
        batch.dedup.pairs.len(),
        format_duration(batch.elapsed),
    );
    for r in batch.results.iter().filter(|r| r.status == FileStatus::Failed) {
        out.push_str(&format!("  ошибка {}: {}\n", r.name, r.error.as_deref().unwrap_or("unknown error")));
    }
    out
}

/// Короткое имя модели для отчёта.
fn model_name(model: Model) -> &'static str {
    match model {
        Model::Sw => "sw",
        Model::SixS => "6s",
    }
}

/// Шапка: когда/где/чем гоняли (дата-время, host/share, remote-dir, модель).
fn markdown_header(config: &Config, run_time: SystemTime) -> String {
    format!(
        "# audio_split — отчёт о прогоне\n\n\
         - Дата/время (UTC): {ts}\n\
         - Хост/шара: //{host}/{share}\n\
         - Удалённая директория: {dir}\n\
         - Модель: {model} ({checkpoint})\n\
         - Формат стемов: {fmt}\n",
        ts = format_utc(run_time),
        host = config.host,
        share = config.share,
        dir = config.remote_dir.display(),
        model = model_name(config.model),
        checkpoint = config.model.checkpoint(),
        fmt = config.output_format,
    )
}

/// Сводная таблица: обработано / успех / провал / дубли / скипы / время.
/// Дубли — не отказ: в `failed` не входят, на exit code не влияют.
fn markdown_summary(batch: &BatchResult, skipped: &Skipped) -> String {
    format!(
        "\n## Сводка\n\n\
         | Метрика | Значение |\n\
         |---|---|\n\
         | Всего обработано | {total} |\n\
         | Успешно | {ok} |\n\
         | С ошибкой | {failed} |\n\
         | Снято как дубли (dedup) | {dups} |\n\
         | Пропущено (resume) | {resume} |\n\
         | Пропущено (CONVERTED_*) | {converted} |\n\
         | Пропущено (DUPLICATE_*) | {dup_skip} |\n\
         | Общее время | {elapsed} |\n",
        total = batch.total,
        ok = batch.ok,
        failed = batch.failed,
        dups = batch.dedup.pairs.len(),
        resume = skipped.resume,
        converted = skipped.converted,
        dup_skip = skipped.duplicates,
        elapsed = format_duration(batch.elapsed),
    )
}

/// Таблица по файлам. Колонка «Стемы» отражает инвариант пайплайна:
/// `Ok` означает «6/6 стемов залиты и исходник переименован».
fn markdown_files(batch: &BatchResult) -> String {
    let rows: String = batch.results.iter().map(file_row).collect();
    format!(
        "\n## Файлы\n\n\
         | Файл | Статус | Стемы | Время | Ошибка |\n\
         |---|---|---|---|---|\n{rows}",
    )
}

/// Строка таблицы одного файла. «Время» — тот же [`format_duration`], что и в
/// сводке (единообразие: «12.5s» | «14m 07s» | «1h 02m 03s»).
fn file_row(r: &FileResult) -> String {
    let stems = match &r.status {
        FileStatus::Ok => format!("{EXPECTED_STEM_COUNT}/{EXPECTED_STEM_COUNT}"),
        _ => "—".to_string(),
    };
    format!(
        "| `{}` | {} | {stems} | {} | {} |\n",
        r.name,
        status_label(&r.status),
        format_duration(r.elapsed),
        r.error.as_deref().unwrap_or(""),
    )
}

/// Секция «Дедупликация»: таблица пар дубль → хранитель (corr по окнам,
/// гейн, крест-факторы, клип, решение) + примечание о файлах с нечитаемым
/// заголовком. Пустая стадия — секция не выводится.
fn markdown_dedup(summary: &DedupSummary) -> String {
    if summary.pairs.is_empty() && summary.unparseable.is_empty() {
        return String::new();
    }
    let mut md = String::from("\n## Дедупликация\n");
    if !summary.pairs.is_empty() {
        md.push_str(
            "\n| Дубль | Хранитель | Corr (старт/середина/конец) | Гейн, дБ | Крест хранителя, дБ | Крест дубля, дБ | Клип | Решение |\n\
             |---|---|---|---|---|---|---|---|\n",
        );
        for pair in &summary.pairs {
            md.push_str(&dedup_pair_row(pair));
        }
    }
    if !summary.unparseable.is_empty() {
        md.push_str("\nЗаголовок не разобран как PCM WAV — считаны уникальными: ");
        md.push_str(
            &summary
                .unparseable
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        md.push('\n');
    }
    md
}

/// Строка таблицы одной пары дубль → хранитель.
fn dedup_pair_row(pair: &PairReport) -> String {
    let decision = if pair.renamed {
        format!("`{DUPLICATE_PREFIX}{}`", pair.duplicate)
    } else if let Some(err) = &pair.rename_error {
        format!("rename не удался: {err}")
    } else {
        "ждёт Ok хранителя".to_string()
    };
    let clip = match (pair.kept_clipping, pair.dup_clipping) {
        (true, true) => "оба",
        (true, false) => "хранитель",
        (false, true) => "дубль",
        (false, false) => "—",
    };
    format!(
        "| `{dup}` | `{kept}` | {c0:.3}/{c1:.3}/{c2:.3} | {gain:+.1} | {crest_kept:.1} | {crest_dup:.1} | {clip} | {decision} |\n",
        dup = pair.duplicate,
        kept = pair.kept,
        c0 = pair.corr[0],
        c1 = pair.corr[1],
        c2 = pair.corr[2],
        gain = pair.gain_db,
        crest_kept = pair.crest_kept_db,
        crest_dup = pair.crest_dup_db,
    )
}

/// Секция деталей ошибок (пуста, если отказов нет).
fn markdown_errors(batch: &BatchResult) -> String {
    let failed: Vec<&FileResult> = batch.results.iter().filter(|r| r.status == FileStatus::Failed).collect();
    if failed.is_empty() {
        return String::new();
    }
    let mut md = String::from("\n## Ошибки\n");
    for r in failed {
        let message = r.error.as_deref().unwrap_or("unknown error");
        md.push_str(&format!("\n### `{}`\n\n```\n{message}\n```\n", r.name));
    }
    md
}

fn status_label(status: &FileStatus) -> &'static str {
    match status {
        FileStatus::Ok => "ok",
        FileStatus::Failed => "FAILED",
        FileStatus::Duplicate { .. } => "duplicate",
    }
}

// --- публикация ---------------------------------------------------------------------

/// Записывает markdown локально (`local_dir/<name>`) и заливает на SMB
/// в корень `remote_dir` через инжектируемый [`SmbOps`]; возвращает удалённый путь.
pub fn publish(
    smb: &dyn SmbOps,
    local_dir: &Path,
    remote_dir: &Path,
    name: &str,
    markdown: &str,
) -> Result<PathBuf, AppError> {
    let local = local_dir.join(name);
    std::fs::write(&local, markdown).map_err(|source| AppError::Io { path: local.clone(), source })?;
    let remote = remote_dir.join(name);
    smb.put(&local, &remote)?;
    Ok(remote)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Mutex;
    use std::time::Instant;

    use super::*;
    use crate::pipeline::summarize_with_dedup;
    use crate::smb::SmbEntry;

    const RUN_TIME_SECS: u64 = 1_000_000_000; // 2001-09-09 01:46:40 UTC

    fn run_time() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(RUN_TIME_SECS)
    }

    // --- фикстуры -------------------------------------------------------------

    fn result_ok(name: &str, elapsed: Duration) -> FileResult {
        FileResult { name: name.into(), status: FileStatus::Ok, error: None, elapsed }
    }

    fn result_failed(name: &str, error: &str, elapsed: Duration) -> FileResult {
        FileResult { name: name.into(), status: FileStatus::Failed, error: Some(error.into()), elapsed }
    }

    fn batch(results: Vec<FileResult>, elapsed: Duration) -> BatchResult {
        let mut summary = summarize_with_dedup(results, DedupSummary::default(), Instant::now());
        summary.elapsed = elapsed;
        summary
    }

    /// Пара дубль → хранитель для секции «Дедупликация» (числа кейса
    /// unknown 2/3).
    fn pair_report() -> PairReport {
        PairReport {
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
        }
    }

    fn config() -> Config {
        Config {
            host: "nas.local".into(),
            share: "music".into(),
            remote_dir: PathBuf::from("kirtan"),
            local_dir: PathBuf::from("/tmp/work"),
            model: Model::Sw,
            venv_path: PathBuf::from("/opt/venv"),
            output_format: "flac".into(),
            smb_creds_file: None,
            local: false,
            no_dedup: false,
        }
    }

    fn skipped() -> Skipped {
        Skipped { converted: 1, resume: 4, duplicates: 2 }
    }

    /// Уникальная локальная директория теста (параллельные тесты не пересекаются).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("audio_split_report_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// SMB-мок: ловит только put (контент+путь); остальные операции не нужны.
    struct PutCapture(Mutex<Option<(PathBuf, String)>>);

    impl SmbOps for PutCapture {
        fn list(&self, _dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
            Err(AppError::Pipeline("unused in this test".into()))
        }

        fn get(&self, _remote: &Path, _local: &Path) -> Result<(), AppError> {
            Err(AppError::Pipeline("unused in this test".into()))
        }

        fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
            let content = fs::read_to_string(local)
                .map_err(|source| AppError::Io { path: local.to_path_buf(), source })?;
            *self.0.lock().unwrap() = Some((remote.to_path_buf(), content));
            Ok(())
        }

        fn create_dir(&self, _dir: &Path) -> Result<(), AppError> {
            Err(AppError::Pipeline("unused in this test".into()))
        }

        fn rename(&self, _from: &Path, _to: &Path) -> Result<(), AppError> {
            Err(AppError::Pipeline("unused in this test".into()))
        }

        fn remote_exists(&self, _path: &Path) -> Result<bool, AppError> {
            Err(AppError::Pipeline("unused in this test".into()))
        }
    }

    // --- время и имена -----------------------------------------------------------

    #[test]
    fn format_utc_renders_known_epochs() {
        assert_eq!(format_utc(UNIX_EPOCH), "1970-01-01 00:00:00");
        // Високосный день.
        assert_eq!(format_utc(UNIX_EPOCH + Duration::from_secs(951_782_400)), "2000-02-29 00:00:00");
        assert_eq!(format_utc(run_time()), "2001-09-09 01:46:40");
    }

    #[test]
    fn report_name_follows_compact_format() {
        assert_eq!(report_name(run_time()), "split_report_20010909-014640.md");
        assert_eq!(report_name(UNIX_EPOCH), "split_report_19700101-000000.md");
    }

    #[test]
    fn format_duration_variants() {
        assert_eq!(format_duration(Duration::ZERO), "0.0s");
        assert_eq!(format_duration(Duration::from_millis(12_500)), "12.5s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 05s");
        // Боевой кейс: файл 14 мин 07 сек (847 c) — не «14.7s».
        assert_eq!(format_duration(Duration::from_secs(847)), "14m 07s");
        assert_eq!(format_duration(Duration::from_secs(3_723)), "1h 02m 03s");
    }

    // --- exit code -----------------------------------------------------------------

    #[test]
    fn exit_code_is_zero_only_without_failures() {
        let all_ok = batch(vec![result_ok("a.wav", Duration::ZERO)], Duration::ZERO);
        let mixed = batch(
            vec![result_ok("a.wav", Duration::ZERO), result_failed("b.wav", "boom", Duration::ZERO)],
            Duration::ZERO,
        );
        assert_eq!(exit_code(&all_ok), 0);
        assert_eq!(exit_code(&batch(vec![], Duration::ZERO)), 0, "empty batch is success");
        assert_eq!(exit_code(&mixed), 1);
    }

    // --- markdown ------------------------------------------------------------------

    #[test]
    fn markdown_mixed_run_matches_golden_snapshot() {
        let results = vec![
            result_ok("a.wav", Duration::from_millis(12_500)),
            result_failed("b.wav", "CUDA out of memory", Duration::from_secs(3)),
            // 14 мин 07 сек — колонка «Время» тем же format_duration, что и сводка.
            result_ok("c.wav", Duration::from_secs(847)),
        ];
        let batch = batch(results, Duration::from_secs(65));
        let expected = r#"# audio_split — отчёт о прогоне

- Дата/время (UTC): 2001-09-09 01:46:40
- Хост/шара: //nas.local/music
- Удалённая директория: kirtan
- Модель: sw (BS-Roformer-SW.ckpt)
- Формат стемов: flac

## Сводка

| Метрика | Значение |
|---|---|
| Всего обработано | 3 |
| Успешно | 2 |
| С ошибкой | 1 |
| Снято как дубли (dedup) | 0 |
| Пропущено (resume) | 4 |
| Пропущено (CONVERTED_*) | 1 |
| Пропущено (DUPLICATE_*) | 2 |
| Общее время | 1m 05s |

## Файлы

| Файл | Статус | Стемы | Время | Ошибка |
|---|---|---|---|---|
| `a.wav` | ok | 6/6 | 12.5s |  |
| `b.wav` | FAILED | — | 3.0s | CUDA out of memory |
| `c.wav` | ok | 6/6 | 14m 07s |  |

## Ошибки

### `b.wav`

```
CUDA out of memory
```
"#;

        let md = generate_markdown(&batch, &skipped(), &config(), run_time());

        assert_eq!(md, expected);
    }

    #[test]
    fn markdown_all_ok_has_no_errors_section() {
        let batch = batch(vec![result_ok("a.wav", Duration::ZERO)], Duration::from_secs(5));

        let md = generate_markdown(&batch, &Skipped::default(), &config(), run_time());

        assert!(md.contains("| Успешно | 1 |"));
        assert!(md.contains("Пропущено (resume) | 0 |"));
        assert!(!md.contains("## Ошибки"));
        assert!(!md.contains("## Дедупликация"), "пустая стадия — секции нет: {md}");
    }

    // --- дедупликация ----------------------------------------------------------------

    #[test]
    fn markdown_dedup_section_renders_pairs_and_unparseable_notes() {
        let results = vec![
            result_ok("unknown 2.wav", Duration::from_secs(10)),
            FileResult {
                name: "unknown 3.wav".into(),
                status: FileStatus::Duplicate { kept: "unknown 2.wav".into() },
                error: None,
                elapsed: Duration::ZERO,
            },
        ];
        let mut summary = summarize_with_dedup(
            results,
            DedupSummary { pairs: vec![pair_report()], unparseable: vec!["broken.wav".into()] },
            Instant::now(),
        );
        summary.elapsed = Duration::from_secs(30);

        let md = generate_markdown(&summary, &skipped(), &config(), run_time());

        assert!(md.contains("## Дедупликация"), "{md}");
        assert!(md.contains("| Снято как дубли (dedup) | 1 |"), "{md}");
        assert!(
            md.contains("| `unknown 3.wav` | `unknown 2.wav` | 0.985/0.999/0.991 | +11.1 | 28.2 | 23.0 | — | `DUPLICATE_unknown 3.wav` |"),
            "{md}"
        );
        assert!(md.contains("Заголовок не разобран как PCM WAV — считаны уникальными: `broken.wav`"), "{md}");
    }

    #[test]
    fn duplicate_status_labeled_and_does_not_affect_exit_code() {
        let results = vec![
            result_ok("kept.wav", Duration::ZERO),
            FileResult {
                name: "dup.wav".into(),
                status: FileStatus::Duplicate { kept: "kept.wav".into() },
                error: None,
                elapsed: Duration::ZERO,
            },
        ];
        let batch = batch(results, Duration::ZERO);

        assert_eq!(exit_code(&batch), 0, "дубли — не отказы");
        let md = generate_markdown(&batch, &Skipped::default(), &config(), run_time());
        assert!(md.contains("| `dup.wav` | duplicate | — |"), "{md}");
        assert!(md.contains("| С ошибкой | 0 |"), "{md}");
    }

    // --- консольная сводка ----------------------------------------------------------

    #[test]
    fn console_summary_all_ok_is_one_line() {
        let batch = batch(
            vec![result_ok("a.wav", Duration::ZERO), result_ok("b.wav", Duration::ZERO)],
            Duration::from_secs(30),
        );

        let out = console_summary(&batch, &Skipped::default());

        assert!(out.contains("ok=2 failed=0"), "{out}");
        assert!(out.contains("skip(resume)=0 skip(converted)=0 dup=0"), "{out}");
        assert!(!out.contains("ошибка"), "no failure lines expected: {out}");
    }

    #[test]
    fn console_summary_failed_shows_error_tails() {
        let batch = batch(vec![result_failed("b.wav", "CUDA out of memory", Duration::ZERO)], Duration::ZERO);

        let out = console_summary(&batch, &skipped());

        assert!(out.contains("ok=0 failed=1"), "{out}");
        assert!(out.contains("ошибка b.wav: CUDA out of memory"), "{out}");
    }

    // --- публикация -----------------------------------------------------------------

    #[test]
    fn publish_writes_locally_and_uploads_to_remote_root() {
        let dir = temp_dir("publish");
        let smb = PutCapture(Mutex::new(None));

        let remote = publish(&smb, &dir, Path::new("/remote"), "split_report_x.md", "# md").unwrap();

        assert_eq!(remote, PathBuf::from("/remote/split_report_x.md"), "report goes to remote_dir root");
        let (put_remote, content) = smb.0.lock().unwrap().clone().expect("put captured");
        assert_eq!(put_remote, PathBuf::from("/remote/split_report_x.md"));
        assert_eq!(content, "# md", "uploaded content equals generated markdown");
    }

    #[test]
    fn publish_fails_when_local_write_is_impossible() {
        let smb = PutCapture(Mutex::new(None));
        let missing = temp_dir("publish_missing_parent").join("no/such/dir");

        let result = publish(&smb, &missing, Path::new("/remote"), "split_report_x.md", "# md");

        assert!(matches!(result, Err(AppError::Io { .. })), "{result:?}");
        assert!(smb.0.lock().unwrap().is_none(), "nothing uploaded after a local write failure");
    }
}
