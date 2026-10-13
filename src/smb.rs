//! Обёртка над smbclient: list/get/put/rename по шаре, без монтирования.
//!
//! Креды — никогда не в коде и не в argv: auth-файл (`--authentication-file=`,
//! путь из `config.smb_creds_file`) либо окружение (`SMB_USER`/`SMB_PASS`
//! пробрасываются дочернему процессу как `USER`/`PASSWD`). Парсинг листинга и
//! сборка аргументов — чистые функции; сам процесс подменяется в тестах
//! фикстурами (поле `execute`), поэтому сеть не нужна.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;

use crate::error::AppError;

/// Путь к бинарю smbclient (установлен в системе, см. docs/hardware.md).
const SMBCLIENT_BIN: &str = "/usr/bin/smbclient";

/// Хвост вывода smbclient в тексте ошибки (диагностика без засорения логов).
const OUTPUT_TAIL_CHARS: usize = 300;

/// Минимум токенов строки листинга: имя(1) + атрибуты + размер + дата(5).
const LS_MIN_TOKENS: usize = 8;

/// Одно имя в листинге SMB-директории.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbEntry {
    /// Имя файла или директории (без пути).
    pub name: String,
    /// Признак директории (атрибут `D` в листинге).
    pub is_dir: bool,
    /// Размер в байтах (0 для директорий).
    pub size: u64,
}

impl SmbEntry {
    /// .wav-файл (регистр расширения не важен; имя без точки — не .wav).
    pub fn is_wav(&self) -> bool {
        !self.is_dir
            && self.name.rfind('.').is_some_and(|dot| self.name[dot + 1..].eq_ignore_ascii_case("wav"))
    }
}

/// Исполнитель процесса smbclient; в тестах подменяется фикстурой вывода (DI).
type Execute = Arc<dyn Fn(&mut Command) -> std::io::Result<Output> + Send + Sync>;

/// Клиент SMB-шары поверх subprocess smbclient.
#[derive(Clone)]
pub struct SmbClient {
    /// Путь к бинарю smbclient.
    program: String,
    /// Хост SMB-сервера.
    host: String,
    /// Имя шары.
    share: String,
    /// Файл креденшелов формата smbclient (`username=`, `password=`, `domain=`).
    creds_file: Option<PathBuf>,
    /// Переменные окружения для дочернего процесса (USER/PASSWD из SMB_*).
    env_overrides: Vec<(String, String)>,
    /// Исполнитель процесса: реальный в продакшене, фикстура в тестах.
    execute: Execute,
}

impl SmbClient {
    /// Продакшн-клиент: `/usr/bin/smbclient`, креды из файла и/или `SMB_*` env.
    pub fn new(host: impl Into<String>, share: impl Into<String>, creds_file: Option<PathBuf>) -> Self {
        Self {
            program: SMBCLIENT_BIN.to_string(),
            host: host.into(),
            share: share.into(),
            creds_file,
            env_overrides: auth_env_overrides(),
            execute: Arc::new(real_execute),
        }
    }

    /// Содержимое директории на шаре (файлы и директории, `.`/`..` отброшены).
    pub fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        let smb_command = list_command(dir)?;
        let stdout = self.run("list", dir, &smb_command)?;
        Ok(parse_ls(&stdout))
    }

    /// Только .wav-файлы директории (регистр расширения не важен).
    pub fn list_wav(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        Ok(self.list(dir)?.into_iter().filter(SmbEntry::is_wav).collect())
    }

    /// Скачивает удалённый файл в локальный путь; родительская локальная
    /// директория создаётся при необходимости.
    pub fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
        if let Some(parent) = local.parent() {
            std::fs::create_dir_all(parent).map_err(|err| AppError::Io {
                path: parent.to_path_buf(),
                source: err,
            })?;
        }
        let smb_command = get_command(remote, local)?;
        self.run("get", remote, &smb_command).map(|_| ())
    }

    /// Заливает локальный файл на шару.
    pub fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
        let smb_command = put_command(local, remote)?;
        self.run("put", remote, &smb_command).map(|_| ())
    }

    /// Создаёт директорию на шаре, обходя mangle длинных имён на FAT-шарах:
    /// `mkdir` длинного имени сохраняет его в 8.3-виде, после чего `cd`/`ls`
    /// по длинному имени дают `NT_STATUS_OBJECT_NAME_NOT_FOUND` и ломают
    /// resume. Проверенный обход (утверждён владельцем 2026-10-13): сначала
    /// `mkdir` короткого 8.3-safe имени `as_xxxxxxxx`, затем rename в
    /// финальное длинное — такое имя хранится и резолвится.
    /// Идемпотентно: существующий финал → Ok; `COLLISION` на mkdir (хвост
    /// упавшего прогона — temp уже есть) и на rename (финал уже создан
    /// повтором/гонкой) — тоже Ok. Нужен потому, что `put` промежуточные
    /// директории не создаёт (на реальном NAS заливка в несуществующую
    /// директорию падает с `NT_STATUS_OBJECT_PATH_NOT_FOUND`).
    pub fn create_dir(&self, remote: &Path) -> Result<(), AppError> {
        if self.remote_exists(remote)? {
            return Ok(());
        }
        let parent =
            remote.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new(""));
        let temp = parent.join(as_temp_name(remote)?);
        match self.run("mkdir", remote, &mkdir_command(&temp)?) {
            // Хвост упавшего прогона (temp существует) — не ошибка, rename ниже.
            Err(err) if !is_object_name_collision(&err) => return Err(err),
            _ => {}
        }
        match self.raw_rename(&temp, remote) {
            Ok(_) => Ok(()),
            // Финал уже существует — цель достигнута.
            Err(err) if is_object_name_collision(&err) => Ok(()),
            Err(err) => Err(err),
        }
    }

    /// Переименовывает путь на шаре; целевой путь обязан отсутствовать —
    /// защита от перезаписи (rename используется только для шага `CONVERTED_`).
    pub fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
        if self.remote_exists(to)? {
            return Err(AppError::Smb {
                operation: "rename",
                path: to.to_path_buf(),
                message: "target already exists, refusing to overwrite".to_string(),
            });
        }
        self.raw_rename(from, to)
    }

    /// Rename без guard'а «цель существует»: для внутреннего шага
    /// `create_dir` (temp → финал), где занятый финал обрабатывается
    /// вызывающим кодом как успех. Публичный [`SmbClient::rename`] guard
    /// сохраняет — это инвариант шага `CONVERTED_`.
    fn raw_rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
        let smb_command = rename_command(from, to)?;
        self.run("rename", to, &smb_command).map(|_| ())
    }

    /// Существует ли путь на шаре: имя ищется в листинге родительской
    /// директории (сравнение без учёта регистра — SMB по умолчанию нечувствителен).
    pub fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
        let name = file_name_str(path)?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        Ok(self.list(parent)?.iter().any(|entry| entry.name.eq_ignore_ascii_case(&name)))
    }

    /// Запускает одну команду smbclient; ненулевой код выхода → `AppError::Smb`
    /// с декодом `NT_STATUS_*` или хвостом вывода.
    fn run(&self, operation: &'static str, path: &Path, smb_command: &str) -> Result<String, AppError> {
        let mut cmd = Command::new(&self.program);
        cmd.arg(service_arg(&self.host, &self.share));
        // -N запрещает интерактивный запрос пароля (иначе процесс зависнет);
        // она не должна перекрывать пароль из auth-файла или PASSWD.
        if !self.has_password_source() {
            cmd.arg("-N");
        }
        if let Some(creds) = &self.creds_file {
            cmd.arg(format!("--authentication-file={}", creds.display()));
        }
        for (key, value) in &self.env_overrides {
            cmd.env(key, value);
        }
        cmd.arg("-c").arg(smb_command);

        let output = (self.execute)(&mut cmd).map_err(|err| AppError::Smb {
            operation,
            path: path.to_path_buf(),
            message: format!("smbclient spawn failed: {err}"),
        })?;
        if !output.status.success() {
            return Err(nt_status_error(operation, path, &output));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Есть ли источник пароля кроме интерактивного запроса.
    fn has_password_source(&self) -> bool {
        self.creds_file.is_some() || self.env_overrides.iter().any(|(key, _)| key == "PASSWD")
    }
}

impl fmt::Debug for SmbClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // env_overrides и creds_file не печатаются: там могут быть секреты.
        f.debug_struct("SmbClient")
            .field("program", &self.program)
            .field("host", &self.host)
            .field("share", &self.share)
            .field("creds_file", &self.creds_file.as_ref().map(|_| "<set>"))
            .finish_non_exhaustive()
    }
}

/// `//host/share` — сервис для smbclient.
fn service_arg(host: &str, share: &str) -> String {
    format!("//{host}/{share}")
}

/// Реальный исполнитель: захватывает stdout и stderr процесса.
fn real_execute(cmd: &mut Command) -> std::io::Result<Output> {
    cmd.output()
}

/// `SMB_USER`/`SMB_PASS` из окружения → `USER`/`PASSWD` дочернего процесса:
/// секреты идут через env, не через argv и не через код.
fn auth_env_overrides() -> Vec<(String, String)> {
    [("SMB_USER", "USER"), ("SMB_PASS", "PASSWD")]
        .into_iter()
        .filter_map(|(src, dst)| std::env::var(src).ok().map(|value| (dst.to_string(), value)))
        .collect()
}

/// Команда листинга: `cd <dir>; ls`. Без маски — пустая директория не даёт
/// `NT_STATUS_OBJECT_NAME_NOT_FOUND`, как случилось бы с `ls dir/*`.
fn list_command(dir: &Path) -> Result<String, AppError> {
    if dir.as_os_str().is_empty() || dir == Path::new(".") {
        return Ok("ls".to_string());
    }
    Ok(format!("cd {}; ls", quote(&path_str("list", dir)?)))
}

/// Команда скачивания: `get <remote> <local>`.
fn get_command(remote: &Path, local: &Path) -> Result<String, AppError> {
    Ok(format!("get {} {}", quote(&path_str("get", remote)?), quote(&path_str("get", local)?)))
}

/// Команда заливки: `put <local> <remote>`.
fn put_command(local: &Path, remote: &Path) -> Result<String, AppError> {
    Ok(format!("put {} {}", quote(&path_str("put", local)?), quote(&path_str("put", remote)?)))
}

/// Команда создания директории: `mkdir <dir>`.
fn mkdir_command(dir: &Path) -> Result<String, AppError> {
    Ok(format!("mkdir {}", quote(&path_str("mkdir", dir)?)))
}

/// 8.3-безопасное временное имя для [`SmbClient::create_dir`]: `as_` + 8
/// hex-символов хеша (DefaultHasher) нормализованного пути — ASCII без
/// точки. Детерминированное: повторный вызов и повторный прогон получают то
/// же имя, поэтому хвост упавшего прогона находится по `COLLISION` и
/// до-переименовывается.
fn as_temp_name(remote: &Path) -> Result<String, AppError> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Нормализация: components схлопывают хвостовой слэш и `.` — один путь
    // в разных записях даёт одно temp-имя.
    let normalized: PathBuf = remote.components().collect();
    let mut hasher = DefaultHasher::new();
    path_str("create_dir", &normalized)?.hash(&mut hasher);
    Ok(format!("as_{:08x}", hasher.finish() as u32))
}

/// Команда переименования: `rename <from> <to>`.
fn rename_command(from: &Path, to: &Path) -> Result<String, AppError> {
    Ok(format!("rename {} {}", quote(&path_str("rename", from)?), quote(&path_str("rename", to)?)))
}

/// Оборачивает аргумент команды smbclient в двойные кавычки (пробелы в именах);
/// внутренние `"` экранируются бэкслешем.
fn quote(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\\\""))
}

/// Путь как UTF-8 строка для подстановки в команду; иначе ошибка (молча
/// потерять байты имени через lossy-конвертацию недопустимо).
fn path_str(operation: &'static str, path: &Path) -> Result<String, AppError> {
    path.to_str().map(str::to_string).ok_or_else(|| AppError::Smb {
        operation,
        path: path.to_path_buf(),
        message: "path is not valid UTF-8".to_string(),
    })
}

/// Имя файла пути (последний компонент) как UTF-8 строка.
fn file_name_str(path: &Path) -> Result<String, AppError> {
    path.file_name().and_then(|name| name.to_str()).map(str::to_string).ok_or_else(|| AppError::Smb {
        operation: "exists",
        path: path.to_path_buf(),
        message: "path has no valid UTF-8 file name".to_string(),
    })
}

/// Разбор листинга `ls`: `<имя> <атрибуты> <размер> <дата из 5 токенов>`.
/// Имя может содержать пробелы; служебные строки (заголовок `Domain=`,
/// футер `blocks of size`, мусор) отбрасываются валидацией.
fn parse_ls(output: &str) -> Vec<SmbEntry> {
    output.lines().filter_map(parse_ls_line).collect()
}

/// Разбор одной строки листинга; `None` — строка не является записью.
fn parse_ls_line(line: &str) -> Option<SmbEntry> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < LS_MIN_TOKENS {
        return None;
    }
    let size = tokens[tokens.len() - 6].parse::<u64>().ok()?;
    let attrs = tokens[tokens.len() - 7];
    if !attrs.chars().all(|c| matches!(c, 'D' | 'A' | 'H' | 'N' | 'R' | 'S')) {
        return None;
    }
    let name = tokens[..tokens.len() - 7].join(" ");
    if name == "." || name == ".." {
        return None;
    }
    Some(SmbEntry { name, is_dir: attrs.contains('D'), size })
}

/// Первый код `NT_STATUS_*` в выводе smbclient (ошибки уходят и в stdout).
fn find_nt_status(text: &str) -> Option<&str> {
    let rest = &text[text.find("NT_STATUS_")?..];
    let end = rest
        .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Ошибка — «цель уже существует» (идемпотентность `create_dir`): код
/// `NT_STATUS_*` smbclient кладёт в текст `AppError::Smb`, его и распознаём.
fn is_object_name_collision(err: &AppError) -> bool {
    matches!(err, AppError::Smb { message, .. } if message.contains("NT_STATUS_OBJECT_NAME_COLLISION"))
}

/// Человекочитаемое пояснение к коду NT_STATUS (для текста ошибки).
fn describe_nt_status(code: &str) -> &'static str {
    match code {
        "NT_STATUS_OBJECT_NAME_NOT_FOUND"
        | "NT_STATUS_OBJECT_PATH_NOT_FOUND"
        | "NT_STATUS_NO_SUCH_FILE"
        | "NT_STATUS_OBJECT_NAME_INVALID" => "object not found",
        "NT_STATUS_ACCESS_DENIED" => "access denied (check credentials and permissions)",
        "NT_STATUS_LOGON_FAILURE" => "logon failure (bad credentials)",
        "NT_STATUS_CONNECTION_REFUSED" | "NT_STATUS_BAD_NETWORK_NAME" | "NT_STATUS_HOST_UNREACHABLE" => {
            "connection failed (check host and share)"
        }
        "NT_STATUS_OBJECT_NAME_COLLISION" => "target name already exists",
        "NT_STATUS_SHARING_VIOLATION" => "file is locked by another client",
        _ => "smb operation failed",
    }
}

/// Ненулевой exit smbclient → `AppError::Smb` с декодом NT_STATUS
/// либо хвостом вывода, если узнаваемого кода нет.
fn nt_status_error(operation: &'static str, path: &Path, output: &Output) -> AppError {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let message = match find_nt_status(&text) {
        Some(code) => format!("{code}: {}", describe_nt_status(code)),
        // Неизвестный код: отдаём код выхода и хвост вывода для диагностики.
        None => match output.status.code() {
            Some(code) => format!("exit code {code}: {}", tail(&text, OUTPUT_TAIL_CHARS)),
            None => format!("killed by signal: {}", tail(&text, OUTPUT_TAIL_CHARS)),
        },
    };
    AppError::Smb { operation, path: path.to_path_buf(), message }
}

/// Последние `max` символов текста (границы символов не ломаются).
fn tail(text: &str, max: usize) -> &str {
    let trimmed = text.trim();
    if trimmed.len() <= max {
        return trimmed;
    }
    let start = trimmed
        .char_indices()
        .map(|(i, _)| i)
        .rev()
        .find(|&i| trimmed.len() - i <= max)
        .unwrap_or(0);
    &trimmed[start..]
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::io;
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// Нормальный листинг: имена с пробелами, директории, разные атрибуты.
    const LISTING: &str = "\
Domain=[WORKGROUP] OS=[Windows 6.1] Server=[Samba 4.15.13-Ubuntu]

  .                                   D        0  Mon Sep 29 18:44:27 2025
  ..                                  D        0  Mon Sep 29 18:44:27 2025
  kirtan morning session.wav          N 52428800  Thu Oct 13 12:34:56 2026
  Hare Krishna - full mix.WAV         A  1234567  Fri Jan  1 03:04:05 2027
  notes.txt                           N       99  Thu Oct 13 09:00:00 2026
  nested recordings                   D        0  Thu Oct 13 12:34:56 2026

    3 blocks of size 1048576. 12 blocks available
";

    /// Пустая директория: только `.`/`..` и футер.
    const EMPTY_DIR: &str = "\
Domain=[WORKGROUP] OS=[Windows 6.1] Server=[Samba 4.15.13-Ubuntu]

  .                                   D        0  Mon Sep 29 18:44:27 2025
  ..                                  D        0  Mon Sep 29 18:44:27 2025

    0 blocks of size 1048576. 42 blocks available
";

    /// Мусорные строки: короткие, без размера/даты, не-атрибутные токены.
    const JUNK: &str = "\
some random junk line
one two three D 100 Thu Oct 13 12:34:56 2026 extra
NT_STATUS_ACCESS_DENIED listing \\*
";

    /// Захваченный вызов smbclient (для проверок аргументов без запуска).
    #[derive(Debug, Clone)]
    struct Captured {
        program: String,
        args: Vec<String>,
        smb_command: Option<String>,
    }

    /// Фиксирует вызов команды, не запуская ничего.
    fn captured(cmd: &Command) -> Captured {
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let smb_command = args.iter().position(|a| a == "-c").and_then(|i| args.get(i + 1)).cloned();
        Captured { program: cmd.get_program().to_string_lossy().into_owned(), args, smb_command }
    }

    /// Output с заданным кодом выхода (unix wait-status) и stdout.
    fn output_exit(code: i32, stdout: &str) -> Output {
        Output {
            status: ExitStatus::from_raw(code.max(0) << 8),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    /// Клиент с подменённым исполнителем + журнал захваченных вызовов.
    /// Ответ на каждый вызов определяет `responder`.
    fn fake_client(
        responder: impl Fn(&Captured) -> io::Result<Output> + Send + Sync + 'static,
    ) -> (SmbClient, Arc<Mutex<Vec<Captured>>>) {
        let log: Arc<Mutex<Vec<Captured>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&log);
        let client = SmbClient {
            program: SMBCLIENT_BIN.to_string(),
            host: "nas.local".to_string(),
            share: "music".to_string(),
            creds_file: None,
            env_overrides: Vec::new(),
            execute: Arc::new(move |cmd: &mut Command| {
                let snap = captured(cmd);
                seen.lock().expect("log lock").push(snap.clone());
                responder(&snap)
            }),
        };
        (client, log)
    }

    fn ok_listing(stdout: &str) -> io::Result<Output> {
        Ok(output_exit(0, stdout))
    }

    // --- парсинг листинга -------------------------------------------------

    #[test]
    fn parse_ls_handles_names_with_spaces() {
        let entries = parse_ls(LISTING);

        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            ["kirtan morning session.wav", "Hare Krishna - full mix.WAV", "notes.txt", "nested recordings"]
        );
        assert_eq!(entries[0].size, 52_428_800);
        assert_eq!(entries[1].size, 1_234_567);
        assert_eq!(entries[2].size, 99);
        assert!(entries.iter().last().is_some_and(|e| e.is_dir));
        assert!(entries[..3].iter().all(|e| !e.is_dir));
    }

    #[test]
    fn parse_ls_empty_directory_yields_no_entries() {
        assert!(parse_ls(EMPTY_DIR).is_empty());
    }

    #[test]
    fn parse_ls_ignores_junk_lines() {
        assert!(parse_ls(JUNK).is_empty());
    }

    // --- маппинг NT_STATUS ------------------------------------------------

    #[test]
    fn find_nt_status_extracts_code() {
        assert_eq!(find_nt_status("oops NT_STATUS_ACCESS_DENIED listing \\*"), Some("NT_STATUS_ACCESS_DENIED"));
        assert_eq!(find_nt_status("code NT_STATUS_OBJECT_NAME_NOT_FOUND."), Some("NT_STATUS_OBJECT_NAME_NOT_FOUND"));
        assert_eq!(find_nt_status("plain failure"), None);
    }

    #[test]
    fn nt_status_codes_map_to_readable_errors() {
        let cases = [
            ("NT_STATUS_OBJECT_NAME_NOT_FOUND", "object not found"),
            ("NT_STATUS_ACCESS_DENIED", "access denied"),
            ("NT_STATUS_CONNECTION_REFUSED", "connection failed"),
            ("NT_STATUS_LOGON_FAILURE", "logon failure"),
            ("NT_STATUS_SOMETHING_ELSE", "smb operation failed"),
        ];
        for (code, expected) in cases {
            let message = format!("{code}: {}", describe_nt_status(code));
            assert!(message.contains(expected), "`{message}` must mention `{expected}`");
        }
    }

    #[test]
    fn non_zero_exit_maps_to_app_error_with_nt_status() {
        let err = nt_status_error("list", Path::new("d"), &output_exit(1, "NT_STATUS_ACCESS_DENIED for '\\*'\n"));
        match err {
            AppError::Smb { operation, message, .. } => {
                assert_eq!(operation, "list");
                assert!(message.contains("NT_STATUS_ACCESS_DENIED"), "{message}");
                assert!(message.contains("access denied"), "{message}");
            }
            other => panic!("expected AppError::Smb, got {other:?}"),
        }
    }

    #[test]
    fn non_zero_exit_without_nt_status_includes_tail() {
        let err = nt_status_error("get", Path::new("f"), &output_exit(2, "session setup failed\n"));
        assert!(err.to_string().contains("exit code 2"), "{err}");
        assert!(err.to_string().contains("session setup failed"), "{err}");
    }

    // --- построение аргументов smbclient -----------------------------------

    #[test]
    fn quote_wraps_and_escapes_double_quotes() {
        assert_eq!(quote("a b.wav"), "\"a b.wav\"");
        assert_eq!(quote("say \"hi\".wav"), "\"say \\\"hi\\\".wav\"");
    }

    #[test]
    fn list_command_uses_cd_and_ls() {
        assert_eq!(list_command(Path::new(".")).unwrap(), "ls");
        assert_eq!(list_command(Path::new("")).unwrap(), "ls");
        assert_eq!(list_command(Path::new("recordings 2024")).unwrap(), "cd \"recordings 2024\"; ls");
    }

    #[test]
    fn get_put_rename_commands_quote_paths() {
        let get = get_command(Path::new("a b.wav"), Path::new("/tmp/w/one.wav")).unwrap();
        assert_eq!(get, "get \"a b.wav\" \"/tmp/w/one.wav\"");

        let put = put_command(Path::new("/tmp/w/one.flac"), Path::new("song/stems/one.flac")).unwrap();
        assert_eq!(put, "put \"/tmp/w/one.flac\" \"song/stems/one.flac\"");

        let rename = rename_command(Path::new("song.wav"), Path::new("CONVERTED_song.wav")).unwrap();
        assert_eq!(rename, "rename \"song.wav\" \"CONVERTED_song.wav\"");
    }

    #[test]
    fn non_utf8_path_is_rejected_not_silently_mangled() {
        use std::os::unix::ffi::OsStrExt;
        let bad = Path::new(OsStr::from_bytes(b"bad\xff.wav"));
        assert!(get_command(bad, Path::new("/tmp/x")).is_err());
    }

    #[test]
    fn service_arg_builds_unc_path() {
        assert_eq!(service_arg("nas.local", "music"), "//nas.local/music");
    }

    // --- флаги аутентификации (без реального запуска) -----------------------

    #[test]
    fn no_pass_flag_only_without_password_source() {
        // Без источника пароля -N обязательна: иначе smbclient зависнет,
        // интерактивно запрашивая пароль.
        let (plain, plain_log) = fake_client(|_| ok_listing(EMPTY_DIR));
        plain.list(Path::new(".")).unwrap();
        assert!(
            plain_log.lock().expect("lock")[0].args.iter().any(|a| a == "-N"),
            "no password source → -N must guard against interactive hang"
        );

        // С auth-файлом или env-паролем -N не ставится: она может перекрыть пароль.
        let (with_creds, creds_log) = fake_client(|_| ok_listing(EMPTY_DIR));
        let mut with_creds = with_creds;
        with_creds.creds_file = Some(PathBuf::from("/run/secrets/smb"));
        with_creds.list(Path::new(".")).unwrap();
        assert!(!creds_log.lock().expect("lock")[0].args.iter().any(|a| a == "-N"));

        let (with_env, env_log) = fake_client(|_| ok_listing(EMPTY_DIR));
        let mut with_env = with_env;
        with_env.env_overrides = vec![("PASSWD".to_string(), String::new())];
        with_env.list(Path::new(".")).unwrap();
        assert!(!env_log.lock().expect("lock")[0].args.iter().any(|a| a == "-N"));
    }

    #[test]
    fn auth_file_is_passed_via_long_option_not_argv_credentials() {
        let (client, log) = fake_client(|_| ok_listing(EMPTY_DIR));
        let mut client = client;
        client.creds_file = Some(PathBuf::from("/run/secrets/smb"));

        client.list(Path::new(".")).unwrap();

        let captured = &log.lock().expect("lock")[0];
        assert_eq!(captured.program, "/usr/bin/smbclient");
        assert!(captured.args.iter().any(|a| a == "--authentication-file=/run/secrets/smb"));
        // Никаких кредов в argv: логин/пароль есть только в файле.
        assert!(!captured.args.iter().any(|a| a.contains("password") || a.contains("username")));
    }

    // --- поведение операций на фикстурах (сеть не нужна) --------------------

    #[test]
    fn list_parses_fixture_output() {
        let (client, log) = fake_client(|_| ok_listing(LISTING));

        let entries = client.list(Path::new("recordings")).unwrap();

        assert_eq!(entries.len(), 4);
        assert_eq!(log.lock().expect("lock")[0].smb_command.as_deref(), Some("cd \"recordings\"; ls"));
    }

    #[test]
    fn list_wav_filters_directories_and_extensions() {
        let (client, _) = fake_client(|_| ok_listing(LISTING));

        let wavs = client.list_wav(Path::new(".")).unwrap();

        let names: Vec<&str> = wavs.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["kirtan morning session.wav", "Hare Krishna - full mix.WAV"]);
    }

    #[test]
    fn nt_status_failure_bubbles_as_app_error() {
        let (client, _) =
            fake_client(|_| Ok(output_exit(1, "NT_STATUS_OBJECT_NAME_NOT_FOUND opening remote file\n")));

        let err = client.get(Path::new("nope.wav"), Path::new("/tmp/w/nope.wav")).expect_err("must fail");

        assert!(err.to_string().contains("object not found"), "{err}");
    }

    #[test]
    fn spawn_failure_maps_to_smb_error() {
        let (client, _) = fake_client(|_| Err(io::Error::other("binary missing")));

        let err = client.list(Path::new(".")).expect_err("must fail");

        assert!(err.to_string().contains("spawn failed"), "{err}");
    }

    #[test]
    fn rename_refuses_existing_target_without_running_rename() {
        let listing_with_target = "  target.wav  N  1  Thu Oct 13 12:00:00 2026\n";
        let (client, log) = fake_client(move |_| ok_listing(listing_with_target));

        let err = client
            .rename(Path::new("a.wav"), Path::new("target.wav"))
            .expect_err("existing target must be refused");

        assert!(err.to_string().contains("already exists"), "{err}");
        let ran = log.lock().expect("lock");
        assert_eq!(ran.len(), 1, "only the existence listing may run");
        assert_eq!(ran[0].smb_command.as_deref(), Some("ls"), "target at share root → root listing");
    }

    #[test]
    fn rename_runs_when_target_absent() {
        let (client, log) = fake_client(|_| ok_listing(LISTING));

        client.rename(Path::new("a.wav"), Path::new("CONVERTED_a.wav")).unwrap();

        let ran = log.lock().expect("lock");
        assert_eq!(ran.len(), 2);
        assert_eq!(ran[1].smb_command.as_deref(), Some("rename \"a.wav\" \"CONVERTED_a.wav\""));
    }

    #[test]
    fn remote_exists_matches_case_insensitively() {
        let (client, _) = fake_client(|_| ok_listing(LISTING));

        assert!(client.remote_exists(Path::new("sub/NOTES.TXT")).unwrap());
        assert!(!client.remote_exists(Path::new("sub/missing.wav")).unwrap());
    }

    // --- create_dir: обход mangle длинных имён на FAT (8.3) -------------------

    /// Первый аргумент в кавычках smb-команды после `prefix` (журнал вызовов).
    fn first_quoted_arg(command: &str, prefix: &str) -> Option<String> {
        let rest = command.strip_prefix(prefix)?;
        let start = rest.find('"')? + 1;
        let end = rest[start..].find('"')? + start;
        Some(rest[start..end].replace("\\\"", "\""))
    }

    /// Имя временной директории из команды mkdir: строго `as_` + 8 hex
    /// (ASCII без точки — 8.3-safe); аргумент — путь в директории финала.
    fn as_temp_of(command: &str) -> String {
        let arg = first_quoted_arg(command, "mkdir").expect("mkdir argument");
        let name = arg.rsplit('/').next().expect("non-empty").to_string();
        assert_eq!(name.len(), 11, "temp = `as_` + 8 hex: {name}");
        assert!(name.starts_with("as_"), "{name}");
        assert!(name[3..].bytes().all(|c| c.is_ascii_hexdigit()), "{name}");
        name
    }

    #[test]
    fn create_dir_makes_short_temp_then_renames_to_final() {
        let (client, log) = fake_client(|cap| match cap.smb_command.as_deref() {
            Some(c) if c.starts_with("mkdir") || c.starts_with("rename") => ok_listing(""),
            _ => ok_listing(EMPTY_DIR), // exists-листинг: финал отсутствует
        });

        client.create_dir(Path::new("stems/kirtan morning")).unwrap();

        let ran = log.lock().expect("lock");
        assert_eq!(ran.len(), 3, "exists-листинг, mkdir temp, rename → финал");
        assert_eq!(ran[0].smb_command.as_deref(), Some("cd \"stems\"; ls"), "листинг родителя");
        let temp = as_temp_of(ran[1].smb_command.as_deref().expect("mkdir"));
        assert_eq!(
            ran[1].smb_command.as_deref(),
            Some(format!("mkdir \"stems/{temp}\"").as_str()),
            "mkdir короткого 8.3-safe имени в той же директории"
        );
        assert_eq!(
            ran[2].smb_command.as_deref(),
            Some(format!("rename \"stems/{temp}\" \"stems/kirtan morning\"").as_str()),
            "rename temp → финальное длинное имя (оно тогда резолвится)"
        );
    }

    #[test]
    fn as_temp_name_is_deterministic_normalized_and_8_3_safe() {
        let a = as_temp_name(Path::new("stems/kirtan morning")).unwrap();
        let slash = as_temp_name(Path::new("stems/kirtan morning/")).unwrap();
        let other = as_temp_name(Path::new("stems/other")).unwrap();

        assert_eq!(a, slash, "хвостовой слэш не меняет temp-имя");
        assert_ne!(a, other, "разные пути → разные temp-имена");
        assert_eq!(as_temp_name(Path::new("stems/kirtan morning")).unwrap(), a, "детерминизм");
        assert_eq!(a.len(), 11);
        assert!(a.starts_with("as_") && a[3..].bytes().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn create_dir_final_already_exists_is_early_ok_without_mkdir() {
        let (client, log) = fake_client(|_| ok_listing(LISTING)); // есть "nested recordings"

        client.create_dir(Path::new("nested recordings")).expect("идемпотентность");

        let ran = log.lock().expect("lock");
        assert_eq!(ran.len(), 1, "только exists-листинг");
        assert_eq!(ran[0].smb_command.as_deref(), Some("ls"), "без mkdir");
    }

    #[test]
    fn create_dir_mkdir_collision_is_ok_and_rename_still_runs() {
        // Хвост упавшего прогона: temp-директория уже существует.
        let (client, log) = fake_client(|cap| match cap.smb_command.as_deref() {
            Some(c) if c.starts_with("mkdir") => {
                Ok(output_exit(1, "NT_STATUS_OBJECT_NAME_COLLISION making remote directory\n"))
            }
            Some(c) if c.starts_with("rename") => ok_listing(""),
            _ => ok_listing(EMPTY_DIR),
        });

        client.create_dir(Path::new("stems/a")).expect("COLLISION на temp — не ошибка");

        let ran = log.lock().expect("lock");
        assert_eq!(ran.len(), 3, "rename выполняется и после COLLISION на mkdir");
        assert!(ran[2].smb_command.as_deref().unwrap_or_default().starts_with("rename"));
    }

    #[test]
    fn create_dir_rename_target_exists_maps_to_ok() {
        // Финал появился между exists-проверкой и rename (повтор/гонка).
        let (client, log) = fake_client(|cap| match cap.smb_command.as_deref() {
            Some(c) if c.starts_with("rename") => {
                Ok(output_exit(1, "NT_STATUS_OBJECT_NAME_COLLISION renaming files\n"))
            }
            Some(c) if c.starts_with("mkdir") => ok_listing(""),
            _ => ok_listing(EMPTY_DIR),
        });

        client.create_dir(Path::new("stems/a")).expect("финал уже существует → Ok");

        assert_eq!(log.lock().expect("lock").len(), 3, "mkdir был, отказ rename погашен");
    }

    #[test]
    fn create_dir_other_nt_status_maps_to_error_with_code() {
        let (client, _) = fake_client(|_| Ok(output_exit(1, "NT_STATUS_ACCESS_DENIED making remote directory\n")));

        let err = client.create_dir(Path::new("stems/a")).expect_err("must fail");

        assert!(err.to_string().contains("NT_STATUS_ACCESS_DENIED"), "{err}");
    }
}
