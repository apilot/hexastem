//! Локальная папка как хранилище: реализация [`SmbOps`] поверх `std::fs`.
//!
//! Назначение — smoke-тесты и отладка пайплайна без сети (флаг `--local`):
//! `remote_dir` читается прямо с диска, без трансляции путей. Семантика
//! повторяет `SmbClient`: rename отказывает при существующем целевом пути
//! (защита от перезаписи — шаг `CONVERTED_`), get создаёт локальную
//! родительскую директорию при необходимости.

use std::fs;
use std::path::Path;

use crate::error::AppError;
use crate::pipeline::SmbOps;
use crate::smb::SmbEntry;

/// Хранилище — локальная папка (без состояния: пути используются как есть).
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalFs;

/// `AppError::Io` с путём (все операции локального диска).
fn io_error(path: &Path, source: std::io::Error) -> AppError {
    AppError::Io { path: path.to_path_buf(), source }
}

/// `true`, если оба пути указывают на один и тот же файл: канонические пути
/// совпадают; при ошибке `canonicalize` (путь ещё не существует) — точное
/// совпадение записей путей. Защита `get` от самокопирования: `fs::copy(p, p)`
/// обнуляет файл.
fn is_same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(canon_a), Ok(canon_b)) => canon_a == canon_b,
        _ => a == b,
    }
}

impl SmbOps for LocalFs {
    fn list(&self, dir: &Path) -> Result<Vec<SmbEntry>, AppError> {
        let mut entries = Vec::new();
        for item in fs::read_dir(dir).map_err(|source| io_error(dir, source))? {
            let entry = item.map_err(|source| io_error(dir, source))?;
            let metadata = entry.metadata().map_err(|source| io_error(&entry.path(), source))?;
            entries.push(SmbEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                is_dir: metadata.is_dir(),
                size: if metadata.is_dir() { 0 } else { metadata.len() },
            });
        }
        // Детерминированный порядок — иначе тесты и отчёты зависят от read_dir.
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    fn get(&self, remote: &Path, local: &Path) -> Result<(), AppError> {
        if is_same_file(remote, local) {
            // remote и local — один файл (например, `--remote-dir == --local-dir`
            // в `--local` режиме): копирование обнулило бы файл, считаем get
            // выполненным без копирования.
            return Ok(());
        }
        if let Some(parent) = local.parent() {
            fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
        }
        fs::copy(remote, local).map(|_| ()).map_err(|source| io_error(remote, source))
    }

    fn put(&self, local: &Path, remote: &Path) -> Result<(), AppError> {
        if let Some(parent) = remote.parent() {
            fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
        }
        fs::copy(local, remote).map(|_| ()).map_err(|source| io_error(remote, source))
    }

    fn create_dir(&self, dir: &Path) -> Result<(), AppError> {
        // create_dir_all идемпотентен: существующая директория — не ошибка.
        fs::create_dir_all(dir).map_err(|source| io_error(dir, source))
    }

    fn rename(&self, from: &Path, to: &Path) -> Result<(), AppError> {
        if to.symlink_metadata().is_ok() {
            // Как и SmbClient: цель существует → отказ, инвариант `CONVERTED_`.
            return Err(io_error(
                to,
                std::io::Error::new(std::io::ErrorKind::AlreadyExists, "rename target already exists"),
            ));
        }
        fs::rename(from, to).map_err(|source| io_error(from, source))
    }

    fn remote_exists(&self, path: &Path) -> Result<bool, AppError> {
        Ok(path.symlink_metadata().is_ok())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    /// Уникальная локальная директория теста (параллельные тесты не пересекаются).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hexastem_local_fs_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn list_reports_files_dirs_and_sizes_sorted() {
        let dir = temp_dir("list");
        fs::write(dir.join("b.wav"), b"12345").unwrap();
        fs::write(dir.join("a.txt"), b"x").unwrap();
        fs::create_dir(dir.join("a_dir")).unwrap();

        let entries = LocalFs.list(&dir).unwrap();

        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["a.txt", "a_dir", "b.wav"], "sorted by name");
        assert!(entries[1].is_dir && entries[1].size == 0);
        assert!(!entries[2].is_dir && entries[2].size == 5);
    }

    #[test]
    fn list_missing_dir_is_error() {
        let dir = temp_dir("missing").join("nope");
        assert!(matches!(LocalFs.list(&dir), Err(AppError::Io { .. })));
    }

    #[test]
    fn get_copies_remote_to_local_creating_parents() {
        let dir = temp_dir("get");
        fs::write(dir.join("song.wav"), b"wav").unwrap();
        let local = dir.join("nested/deep/song.wav");

        LocalFs.get(&dir.join("song.wav"), &local).unwrap();

        assert_eq!(fs::read(&local).unwrap(), b"wav");
    }

    #[test]
    fn get_same_path_is_noop_and_keeps_content() {
        let dir = temp_dir("self_copy");
        let file = dir.join("song.wav");
        fs::write(&file, b"payload").unwrap();

        LocalFs.get(&file, &file).unwrap();

        assert_eq!(fs::read(&file).unwrap(), b"payload", "self-get must not truncate the file");
    }

    #[test]
    fn get_different_paths_still_copies() {
        let dir = temp_dir("get_diff");
        fs::write(dir.join("src.wav"), b"copy-me").unwrap();
        let local = dir.join("out/src.wav");

        LocalFs.get(&dir.join("src.wav"), &local).unwrap();

        assert_eq!(fs::read(&local).unwrap(), b"copy-me");
    }

    #[test]
    fn put_copies_local_to_remote_creating_parents() {
        let dir = temp_dir("put");
        let local = dir.join("in/stem.flac");
        fs::create_dir_all(local.parent().unwrap()).unwrap();
        fs::write(&local, b"flac").unwrap();

        LocalFs.put(&local, &dir.join("song/vocals.flac")).unwrap();

        assert_eq!(fs::read(dir.join("song/vocals.flac")).unwrap(), b"flac");
    }

    #[test]
    fn rename_moves_file_and_refuses_existing_target() {
        let dir = temp_dir("rename");
        fs::write(dir.join("song.wav"), b"x").unwrap();
        fs::write(dir.join("CONVERTED_song.wav"), b"old").unwrap();

        let refused = LocalFs.rename(&dir.join("song.wav"), &dir.join("CONVERTED_song.wav"));
        assert!(refused.is_err(), "existing target must be refused");
        assert!(dir.join("song.wav").exists(), "source untouched after refusal");

        LocalFs.rename(&dir.join("song.wav"), &dir.join("CONVERTED_song.wav.bak")).unwrap();
        assert!(!dir.join("song.wav").exists());
        assert!(dir.join("CONVERTED_song.wav.bak").exists());
    }

    #[test]
    fn remote_exists_matches_disk() {
        let dir = temp_dir("exists");
        fs::write(dir.join("a.wav"), b"x").unwrap();
        fs::create_dir(dir.join("sub")).unwrap();

        assert!(LocalFs.remote_exists(&dir.join("a.wav")).unwrap());
        assert!(LocalFs.remote_exists(&dir.join("sub")).unwrap());
        assert!(!LocalFs.remote_exists(&dir.join("ghost")).unwrap());
    }

    #[test]
    fn create_dir_makes_parents_and_is_idempotent() {
        let dir = temp_dir("create_dir");

        LocalFs.create_dir(&dir.join("song/stems")).unwrap();
        LocalFs.create_dir(&dir.join("song/stems")).unwrap();

        assert!(dir.join("song/stems").is_dir());
    }
}
