//! Единый тип ошибок приложения: все модули возвращают `Result<_, AppError>`.

use std::path::PathBuf;

/// Ошибки всех подсистем пайплайна.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// Сбой SMB-операции (list/get/put/rename через smbclient).
    #[error("SMB {operation} failed for `{path}`: {message}")]
    Smb {
        operation: &'static str,
        path: PathBuf,
        message: String,
    },

    /// Сбой движка разделения (diagnostics — хвост stderr движка).
    #[error("separation engine failed for `{file}` (exit {exit_code:?}): {stderr_tail}")]
    Engine {
        file: PathBuf,
        exit_code: Option<i32>,
        stderr_tail: String,
    },

    /// Нарушение инварианта пайплайна (например, неполный набор стемов).
    #[error("pipeline invariant violated: {0}")]
    Pipeline(String),

    /// Ошибка локального файлового I/O.
    #[error("I/O error for `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Некорректная конфигурация прогона.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// Ошибка DSP-модуля нативного движка (формат аудио, математика, границы).
    #[error("dsp error: {0}")]
    Dsp(String),
}
