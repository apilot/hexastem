//! hexastem — SMB batch stem-separation pipeline.
//!
//! Пайплайн: скан SMB-директории → download → separation → upload стемов
//! → rename `CONVERTED_*` → отчёт. Инференс — subprocess, не в Rust.

pub mod config;
pub mod dedup;
pub mod engine;
pub mod envelope;
pub mod error;
pub mod local_fs;
pub mod pipeline;
pub mod report;
pub mod smb;
pub mod tui;

pub use config::{default_onnx_model, Config, EngineKind, Model};
pub use error::AppError;
