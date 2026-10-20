//! Конфигурация прогона: иммутабельный набор параметров (см. docs/architecture.md).
//!
//! Парсинг — clap derive; doc-комментарии полей становятся help-текстом CLI.

use std::path::PathBuf;

use clap::Parser;

/// Дефолт ONNX-модели движка: крейто-относительный путь.
/// Статическая строка — единая точка для CLI-дефолта (`default_value`
/// принимает только статические строки без фичи clap «string»).
pub const DEFAULT_ONNX_MODEL: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/.models/onnx/bs_roformer_sw_6stem_fp32.onnx");

/// Дефолт ONNX-модели как путь: единая точка для TUI.
pub fn default_onnx_model() -> PathBuf {
    PathBuf::from(DEFAULT_ONNX_MODEL)
}

/// Настройки одного прогона пайплайна.
#[derive(Debug, Clone, Parser)]
#[command(name = "hexastem", about = "SMB batch stem-separation pipeline")]
pub struct Config {
    /// Хост SMB-сервера (сервис будет //host/share).
    #[arg(long)]
    pub host: String,

    /// Имя SMB-шары.
    #[arg(long)]
    pub share: String,

    /// Исходная директория с записями на шаре.
    #[arg(long)]
    pub remote_dir: PathBuf,

    /// Локальная рабочая директория (загрузка исходников и стемов).
    #[arg(long)]
    pub local_dir: PathBuf,

    /// Формат выходных стемов (движок пишет только FLAC).
    #[arg(long, default_value = "flac")]
    pub output_format: String,

    /// ONNX-модель движка.
    #[arg(long, default_value = DEFAULT_ONNX_MODEL)]
    pub onnx_model: PathBuf,

    /// Файл SMB-креденшелов для smbclient (-A); опционально.
    #[arg(long)]
    pub smb_creds_file: Option<PathBuf>,

    /// Локальная папка вместо SMB: remote_dir читается прямо с диска (тот же
    /// трейт SmbOps) — smoke-тесты и отладка пайплайна без сети.
    #[arg(long)]
    pub local: bool,

    /// Отключить дедупликацию дублей-записей (по умолчанию включена: файлы
    /// с одинаковым контентом обрабатываются один раз, см. README).
    #[arg(long)]
    pub no_dedup: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Минимальный валидный набор аргументов (без опциональных).
    fn base_args() -> Vec<&'static str> {
        vec![
            "hexastem",
            "--host", "nas.local",
            "--share", "music",
            "--remote-dir", "kirtan",
            "--local-dir", "/tmp/work",
        ]
    }

    #[test]
    fn defaults_are_flac_and_no_optional_flags() {
        let cfg = Config::try_parse_from(base_args()).expect("valid args");

        assert_eq!(cfg.output_format, "flac");
        assert_eq!(cfg.smb_creds_file, None);
    }

    #[test]
    fn local_flag_is_opt_in() {
        let minimal = Config::try_parse_from(base_args()).expect("valid args");
        assert!(!minimal.local, "SMB is the default backend");

        let local = Config::try_parse_from(
            base_args().into_iter().chain(["--local"]).collect::<Vec<_>>(),
        )
        .expect("valid args with --local");
        assert!(local.local, "--local switches the backend to a local folder");
    }

    #[test]
    fn dedup_is_on_by_default_and_disabled_by_no_dedup_flag() {
        let default = Config::try_parse_from(base_args()).expect("valid args");
        assert!(!default.no_dedup, "дедупликация включена по умолчанию");

        let disabled = Config::try_parse_from(
            base_args().into_iter().chain(["--no-dedup"]).collect::<Vec<_>>(),
        )
        .expect("valid args with --no-dedup");
        assert!(disabled.no_dedup, "--no-dedup выключает дедупликацию");
    }

    #[test]
    fn onnx_model_default_is_crate_relative() {
        let cfg = Config::try_parse_from(base_args()).expect("valid args");
        assert!(
            cfg.onnx_model.starts_with(env!("CARGO_MANIFEST_DIR")),
            "дефолт ONNX-модели — относительно крейта: {}",
            cfg.onnx_model.display()
        );
    }

    #[test]
    fn onnx_model_flag_overrides_default() {
        let cfg = Config::try_parse_from(
            base_args().into_iter().chain(["--onnx-model", "/opt/model.onnx"]).collect::<Vec<_>>(),
        )
        .expect("valid args with --onnx-model");
        assert_eq!(cfg.onnx_model, PathBuf::from("/opt/model.onnx"));
    }

    #[test]
    fn default_onnx_model_names_bs_roformer_sw() {
        let model = default_onnx_model();
        assert!(
            model.ends_with(".models/onnx/bs_roformer_sw_6stem_fp32.onnx"),
            "единая точка дефолта ONNX: {}",
            model.display()
        );
    }
}
