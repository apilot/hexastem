//! Конфигурация прогона: иммутабельный набор параметров (см. docs/architecture.md).
//!
//! Парсинг — clap derive; doc-комментарии полей становятся help-текстом CLI.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Движок разделения стемов.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    /// Python-драйвер audio-separator (subprocess) — legacy-путь, требует venv.
    Python,
    /// Нативный Rust-DSP движок (ONNX Runtime CUDA, FLAC 24-bit) — дефолт.
    Native,
}

/// Модель разделения и её чекпойнт для движка audio-separator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Model {
    /// BS-Roformer-SW: SOTA-качество, ~2.6 GiB VRAM (дефолт).
    Sw,
    /// htdemucs_6s: быстрее и легче, ~1.1 GiB VRAM.
    #[value(name = "6s")]
    SixS,
}

impl Model {
    /// Имя файла модели, передаваемое движку (`-m <checkpoint>`).
    pub fn checkpoint(self) -> &'static str {
        match self {
            Model::Sw => "BS-Roformer-SW.ckpt",
            Model::SixS => "htdemucs_6s.yaml",
        }
    }

    /// Стартовая оценка RTF (время обработки / длительность) для
    /// оценочного separate-прогресса драйвера; уточняется по бою.
    pub fn rtf_estimate(self) -> f64 {
        match self {
            Model::Sw => 0.45,
            Model::SixS => 0.15,
        }
    }
}

/// Дефолт ONNX-модели нативного движка: крейто-относительный путь.
/// Статическая строка — единая точка для CLI-дефолта (`default_value`
/// принимает только статические строки без фичи clap «string»).
pub const DEFAULT_ONNX_MODEL: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/.models/onnx/bs_roformer_sw_6stem_fp32.onnx");

/// Дефолт ONNX-модели как путь: единая точка для TUI (`--engine native`).
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

    /// Модель разделения: sw (BS-Roformer-SW.ckpt) или 6s (htdemucs_6s.yaml).
    #[arg(long, value_enum, default_value = "sw")]
    pub model: Model,

    /// Путь к python venv с audio-separator (дефолт — `.venv` в крейте).
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/.venv"))]
    pub venv_path: PathBuf,

    /// Формат выходных стемов.
    #[arg(long, default_value = "flac")]
    pub output_format: String,

    /// Движок разделения: native (Rust DSP + ONNX Runtime CUDA, дефолт)
    /// или python (legacy subprocess audio-separator, требует venv).
    #[arg(long, value_enum, default_value = "native")]
    pub engine: EngineKind,

    /// ONNX-модель нативного движка (--engine native).
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
            "--venv-path", "/opt/venv",
        ]
    }

    #[test]
    fn defaults_are_sw_model_and_flac() {
        let cfg = Config::try_parse_from(base_args()).expect("valid args");

        assert_eq!(cfg.model, Model::Sw);
        assert_eq!(cfg.model.checkpoint(), "BS-Roformer-SW.ckpt");
        assert_eq!(cfg.output_format, "flac");
        assert_eq!(cfg.smb_creds_file, None);
    }

    #[test]
    fn model_6s_maps_to_htdemucs_yaml() {
        let args = base_args()
            .into_iter()
            .chain(["--model", "6s"])
            .collect::<Vec<_>>();
        let cfg = Config::try_parse_from(args).expect("valid args");

        assert_eq!(cfg.model, Model::SixS);
        assert_eq!(cfg.model.checkpoint(), "htdemucs_6s.yaml");
    }

    #[test]
    fn rtf_estimate_matches_model_speed() {
        assert_eq!(Model::Sw.rtf_estimate(), 0.45);
        assert_eq!(Model::SixS.rtf_estimate(), 0.15);
    }

    #[test]
    fn unknown_model_is_rejected() {
        let args = base_args()
            .into_iter()
            .chain(["--model", "bogus"])
            .collect::<Vec<_>>();

        let result = Config::try_parse_from(args);
        assert!(result.is_err(), "unknown model must fail parsing");
    }

    #[test]
    fn venv_defaults_to_crate_and_local_flag_is_opt_in() {
        let minimal = Config::try_parse_from([
            "hexastem", "--host", "nas.local", "--share", "music", "--remote-dir", "kirtan", "--local-dir", "/tmp/work",
        ])
        .expect("valid args without optional flags");

        assert!(minimal.venv_path.ends_with(".venv"), "default venv is crate-relative: {}", minimal.venv_path.display());
        assert!(!minimal.local, "SMB is the default backend");

        let local = Config::try_parse_from([
            "hexastem", "--host", "h", "--share", "s", "--remote-dir", "r", "--local-dir", "/tmp/w", "--local",
        ])
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
    fn engine_defaults_to_native_and_switches_to_python() {
        // Паритет native-пути принят (T11) — с T13 дефолт движка — native.
        let default = Config::try_parse_from(base_args()).expect("valid args");
        assert_eq!(default.engine, EngineKind::Native);
        assert!(default.onnx_model.ends_with("bs_roformer_sw_6stem_fp32.onnx"));

        let python = Config::try_parse_from(
            base_args().into_iter().chain(["--engine", "python"]).collect::<Vec<_>>(),
        )
        .expect("valid args with --engine python");
        assert_eq!(python.engine, EngineKind::Python);
    }

    #[test]
    fn unknown_engine_is_rejected() {
        let args = base_args()
            .into_iter()
            .chain(["--engine", "bogus"])
            .collect::<Vec<_>>();
        assert!(Config::try_parse_from(args).is_err(), "unknown engine must fail parsing");
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
    fn default_onnx_model_names_bs_roformer_sw() {
        let model = default_onnx_model();
        assert!(
            model.ends_with(".models/onnx/bs_roformer_sw_6stem_fp32.onnx"),
            "единая точка дефолта ONNX: {}",
            model.display()
        );
    }
}
