//! Persistence пресета TUI: `~/.config/hexastem/tui.toml` (serde + toml).
//!
//! Хранит последние значения полей попапа запуска: загрузка — при открытии
//! попапа (префилл), сохранение — при успешном старте прогона. Отсутствующий
//! файл = дефолты; битый файл / ошибка записи — graceful (не мешают запуску):
//! чтение возвращает `Default`, запись отдаёт `Err` вызывающему коду, который
//! её глотает.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::Model;

/// Имя файла пресета (в `~/.config/hexastem/`).
pub const SETTINGS_FILE: &str = "tui.toml";

/// Пресет попапа запуска: последние значения полей.
/// `serde(default)` — неполный файл дозаполняется из `Default` (graceful).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TuiSettings {
    /// Хост SMB-сервера.
    pub host: String,
    /// Имя шары.
    pub share: String,
    /// Исходная директория на шаре.
    pub remote_dir: String,
    /// Локальная рабочая директория.
    pub local_dir: String,
    /// Модель разделения: `sw` | `6s`.
    pub model: String,
    /// Дедупликация включена.
    pub dedup: bool,
    /// Локальная папка вместо SMB (remote читается с диска).
    pub local_mode: bool,
    /// Язык интерфейса: `ru` | `en` (см. `i18n::Lang`).
    pub lang: String,
}

impl Default for TuiSettings {
    fn default() -> Self {
        Self {
            host: String::new(),
            share: String::new(),
            remote_dir: String::new(),
            local_dir: String::new(),
            model: model_label(Model::Sw).to_string(),
            dedup: true,
            local_mode: false,
            lang: crate::tui::i18n::Lang::default().label().to_string(),
        }
    }
}

impl TuiSettings {
    /// Модель из пресета: неизвестное значение → дефолт `sw` (graceful).
    pub fn model(&self) -> Model {
        if self.model == model_label(Model::SixS) {
            Model::SixS
        } else {
            Model::Sw
        }
    }

    /// Язык из пресета: неизвестное значение → дефолт `Ru` (graceful).
    pub fn lang(&self) -> crate::tui::i18n::Lang {
        crate::tui::i18n::Lang::from_label(&self.lang)
    }
}

/// Метка модели как в CLI (`--model sw|6s`).
pub fn model_label(model: Model) -> &'static str {
    match model {
        Model::Sw => "sw",
        Model::SixS => "6s",
    }
}

/// Полный путь пресета: `~/.config/hexastem/tui.toml`.
/// `None` — домашняя директория не определена (грузим дефолты, не сохраняем).
pub fn path() -> Option<PathBuf> {
    std::env::home_dir().map(|home| {
        home.join(".config").join("hexastem").join(SETTINGS_FILE)
    })
}

/// Загрузить пресет: отсутствующий/битый файл → дефолты (graceful).
pub fn load() -> TuiSettings {
    match path() {
        Some(path) => load_from(&path),
        None => TuiSettings::default(),
    }
}

/// Загрузка из конкретного файла (тестируемый вариант [`load`]).
pub fn load_from(path: &Path) -> TuiSettings {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

/// Сохранить пресет по стандартному пути (создаёт директории).
pub fn save(settings: &TuiSettings) -> io::Result<()> {
    let path = path().ok_or_else(|| io::Error::other("home directory not found"))?;
    save_to(&path, settings)
}

/// Сохранение в конкретный файл (тестируемый вариант [`save`]).
pub fn save_to(path: &Path, settings: &TuiSettings) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, toml::to_string(settings).map_err(io::Error::other)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Уникальный файл пресета в temp-директории теста.
    fn temp_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hexastem_persist_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir.join(SETTINGS_FILE)
    }

    fn fixture() -> TuiSettings {
        TuiSettings {
            host: "nas.local".into(),
            share: "media".into(),
            remote_dir: "/recordings 2026".into(),
            local_dir: "/tmp/work".into(),
            model: "6s".into(),
            dedup: false,
            local_mode: true,
            lang: "en".into(),
        }
    }

    #[test]
    fn roundtrip_preserves_all_fields() {
        let path = temp_path("roundtrip");
        let settings = fixture();

        save_to(&path, &settings).expect("save");
        assert_eq!(load_from(&path), settings);
    }

    #[test]
    fn save_creates_missing_parent_directories() {
        let path = temp_path("mkdirs").join("deep/nested").join(SETTINGS_FILE);

        save_to(&path, &fixture()).expect("save");

        assert!(path.is_file());
    }

    #[test]
    fn missing_file_loads_defaults() {
        let path = temp_path("missing").join("nope").join(SETTINGS_FILE);

        assert_eq!(load_from(&path), TuiSettings::default());
    }

    #[test]
    fn corrupt_file_loads_defaults() {
        let path = temp_path("corrupt");
        fs::create_dir_all(path.parent().expect("parent")).unwrap();
        fs::write(&path, "not [valid toml {{{{").unwrap();

        assert_eq!(load_from(&path), TuiSettings::default());
    }

    #[test]
    fn partial_file_fills_missing_fields_from_defaults() {
        let path = temp_path("partial");
        fs::create_dir_all(path.parent().expect("parent")).unwrap();
        fs::write(&path, "host = \"nas\"\n").unwrap();

        let settings = load_from(&path);

        assert_eq!(settings.host, "nas");
        assert_eq!(settings.share, "", "остальные поля — дефолтные");
        assert!(settings.dedup, "дедупликация по умолчанию включена");
        assert!(!settings.local_mode);
    }

    #[test]
    fn model_parses_known_labels_and_falls_back_to_sw() {
        let mut settings = TuiSettings::default();
        assert_eq!(settings.model(), Model::Sw);

        settings.model = "6s".into();
        assert_eq!(settings.model(), Model::SixS);

        settings.model = "bogus".into();
        assert_eq!(settings.model(), Model::Sw, "неизвестное значение → sw");
    }

    #[test]
    fn lang_roundtrip_and_graceful_fallback() {
        // Выбор языка переживает сохранение/загрузку.
        let path = temp_path("lang");
        let settings = TuiSettings { lang: "en".into(), ..TuiSettings::default() };
        save_to(&path, &settings).expect("save");
        assert_eq!(load_from(&path).lang(), crate::tui::i18n::Lang::En);

        // Дефолт — ru (решение владельца); неизвестное значение → ru.
        assert_eq!(TuiSettings::default().lang(), crate::tui::i18n::Lang::Ru);
        let bogus = TuiSettings { lang: "de".into(), ..TuiSettings::default() };
        assert_eq!(bogus.lang(), crate::tui::i18n::Lang::Ru, "неизвестное → дефолт");

        // Файл без поля lang (старый пресет) — дефолт.
        let legacy = temp_path("lang_legacy");
        fs::create_dir_all(legacy.parent().expect("parent")).unwrap();
        fs::write(&legacy, "host = \"nas\"\n").unwrap();
        assert_eq!(load_from(&legacy).lang(), crate::tui::i18n::Lang::Ru);
    }

    #[test]
    fn default_preset_is_smb_with_dedup() {
        let settings = TuiSettings::default();

        assert_eq!(settings.model(), Model::Sw);
        assert_eq!(settings.model, "sw");
        assert!(settings.dedup);
        assert!(!settings.local_mode);
        assert!(settings.host.is_empty() && settings.remote_dir.is_empty());
    }

    #[test]
    fn path_lays_under_config_hexastem() {
        let Some(path) = path() else {
            panic!("в тестовом окружении HOME определён");
        };
        assert!(path.ends_with(Path::new("hexastem").join(SETTINGS_FILE)));
        assert_eq!(path.file_name().and_then(|n| n.to_str()), Some(SETTINGS_FILE));
    }
}
