//! Разбор вставленных путей: GVFS-монтирование и `smb://` → подсказка SMB.
//!
//! Drag&drop папки из файлового менеджера (GNOME) даёт FUSE-путь GVFS
//! (`/run/user/<uid>/gvfs/smb-share:server=…,share=…/…`), а «копировать путь»
//! сетевой папки — `smb://…`. Оба вида — не remote_dir внутри шары: без
//! разбора TUI подставлял путь целиком в поле Remote dir, smbclient не находил
//! такую директорию, и прогон «успешно» завершался с пустым планом. Чистый
//! нормалайзер [`normalize_pasted_path`] возвращает [`RemotePathHint`];
//! интеграцию в попап делает `app.rs` (`ConfigState::paste_text`,
//! `App::prefill_remote_dir`).

/// Распознанный SMB-путь: подключение и директория внутри шары.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemotePathHint {
    /// Хост SMB-сервера.
    pub host: String,
    /// Имя шары.
    pub share: String,
    /// Путь внутри шары без ведущего слэша (пустой — корень шары).
    pub remote_dir: String,
}

/// Префикс FUSE-монтирований GVFS: `/run/user/<uid>/gvfs/`.
const GVFS_USER_PREFIX: &str = "/run/user/";
/// Каталог монтирований внутри `/run/user/<uid>/`.
const GVFS_DIR: &str = "gvfs/";
/// Схема SMB-монтирований GVFS (перед `:` — маркер, после — параметры).
const SMB_SHARE_MARKER: &str = "smb-share";
/// Схема сетевых путей GNOME/Windows-клиентов.
const SMB_URL_PREFIX: &str = "smb://";

/// Нормализовать вставленный текст: SMB/GVFS-путь → [`RemotePathHint`],
/// локальный путь или мусор → `None` (вставляем как есть).
///
/// - GVFS: `/run/user/<uid>/gvfs/smb-share:server=<HOST>,share=<SHARE>/<rel>`
///   (uid — любые цифры; параметры `server=`/`share=`/`domain=` в любом
///   порядке и регистре);
/// - `smb://[user@]host[:port]/share/<rel>` — пользователь игнорируется,
///   порт отбрасывается;
/// - слэши в `rel` схлопываются, завершающий `/` отбрасывается.
pub fn normalize_pasted_path(text: &str) -> Option<RemotePathHint> {
    let trimmed = text.trim();
    parse_gvfs_mount(trimmed).or_else(|| parse_smb_url(trimmed))
}

/// GVFS-монтирование: `/run/user/<uid>/gvfs/smb-share:<параметры>/<rel>`.
fn parse_gvfs_mount(path: &str) -> Option<RemotePathHint> {
    let rest = path.strip_prefix(GVFS_USER_PREFIX)?;
    let (uid, rest) = rest.split_once('/')?;
    if uid.is_empty() || !uid.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let rest = rest.strip_prefix(GVFS_DIR)?;
    let (marker, tail) = rest.split_once(':')?;
    if !marker.eq_ignore_ascii_case(SMB_SHARE_MARKER) {
        return None;
    }
    // Параметры не содержат `/` — всё после первого слэша относится к rel.
    let (params, rel) = tail.split_once('/').unwrap_or((tail, ""));
    let mut host = None;
    let mut share = None;
    for param in params.split(',') {
        let (key, value) = param.split_once('=')?;
        let (key, value) = (key.trim(), value.trim());
        match key.to_ascii_lowercase().as_str() {
            "server" if !value.is_empty() => host = Some(value.to_string()),
            "share" if !value.is_empty() => share = Some(value.to_string()),
            // domain=… и неизвестные параметры разбору не мешают.
            _ => {}
        }
    }
    Some(RemotePathHint { host: host?, share: share?, remote_dir: normalize_rel(rel) })
}

/// `smb://[user@]host[:port]/share/<rel>`: первый сегмент пути — шара.
fn parse_smb_url(url: &str) -> Option<RemotePathHint> {
    let rest = url.strip_prefix(SMB_URL_PREFIX)?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let host = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
    let host = host_without_port(host)?;
    let mut segments = path.split('/');
    let share = segments.next().unwrap_or("");
    if share.is_empty() {
        return None; // без шары прогон невозможен — это не подсказка
    }
    let rel = segments.filter(|segment| !segment.is_empty()).collect::<Vec<_>>();
    Some(RemotePathHint { host: host.to_string(), share: share.to_string(), remote_dir: rel.join("/") })
}

/// Хост до `:`; порт (если указан) должен быть числом — иначе это мусор.
fn host_without_port(host: &str) -> Option<&str> {
    match host.split_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit()) => {
            Some(host)
        }
        Some(_) => None,
        None => (!host.is_empty()).then_some(host),
    }
}

/// Схлопнуть слэши в относительном пути: ведущий, дублирующиеся и хвостовой.
fn normalize_rel(rel: &str) -> String {
    rel.split('/').filter(|segment| !segment.is_empty()).collect::<Vec<_>>().join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint(host: &str, share: &str, remote_dir: &str) -> Option<RemotePathHint> {
        Some(RemotePathHint { host: host.into(), share: share.into(), remote_dir: remote_dir.into() })
    }

    /// Золотая таблица: боевой сценарий владельца + варианты GVFS и `smb://`.
    #[test]
    fn gvfs_mounts_and_smb_urls_normalize_to_hints() {
        let cases = [
            // Боевой случай: drag&drop папки из файлового менеджера.
            (
                "/run/user/1000/gvfs/smb-share:server=192.168.1.88,share=public/from_share_BBGS_archive/2024/ZOOM0003",
                hint("192.168.1.88", "public", "from_share_BBGS_archive/2024/ZOOM0003"),
            ),
            // Корень шары: без rel и с завершающим слэшем.
            ("/run/user/1000/gvfs/smb-share:server=nas.local,share=media", hint("nas.local", "media", "")),
            ("/run/user/1000/gvfs/smb-share:server=nas,share=media/rec/", hint("nas", "media", "rec")),
            // Домен перед server; rel с кириллицей и пробелом.
            (
                "/run/user/1000/gvfs/smb-share:domain=WORKGROUP,server=nas,share=media/rec/запись 1",
                hint("nas", "media", "rec/запись 1"),
            ),
            // Другой uid, порядок и регистр параметров, регистр маркера.
            ("/run/user/7/gvfs/smb-share:SHARE=Media,Server=Nas/x", hint("Nas", "Media", "x")),
            ("/run/user/65534/gvfs/SMB-Share:server=h,share=s", hint("h", "s", "")),
            // smb:// с пользователем и портом (оба отбрасываются).
            (
                "smb://admin@nas.local:445/media/rec/a b/Привет.wav",
                hint("nas.local", "media", "rec/a b/Привет.wav"),
            ),
            ("smb://192.168.1.88/public/from_share/ZOOM0003", hint("192.168.1.88", "public", "from_share/ZOOM0003")),
            ("smb://nas/media", hint("nas", "media", "")),
            ("smb://NAS/Media/", hint("NAS", "Media", "")),
            ("smb://@host/share", hint("host", "share", "")),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_pasted_path(input), expected, "вход: {input}");
        }
    }

    /// Локальные пути и мусор не распознаются — вставляются как есть.
    #[test]
    fn local_paths_and_garbage_are_none() {
        let garbage = [
            "",
            "   ",
            "/mnt/docs/develop/split_audio",
            "records/ZOOM0003",
            "file:///mnt/x",
            "//nas/share",
            "\\\\nas\\share",
            "/run/user/abc/gvfs/smb-share:server=h,share=s",
            "/run/user/1000/gvfs/nfs-share:server=h,share=s",
            "/run/user/1000/gvfs/smb-share:server=,share=s",
            "/run/user/1000/gvfs/smb-share:share=s",
            "/run/user/1000/gvfs/smb-share:server",
            "smb://",
            "smb://host",
            "smb://host/",
            "smb://host:abc/share",
            "smb://host:/share",
        ];
        for input in garbage {
            assert_eq!(normalize_pasted_path(input), None, "вход: {input}");
        }
    }

    /// Пробелы по краям и дублирующиеся слэши нормализуются.
    #[test]
    fn whitespace_and_duplicate_slashes_are_normalized() {
        let gvfs = "  /run/user/1000/gvfs/smb-share:server=nas,share=media//a///b/  ";
        assert_eq!(normalize_pasted_path(gvfs), hint("nas", "media", "a/b"));

        let smb = "  smb://nas/media//x//y//  ";
        assert_eq!(normalize_pasted_path(smb), hint("nas", "media", "x/y"));
    }
}
