//! Кольцевой буфер журнала событий для таба Log.
//!
//! Буфер фиксированной ёмкости (`LOG_CAPACITY`): при переполнении самые
//! старые записи вытесняются. Поддерживает фильтрацию по подстроке
//! (без учёта регистра) и прокрутку: `scroll == 0` означает «следовать за
//! хвостом» (автоскролл), значения больше 0 — отступ от хвоста вверх.
//!
//! Timestamp — UTC «HH:MM:SS» (см. [`format_hhmmss`]): в проекте нет
//! зависимости `chrono`, время берётся из `SystemTime::UNIX_EPOCH`.

use std::collections::VecDeque;

/// Ёмкость кольцевого буфера журнала.
pub const LOG_CAPACITY: usize = 1000;
/// На сколько строк сдвигается прокрутка за один PgUp/PgDn.
pub const PAGE_LINES: usize = 10;

/// Одна запись журнала: время (UTC «HH:MM:SS») и текст сообщения.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    pub time: String,
    pub message: String,
}

impl LogEntry {
    pub fn new(time: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            time: time.into(),
            message: message.into(),
        }
    }

    /// Полная строка для рендера: «HH:MM:SS  сообщение».
    pub fn line(&self) -> String {
        format!("{}  {}", self.time, self.message)
    }
}

/// Кольцевой буфер журнала с фильтром и прокруткой.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LogBuffer {
    entries: VecDeque<LogEntry>,
    /// Активный фильтр по подстроке (case-insensitive); `None` — без фильтра.
    filter: Option<String>,
    /// Отступ от хвоста в строках; 0 — автоскролл (хвост).
    scroll: usize,
}

impl LogBuffer {
    pub fn push(&mut self, time: impl Into<String>, message: impl Into<String>) {
        if self.entries.len() == LOG_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back(LogEntry::new(time, message));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Очищает журнал целиком: записи, прокрутку и фильтр.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.filter = None;
        self.scroll = 0;
    }

    pub fn filter(&self) -> Option<&str> {
        self.filter.as_deref()
    }

    /// Устанавливает фильтр; пустая строка трактуется как «без фильтра».
    pub fn set_filter(&mut self, filter: Option<String>) {
        self.filter = filter.filter(|f| !f.is_empty());
        self.scroll = 0;
    }

    /// Записи, прошедшие фильтр, в хронологическом порядке.
    pub fn visible(&self) -> Vec<&LogEntry> {
        match &self.filter {
            None => self.entries.iter().collect(),
            Some(f) => {
                let needle = f.to_lowercase();
                self.entries
                    .iter()
                    .filter(|e| e.message.to_lowercase().contains(&needle))
                    .collect()
            }
        }
    }

    pub fn visible_len(&self) -> usize {
        self.visible().len()
    }

    pub fn scroll(&self) -> usize {
        self.scroll
    }

    /// Прокрутка вверх (от хвоста) на `lines` строк.
    pub fn scroll_up(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_add(lines);
    }

    /// Прокрутка вниз (к хвосту) на `lines` строк; 0 — автоскролл.
    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    /// Вернуть прокрутку в режим автоскролла (хвост).
    pub fn scroll_to_tail(&mut self) {
        self.scroll = 0;
    }
}

/// Форматирует unix-время (секунды с `UNIX_EPOCH`) в UTC «HH:MM:SS».
///
/// Чистая функция без внешних зависимостей: в проекте нет `chrono`.
pub fn format_hhmmss(unix_secs: u64) -> String {
    let secs_of_day = unix_secs % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60
    )
}

/// Текущее unix-время в секундах; при ошибке системных часов — 0.
pub fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(n: usize) -> LogEntry {
        LogEntry::new(format!("{n:02}:00:00"), format!("msg {n}"))
    }

    #[test]
    fn push_keeps_chronological_order() {
        let mut buf = LogBuffer::default();
        buf.push("00:00:01", "first");
        buf.push("00:00:02", "second");

        let visible: Vec<String> = buf.visible().iter().map(|e| e.message.clone()).collect();
        assert_eq!(visible, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn capacity_evicts_oldest_entries() {
        let mut buf = LogBuffer::default();
        for i in 0..(LOG_CAPACITY + 5) {
            buf.push("00:00:00", format!("msg {i}"));
        }

        assert_eq!(buf.len(), LOG_CAPACITY);
        let first = buf.visible().first().expect("non-empty").message.clone();
        assert_eq!(first, format!("msg {}", 5));
    }

    #[test]
    fn filter_matches_case_insensitive_substring() {
        let mut buf = LogBuffer::default();
        buf.push("00:00:01", "download a.wav");
        buf.push("00:00:02", "separate b.wav");
        buf.set_filter(Some("DOWN".to_string()));

        let visible: Vec<String> = buf.visible().iter().map(|e| e.message.clone()).collect();
        assert_eq!(visible, vec!["download a.wav".to_string()]);
    }

    #[test]
    fn empty_filter_string_disables_filter() {
        let mut buf = LogBuffer::default();
        buf.push("00:00:01", "a");
        buf.set_filter(Some(String::new()));

        assert_eq!(buf.filter(), None);
        assert_eq!(buf.visible_len(), 1);
    }

    #[test]
    fn clear_resets_entries_scroll_and_filter() {
        let mut buf = LogBuffer::default();
        buf.push("00:00:01", "a");
        buf.set_filter(Some("a".to_string()));
        buf.scroll_up(3);

        buf.clear();

        assert!(buf.is_empty());
        assert_eq!(buf.filter(), None);
        assert_eq!(buf.scroll(), 0);
    }

    #[test]
    fn scroll_up_down_roundtrip_stops_at_tail() {
        let mut buf = LogBuffer::default();
        for i in 0..20 {
            buf.push("00:00:00", format!("m{i}"));
        }

        buf.scroll_up(5);
        assert_eq!(buf.scroll(), 5);

        buf.scroll_down(PAGE_LINES);
        assert_eq!(buf.scroll(), 0);
    }

    #[test]
    fn visible_len_reflects_filter() {
        let mut buf = LogBuffer::default();
        buf.push("00:00:01", "download a.wav");
        buf.push("00:00:02", "separate b.wav");
        buf.push("00:00:03", "download c.wav");

        assert_eq!(buf.visible_len(), 3);
        buf.set_filter(Some("download".to_string()));
        assert_eq!(buf.visible_len(), 2);
    }

    #[test]
    fn format_hhmmss_zero_is_midnight() {
        assert_eq!(format_hhmmss(0), "00:00:00");
    }

    #[test]
    fn format_hhmmss_formats_known_instants() {
        assert_eq!(format_hhmmss(3_661), "01:01:01");
        assert_eq!(format_hhmmss(86_399), "23:59:59");
    }

    #[test]
    fn entry_line_prefixes_timestamp() {
        let e = entry(7);
        assert_eq!(e.line(), "07:00:00  msg 7");
    }

    #[test]
    fn now_unix_secs_is_monotonic_positive() {
        let a = now_unix_secs();
        let b = now_unix_secs();
        assert!(a > 0);
        assert!(b >= a);
    }
}
