//! TUI-бинарник `audio_split_tui` (ratatui + crossterm + tokio).
//!
//! Вся логика — в модуле `audio_split::tui`; здесь только запуск.

#[tokio::main]
async fn main() -> std::io::Result<()> {
    audio_split::tui::run().await
}
