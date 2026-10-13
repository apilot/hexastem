//! TUI-бинарник `audio_split_tui` (ratatui + crossterm + tokio).
//!
//! Вся логика — в модуле `audio_split::tui`; здесь только запуск.
//! Код выхода — итог последнего прогона (`runner::exit_code`: 0 — без
//! отказов, 1 — есть отказы/прогон не состоялся; 0 — прогонов не было).

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let exit_code = audio_split::tui::run().await?;
    std::process::exit(exit_code);
}
