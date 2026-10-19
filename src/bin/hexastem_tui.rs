//! TUI-бинарник `hexastem_tui` (ratatui + crossterm + tokio).
//!
//! Вся логика — в модуле `hexastem::tui`; здесь только запуск.
//! Код выхода — итог последнего прогона (`runner::exit_code`: 0 — без
//! отказов, 1 — есть отказы/прогон не состоялся; 0 — прогонов не было).

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let exit_code = hexastem::tui::run().await?;
    std::process::exit(exit_code);
}
