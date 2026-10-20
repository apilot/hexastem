//! TUI-бинарник `hexastem_tui` (ratatui + crossterm + tokio).
//!
//! Вся логика — в модуле `hexastem::tui`; здесь только парсинг аргументов
//! (`--engine`, ошибки парсинга — сообщение clap и код 2) и запуск.
//! Код выхода — итог последнего прогона (`runner::exit_code`: 0 — без
//! отказов, 1 — есть отказы/прогон не состоялся; 0 — прогонов не было).

use clap::Parser as _;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let args = hexastem::tui::TuiArgs::parse();
    let exit_code = hexastem::tui::run(args.engine).await?;
    std::process::exit(exit_code);
}
