//! GPU-поллер шапки: опрос `nvidia-smi` раз в [`GPU_POLL_INTERVAL`].
//!
//! Контракт subtask 06: [`spawn_poller`] крутит tokio-таск (interval →
//! `nvidia-smi --query-gpu=…`), снимок уходит в канал как [`GpuEvent`].
//! `Info(None)` — GPU недоступна (нет бинарника, exit≠0, таймаут, мусор
//! в выводе): в шапке остаётся «GPU: n/a» (виджет — `widgets::gpu_spans`).
//!
//! Остановка поллера — чистая в обоих вариантах: `abort` хэндла либо
//! дроп приёмника (неудачная отправка завершает цикл) — dangling-тасков нет.
//! Несколько GPU: берётся первая строка вывода. Поля-заглушки `[N/A]` /
//! `[Not Supported]`: утилизация → `util_percent: None`; память в
//! [`GpuInfo`] не опциональна, поэтому такая строка невалидна целиком.
//!
//! Логи: первая неудача опроса даёт одно сообщение в stderr, дальше —
//! тихий поллинг (без спама на каждый тик).

use std::time::Duration;

use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tokio::time::{MissedTickBehavior, timeout};

/// Период опроса nvidia-smi (решение владельца: 2 с).
pub const GPU_POLL_INTERVAL: Duration = Duration::from_secs(2);
/// Таймаут одного опроса: зависший бинарник не должен блокировать цикл
/// дольше интервала.
const QUERY_TIMEOUT: Duration = Duration::from_secs(2);
/// Бинарник и аргументы запроса (`csv,noheader,nounits`: одна строка на GPU,
/// значения без единиц — «2100», не «2100 MiB»).
const NVIDIA_SMI: &str = "nvidia-smi";
const NVIDIA_SMI_ARGS: [&str; 2] = [
    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
];

/// Снимок состояния GPU (первая строка вывода nvidia-smi).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// Имя GPU (`NVIDIA GeForce RTX 4090`).
    pub name: String,
    /// Утилизация, % — `None`, если драйвер не даёт значение (`[N/A]`).
    pub util_percent: Option<u8>,
    /// Занято VRAM, MiB.
    pub mem_used_mib: u32,
    /// Всего VRAM, MiB.
    pub mem_total_mib: u32,
}

/// События поллера для event loop (маппинг в `AppEvent` — следующий этап).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuEvent {
    /// Снимок GPU; `None` — GPU недоступна (показывать «GPU: n/a»).
    Info(Option<GpuInfo>),
}

/// Запускает поллер: `tokio::spawn` + `interval(GPU_POLL_INTERVAL)`.
///
/// Первый тик интервала срабатывает мгновенно — данные приходят сразу,
/// без стартовой паузы. Таск завершается сам при дропе приёмника
/// (quit) либо принудительно через `JoinHandle::abort`.
pub fn spawn_poller(tx: UnboundedSender<GpuEvent>) -> JoinHandle<()> {
    tokio::spawn(poll_loop(tx, GPU_POLL_INTERVAL))
}

/// Цикл поллера. Интервал — параметр: продакшен использует
/// [`GPU_POLL_INTERVAL`], тесты — малый интервал без ожидания 2 с.
async fn poll_loop(tx: UnboundedSender<GpuEvent>, period: Duration) {
    let mut ticker = tokio::time::interval(period);
    // Отставший тик не догоняем серией: опрос не чаще `period`.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut announced = false; // о недоступности уже сообщили — дальше тихо

    loop {
        ticker.tick().await;
        let info = query_async().await;
        match &info {
            Some(_) => announced = false,
            None if !announced => {
                announced = true;
                // Допустимый единственный eprintln в TUI-пути: поллер стартует уже на
                // alt-screen, строку затирает первый draw — видимого вывода не создаёт
                // (дублировать её в общий канал событий рано: UI ещё не читает лог).
                eprintln!("hexastem_tui: nvidia-smi недоступна — в шапке «GPU: n/a»");
            }
            None => {} // тихий поллинг: одно сообщение, не на каждый тик
        }
        if tx.send(GpuEvent::Info(info)).is_err() {
            break; // приёмник закрыт (quit) — поллер останавливается чисто
        }
    }
}

/// Один async-опрос: нет бинарника / exit≠0 / таймаут → `None`.
async fn query_async() -> Option<GpuInfo> {
    let output = timeout(QUERY_TIMEOUT, Command::new(NVIDIA_SMI).args(NVIDIA_SMI_ARGS).output())
        .await
        .ok()?
        .ok()?;
    decode_output(&output.status, &output.stdout)
}

/// Синхронный опрос nvidia-smi (разовая диагностика вне рантайма).
/// Та же graceful-семантика, что у [`query_async`].
pub fn query_once() -> Option<GpuInfo> {
    query_program(NVIDIA_SMI)
}

/// Опрос указанной программы: инъекция точки входа позволяет тестировать
/// graceful-ветки (нет бинарника / exit≠0) без манипуляций с PATH.
fn query_program(program: &str) -> Option<GpuInfo> {
    let output = std::process::Command::new(program).args(NVIDIA_SMI_ARGS).output().ok()?;
    decode_output(&output.status, &output.stdout)
}

/// Общий хвост sync/async путей: exit-код + stdout → [`GpuInfo`].
fn decode_output(status: &std::process::ExitStatus, stdout: &[u8]) -> Option<GpuInfo> {
    if !status.success() {
        return None;
    }
    parse_nvidia_smi_csv(&String::from_utf8_lossy(stdout))
}

/// Парсит вывод `nvidia-smi --format=csv,noheader,nounits` — чистая функция.
///
/// Берёт первую непустую строку (несколько GPU → первая). Формат поля:
/// `name, util%, mem.used, mem.total`. `None` — пусто/мусор/нехватка полей/
/// нечисловые память или утилизация; `[N/A]`/`[Not Supported]` в утилизации —
/// не ошибка (`util_percent: None`).
pub fn parse_nvidia_smi_csv(output: &str) -> Option<GpuInfo> {
    let line = output.lines().find(|line| !line.trim().is_empty())?;
    let mut fields = line.split(',');

    let name = fields.next()?.trim();
    if name.is_empty() {
        return None;
    }

    let util = fields.next()?.trim();
    let util_percent = if util.is_empty() || is_unavailable(util) {
        None
    } else {
        Some(util.parse().ok()?)
    };

    Some(GpuInfo {
        name: name.to_string(),
        util_percent,
        mem_used_mib: fields.next()?.trim().parse().ok()?,
        mem_total_mib: fields.next()?.trim().parse().ok()?,
    })
}

/// Поле-заглушка nvidia-smi для значений, которые драйвер не отдаёт.
fn is_unavailable(field: &str) -> bool {
    matches!(field, "[N/A]" | "[Not Supported]")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- парсер: чистая функция, тестируется без nvidia-smi ---

    #[test]
    fn parse_valid_line_full_info() {
        let info = parse_nvidia_smi_csv("NVIDIA GeForce RTX 4090, 63, 2100, 24564");
        assert_eq!(
            info,
            Some(GpuInfo {
                name: "NVIDIA GeForce RTX 4090".into(),
                util_percent: Some(63),
                mem_used_mib: 2100,
                mem_total_mib: 24564,
            })
        );
    }

    #[test]
    fn parse_trims_spaces_around_fields() {
        let info = parse_nvidia_smi_csv(" RTX 3090 ,  35 , 1203 , 24576 ");
        assert_eq!(
            info,
            Some(GpuInfo {
                name: "RTX 3090".into(),
                util_percent: Some(35),
                mem_used_mib: 1203,
                mem_total_mib: 24576,
            })
        );
    }

    #[test]
    fn parse_na_utilization_becomes_none() {
        let info = parse_nvidia_smi_csv("RTX, [N/A], 100, 200");
        assert_eq!(info.as_ref().map(|g| g.util_percent), Some(None));
        assert_eq!(info.as_ref().map(|g| g.mem_used_mib), Some(100));
        assert_eq!(info.as_ref().map(|g| g.mem_total_mib), Some(200));
    }

    #[test]
    fn parse_not_supported_utilization_becomes_none() {
        let info = parse_nvidia_smi_csv("RTX, [Not Supported], 100, 200");
        assert_eq!(info.as_ref().map(|g| g.util_percent), Some(None));
        assert_eq!(info.as_ref().map(|g| g.name.as_str()), Some("RTX"));
    }

    #[test]
    fn parse_unavailable_memory_invalidates_line() {
        // Память не опциональна в GpuInfo: строка невалидна целиком → n/a.
        assert_eq!(parse_nvidia_smi_csv("RTX, 10, [N/A], 200"), None);
        assert_eq!(parse_nvidia_smi_csv("RTX, 10, 100, [Not Supported]"), None);
    }

    #[test]
    fn parse_garbage_is_none() {
        assert_eq!(parse_nvidia_smi_csv("hello world"), None);
        assert_eq!(parse_nvidia_smi_csv("RTX, abc, 1, 2"), None);
        assert_eq!(parse_nvidia_smi_csv("RTX, 1, 2"), None); // мало полей
        assert_eq!(parse_nvidia_smi_csv(", 1, 2, 3"), None); // пустое имя
    }

    #[test]
    fn parse_empty_output_is_none() {
        assert_eq!(parse_nvidia_smi_csv(""), None);
        assert_eq!(parse_nvidia_smi_csv("   \n\t "), None);
    }

    #[test]
    fn parse_multiple_gpus_takes_first_line() {
        let output = "GPU A, 10, 100, 200\nGPU B, 20, 300, 400";
        let info = parse_nvidia_smi_csv(output).expect("первая строка валидна");
        assert_eq!(info.name, "GPU A");
        assert_eq!(info.mem_used_mib, 100);
        assert_eq!(info.mem_total_mib, 200);
    }

    #[test]
    fn parse_handles_crlf_lines() {
        // Windows-вывод nvidia-smi завершает строки \r\n.
        let info = parse_nvidia_smi_csv("GPU A, 10, 100, 200\r\n");
        assert_eq!(info.as_ref().map(|g| g.name.as_str()), Some("GPU A"));
    }

    // --- query: graceful без nvidia-smi (без манипуляций с PATH) ---

    #[test]
    fn query_missing_binary_is_none() {
        assert_eq!(query_program("nvidia-smi-missing-for-tests-xyz"), None);
    }

    #[cfg(unix)]
    #[test]
    fn query_nonzero_exit_is_none() {
        // `false` игнорирует аргументы и завершается с кодом 1.
        assert_eq!(query_program("false"), None);
    }

    #[cfg(unix)]
    #[test]
    fn query_zero_exit_but_unparsable_output_is_none() {
        // `true`: exit 0 и пустой stdout — нечего парсить.
        assert_eq!(query_program("true"), None);
    }

    // --- поллер: событие за первый тик + чистая остановка ---

    #[tokio::test]
    async fn poller_delivers_event_on_first_tick_and_stops_cleanly() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = tokio::spawn(poll_loop(tx, Duration::from_millis(10)));

        // Первый тик интервала мгновенный; Some/None зависит от наличия
        // nvidia-smi на машине — канал доставки одинаков.
        let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("поллер не ответил за 5 с");
        assert!(matches!(event, Some(GpuEvent::Info(_))));

        // Дроп приёмника (quit) → отправка фейлится → цикл завершается сам.
        drop(rx);
        let stopped = tokio::time::timeout(Duration::from_secs(5), handle).await;
        assert!(stopped.is_ok(), "поллер не завершился после дропа приёмника");
    }

    #[tokio::test]
    async fn poller_stops_on_abort() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = tokio::spawn(poll_loop(tx, Duration::from_millis(10)));
        handle.abort();
        // abort не паникует и не оставляет dangling: таск больше не тикает.
        assert!(handle.await.is_err(), "таск должен быть отменён");
    }
}
