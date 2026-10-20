# HexaStem

SMB batch stem-separation pipeline: берёт `.wav`-записи из директории на NAS
(по SMB, без монтирования), разделяет их нейросетью на 6 стемов
(vocals/drums/bass/guitar/piano/other) и возвращает стемы обратно на шару.

Движки разделения: **native** (дефолт) — инференс в Rust через ONNX Runtime
(CUDA), пишет FLAC 24-bit; **python** — legacy через [audio-separator]
(python-драйвер `python/separator_driver.py` из venv: грузит модель один раз
на батч и отдаёт прогресс JSON-строками в stdout).

## Как работает

1. **План**: скан `remote_dir` → только `*.wav`; скип `CONVERTED_*`,
   `DUPLICATE_*` и файлов, у которых на шаре уже есть `<basename>/` со всеми
   6 стемами (resume).
2. **Download** исходников в `local_dir` (отказ одного файла — в отчёт, батч продолжается).
3. **Дедупликация**: файлы с одинаковыми `(data_size, frames)` попарно
   сравниваются (см. «Дедупликация» ниже); дубли снимаются с обработки,
   их rename на NAS — строго после `Ok` хранителя.
4. **Separate**: один вызов драйвера на батч хранителей и уникальных
   (модель «горячая»); по событию `file_done` — upload стемов
   в `<basename>/<stem>.flac`.
5. **Rename** `CONVERTED_<имя>` — строго после успешной заливки всех 6 стемов
   (сбой раньше шага оставляет исходник нетронутым), затем локальный cleanup.
6. **Отчёт**: markdown (`split_report_YYYYMMDD-HHMMSS.md`) на шару +
   консольная сводка + код возврата (0 — нет отказов, 1 — есть).

## FAT-шары: обход mangle 8.3

На FAT-шаре `mkdir` длинного имени сохраняет его в 8.3-виде, после чего
`cd`/`ls` по длинному имени дают `NT_STATUS_OBJECT_NAME_NOT_FOUND` — ломая
resume. Обход в `SmbClient::create_dir`: директория стемов создаётся
коротким 8.3-safe именем `as_xxxxxxxx` (детерминированный хеш пути) и
переименовывается в финальное длинное — такое имя хранится и резолвится.
Повторы и хвосты упавших прогонов погашены (существующий финал /
`COLLISION` → `Ok`); на локальных папках (`--local`) обход не нужен.

## Дедупликация

На рекордере появляются файлы-дубли: одна и та же запись, но одна дорожка
громче (цифровой гейн + лимитер). Реальный кейс: `unknown 2` (чистая, пик
−6.6 дБФС, крест-фактор ~28 дБ) и `unknown 3` (+11.1 дБ, пережата лимитером,
пик −0.32 дБФС, крест-фактор ~23 дБ). Дедупликация включена по умолчанию,
отключается флагом `--no-dedup`.

### Методика

- Кандидаты группируются по `(data_size, frames)`; группы ≥ 2 разбираются
  попарно. Равное число фреймов — обязательное условие дубля (разница даже
  в 1 фрейм → не дубль).
- Каждый PCM WAV читается потоково (файлы ~600 МБ в память не помещаются):
  RIFF-заголовок, три окна по 30 с (старт+60 с, середина, конец−90 с,
  фрейм огибающей 50 мс) и блок-скан всего файла на клип (true peak, доля
  сэмплов у потолка, максимальная серия).
- Файл — дубль, если корреляция нормированных пик-огибающих ≥ **0.97**
  в каждом из трёх окон при прямом выравнивании (сдвиг 0). Порог взят
  с запасом: реальные дубли дают 0.985–0.999, разный контент — < 0.5.
- Заголовок не разобрался (не PCM WAV 16/24/32 бит, 1/2 каналов) → файл
  считается уникальным, примечание попадает в отчёт.

### Правило хранителя

Из кластера дублей обрабатывается один — хранитель:

1. **Клипующие** (пик у потолка, `true_peak ≥ 0.999`) исключаются;
2. различие **глобального** крест-фактора (dbfs пика всего файла − медиана
   медианных RMS трёх окон) > **2 дБ** → хранитель более динамичный
   (НЕ лимитерованный), даже если тише;
3. иначе — более громкий (выше RMS).

Крест-фактор — именно глобальный, не средний по окнам: окна покрывают лишь
90 с записи, и в реальном кейсе unknown 2/3 оконный средний давал 21.6 против
20.8 (Δ 0.8 ≤ 2, хранителем ошибочно становилась лимитерованная копия),
глобальный — 28.7 против 23.9 (Δ 4.8 > 2, хранитель — чистая).

### Судьба дубля

- Хранитель дошёл до `Ok` (6 стемов залиты, исходник → `CONVERTED_*`) →
  дубль переименовывается на NAS в `DUPLICATE_<имя>`.
- Хранитель не дошёл до `Ok` → дубль не трогается, в отчёте — «ждёт Ok
  хранителя». Дубли никогда не удаляются; файлы с префиксом `DUPLICATE_*`
  при следующих прогонах скипаются сканом.
- Статус дубли не влияет на exit code; в отчёте — секция «Дедупликация»
  с таблицей пар (corr по окнам, гейн, крест-факторы, клип, решение).

## Сборка

```bash
cd hexastem
cargo build --release   # бинарник target/release/hexastem
cargo test              # юнит- и e2e-тесты (моки: без сети и без GPU)
```

Требования: Rust stable ≥ 1.99; для реального прогона с `--engine python` —
python 3.11 venv с
`audio-separator[gpu]==0.47.0` (torch пин `2.7.1+cu126` для Pascal-видеокарт),
`smbclient` в системе, ffmpeg (для декодирования не-wav источников библиотекой).

## Запуск

```bash
hexastem --host <host> --share <share> --remote-dir <dir-on-share> \
            --local-dir <work-dir> [--model sw|6s] [--engine native|python] \
            [--onnx-model <file>] [--output-format flac] [--venv-path <dir>] \
            [--smb-creds-file <file>] [--local]
```

| Параметр | Дефолт | Описание |
|---|---|---|
| `--host` | — | Хост SMB-сервера |
| `--share` | — | Имя шары |
| `--remote-dir` | — | Директория с записями на шаре |
| `--local-dir` | — | Локальная рабочая директория (загрузка/выгрузка) |
| `--model` | `sw` | `sw` = BS-Roformer-SW.ckpt (SOTA, ~2.6 GiB VRAM); `6s` = htdemucs_6s.yaml (быстрее, ~1.1 GiB) |
| `--engine` | `native` | Движок разделения: `native` = ONNX Runtime CUDA в Rust (FLAC 24-bit); `python` = legacy через audio-separator (subprocess, требует venv) |
| `--onnx-model` | `<крейт>/.models/onnx/bs_roformer_sw_6stem_fp32.onnx` | ONNX-модель нативного движка (`--engine native`) |
| `--output-format` | `flac` | Формат стемов (`flac`/`wav`/`mp3`) |
| `--venv-path` | `<крейт>/.venv` | Python venv с audio-separator |
| `--smb-creds-file` | — | Файл кредов smbclient (`username=`, `password=`); иначе `SMB_USER`/`SMB_PASS` из env |
| `--local` | выкл | Локальная папка вместо SMB (`remote-dir` читается с диска) — отладка и smoke без сети |
| `--no-dedup` | выкл | Отключить дедупликацию дублей-записей (каждый файл обрабатывается как есть) |

Креды никогда не в коде и не в argv: auth-файл или переменные `SMB_USER`/`SMB_PASS`.

### Пример: тестовая шара

```bash
SMB_USER=user SMB_PASS=pass \
./target/release/hexastem \
  --host nas.local --share music \
  --remote-dir kirtan/2026-10 \
  --local-dir /var/tmp/hexastem_work \
  --model sw
```

Ожидаемый результат на шаре: `kirtan/2026-10/<трек>/<stem>.flac` × 6,
исходник переименован в `CONVERTED_<трек>.wav`, в корне `remote-dir` —
`split_report_*.md`; в stdout — строка сводки вида
`hexastem: всего=2 ok=2 failed=0 skip(resume)=0 skip(converted)=1 время=…`.

### Пример: локальная отладка без сети

```bash
./target/release/hexastem --local \
  --host local --share local \
  --remote-dir /tmp/music_in --local-dir /tmp/music_work \
  --venv-path .venv
```

### Venv и модели

Venv по умолчанию — `.venv` внутри крейта; развёртывание (см. также сессию
`2026-10-13-audio-split-pipeline`):

```bash
python3.11 -m venv .venv
.venv/bin/pip install "audio-separator[gpu]==0.47.0" "torch==2.7.1+cu126"
```

Кэш моделей — `.models` внутри крейта (драйвер ведёт его сам); при первом
прогоне модели скачаются автоматически (~720 MB для `sw`).

## Структура

```
src/config.rs    — clap-конфиг прогона (иммутабельный)
src/smb.rs       — обёртка smbclient: list/get/put/rename (+ парсинг листинга)
src/local_fs.rs  — локальная папка как SmbOps (--local: тесты/отладка)
src/dedup.rs     — дедупликация: парсер PCM WAV, окна/огибающие, compare_pair,
                   правило хранителя (чистые функции, std only)
src/pipeline.rs  — план батча (resume/CONVERTED_*/DUPLICATE_*), DI-трейты
                   SmbOps/EngineOps, цикл download→dedup→separate→upload→rename,
                   адаптер драйвера
src/report.rs    — markdown-отчёт (+ секция «Дедупликация») + консольная сводка
                   + exit code
src/main.rs      — сборка шагов + e2e-смоук (InMemorySmb + StubEngine)
python/separator_driver.py — python-драйвер audio-separator (JSON-lines в stdout)
```

[audio-separator]: https://pypi.org/project/audio-separator/
