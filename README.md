# HexaStem

[Русский](README.ru.md)

SMB batch stem-separation pipeline: it picks up `.wav` recordings from a NAS
directory (over SMB, no mounting), splits them with a neural network into
6 stems (vocals/drums/bass/guitar/piano/other), and uploads the stems back
to the share.

HexaStem is job management end to end: batch planning with resume, duplicate
detection, per-file progress, reports, and exit codes — plus a terminal UI on
top of the same engine.

The separation engine is a single **native** one: inference entirely in Rust
via ONNX Runtime (NVIDIA GPU through CUDA, with an automatic CPU fallback),
writing 24-bit FLAC.

| Configure & run (TUI) | Live progress | Batch summary |
|---|---|---|
| ![Config popup](https://raw.githubusercontent.com/apilot/hexastem/main/screenshots/03-config-en.png) | ![Running progress](https://raw.githubusercontent.com/apilot/hexastem/main/screenshots/10-running-progress-en.png) | ![Summary](https://raw.githubusercontent.com/apilot/hexastem/main/screenshots/12-summary-en.png) |

## How it works

1. **Plan**: scan `remote-dir` → only `*.wav`; skips `CONVERTED_*`,
   `DUPLICATE_*` and files whose `<basename>/` with all 6 stems already
   exists on the share (resume).
2. **Download** sources into `local-dir` (a single-file failure goes to the
   report, the batch continues).
3. **Deduplication**: files with identical `(data_size, frames)` are compared
   pairwise (see «Deduplication» below); duplicates are pulled from
   processing and renamed on the NAS strictly after the keeper's `Ok`.
4. **Separate**: one engine call per batch of keepers and uniques (the ONNX
   session stays hot — loaded once per batch); on each `file_done` event the
   stems are uploaded to `<basename>/<stem>.flac`.
5. **Rename** to `CONVERTED_<name>` — strictly after all 6 stems uploaded
   successfully (a failure before that step leaves the source untouched),
   then local cleanup.
6. **Report**: markdown (`split_report_YYYYMMDD-HHMMSS.md`) on the share +
   console summary + exit code (0 — no failures, 1 — some).

## FAT shares: working around 8.3 mangling

On a FAT share, `mkdir` with a long name stores it in 8.3 form, after which
`cd`/`ls` by the long name fail with `NT_STATUS_OBJECT_NAME_NOT_FOUND` —
breaking resume. The workaround in `SmbClient::create_dir`: the stem
directory is first created under a short 8.3-safe name `as_xxxxxxxx`
(deterministic path hash) and renamed to the final long name, which is then
remembered and resolved. Retries and tails of crashed runs are absorbed
(existing final dir / `COLLISION` → `Ok`); local folders (`--local`) don't
need any of this.

## Deduplication

Recorders produce duplicate files: the same take, but one track is louder
(digital gain + limiting). A real case: `unknown 2` (clean, peak −6.6 dBFS,
crest factor ~28 dB) and `unknown 3` (+11.1 dB, limited, peak −0.32 dBFS,
crest factor ~23 dB). Deduplication is on by default, disable with
`--no-dedup`.

### Method

- Candidates are grouped by `(data_size, frames)`; groups ≥ 2 are compared
  pairwise. Equal frame counts are a hard requirement (even a 1-frame
  difference → not a duplicate).
- Each PCM WAV is streamed (files ~600 MB don't fit in memory): RIFF header,
  three 30 s windows (start+60 s, middle, end−90 s, 50 ms envelope frames)
  and a full-file clip scan (true peak, sample share at the ceiling, longest
  run).
- A file is a duplicate if the correlation of normalized peak envelopes is
  ≥ **0.97** in all three windows at direct alignment (0 shift). The
  threshold has headroom: real duplicates score 0.985–0.999, different
  content — below 0.5.
- An unparsable header (not PCM WAV 16/24/32 bit, 1/2 channels) → the file
  counts as unique, with a note in the report.

### Keeper rule

One file per duplicate cluster is processed — the keeper:

1. **Clipping** files (peak at the ceiling, `true_peak ≥ 0.999`) are
   excluded;
2. a **global** crest-factor difference (whole-file peak dbfs − median of
   the median RMS of the three windows) > **2 dB** → the more dynamic one
   wins (NOT the limited one), even if quieter;
3. otherwise — the louder one (higher RMS).

The crest factor is specifically the global one, not the window average:
windows cover only 90 s of audio, and in the real unknown 2/3 case the
windowed average gave 21.6 vs 20.8 (Δ 0.8 ≤ 2 — the limited copy would
wrongly win), while the global one gave 28.7 vs 23.9 (Δ 4.8 > 2 — the clean
take wins).

### Duplicate fate

- Keeper reached `Ok` (6 stems uploaded, source → `CONVERTED_*`) → the
  duplicate is renamed on the NAS to `DUPLICATE_<name>`.
- Keeper didn't reach `Ok` → the duplicate is left untouched and reported
  as "waiting for the keeper's Ok". Duplicates are never deleted; files with
  the `DUPLICATE_*` prefix are skipped by later scans.
- Duplicate status doesn't affect the exit code; the report has a
  «Deduplication» section with a table of pairs (per-window corr, gain,
  crest factors, clip, decision).

## Build

```bash
cd hexastem
cargo build --release   # binaries: target/release/hexastem, hexastem_tui
cargo test              # unit + e2e tests (mocked: no network, no GPU)
```

Requirements: Rust stable ≥ 1.99, `smbclient` on the system. An NVIDIA GPU
with CUDA is needed for speed (ONNX Runtime inference); CPU works too —
slowly.

## Usage

```bash
hexastem --host <host> --share <share> --remote-dir <dir-on-share> \
         --local-dir <work-dir> [--onnx-model <file>] \
         [--output-format flac] [--smb-creds-file <file>] [--local]
```

| Option | Default | Description |
|---|---|---|
| `--host` | — | SMB server host |
| `--share` | — | Share name |
| `--remote-dir` | — | Directory with recordings on the share |
| `--local-dir` | — | Local working directory (download/upload) |
| `--onnx-model` | `<crate>/.models/onnx/bs_roformer_sw_6stem_fp32.onnx` | Engine ONNX model |
| `--output-format` | `flac` | Stem format (the engine writes FLAC only) |
| `--smb-creds-file` | — | smbclient auth file (`username=`, `password=`); otherwise `SMB_USER`/`SMB_PASS` from the environment |
| `--local` | off | Local folder instead of SMB (`remote-dir` is read from disk) — debugging and smoke runs without network |
| `--no-dedup` | off | Disable duplicate-record deduplication (every file is processed as-is) |

Credentials never live in code or argv: an auth file or `SMB_USER`/`SMB_PASS`
environment variables.

### Example: test share

```bash
SMB_USER=user SMB_PASS=pass \
./target/release/hexastem \
  --host nas.local --share music \
  --remote-dir kirtan/2026-10 \
  --local-dir /var/tmp/hexastem_work
```

Expected result on the share: `kirtan/2026-10/<track>/<stem>.flac` × 6, the
source renamed to `CONVERTED_<track>.wav`, `split_report_*.md` in the root
of `remote-dir`; stdout prints a summary like
`hexastem: total=2 ok=2 failed=0 skip(resume)=0 skip(converted)=1 time=…`.

### Example: local debugging without network

```bash
./target/release/hexastem --local \
  --host local --share local \
  --remote-dir /tmp/music_in --local-dir /tmp/music_work
```

### TUI

`hexastem_tui` drives the same engine interactively: edit the job preset in
a popup, watch per-file progress with a live waveform, browse the engine log
and dedup pairs, switch interface language (`l` — Russian/English), and run
in `--local` mode for offline testing. Presets persist in
`~/.config/hexastem/tui.toml`.

## Model & hardware

The default model is `.models/onnx/bs_roformer_sw_6stem_fp32.onnx` inside
the crate — [BS-Roformer SW, 6 stems, fp32](https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx)
(~700 MB). The path is overridden with `--onnx-model`.

The ONNX Runtime shared library can be swapped via `ORT_DYLIB_PATH`
(1.31.0 or newer recommended); otherwise the system/linked runtime is used.

Backend selection: the CUDA execution provider is attempted first; if
CUDA libraries are missing or the device is unsupported, HexaStem logs a
short human-readable reason and falls back to CPU EP automatically.

| Backend | 10 s stereo clip (BS-Roformer SW fp32) |
|---|---|
| NVIDIA GPU (CUDA) | real-time factor < 1 |
| CPU | real-time factor ~2.5–4 |

Known limitation: GPUs with 4 GB VRAM (e.g. GTX 1050 Ti) run out of memory
during full fp32 inference — a CPU fallback keeps the run alive. Hardware-aware
model selection (smaller models for modest GPUs) is planned.

## Layout

```
src/config.rs    — immutable clap config for a run
src/smb.rs       — smbclient wrapper: list/get/put/rename (+ listing parsing)
src/local_fs.rs  — local folder as SmbOps (--local: tests/debugging)
src/dedup.rs     — dedup: PCM WAV parser, windows/envelopes, compare_pair,
                   keeper rule (pure functions, std only)
src/pipeline.rs  — batch plan (resume/CONVERTED_*/DUPLICATE_*), DI traits
                   SmbOps/EngineOps, download→dedup→separate→upload→rename loop
src/engine/      — native engine: DSP chain (wav → resample → chunker →
                   stft → ONNX inference → iSTFT → crossfade → FLAC 24-bit),
                   CUDA→CPU fallback, progress event protocol
src/report.rs    — markdown report (+ «Deduplication» section) + console
                   summary + exit code
src/tui/         — hexastem_tui binary (ratatui): run popup, progress, log,
                   dedup tab, GPU panel
src/main.rs      — step assembly + e2e smoke (InMemorySmb + StubEngine)
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
