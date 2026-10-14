#!/usr/bin/env python3
"""Batch stem-separation driver over audio-separator (JSON-lines protocol for Rust).

One Separator per batch ("model hot"): the model is loaded once and reused
for every file (load_model() caches model_instance internally). All driver
protocol output goes to stdout as one JSON object per line; library chatter
goes to stderr:

  {"event": "progress", "file": "...", "stage": "load|separate", "pct": 0..100,
   "est": true|false}
  # est=true — оценка по времени (audio-separator не отдаёт прогресс separate);
  # est=false — точное значение (старт 0%, финал 100%).
  {"event": "device", "name": "..."}
  # сразу после load_model(): имя CUDA-GPU (torch.cuda.get_device_name(0))
  # или "cpu", если CUDA недоступен.
  {"event": "file_done", "file": "...", "stems": ["...", ...], "seconds": 12.3}
  {"event": "file_error", "file": "...", "error": "..."}
  {"event": "batch_done", "ok": N, "failed": M}

A failing file never aborts the batch: every exception is caught per file and
reported as file_error after exactly one in-place retry. Stems are written as
FLAC into <output_dir>/<file stem>/ with clean names via custom_output_names.

API verified against installed audio-separator 0.47.0:
  Separator(model_file_dir=..., output_dir=..., output_format=..., mdxc_params=...)
  separator.load_model(model_filename, force_reload=False)
  separator.separate(audio_file_path, custom_output_names=None) -> list[str]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEFAULT_MODEL = "BS-Roformer-SW.ckpt"
FALLBACK_MODEL = "htdemucs_6s.yaml"
DEFAULT_OUTPUT_FORMAT = "FLAC"
# audio_split/.models — persistent model cache (git-ignored)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / ".models"
ATTEMPTS_PER_FILE = 2  # 1 try + 1 retry
# audio-separator не отдаёт прогресс separate: пока работает worker-поток,
# главный цикл раз в EST_TICK_SEC шлёт оценку прогресса (progress "est": true).
EST_TICK_SEC = 1.0
EST_CAP_PCT = 95.0  # оценка не заявляет завершение — финальные 100% только реальные
DEFAULT_RTF_ESTIMATE = 0.5  # стартовая оценка RTF (elapsed/duration), уточняется по бою
# Stems of BS-Roformer-SW / htdemucs_6s; used as clean custom output names.
EXPECTED_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")

Emitter = Callable[[dict], None]
SeparatorFactory = Callable[["DriverConfig"], Any]


@dataclass(frozen=True)
class DriverConfig:
    model_file: str
    model_file_dir: Path
    output_dir: Path
    output_format: str
    files: tuple[Path, ...]
    rtf_estimate: float = DEFAULT_RTF_ESTIMATE


# --- protocol: pure event builders -------------------------------------------


def progress_event(file: str, stage: str, pct: float, est: bool = False) -> dict:
    return {
        "event": "progress",
        "file": file,
        "stage": stage,
        "pct": round(pct, 1),
        "est": est,
    }


def file_done_event(file: str, stems: list[str], seconds: float) -> dict:
    return {
        "event": "file_done",
        "file": file,
        "stems": stems,
        "seconds": round(seconds, 2),
    }


def file_error_event(file: str, error: str) -> dict:
    return {"event": "file_error", "file": file, "error": error}


def batch_done_event(ok: int, failed: int) -> dict:
    return {"event": "batch_done", "ok": ok, "failed": failed}


def device_event(name: str) -> dict:
    """Compute device report: CUDA GPU name or "cpu" (for the TUI header)."""
    return {"event": "device", "name": name}


def probe_device() -> str:
    """GPU name via torch; "cpu" when CUDA is unavailable or torch is missing."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 — probe must never kill the batch
        pass
    return "cpu"


def emit(event: dict) -> None:
    """One JSON object per line on stdout; flush so the Rust reader sees it live."""
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- separation ---------------------------------------------------------------


def log_environment() -> None:
    """GPU/CUDA status to stderr before any work starts."""
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "cpu"
        _log(f"torch {torch.__version__} | cuda_available={cuda} | device={device}")
    except Exception as error:  # noqa: BLE001 — probe must never kill the batch
        _log(f"torch probe failed: {type(error).__name__}: {error}")


def build_separator(config: DriverConfig) -> Any:
    """Single Separator for the whole batch; model loaded once ("model hot")."""
    from audio_separator.separator import Separator  # lazy: keeps unit tests dependency-free

    separator = Separator(
        model_file_dir=str(config.model_file_dir),
        output_dir=str(config.output_dir),
        output_format=config.output_format,
    )
    separator.load_model(config.model_file)
    return separator


def point_output_dir(separator: Any, out_dir: Path) -> None:
    """The library writes via model_instance.output_dir — set it on both objects."""
    separator.output_dir = str(out_dir)
    model_instance = getattr(separator, "model_instance", None)
    if model_instance is not None:
        model_instance.output_dir = str(out_dir)


def resolve_stem_paths(out_dir: Path, names: list[str]) -> list[str]:
    """Normalize whatever separate() returned into existing absolute paths."""
    paths: list[str] = []
    for name in names:
        candidate = Path(name)
        if not candidate.is_file():
            candidate = out_dir / Path(name).name
        if not candidate.is_file():
            raise FileNotFoundError(f"stem missing after separation: {name}")
        paths.append(str(candidate.resolve()))
    return paths


def separate_one(separator: Any, audio_path: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    point_output_dir(separator, out_dir)
    custom_names = {stem: stem for stem in EXPECTED_STEMS}
    names = separator.separate(str(audio_path), custom_output_names=custom_names)
    if not names:
        raise RuntimeError("separation returned no stems")
    return resolve_stem_paths(out_dir, names)


def duration_secs(path: Path) -> float | None:
    """Audio duration from the file header only (soundfile.info, no decoding)."""
    try:
        import soundfile  # lazy: keeps unit tests dependency-free

        info = soundfile.info(str(path))
    except Exception:  # noqa: BLE001 — unknown/corrupt format → no estimate, just 0%
        return None
    if info.samplerate <= 0:
        return None
    return info.frames / info.samplerate


def estimated_pct(elapsed_sec: float, duration_sec: float, rtf: float) -> float:
    """Time-based progress estimate; capped below 100 until the real finish."""
    if duration_sec <= 0 or rtf <= 0:
        return 0.0
    return min(EST_CAP_PCT, elapsed_sec / (duration_sec * rtf) * 100.0)


class RtfTracker:
    """Self-learning RTF: starts from the CLI estimate, learns from finished files."""

    def __init__(self, initial: float = DEFAULT_RTF_ESTIMATE):
        self.rtf = initial if initial > 0 else DEFAULT_RTF_ESTIMATE

    def learn(self, elapsed_sec: float, duration_sec: float | None) -> None:
        """Feed a measured RTF back (ignored when inputs are unusable)."""
        if duration_sec is not None and duration_sec > 0 and elapsed_sec > 0:
            self.rtf = elapsed_sec / duration_sec


def separate_with_progress(
    separator: Any,
    audio_path: Path,
    out_dir: Path,
    emit_fn: Emitter,
    rtf: float,
    duration: float | None,
    tick: float = EST_TICK_SEC,
) -> tuple[list[str], float]:
    """Run separate() in a worker thread, ticking time-based progress meanwhile.

    Returns (stem_paths, elapsed_seconds); raises the worker's exception on the
    caller side so process_file's retry logic sees it as before.
    """
    started = time.perf_counter()
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["stems"] = separate_one(separator, audio_path, out_dir)
        except Exception as error:  # noqa: BLE001 — re-raised on the caller side
            result["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(tick)
        if thread.is_alive() and duration is not None:
            elapsed = time.perf_counter() - started
            emit_fn(progress_event(
                str(audio_path), "separate", estimated_pct(elapsed, duration, rtf), est=True,
            ))
    if "error" in result:
        raise result["error"]
    if "stems" not in result:
        raise RuntimeError("separation thread finished without result")
    return result["stems"], time.perf_counter() - started


def process_file(
    separator: Any,
    audio_path: Path,
    output_dir: Path,
    emit_fn: Emitter,
    tracker: RtfTracker,
    attempts: int = ATTEMPTS_PER_FILE,
    tick: float = EST_TICK_SEC,
) -> bool:
    """One file, retried in place; any failure is reported, never raised."""
    out_dir = output_dir / audio_path.stem
    duration = duration_secs(audio_path)
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        emit_fn(progress_event(str(audio_path), "separate", 0))
        try:
            stems, seconds = separate_with_progress(
                separator, audio_path, out_dir, emit_fn, tracker.rtf, duration, tick,
            )
            tracker.learn(seconds, duration)
            emit_fn(file_done_event(str(audio_path), stems, seconds))
            return True
        except Exception as error:  # noqa: BLE001 — one bad file must not stop the batch
            last_error = f"{type(error).__name__}: {error}"
            _log(f"attempt {attempt}/{attempts} failed for {audio_path.name}: {last_error}")
    emit_fn(file_error_event(str(audio_path), last_error))
    return False


def run_batch(
    config: DriverConfig,
    factory: SeparatorFactory,
    emit_fn: Emitter,
) -> tuple[int, int]:
    """Batch driver: one model load, then per-file isolation. Returns (ok, failed)."""
    first = str(config.files[0]) if config.files else ""
    emit_fn(progress_event(first, "load", 0))
    try:
        separator = factory(config)
    except Exception as error:  # noqa: BLE001 — load failure fails every file
        message = f"model load failed: {type(error).__name__}: {error}"
        _log(message)
        for path in config.files:
            emit_fn(file_error_event(str(path), message))
        return 0, len(config.files)
    emit_fn(progress_event(first, "load", 100))
    emit_fn(device_event(probe_device()))
    tracker = RtfTracker(config.rtf_estimate)
    results = [
        process_file(separator, path, config.output_dir, emit_fn, tracker)
        for path in config.files
    ]
    ok = sum(results)
    return ok, len(results) - ok


# --- CLI ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch stem separation over audio-separator; JSON-lines protocol on stdout.",
    )
    parser.add_argument("--model-file", default=DEFAULT_MODEL,
                        help=f"model checkpoint name (default: {DEFAULT_MODEL}; "
                             f"alternative: {FALLBACK_MODEL})")
    parser.add_argument("--files", nargs="+", required=True, type=Path,
                        help="audio files to separate")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="root output dir; stems go to <output-dir>/<file stem>/")
    parser.add_argument("--model-file-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="model cache dir (default: audio_split/.models)")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT,
                        help="stem format: FLAC | WAV | MP3 (default: FLAC)")
    parser.add_argument("--rtf-estimate", type=float, default=DEFAULT_RTF_ESTIMATE,
                        help="initial real-time-factor guess for estimated separate "
                             "progress; refined after each finished file "
                             f"(default: {DEFAULT_RTF_ESTIMATE})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = DriverConfig(
        model_file=args.model_file,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        output_format=args.output_format.upper(),
        files=tuple(args.files),
        rtf_estimate=args.rtf_estimate,
    )
    log_environment()
    _log(
        f"model={config.model_file} | model_dir={config.model_file_dir} | "
        f"output_dir={config.output_dir} | format={config.output_format} | "
        f"files={len(config.files)}"
    )
    ok, failed = run_batch(config, build_separator, emit)
    emit(batch_done_event(ok, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
