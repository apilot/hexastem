#!/usr/bin/env python3
"""Protocol and CLI tests for separator_driver (no GPU, no audio_separator needed).

Runs standalone: python3 test_driver_protocol.py  (also collectable by pytest).
Covers: JSON event shapes, JSON-lines emission, per-file failure isolation,
single retry per file, model-load failure handling, CLI defaults.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import wave
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import separator_driver as sd  # noqa: E402


class FakeSeparator:
    """Stands in for audio_separator.separator.Separator; behavior keyed by input path."""

    def __init__(self, behavior: dict[str, object], delay: float = 0.0):
        self._behavior = behavior
        self._delay = delay
        self.calls: list[str] = []
        self.output_dir: str | None = None
        self.model_instance = None

    def separate(self, audio_path: str, custom_output_names=None) -> list[str]:
        self.calls.append(audio_path)
        if self._delay:
            time.sleep(self._delay)  # simulates GPU work for estimate ticks
        result = self._behavior[audio_path]
        if isinstance(result, Exception):
            raise result
        names = sorted(custom_output_names.values())
        for stem in names:  # the driver checks stems exist on disk after separation
            (Path(self.output_dir) / f"{stem}.flac").write_text("stem")
        return [f"{stem}.flac" for stem in names]

    def load_model(self, model_filename: str) -> None:
        self.model_filename = model_filename


class FakeFactory:
    """Configurable separator factory: either yields a FakeSeparator or raises."""

    def __init__(self, separator: FakeSeparator | None = None, error: Exception | None = None):
        self._separator = separator
        self._error = error
        self.built = 0

    def __call__(self, config: sd.DriverConfig):
        self.built += 1
        if self._error is not None:
            raise self._error
        return self._separator


def collect_events(config: sd.DriverConfig, factory) -> list[dict]:
    events: list[dict] = []
    ok, failed = sd.run_batch(config, factory, events.append)
    events.append(sd.batch_done_event(ok, failed))
    events.append({"event": "exit_check", "ok": ok, "failed": failed})
    return events


def make_config(tmp: Path, names: list[str], **overrides) -> sd.DriverConfig:
    files = []
    for name in names:
        path = tmp / name
        path.write_text("fake audio")
        files.append(path)
    defaults = {
        "model_file": sd.DEFAULT_MODEL,
        "model_file_dir": tmp / "models",
        "output_dir": tmp / "out",
        "output_format": "FLAC",
        "files": tuple(files),
        "rtf_estimate": sd.DEFAULT_RTF_ESTIMATE,
    }
    return sd.DriverConfig(**{**defaults, **overrides})


def make_wav(path: Path, seconds: float, rate: int = 8000) -> Path:
    """Synthetic mono 16-bit wav (silence) with an exact frames/rate duration."""
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def test_event_builders_shapes():
    progress = sd.progress_event("/a.wav", "separate", 33.333)
    assert progress == {"event": "progress", "file": "/a.wav", "stage": "separate",
                        "pct": 33.3, "est": False}
    estimated = sd.progress_event("/a.wav", "separate", 50.0, est=True)
    assert estimated["est"] is True
    done = sd.file_done_event("/a.wav", ["/out/vocals.flac"], 1.234)
    assert done == {"event": "file_done", "file": "/a.wav", "stems": ["/out/vocals.flac"], "seconds": 1.23}
    error = sd.file_error_event("/a.wav", "boom")
    assert error == {"event": "file_error", "file": "/a.wav", "error": "boom"}
    summary = sd.batch_done_event(2, 1)
    assert summary == {"event": "batch_done", "ok": 2, "failed": 1}
    device = sd.device_event("NVIDIA GeForce RTX 4090")
    assert device == {"event": "device", "name": "NVIDIA GeForce RTX 4090"}


def test_probe_device_returns_a_device_name():
    # В среде с torch/cuda — имя GPU, без него — "cpu"; в любом случае строка.
    assert isinstance(sd.probe_device(), str) and sd.probe_device()


def test_probe_device_falls_back_to_cpu_when_torch_missing():
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch blocked for test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_torch
    try:
        assert sd.probe_device() == "cpu"
    finally:
        builtins.__import__ = real_import


def test_emit_writes_one_json_line():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        sd.emit(sd.batch_done_event(1, 0))
    lines = [line for line in buffer.getvalue().splitlines() if line]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "batch_done", "ok": 1, "failed": 0}


def test_batch_success_events():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        config = make_config(tmp, ["a.wav"])
        fake = FakeSeparator({str(config.files[0]): "ok"})
        events = collect_events(config, FakeFactory(fake))
        kinds = [e["event"] for e in events]
        assert kinds == [
            "progress", "progress", "device", "progress",
            "file_done", "batch_done", "exit_check",
        ]
        assert events[0]["stage"] == "load" and events[0]["pct"] == 0
        assert events[1]["stage"] == "load" and events[1]["pct"] == 100
        # Событие device идёт сразу после загрузки модели: имя GPU либо cpu.
        assert events[2] == {"event": "device", "name": sd.probe_device()}
        assert events[3]["stage"] == "separate"
        done = events[4]
        assert done["file"] == str(config.files[0])
        assert len(done["stems"]) == 6
        assert all(s.endswith(".flac") for s in done["stems"])
        assert done["seconds"] >= 0
        assert events[5] == {"event": "batch_done", "ok": 1, "failed": 0}


def test_failure_isolation_and_retry():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        config = make_config(tmp, ["good1.wav", "bad.wav", "good2.wav"])
        behavior = {
            str(config.files[0]): "ok",
            str(config.files[1]): RuntimeError("gpu exploded"),
            str(config.files[2]): "ok",
        }
        fake = FakeSeparator(behavior)
        events = collect_events(config, FakeFactory(fake))
        kinds = [e["event"] for e in events]
        # both good files survive the failure of the middle one
        assert kinds == [
            "progress", "progress", "device",
            "progress", "file_done",
            "progress", "progress", "file_error",
            "progress", "file_done",
            "batch_done", "exit_check",
        ]
        bad_calls = [c for c in fake.calls if "bad.wav" in c]
        assert len(bad_calls) == 2, f"expected exactly 1 retry, got {len(bad_calls)} attempts"
        assert events[-2] == {"event": "batch_done", "ok": 2, "failed": 1}
        assert events[7]["event"] == "file_error" and "gpu exploded" in events[7]["error"]
        assert events[9]["event"] == "file_done" and "good2" in events[9]["file"]


def test_model_load_failure_fails_all_files():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        config = make_config(tmp, ["a.wav", "b.wav"])
        events = collect_events(config, FakeFactory(error=RuntimeError("ckpt corrupted")))
        kinds = [e["event"] for e in events]
        # Device event never fires: the model (and its device) never loaded.
        assert kinds == ["progress", "file_error", "file_error", "batch_done", "exit_check"]
        assert all(e["event"] != "device" for e in events)
        assert events[0]["stage"] == "load" and events[0]["pct"] == 0
        assert events[-2] == {"event": "batch_done", "ok": 0, "failed": 2}
        assert all("ckpt corrupted" in e["error"] for e in events if e["event"] == "file_error")


def test_cli_defaults():
    args = sd.parse_args([
        "--files", "/x/a.wav",
        "--output-dir", "/tmp/out",
    ])
    assert args.model_file == "BS-Roformer-SW.ckpt"
    assert args.output_format.upper() == "FLAC"
    assert args.model_file_dir == sd.DEFAULT_MODEL_DIR
    assert args.rtf_estimate == sd.DEFAULT_RTF_ESTIMATE
    assert [str(f) for f in args.files] == ["/x/a.wav"]


def test_cli_explicit_options():
    args = sd.parse_args([
        "--model-file", "htdemucs_6s.yaml",
        "--files", "/x/a.wav", "/x/b.wav",
        "--output-dir", "/tmp/out",
        "--model-file-dir", "/tmp/models",
        "--output-format", "flac",
        "--rtf-estimate", "0.2",
    ])
    assert args.model_file == "htdemucs_6s.yaml"
    assert args.model_file_dir == Path("/tmp/models")
    assert args.rtf_estimate == 0.2
    assert len(args.files) == 2


def test_duration_secs_reads_header():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        wav = make_wav(tmp / "two_sec.wav", 2.0, rate=4000)
        duration = sd.duration_secs(wav)
        assert duration is not None and abs(duration - 2.0) < 1e-6
        text = tmp / "fake.wav"
        text.write_text("not audio at all")
        assert sd.duration_secs(text) is None


def test_estimated_pct_caps_and_guards():
    assert sd.estimated_pct(1.0, 10.0, 0.5) == 20.0
    assert sd.estimated_pct(1000.0, 10.0, 0.5) == sd.EST_CAP_PCT
    assert sd.estimated_pct(1.0, 0.0, 0.5) == 0.0  # no duration → no estimate
    assert sd.estimated_pct(1.0, 10.0, 0.0) == 0.0  # no rtf → no estimate


def test_est_ticks_monotonic_with_real_finish():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        wav = make_wav(tmp / "slow.wav", 2.0)  # 2s duration, rtf 0.5 → full 1.0s
        fake = FakeSeparator({str(wav): "ok"}, delay=0.35)
        events: list[dict] = []
        assert sd.process_file(fake, wav, tmp / "out", events.append, sd.RtfTracker(), tick=0.05)
        est = [e for e in events if e["event"] == "progress" and e["est"]]
        assert est, "expected time-based estimate ticks while separate() runs"
        pcts = [e["pct"] for e in est]
        assert all(e["est"] for e in est)
        assert all(a <= b for a, b in zip(pcts, pcts[1:])), f"not monotonic: {pcts}"
        assert all(p < 100 for p in pcts)
        done = [e for e in events if e["event"] == "file_done"]
        assert len(done) == 1 and done[0]["seconds"] >= 0.3


def test_actual_rtf_learns_for_next_file():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        wav1 = make_wav(tmp / "first.wav", 2.0)
        wav2 = make_wav(tmp / "second.wav", 2.0)
        behavior = {str(wav1): "ok", str(wav2): "ok"}
        fake = FakeSeparator(behavior, delay=0.3)  # actual rtf = 0.3/2 = 0.15
        tracker = sd.RtfTracker(initial=1000.0)  # absurd default → ~0% for the 1st file
        out = tmp / "out"

        first: list[dict] = []
        assert sd.process_file(fake, wav1, out, first.append, tracker, tick=0.05)
        assert tracker.rtf < 1.0, f"tracker did not learn: {tracker.rtf}"

        second: list[dict] = []
        assert sd.process_file(fake, wav2, out, second.append, tracker, tick=0.05)

        def max_est(events: list[dict]) -> float:
            return max((e["pct"] for e in events if e.get("est")), default=0.0)

        assert max_est(second) > max_est(first) + 5.0, (
            f"learned rtf must raise estimates: first={max_est(first)} second={max_est(second)}"
        )


def run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS {name}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
    passed = len(tests) - failures
    print(f"\n{passed}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_all())
