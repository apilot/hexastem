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
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import separator_driver as sd  # noqa: E402


class FakeSeparator:
    """Stands in for audio_separator.separator.Separator; behavior keyed by input path."""

    def __init__(self, behavior: dict[str, object]):
        self._behavior = behavior
        self.calls: list[str] = []
        self.output_dir: str | None = None
        self.model_instance = None

    def separate(self, audio_path: str, custom_output_names=None) -> list[str]:
        self.calls.append(audio_path)
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
    }
    return sd.DriverConfig(**{**defaults, **overrides})


def test_event_builders_shapes():
    progress = sd.progress_event("/a.wav", "separate", 33.333)
    assert progress == {"event": "progress", "file": "/a.wav", "stage": "separate", "pct": 33.3}
    done = sd.file_done_event("/a.wav", ["/out/vocals.flac"], 1.234)
    assert done == {"event": "file_done", "file": "/a.wav", "stems": ["/out/vocals.flac"], "seconds": 1.23}
    error = sd.file_error_event("/a.wav", "boom")
    assert error == {"event": "file_error", "file": "/a.wav", "error": "boom"}
    summary = sd.batch_done_event(2, 1)
    assert summary == {"event": "batch_done", "ok": 2, "failed": 1}


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
            "progress", "progress", "progress",
            "file_done", "batch_done", "exit_check",
        ]
        assert events[0]["stage"] == "load" and events[0]["pct"] == 0
        assert events[1]["stage"] == "load" and events[1]["pct"] == 100
        assert events[2]["stage"] == "separate"
        done = events[3]
        assert done["file"] == str(config.files[0])
        assert len(done["stems"]) == 6
        assert all(s.endswith(".flac") for s in done["stems"])
        assert done["seconds"] >= 0
        assert events[4] == {"event": "batch_done", "ok": 1, "failed": 0}


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
            "progress", "progress",
            "progress", "file_done",
            "progress", "progress", "file_error",
            "progress", "file_done",
            "batch_done", "exit_check",
        ]
        bad_calls = [c for c in fake.calls if "bad.wav" in c]
        assert len(bad_calls) == 2, f"expected exactly 1 retry, got {len(bad_calls)} attempts"
        assert events[-2] == {"event": "batch_done", "ok": 2, "failed": 1}
        assert events[6]["event"] == "file_error" and "gpu exploded" in events[6]["error"]
        assert events[8]["event"] == "file_done" and "good2" in events[8]["file"]


def test_model_load_failure_fails_all_files():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        config = make_config(tmp, ["a.wav", "b.wav"])
        events = collect_events(config, FakeFactory(error=RuntimeError("ckpt corrupted")))
        kinds = [e["event"] for e in events]
        assert kinds == ["progress", "file_error", "file_error", "batch_done", "exit_check"]
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
    assert [str(f) for f in args.files] == ["/x/a.wav"]


def test_cli_explicit_options():
    args = sd.parse_args([
        "--model-file", "htdemucs_6s.yaml",
        "--files", "/x/a.wav", "/x/b.wav",
        "--output-dir", "/tmp/out",
        "--model-file-dir", "/tmp/models",
        "--output-format", "flac",
    ])
    assert args.model_file == "htdemucs_6s.yaml"
    assert args.model_file_dir == Path("/tmp/models")
    assert len(args.files) == 2


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
