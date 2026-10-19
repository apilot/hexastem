#!/usr/bin/env python3
"""Compatibility launcher: the full driver lives in ../python/separator_driver.py.

This thin entry point exists so the driver can be invoked from the canonical
orchestrator path hexastem/driver/split_driver.py without duplicating logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from separator_driver import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
