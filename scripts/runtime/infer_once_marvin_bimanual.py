#!/usr/bin/env python3
"""Validate and execute one Marvin bimanual request without ROS2."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.inference_marvin_bimanual import main


if __name__ == "__main__":
    raise SystemExit(main())
