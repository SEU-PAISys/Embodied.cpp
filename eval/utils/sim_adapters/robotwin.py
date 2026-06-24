# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility imports for RobotWin simulator adapters.

The implementation lives in `adapter.sim.robotwin` so simulator I/O conversion
is no longer owned by the eval client tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.sim.robotwin import RobotWinHyVLAAdapter, RobotWinObservationAdapter

__all__ = [
    "RobotWinHyVLAAdapter",
    "RobotWinObservationAdapter",
]
