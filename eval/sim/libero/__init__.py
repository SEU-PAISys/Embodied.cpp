# Copyright 2026 VinRobotics
# Source: VinRobotics/vla.cpp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The upstream checkout uses ``libero/libero`` without an outer
# ``__init__.py``.  Add the checkout root so ``libero.libero`` resolves after
# setup_libero.sh, without requiring every evaluator command to set PYTHONPATH.
LIBERO_CHECKOUT = Path(__file__).resolve().parent / "LIBERO"
if LIBERO_CHECKOUT.is_dir():
    sys.path.insert(0, str(LIBERO_CHECKOUT))

from sim.libero.libero_env import register_libero_envs
register_libero_envs()
