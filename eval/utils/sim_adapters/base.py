# Copyright 2026 VinRobotics
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

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.pipeline import AdapterPipeline
from adapter.typed_io import EmbodiedObservation

class BasePipelineAdapter:
    def __init__(self, client: Any, parser: Any, arch: str):
        self._client = client
        self._parser = parser
        self._adapter_pipeline = AdapterPipeline(parser)
        self.arch = arch

    def reset(self):
        return self._client.reset()

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        return self._adapter_pipeline.parse_embodied_observation(obs)

    def get_action(self, obs: dict[str, Any]) -> Any:
        embodied_obs = self.parse_embodied_observation(obs)
        action = self._client.get_action(embodied_obs.model_inputs)
        parsed_action = self._parser.parse_action(action)
        return parsed_action
