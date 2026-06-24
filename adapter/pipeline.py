# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Adapter pipeline helpers shared by Python evaluation clients."""

from __future__ import annotations

from typing import Any

from .typed_io import EmbodiedObservation


class AdapterPipeline:
    """Wrap a parser and expose an embodied observation boundary.

    Existing parsers may only implement `parse_observation`. Newer parsers can
    implement `parse_embodied_observation` and return richer typed metadata.
    """

    def __init__(self, parser: Any):
        self.parser = parser

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        if hasattr(self.parser, "parse_embodied_observation"):
            parsed = self.parser.parse_embodied_observation(obs)
            if not isinstance(parsed, EmbodiedObservation):
                raise TypeError(
                    "parse_embodied_observation must return EmbodiedObservation")
            return parsed
        model_inputs = self.parser.parse_observation(obs)
        return EmbodiedObservation(model_inputs=model_inputs, raw=obs)

    def parse_model_inputs(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.parse_embodied_observation(obs).model_inputs
