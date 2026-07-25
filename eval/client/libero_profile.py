# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Aggregate full-suite LIBERO fidelity and single-request profile metrics."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
import threading
from pathlib import Path
from typing import Any


def find_server_pid(address: str) -> int | None:
    match = re.search(r"tcp://[^:]+:(\d+)$", address)
    if match is None:
        return None
    port = match.group(1)
    try:
        output = subprocess.check_output(
            ["ss", "-tlnp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        if f":{port} " not in line:
            continue
        pid_match = re.search(r"pid=(\d+)", line)
        if pid_match is not None:
            return int(pid_match.group(1))
    return None


def _parse_memory_mib(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def sample_vram(pid: int) -> tuple[int | None, str | None, str | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None, None
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[1].isdigit() or int(parts[1]) != pid:
            continue
        gpu_uuid = parts[0]
        process_memory = _parse_memory_mib(parts[2])
        if process_memory is not None:
            return process_memory, "process_used_memory", gpu_uuid

        # WSL exposes the compute-process PID and GPU UUID but commonly reports
        # per-process used_memory as N/A. On a dedicated evaluation GPU, total
        # memory.used is the best available compatible measurement.
        try:
            device_output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_uuid}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None, None, gpu_uuid
        device_memory = _parse_memory_mib(device_output.splitlines()[0])
        return device_memory, "device_total_fallback", gpu_uuid
    return None, None, None


def sample_vram_mib(pid: int) -> int | None:
    value, _source, _gpu_uuid = sample_vram(pid)
    return value


class VramSampler(threading.Thread):
    def __init__(self, pid: int, interval_s: float) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.interval_s = interval_s
        self.samples_mib: list[int] = []
        self.sources: list[str] = []
        self.gpu_uuids: list[str] = []
        self._stop_event = threading.Event()

    def sample_once(self) -> None:
        value, source, gpu_uuid = sample_vram(self.pid)
        if value is not None:
            self.samples_mib.append(value)
        if source is not None:
            self.sources.append(source)
        if gpu_uuid is not None:
            self.gpu_uuids.append(gpu_uuid)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    probability = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z_squared / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def _table_number(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


class LiberoSuiteProfiler:
    def __init__(
        self,
        *,
        output_path: Path,
        model_label: str,
        backbone_label: str,
        arch: str,
        suite: str,
        replay_chunk_size: int,
        expected_episodes: int,
        server_address: str,
        server_pid: int | None,
        vram_interval_s: float,
        warmup_requests: int,
        step_definition: str = "client get_action wall-clock per environment step, amortized over replayed actions",
        inference_definition: str = "server-side model forward (PredictResponse.latency_ms_total), unique RPC requests",
        vram_target_label: str = "VLA server",
    ) -> None:
        self.output_path = output_path if output_path.suffix else output_path.with_suffix(".json")
        self.model_label = model_label
        self.backbone_label = backbone_label
        self.arch = arch
        self.suite = suite
        self.replay_chunk_size = replay_chunk_size
        self.expected_episodes = expected_episodes
        self.server_address = server_address
        self.server_pid = server_pid or find_server_pid(server_address)
        self.vram_interval_s = vram_interval_s
        self.warmup_requests = warmup_requests
        self.step_definition = step_definition
        self.inference_definition = inference_definition
        self.vram_target_label = vram_target_label

        self.step_wall_ms: list[float] = []
        self.server_requests: list[dict[str, float | int | None]] = []
        self.episodes: list[dict[str, Any]] = []
        self._last_inference_sequence: int | None = None
        self._sampler: VramSampler | None = None

    def start(self) -> None:
        if self.server_pid is None:
            print(
                f"warning: could not determine the {self.vram_target_label} PID; "
                "VRAM (MiB) will be unavailable. Pass --profile-server-pid.",
                flush=True,
            )
            return
        self._sampler = VramSampler(self.server_pid, self.vram_interval_s)
        self._sampler.sample_once()
        self._sampler.start()
        source = self._sampler.sources[-1] if self._sampler.sources else "unavailable"
        print(
            f"profiling {self.vram_target_label} pid={self.server_pid}, VRAM source={source}",
            flush=True,
        )

    def stop(self) -> None:
        if self._sampler is not None:
            self._sampler.stop()

    def record_step(self, wall_ms: float) -> None:
        self.step_wall_ms.append(float(wall_ms))

    def record_inference(
        self,
        total_ms: float,
        *,
        model_chunk_size: int,
        vision_ms: float | None = None,
        action_inference_ms: float | None = None,
        prefill_ms: float | None = None,
        denoise_ms: float | None = None,
    ) -> None:
        self.server_requests.append(
            {
                "sequence": len(self.server_requests),
                "server_total_ms": float(total_ms),
                "server_vision_ms": vision_ms,
                "server_inference_ms": action_inference_ms,
                "server_prefill_ms": prefill_ms,
                "server_denoise_ms": denoise_ms,
                "model_chunk_size": int(model_chunk_size),
            }
        )

    def capture_inference(self, client: Any) -> None:
        getter = getattr(client, "get_last_inference_profile", None)
        if getter is None:
            return
        profile = getter()
        if profile is None:
            return
        sequence = int(profile["sequence"])
        if sequence == self._last_inference_sequence:
            return
        self._last_inference_sequence = sequence
        self.server_requests.append(profile)

    def record_episode(
        self,
        *,
        task: str,
        task_id: int,
        episode: int,
        success: bool,
        skipped: bool,
        environment_steps: int | None,
    ) -> None:
        self.episodes.append(
            {
                "task": task,
                "task_id": task_id,
                "episode": episode,
                "success": bool(success),
                "skipped": bool(skipped),
                "environment_steps": (
                    int(environment_steps) if environment_steps is not None else None
                ),
            }
        )
        self.write(complete=False)

    def _vram_samples(self) -> list[int]:
        return list(self._sampler.samples_mib) if self._sampler is not None else []

    def result(self, *, complete: bool) -> dict[str, Any]:
        evaluable = [episode for episode in self.episodes if not episode["skipped"]]
        successes = sum(bool(episode["success"]) for episode in evaluable)
        trials = len(evaluable)
        interval_low, interval_high = wilson_interval(successes, trials)

        request_profiles = self.server_requests
        warmup_used = min(self.warmup_requests, max(0, len(request_profiles) - 1))
        measured_requests = request_profiles[warmup_used:]
        def request_values(key: str) -> list[float]:
            return [float(item[key]) for item in measured_requests if item.get(key) is not None]

        server_total_ms = request_values("server_total_ms")
        server_vision_ms = request_values("server_vision_ms")
        server_inference_ms = request_values("server_inference_ms")
        server_prefill_ms = request_values("server_prefill_ms")
        server_denoise_ms = request_values("server_denoise_ms")
        model_chunk_sizes = sorted(
            {
                int(item["model_chunk_size"])
                for item in request_profiles
                if item.get("model_chunk_size") is not None
            }
        )

        vram_samples = self._vram_samples()
        vram_sources = sorted(set(getattr(self._sampler, "sources", []))) \
            if self._sampler is not None else []
        gpu_uuids = sorted(set(getattr(self._sampler, "gpu_uuids", []))) \
            if self._sampler is not None else []
        vram_source = vram_sources[0] if len(vram_sources) == 1 else (
            "mixed" if vram_sources else None
        )
        vram_definition = (
            "target process used_memory reported by nvidia-smi"
            if vram_source == "process_used_memory"
            else "total memory.used for the target process GPU; WSL per-process memory was N/A"
            if vram_source == "device_total_fallback"
            else "VRAM measurement unavailable"
        )
        skipped = sum(bool(episode["skipped"]) for episode in self.episodes)
        recorded = len(self.episodes)
        table_ready = (
            complete
            and recorded == self.expected_episodes
            and skipped == 0
            and trials == self.expected_episodes
            and bool(self.step_wall_ms)
            and bool(server_total_ms)
            and bool(vram_samples)
        )

        sr_percent = 100.0 * successes / trials if trials else None
        return {
            "schema_version": 2,
            "complete": complete,
            "table_ready": table_ready,
            "model": self.model_label,
            "backbone": self.backbone_label,
            "arch": self.arch,
            "suite": self.suite,
            "episodes": {
                "expected": self.expected_episodes,
                "recorded": recorded,
                "evaluable": trials,
                "skipped": skipped,
                "successes": successes,
                "success_rate_percent": _rounded(sr_percent),
                "wilson_95_percent": [
                    round(100.0 * interval_low, 3),
                    round(100.0 * interval_high, 3),
                ],
                "records": self.episodes,
            },
            "n_a": self.replay_chunk_size,
            "model_chunk_sizes": model_chunk_sizes,
            "step_ms": {
                "definition": self.step_definition,
                "n": len(self.step_wall_ms),
                "mean": _rounded(_mean(self.step_wall_ms)),
                "total": _rounded(sum(self.step_wall_ms)),
            },
            "inf_ms": {
                "definition": self.inference_definition,
                "n": len(server_total_ms),
                "warmup_requests_excluded": warmup_used,
                "mean": _rounded(_mean(server_total_ms)),
                "vision_mean": _rounded(_mean(server_vision_ms)),
                "action_inference_mean": _rounded(_mean(server_inference_ms)),
                "prefill_mean": _rounded(_mean(server_prefill_ms)),
                "denoise_mean": _rounded(_mean(server_denoise_ms)),
            },
            "vram_mib": {
                "server_pid": self.server_pid,
                "target": self.vram_target_label,
                "gpu_uuids": gpu_uuids,
                "source": vram_source,
                "definition": vram_definition,
                "sample_interval_s": self.vram_interval_s,
                "n": len(vram_samples),
                "peak": max(vram_samples) if vram_samples else None,
                "mean": _rounded(_mean([float(value) for value in vram_samples]), 1),
            },
        }

    def write(self, *, complete: bool) -> dict[str, Any]:
        result = self.result(complete=complete)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if complete:
            self._write_csv(result)
            self._write_markdown(result)
        return result

    def _table_row(self, result: dict[str, Any]) -> dict[str, str]:
        episodes = result["episodes"]
        interval = episodes["wilson_95_percent"]
        success_rate = episodes["success_rate_percent"]
        sr_text = "-" if success_rate is None else (
            f"{success_rate:.1f} [{interval[0]:.1f}, {interval[1]:.1f}]"
        )
        return {
            "Model": result["model"],
            "Backbone": result["backbone"],
            "n_a": str(result["n_a"]),
            "SR (%)": sr_text,
            "step (ms)": _table_number(result["step_ms"]["mean"]),
            "inf (ms)": _table_number(result["inf_ms"]["mean"]),
            "VRAM (MiB)": _table_number(result["vram_mib"]["peak"], digits=0),
        }

    def _write_csv(self, result: dict[str, Any]) -> None:
        row = self._table_row(result)
        path = self.output_path.with_suffix(".csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def _write_markdown(self, result: dict[str, Any]) -> None:
        row = self._table_row(result)
        headers = list(row)
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            "| " + " | ".join(row[header] for header in headers) + " |",
            "",
            f"Complete: `{result['complete']}`; table ready: `{result['table_ready']}`.",
            f"`step`: {result['step_ms']['definition']}; `inf`: "
            f"{result['inf_ms']['definition']} after warmup. "
            "SR brackets are 95% Wilson intervals.",
            f"VRAM source: `{result['vram_mib']['source']}`. "
            f"{result['vram_mib']['definition']}.",
        ]
        self.output_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
