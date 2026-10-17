#!/usr/bin/env python3
"""Strictly validate and aggregate a SmolVLA Q8_0 LIBERO-Object run.

The validator consumes only machine-readable artifacts. It requires the
complete ten-task, twenty-episode Object protocol, recomputes all counts,
rates, Wilson intervals, and raw metric summaries, and never manufactures a
cross-model or normalized Table 3 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from client.reproducibility import derive_episode_noise_seed  # noqa: E402


DEFAULT_TASK_IDS = tuple(range(10))
DEFAULT_EPISODES = 20
EXPECTED_TOTAL_EPISODES = len(DEFAULT_TASK_IDS) * DEFAULT_EPISODES
SMOKE_TASK_IDS = (0,)
SMOKE_EPISODES = 3
SUITE_PREFLIGHT_EPISODES = 3
Z_95 = 1.959963984540054
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
NOISE_CHECKSUM_RE = re.compile(r"^[0-9a-f]{16,64}$")
TASK_DIR_RE = re.compile(r"^task_(\d+)$")


def wilson(successes: int, trials: int) -> tuple[float | None, float | None]:
    """Return a 95% Wilson interval in percentage points."""
    if trials <= 0:
        return None, None
    p = successes / trials
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denom
    radius = Z_95 * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    ) / denom
    return 100.0 * (centre - radius), 100.0 * (centre + radius)


def _fail(path: Path, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(path, "required JSON artifact is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(path, f"unreadable JSON: {exc}")
    if not isinstance(value, dict):
        _fail(path, "top-level value must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_kind(task_ids: tuple[int, ...], episodes: int) -> str:
    if task_ids == DEFAULT_TASK_IDS and episodes == DEFAULT_EPISODES:
        return "full_suite"
    if task_ids == SMOKE_TASK_IDS and episodes == SMOKE_EPISODES:
        return "smoke_preflight"
    if task_ids == DEFAULT_TASK_IDS and episodes == SUITE_PREFLIGHT_EPISODES:
        return "suite_preflight"
    raise ValueError(
        "LIBERO-Object validation supports only full task IDs 0..9 x20, "
        "suite preflight task IDs 0..9 x3, or explicit smoke task ID 0 x3"
    )


def _resolve_in_root(root: Path, relative: str, source: Path, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail(source, f"{label} escapes the output directory")
    if not candidate.is_file():
        _fail(candidate, f"{label} is missing")
    return candidate


def _required(mapping: dict[str, Any], key: str, path: Path) -> Any:
    if key not in mapping:
        _fail(path, f"required field {key!r} is missing")
    return mapping[key]


def _nonempty_string(value: Any, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, path: Path, field: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(path, f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, path: Path, field: str) -> int:
    if type(value) is not int or value < 0:
        _fail(path, f"{field} must be a non-negative integer")
    return value


def _finite(value: Any, path: Path, field: str) -> float:
    if isinstance(value, bool):
        _fail(path, f"{field} must be finite numeric data")
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(path, f"{field} must be finite numeric data")
    if not math.isfinite(number):
        _fail(path, f"{field} must be finite")
    return number


def _raw_samples(profile: dict[str, Any], key: str, label: str, path: Path) -> list[float]:
    section = _required(profile, key, path)
    if not isinstance(section, dict):
        _fail(path, f"{label} must be an object containing raw samples")
    samples = _required(section, "samples", path)
    if not isinstance(samples, list) or not samples:
        _fail(path, f"missing raw {label} samples")
    parsed = [_finite(value, path, f"raw {label} sample") for value in samples]
    if any(value < 0.0 for value in parsed):
        _fail(path, f"raw {label} samples must be non-negative")
    return parsed


def _percentile(ordered: list[float], fraction: float) -> float:
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _distribution(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "samples": samples,
        "mean": statistics.fmean(samples),
        "std": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "peak": max(samples),
    }


def _validate_optional_summary(
    section: dict[str, Any],
    recomputed: dict[str, Any],
    path: Path,
    *,
    digits: int,
) -> None:
    """Check summaries when present, while deriving output from raw samples."""
    if "n" in section and section["n"] != recomputed["n"]:
        _fail(path, f"metric n={section['n']} disagrees with raw samples")
    tolerance = 0.5 * 10.0 ** (-digits) + 1e-12
    for key in ("mean", "std", "p50", "p95", "p99"):
        if key in section:
            value = _finite(section[key], path, f"metric {key}")
            expected = round(recomputed[key], digits)
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
                _fail(path, f"metric {key} disagrees with raw samples")


def _validate_run_manifest(
    root: Path,
    task_ids: tuple[int, ...] = DEFAULT_TASK_IDS,
    episodes: int = DEFAULT_EPISODES,
) -> dict[str, Any]:
    path = root / "run_manifest.json"
    data = _load_json(path)
    if data.get("schema") != "smolvla_libero_object_run.v1":
        _fail(path, "schema must be smolvla_libero_object_run.v1")
    if data.get("suite") != "libero_object":
        _fail(path, "suite must be libero_object")
    if data.get("task_ids") != list(task_ids):
        _fail(path, f"task IDs must be exactly {list(task_ids)!r} with no duplicates or skips")
    if data.get("episodes_per_task") != episodes:
        _fail(path, f"episodes_per_task must be exactly {episodes}")
    for key, expected in (
        ("environment_seed", 1000),
        ("action_noise_seed", 1000),
        ("action_replay", 1),
        ("flow_steps", 10),
        ("generated_action_horizon", 50),
        ("model_image_size", 512),
        ("observation_width", 360),
        ("observation_height", 360),
        ("settling_action", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ):
        if data.get(key) != expected:
            _fail(path, f"{key} must be {expected!r}")
    if data.get("control_mode") != "relative":
        _fail(path, "control_mode must be relative")
    if data.get("video_enabled") is not False:
        _fail(path, "video_enabled must be false for the default Object protocol")

    artifacts = _required(data, "artifacts", path)
    if not isinstance(artifacts, dict):
        _fail(path, "artifacts must be an object")
    for key in ("profile", "quantizer", "server_startup", "config_snapshot"):
        _nonempty_string(_required(artifacts, key, path), path, f"artifacts.{key}")

    provenance = _required(data, "provenance", path)
    if not isinstance(provenance, dict):
        _fail(path, "provenance must be an object")
    for key in ("arxiv_reference", "arxiv_revision_date"):
        _nonempty_string(_required(provenance, key, path), path, f"provenance.{key}")

    repository = _required(provenance, "repository", path)
    if not isinstance(repository, dict):
        _fail(path, "provenance.repository must be an object")
    for key in ("remote", "commit", "branch", "dirty_state"):
        _nonempty_string(_required(repository, key, path), path, f"repository.{key}")

    commands = _required(provenance, "commands", path)
    if not isinstance(commands, dict):
        _fail(path, "provenance.commands must be an object")
    for key in ("server", "client"):
        _nonempty_string(_required(commands, key, path), path, f"commands.{key}")

    config = _required(provenance, "config", path)
    if not isinstance(config, dict):
        _fail(path, "provenance.config must be an object")
    config_path = _nonempty_string(_required(config, "path", path), path, "config.path")
    config_hash = _nonempty_string(_required(config, "sha256", path), path, "config.sha256")
    if not HEX64_RE.fullmatch(config_hash):
        _fail(path, "config.sha256 must be a lowercase SHA-256 hex digest")
    if not isinstance(_required(config, "snapshot", path), dict):
        _fail(path, "config.snapshot must be an object")
    if config_path != artifacts["config_snapshot"]:
        _fail(path, "config.path must reference artifacts.config_snapshot")
    snapshot_path = _resolve_in_root(root, config_path, path, "config snapshot")
    if _sha256(snapshot_path) != config_hash:
        _fail(path, "config.sha256 disagrees with the copied config snapshot")
    try:
        parsed_snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _fail(snapshot_path, f"config snapshot is unreadable: {exc}")
    if parsed_snapshot != config["snapshot"]:
        _fail(path, "config.snapshot disagrees with the copied config snapshot")
    for key, expected in (
        ("observation_width", 360),
        ("observation_height", 360),
        ("image_size", 512),
        ("settling_gripper_action", -1.0),
    ):
        if parsed_snapshot.get(key) != expected:
            _fail(snapshot_path, f"{key} must be {expected}")

    build = _required(provenance, "build", path)
    if not isinstance(build, dict):
        _fail(path, "provenance.build must be an object")
    for key in (
        "type", "cmake_flags", "cuda_architecture", "compiler", "cuda",
        "driver", "os", "gpu", "python",
    ):
        _nonempty_string(_required(build, key, path), path, f"build.{key}")

    hashes = _required(provenance, "hashes", path)
    if not isinstance(hashes, dict):
        _fail(path, "provenance.hashes must be an object")
    for key in ("source_policy", "quantized_policy", "mmproj", "tokenizer"):
        digest = _nonempty_string(_required(hashes, key, path), path, f"hashes.{key}")
        if not HEX64_RE.fullmatch(digest):
            _fail(path, f"hashes.{key} must be a lowercase SHA-256 hex digest")

    quantizer = _required(provenance, "quantizer", path)
    if not isinstance(quantizer, dict):
        _fail(path, "provenance.quantizer must be an object")
    if quantizer.get("qtype") != "Q8_0":
        _fail(path, "provenance.quantizer.qtype must be Q8_0")
    if quantizer.get("scope") != "main_model_full":
        _fail(path, "provenance.quantizer.scope must be main_model_full")
    _nonempty_string(_required(quantizer, "script", path), path, "quantizer.script")
    _positive_int(_required(quantizer, "selected_tensor_count", path), path, "selected_tensor_count")
    for key in ("selected_quantized_bytes", "source_tensor_bytes", "output_tensor_bytes"):
        _positive_int(_required(quantizer, key, path), path, key)
    for key in ("source_dtype_counts", "output_dtype_counts"):
        counts = _required(quantizer, key, path)
        if not isinstance(counts, dict) or not counts:
            _fail(path, f"quantizer.{key} must be a non-empty dtype count map")
        for dtype, count in counts.items():
            _nonempty_string(dtype, path, f"quantizer.{key} dtype")
            _nonnegative_int(count, path, f"quantizer.{key}.{dtype}")
    if quantizer["output_dtype_counts"].get("Q8_0", 0) != quantizer["selected_tensor_count"]:
        _fail(path, "quantizer output Q8_0 count disagrees with selected_tensor_count")
    return data


def _resolve_artifact(root: Path, manifest: dict[str, Any], key: str) -> Path:
    manifest_path = root / "run_manifest.json"
    relative = _nonempty_string(manifest["artifacts"][key], manifest_path, f"artifacts.{key}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail(manifest_path, f"artifacts.{key} escapes the output directory")
    if not candidate.is_file():
        _fail(candidate, f"artifacts.{key} is missing")
    return candidate


def _validate_quantizer_manifest(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if data.get("schema") != "smolvla_q8_0_quantization.v1":
        _fail(path, "schema must be smolvla_q8_0_quantization.v1")
    if data.get("qtype") != "Q8_0":
        _fail(path, "qtype must be Q8_0")
    if data.get("quantization_scope") != "main_model_full":
        _fail(path, "quantization_scope must be main_model_full")
    selected = _positive_int(_required(data, "quantized_tensor_count", path), path, "quantized_tensor_count")
    selected_bytes = _positive_int(
        _required(data, "selected_quantized_bytes", path), path, "selected_quantized_bytes"
    )
    for key in ("source_tensor_bytes", "output_tensor_bytes"):
        _positive_int(_required(data, key, path), path, key)
    for key in ("source_sha256", "output_sha256"):
        digest = _nonempty_string(_required(data, key, path), path, key)
        if not HEX64_RE.fullmatch(digest):
            _fail(path, f"{key} must be a lowercase SHA-256 hex digest")
    source_counts = _required(data, "source_dtype_counts", path)
    output_counts = _required(data, "output_dtype_counts", path)
    for key, counts in (("source_dtype_counts", source_counts), ("output_dtype_counts", output_counts)):
        if not isinstance(counts, dict) or not counts:
            _fail(path, f"{key} must be a non-empty map")
        for dtype, count in counts.items():
            _nonempty_string(dtype, path, f"{key} dtype")
            _nonnegative_int(count, path, f"{key}.{dtype}")
    if output_counts.get("Q8_0", 0) != selected:
        _fail(path, "output Q8_0 count disagrees with quantized_tensor_count")
    tensors = _required(data, "tensors", path)
    if not isinstance(tensors, list) or not tensors:
        _fail(path, "tensors must contain the deterministic tensor inventory")
    names: set[str] = set()
    selected_records = 0
    selected_output_bytes = 0
    source_bytes = 0
    output_bytes = 0
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            _fail(path, f"tensors[{index}] must be an object")
        name = _nonempty_string(_required(tensor, "name", path), path, f"tensors[{index}].name")
        if name in names:
            _fail(path, f"duplicate tensor name {name!r}")
        names.add(name)
        for key in ("category", "reason", "source_type", "output_type"):
            _nonempty_string(_required(tensor, key, path), path, f"tensors[{index}].{key}")
        if type(_required(tensor, "selected", path)) is not bool:
            _fail(path, f"tensors[{index}].selected must be boolean")
        if tensor["selected"]:
            selected_records += 1
            if tensor["output_type"] != "Q8_0":
                _fail(path, f"selected tensor {name} is not resident Q8_0")
        for key in ("source_bytes", "output_bytes"):
            value = _positive_int(_required(tensor, key, path), path, f"tensors[{index}].{key}")
            if key == "source_bytes":
                source_bytes += value
            else:
                output_bytes += value
                if tensor["selected"]:
                    selected_output_bytes += value
    if selected_records != selected:
        _fail(path, "tensor inventory selected count disagrees with quantized_tensor_count")
    if source_bytes != data["source_tensor_bytes"]:
        _fail(path, "tensor inventory source bytes disagree with source_tensor_bytes")
    if output_bytes != data["output_tensor_bytes"]:
        _fail(path, "tensor inventory output bytes disagree with output_tensor_bytes")
    if selected_output_bytes != selected_bytes:
        _fail(path, "selected tensor output bytes disagree with selected_quantized_bytes")
    return data


def _validate_startup(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if data.get("schema") != "smolvla_server_startup.v1":
        _fail(path, "schema must be smolvla_server_startup.v1")
    for key in ("backend", "model_path", "mmproj_path"):
        _nonempty_string(_required(data, key, path), path, key)
    if data.get("storage_qtype") != "Q8_0":
        _fail(path, "storage_qtype must be Q8_0")
    if data.get("resident_qtype") != "Q8_0":
        _fail(path, "resident_qtype must be Q8_0; BF16 full-model residency is not accepted")
    count = _positive_int(
        _required(data, "resident_quantized_tensor_count", path),
        path,
        "resident_quantized_tensor_count",
    )
    qbytes = _positive_int(_required(data, "resident_quantized_bytes", path), path, "resident_quantized_bytes")
    for key in ("resident_f32_bytes", "resident_bf16_bytes", "resident_weight_bytes"):
        _nonnegative_int(_required(data, key, path), path, key)
    if data["resident_weight_bytes"] != (
        qbytes + data["resident_f32_bytes"] + data["resident_bf16_bytes"]
    ):
        _fail(path, "resident_weight_bytes does not reconcile resident tensor bytes")
    if count <= 0 or qbytes <= 0:
        _fail(path, "native Q8_0 residency evidence is empty")
    if data.get("flow_steps") != 10:
        _fail(path, "flow_steps must be 10")
    if data.get("generated_action_horizon") != 50:
        _fail(path, "generated_action_horizon must be 50")
    return data


def _validate_task(
    path: Path, task_id: int, episodes: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = path / "result.json"
    data = _load_json(result_path)
    for key, expected in (
        ("arch", "smolvla"),
        ("implementation", "cpp"),
        ("suite", "libero_object"),
        ("task_id", task_id),
        ("episodes_requested", episodes),
        ("seed", 1000),
        ("noise_seed", 1000),
        ("derive_episode_noise", True),
        ("n_action_steps", 1),
        ("flow_steps", 10),
        ("image_size", 512),
        ("observation_width", 360),
        ("observation_height", 360),
        ("max_state_dim", 32),
        ("real_action_dim", 7),
        ("max_length", 48),
        ("image_keys", ["observation.images.image", "observation.images.image2"]),
        ("prompt_policy", "smolvla_trailing_newline"),
        ("generated_action_horizon", 50),
        ("num_steps_wait", 10),
        ("settling_action", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        ("control_mode", "relative"),
    ):
        if data.get(key) != expected:
            label = "task IDs" if key == "task_id" else key
            _fail(result_path, f"{label}={data.get(key)!r}, expected {expected!r}")
    records = _required(data, "episodes", result_path)
    if not isinstance(records, list) or len(records) != episodes:
        _fail(result_path, f"episode IDs must contain exactly 0..{episodes - 1} with no missing IDs")
    if any(not isinstance(record, dict) for record in records):
        _fail(result_path, "episode IDs must be objects with exact integer IDs")
    ids = [record["episode"] for record in records]
    if any(type(value) is not int for value in ids):
        _fail(result_path, "episode IDs must be exact integers")
    if len(set(ids)) != len(ids) or sorted(ids) != list(range(episodes)):
        _fail(result_path, f"episode IDs {ids!r} do not cover exactly 0..{episodes - 1}")

    for record in records:
        episode = record["episode"]
        if record.get("task_id") != task_id:
            _fail(result_path, f"episode {episode}: task IDs are inconsistent")
        expected_seed = derive_episode_noise_seed(1000, "libero_object", task_id, episode)
        if record.get("noise_seed") != expected_seed:
            _fail(result_path, f"episode {episode}: derived noise_seed is incorrect")
        checksum = record.get("noise_checksum")
        if not isinstance(checksum, str) or not NOISE_CHECKSUM_RE.fullmatch(checksum):
            _fail(result_path, f"episode {episode}: noise_checksum is missing or invalid")
        if record.get("noise_mode") != "derived":
            _fail(result_path, f"episode {episode}: noise_mode must be derived")
        if record.get("noise_device") != "cpu":
            _fail(result_path, f"episode {episode}: noise_device must be cpu")
        if record.get("noise_dtype") != "float32":
            _fail(result_path, f"episode {episode}: noise_dtype must be float32")
        if type(record.get("success")) is not bool or type(record.get("skipped")) is not bool:
            _fail(result_path, f"episode {episode}: success/skipped must be bool")
        if record["skipped"]:
            _fail(result_path, f"episode {episode}: skip/aborted episodes are not accepted")
        _positive_int(record.get("environment_steps"), result_path, f"episode {episode}.environment_steps")
        _finite(record.get("average_step_ms"), result_path, f"episode {episode}.average_step_ms")

    successes = sum(record["success"] for record in records)
    skipped = sum(record["skipped"] for record in records)
    for key, expected in (
        ("episodes_counted", episodes),
        ("successes", successes),
        ("skipped", skipped),
    ):
        if data.get(key) != expected:
            _fail(result_path, f"{key}={data.get(key)!r}, recomputed {expected!r}")
    expected_rate = successes / episodes
    recorded_rate = _finite(_required(data, "success_rate", result_path), result_path, "success_rate")
    if not math.isclose(recorded_rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9):
        _fail(result_path, "success_rate does not match recomputed episode counts")
    return (
        {
            "task_id": task_id,
            "episodes_requested": episodes,
            "episodes_counted": episodes,
            "successes": successes,
            "skipped": skipped,
            "success_rate_pct": 100.0 * expected_rate,
            "wilson_95_pct": list(wilson(successes, episodes)),
            "result": str(result_path).replace("\\", "/"),
        },
        records,
    )


def _validate_profile(path: Path, expected_keys: set[tuple[int, int]]) -> dict[str, Any]:
    data = _load_json(path)
    if data.get("schema_version") != 2:
        _fail(path, "schema_version must be 2")
    if data.get("complete") is not True:
        _fail(path, "complete must be true")
    if data.get("table_ready") is not True:
        _fail(path, "table_ready must be true")
    if data.get("arch") != "smolvla" or data.get("suite") != "libero_object":
        _fail(path, "profile arch/suite provenance is incorrect")
    if data.get("implementation") != "cpp":
        _fail(path, "profile implementation must be cpp")
    if data.get("n_a") != 1:
        _fail(path, "profile n_a must be 1")

    episodes = _required(data, "episodes", path)
    if not isinstance(episodes, dict):
        _fail(path, "episodes must be an object")
    records = _required(episodes, "records", path)
    expected_total = len(expected_keys)
    if not isinstance(records, list) or len(records) != expected_total:
        _fail(path, f"profile episode IDs must contain exactly {expected_total} records")
    observed: set[tuple[int, int]] = set()
    successes = 0
    for record in records:
        if not isinstance(record, dict):
            _fail(path, "profile episode records must be objects")
        key = (record.get("task_id"), record.get("episode"))
        if any(type(value) is not int for value in key):
            _fail(path, f"profile episode IDs must be integer pairs, got {key!r}")
        if key in observed:
            _fail(path, f"duplicate profile episode IDs: {key!r}")
        observed.add(key)
        if type(record.get("success")) is not bool or type(record.get("skipped")) is not bool:
            _fail(path, f"profile record {key!r}: success/skipped must be bool")
        if record["skipped"]:
            _fail(path, f"profile record {key!r}: skip/aborted episode is not accepted")
        _positive_int(record.get("environment_steps"), path, f"profile record {key!r}.environment_steps")
        successes += int(record["success"])
    if observed != expected_keys:
        _fail(path, "profile episode IDs have missing, duplicate, or skipped task/episode IDs")
    for key, expected in (
        ("expected", expected_total),
        ("recorded", expected_total),
        ("evaluable", expected_total),
        ("skipped", 0),
        ("successes", successes),
    ):
        if episodes.get(key) != expected:
            _fail(path, f"profile episodes.{key}={episodes.get(key)!r}, recomputed {expected!r}")
    expected_rate = 100.0 * successes / expected_total
    recorded_rate = _finite(
        _required(episodes, "success_rate_percent", path),
        path,
        "profile success_rate_percent",
    )
    # LiberoSuiteProfiler serializes percentages to three decimal places.
    # Compare against that public representation, not the unrounded fraction
    # (for example, one success in three is stored as 33.333).
    if not math.isclose(
        recorded_rate, round(expected_rate, 3), rel_tol=0.0, abs_tol=0.0005 + 1e-12
    ):
        _fail(path, "profile success_rate_percent does not match raw episode records")
    interval = episodes.get("wilson_95_percent")
    expected_interval = wilson(successes, expected_total)
    if not isinstance(interval, list) or len(interval) != 2:
        _fail(path, "profile Wilson interval does not match raw episode records")
    for index, value in enumerate(interval):
        recorded_bound = _finite(value, path, "profile Wilson interval")
        if not math.isclose(recorded_bound, expected_interval[index], rel_tol=1e-6, abs_tol=0.1):
            _fail(path, "profile Wilson interval does not match raw episode records")

    chunk_sizes = _required(data, "model_chunk_sizes", path)
    if chunk_sizes != [50]:
        _fail(path, "model_chunk_sizes must be exactly [50]")

    metrics: dict[str, Any] = {}
    raw_metric_samples: dict[str, list[float]] = {}
    model_step = _required(data, "model_step_ms", path)
    if not isinstance(model_step, dict):
        _fail(path, "model_step_ms must be an object")
    _nonempty_string(
        _required(model_step, "definition", path), path, "model_step_ms.definition"
    )
    for key, label in (
        ("inf_ms", "server latency"),
        ("generated_action_step_ms", "generated action-step latency"),
        ("step_ms", "step latency"),
    ):
        samples = _raw_samples(data, key, label, path)
        definition = _nonempty_string(
            _required(data[key], "definition", path), path, f"{key}.definition"
        )
        del definition
        if key in {"inf_ms", "generated_action_step_ms"} and len(samples) < 100:
            _fail(path, f"{label} requires at least 100 post-warmup samples")
        if key in {"inf_ms", "step_ms"}:
            warmup = _nonnegative_int(
                _required(data[key], "warmup_requests_excluded", path),
                path,
                f"{key}.warmup_requests_excluded",
            )
            if warmup < 5:
                _fail(path, f"{key} must exclude at least 5 warmup requests")
        recomputed = _distribution(samples)
        _validate_optional_summary(data[key], recomputed, path, digits=3)
        metrics[key] = recomputed
        raw_metric_samples[key] = samples

    inf_samples = raw_metric_samples["inf_ms"]
    generated_samples = raw_metric_samples["generated_action_step_ms"]
    if len(inf_samples) != len(generated_samples):
        _fail(path, "inf_ms and generated_action_step_ms sample counts must match")
    chunk_size = chunk_sizes[0]
    for index, (inf_ms, generated_ms) in enumerate(zip(inf_samples, generated_samples)):
        expected_generated_ms = inf_ms / chunk_size
        if not math.isclose(
            generated_ms,
            expected_generated_ms,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            _fail(
                path,
                f"generated_action_step_ms.samples[{index}] must equal "
                f"inf_ms.samples[{index}] / {chunk_size}",
            )

    vram_samples = _raw_samples(data, "vram_mib", "VRAM", path)
    vram = _distribution(vram_samples)
    _validate_optional_summary(data["vram_mib"], vram, path, digits=1)
    source = _nonempty_string(_required(data["vram_mib"], "source", path), path, "vram_mib.source")
    if source not in {"process_used_memory", "device_total_fallback"}:
        _fail(path, "vram_mib.source must identify the raw VRAM measurement")
    _positive_int(_required(data["vram_mib"], "server_pid", path), path, "vram_mib.server_pid")
    gpu_uuids = _required(data["vram_mib"], "gpu_uuids", path)
    if not isinstance(gpu_uuids, list) or not gpu_uuids:
        _fail(path, "vram_mib.gpu_uuids must contain at least one GPU UUID")
    for uuid in gpu_uuids:
        _nonempty_string(uuid, path, "vram_mib.gpu_uuids entry")
    interval_s = _finite(
        _required(data["vram_mib"], "sample_interval_s", path),
        path,
        "vram_mib.sample_interval_s",
    )
    if interval_s <= 0:
        _fail(path, "vram_mib.sample_interval_s must be positive")
    _nonempty_string(
        _required(data["vram_mib"], "definition", path), path, "vram_mib.definition"
    )
    metrics["vram_mib"] = vram
    metrics["vram_source"] = source
    metrics["model_chunk_sizes"] = list(chunk_sizes)
    return metrics


def aggregate(
    outputs_dir: str | Path,
    *,
    task_ids: Iterable[int] = DEFAULT_TASK_IDS,
    episodes: int = DEFAULT_EPISODES,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate the exact Object protocol and return recomputed raw metrics."""
    del require_complete  # strict validation never permits a partial run
    root = Path(outputs_dir)
    task_ids = tuple(task_ids)
    evidence_kind = _evidence_kind(task_ids, episodes)
    manifest = _validate_run_manifest(root, task_ids, episodes)
    quantizer_path = _resolve_artifact(root, manifest, "quantizer")
    startup_path = _resolve_artifact(root, manifest, "server_startup")
    profile_path = _resolve_artifact(root, manifest, "profile")
    quantizer = _validate_quantizer_manifest(quantizer_path)
    startup = _validate_startup(startup_path)
    manifest_quantizer = manifest["provenance"]["quantizer"]
    for manifest_key, artifact_key in (
        ("qtype", "qtype"),
        ("scope", "quantization_scope"),
        ("selected_tensor_count", "quantized_tensor_count"),
        ("selected_quantized_bytes", "selected_quantized_bytes"),
        ("source_tensor_bytes", "source_tensor_bytes"),
        ("output_tensor_bytes", "output_tensor_bytes"),
        ("source_dtype_counts", "source_dtype_counts"),
        ("output_dtype_counts", "output_dtype_counts"),
    ):
        if manifest_quantizer.get(manifest_key) != quantizer.get(artifact_key):
            _fail(
                root / "run_manifest.json",
                f"quantizer provenance {manifest_key} disagrees with manifest artifact",
            )
    if startup["resident_quantized_tensor_count"] != quantizer["quantized_tensor_count"]:
        _fail(root / "server_startup.json", "resident Q8_0 tensor count disagrees with quantizer manifest")
    if startup["resident_quantized_bytes"] != quantizer["selected_quantized_bytes"]:
        _fail(root / "server_startup.json", "resident Q8_0 bytes disagree with selected quantizer bytes")
    hashes = manifest["provenance"]["hashes"]
    if quantizer["source_sha256"] != hashes["source_policy"]:
        _fail(quantizer_path, "source_sha256 disagrees with run_manifest provenance")
    if quantizer["output_sha256"] != hashes["quantized_policy"]:
        _fail(quantizer_path, "output_sha256 disagrees with run_manifest provenance")

    tasks_root = root / "smolvla" / "libero_object"
    if not tasks_root.is_dir():
        _fail(tasks_root, "task IDs directory is missing")
    observed_dirs = {
        int(match.group(1))
        for child in tasks_root.iterdir()
        if child.is_dir() and (match := TASK_DIR_RE.fullmatch(child.name))
    }
    if observed_dirs != set(task_ids):
        _fail(tasks_root, f"task IDs must be exactly {list(task_ids)!r} with no duplicates or skips")
    task_summaries: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        summary, records = _validate_task(tasks_root / f"task_{task_id}", task_id, episodes)
        task_summaries.append(summary)
        episode_records.extend(records)

    expected_keys = {
        (task_id, episode)
        for task_id in task_ids
        for episode in range(episodes)
    }
    metrics = _validate_profile(profile_path, expected_keys)
    profile_data = _load_json(profile_path)
    profile_keys = {
        (record.get("task_id"), record.get("episode"))
        for record in profile_data["episodes"]["records"]
    }
    result_keys = {(record["task_id"], record["episode"]) for record in episode_records}
    if profile_keys != result_keys:
        _fail(profile_path, "profile and result episode IDs do not match")
    result_by_key = {
        (record["task_id"], record["episode"]): record
        for record in episode_records
    }
    profile_by_key = {
        (record["task_id"], record["episode"]): record
        for record in profile_data["episodes"]["records"]
    }
    for key in sorted(result_keys):
        result_record = result_by_key[key]
        profile_record = profile_by_key[key]
        for field in ("success", "skipped"):
            if result_record[field] != profile_record[field]:
                _fail(profile_path, f"profile {field} mismatch for episode {key!r}")
        for field in ("environment_steps", "steps", "step_count"):
            if (
                field in result_record
                and field in profile_record
                and result_record[field] != profile_record[field]
            ):
                _fail(profile_path, f"profile {field} mismatch for episode {key!r}")

    total_successes = sum(task["successes"] for task in task_summaries)
    total_counted = sum(task["episodes_counted"] for task in task_summaries)
    total_skipped = sum(task["skipped"] for task in task_summaries)
    low, high = wilson(total_successes, total_counted)
    return {
        "schema": "smolvla_libero_object_eval.v2",
        "evidence_kind": evidence_kind,
        "promotion_eligible": False,
        "arch": "smolvla",
        "suite": "libero_object",
        "protocol": {
            "task_ids": list(task_ids),
            "episodes_per_task": episodes,
            "environment_seed": 1000,
            "action_noise_seed": 1000,
            "control_mode": "relative",
            "action_replay": 1,
            "flow_steps": 10,
            "generated_action_horizon": 50,
            "observation_width": 360,
            "observation_height": 360,
            "model_image_size": 512,
            "video_enabled": False,
        },
        "tasks": task_summaries,
        "totals": {
            "episodes_requested": len(task_ids) * episodes,
            "episodes_counted": total_counted,
            "successes": total_successes,
            "skipped": total_skipped,
            "success_rate_pct": 100.0 * total_successes / total_counted,
            "wilson_95_pct": [low, high],
        },
        "raw_metrics": metrics,
        "quantizer": {
            "qtype": quantizer["qtype"],
            "quantization_scope": quantizer["quantization_scope"],
            "quantized_tensor_count": quantizer["quantized_tensor_count"],
            "selected_quantized_bytes": quantizer["selected_quantized_bytes"],
            "source_tensor_bytes": quantizer["source_tensor_bytes"],
            "output_tensor_bytes": quantizer["output_tensor_bytes"],
        },
        "startup": {
            "storage_qtype": startup["storage_qtype"],
            "resident_qtype": startup["resident_qtype"],
            "resident_quantized_tensor_count": startup["resident_quantized_tensor_count"],
            "resident_quantized_bytes": startup["resident_quantized_bytes"],
            "resident_weight_bytes": startup["resident_weight_bytes"],
        },
        "provenance": manifest["provenance"],
    }


def to_markdown(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    totals = data["totals"]
    evidence_note = (
        "This is full 10x20 suite evidence."
        if data["evidence_kind"] == "full_suite"
        else "This is smoke/integration evidence."
    )
    lines = [
        "# SmolVLA LIBERO-Object Q8_0 native-resident run",
        "",
        evidence_note,
        "",
        "Table 3 promotion requires a matched baseline.",
        "This integration report is not promotion-eligible.",
        "",
        "This report is generated from strict per-task results, raw profile",
        "samples, quantizer manifest metadata, and server startup evidence.",
        "",
    ]
    if metadata:
        lines.extend(["## Provenance supplement", ""])
        for key, value in metadata.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend([
        "## Result",
        "",
        "| task | success / counted | skipped | success rate | Wilson 95% |",
        "|---:|---:|---:|---:|---:|",
    ])
    for task in data["tasks"]:
        interval = task["wilson_95_pct"]
        lines.append(
            f"| {task['task_id']} | {task['successes']} / {task['episodes_counted']} | "
            f"{task['skipped']} | {task['success_rate_pct']:.2f}% | "
            f"{interval[0]:.2f}%–{interval[1]:.2f}% |"
        )
    interval = totals["wilson_95_pct"]
    lines.extend([
        f"| **total** | **{totals['successes']} / {totals['episodes_counted']}** | "
        f"**{totals['skipped']}** | **{totals['success_rate_pct']:.2f}%** | "
        f"**{interval[0]:.2f}%–{interval[1]:.2f}%** |",
        "",
        "## Raw metrics",
        "",
        "Raw latency samples are retained for server inference, generated action-step, and environment step timing.",
        f"- Server latency samples: `{data['raw_metrics']['inf_ms']['n']}`",
        f"- Generated action-step samples: `{data['raw_metrics']['generated_action_step_ms']['n']}`",
        f"- Raw VRAM samples: `{data['raw_metrics']['vram_mib']['n']}` "
        f"(`{data['raw_metrics']['vram_source']}`)",
        "- Native resident storage: `Q8_0`",
        "- Normalized Table 3 ratios: unavailable without matching baseline evidence.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--metadata-json", type=Path)
    args = parser.parse_args()
    metadata = None
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            parser.error("--metadata-json must contain an object")
    data = aggregate(args.outputs, task_ids=args.task_ids, episodes=args.episodes)
    if metadata:
        data["metadata"] = metadata
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "result.json"
    markdown_path = args.out_dir / "result.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(to_markdown(data, metadata), encoding="utf-8")
    totals = data["totals"]
    print(
        f"validated {len(args.task_ids)} task(s) x {args.episodes} episodes "
        f"as {data['evidence_kind']}"
    )
    print(f"result={totals['successes']}/{totals['episodes_counted']} skipped={totals['skipped']}")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
