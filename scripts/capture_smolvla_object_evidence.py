#!/usr/bin/env python3
"""Capture immutable pre-rollout evidence for SmolVLA Q8_0 Object runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable

import yaml


STARTUP_MARKER = "vla(smolvla): startup_evidence "
TOKEN_RE = re.compile(r"([a-z0-9_]+)=([^\s]+)")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
FULL_TASK_IDS = tuple(range(10))
OBJECT_IMAGE_PROTOCOL = {
    "observation_width": 360,
    "observation_height": 360,
    "image_size": 512,
    "settling_gripper_action": -1.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"tokenizer directory is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"tokenizer directory is empty: {root}")
    digest = hashlib.sha256()
    resolved_root = root.resolve()
    for path in files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"tokenizer entry escapes tokenizer root: {path}") from exc
        if path.is_symlink():
            raise ValueError(f"tokenizer tree may not contain symlinks: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"repository identity discovery failed for git {' '.join(args)}: {detail}")
    return completed.stdout.strip()


def _repository_identity(repo_root: Path) -> dict[str, str]:
    remote = _nonempty(_run_git(repo_root, "remote", "get-url", "origin"), "repository.remote")
    commit = _nonempty(_run_git(repo_root, "rev-parse", "HEAD"), "repository.commit")
    branch = _nonempty(_run_git(repo_root, "branch", "--show-current"), "repository.branch")
    status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        lines = status.splitlines()
        status_hash = hashlib.sha256(status.encode("utf-8")).hexdigest()
        dirty_state = f"dirty:{len(lines)}:{status_hash}"
    else:
        dirty_state = "clean"
    return {
        "remote": remote,
        "commit": commit,
        "branch": branch,
        "dirty_state": dirty_state,
    }


def _validated_repository_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("repository identity must be an object")
    required = ("remote", "commit", "branch", "dirty_state")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(
            f"repository identity is missing required field repository.{missing[0]}"
        )
    extra = sorted(set(value) - set(required))
    if extra:
        raise ValueError(f"repository identity has unexpected field repository.{extra[0]}")
    result = {key: _nonempty(value[key], f"repository.{key}") for key in required}
    if not GIT_COMMIT_RE.fullmatch(result["commit"]):
        raise ValueError("repository.commit must be a lowercase 40- or 64-digit Git object ID")
    return result


def _resolved_logged_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def parse_startup_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"server startup log is missing: {path}")
    lines = [
        line.split(STARTUP_MARKER, 1)[1].strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if STARTUP_MARKER in line
    ]
    if len(lines) != 1:
        raise ValueError(f"startup evidence requires exactly one parsed startup line; found {len(lines)}")
    fields = dict(TOKEN_RE.findall(lines[0]))
    required_text = (
        "backend", "model_path", "mmproj_path", "storage_qtype", "resident_qtype",
    )
    required_int = (
        "resident_quantized_tensors", "resident_quantized_bytes", "resident_f32_bytes",
        "resident_bf16_bytes", "resident_weight_bytes", "resident_buffer_bytes",
        "flow_steps", "generated_action_horizon",
    )
    missing = [key for key in (*required_text, *required_int) if key not in fields]
    if missing:
        raise ValueError(f"startup evidence is missing required fields: {', '.join(missing)}")
    result: dict[str, Any] = {
        "schema": "smolvla_server_startup.v1",
        **{key: _nonempty(fields[key], f"startup.{key}") for key in required_text},
    }
    for key in required_int:
        try:
            result_key = (
                "resident_quantized_tensor_count"
                if key == "resident_quantized_tensors" else key
            )
            result[result_key] = int(fields[key])
        except ValueError as exc:
            raise ValueError(f"startup field {key} must be an integer") from exc
    if result["storage_qtype"] != "Q8_0" or result["resident_qtype"] != "Q8_0":
        raise ValueError("startup evidence must report Q8_0 storage and resident qtypes")
    if result["resident_quantized_tensor_count"] <= 0 or result["resident_quantized_bytes"] <= 0:
        raise ValueError("startup evidence must report non-empty native Q8_0 residency")
    if result["resident_weight_bytes"] != (
        result["resident_quantized_bytes"]
        + result["resident_f32_bytes"]
        + result["resident_bf16_bytes"]
    ):
        raise ValueError("startup resident_weight_bytes does not reconcile resident tensor bytes")
    if result["flow_steps"] != 10 or result["generated_action_horizon"] != 50:
        raise ValueError("startup evidence must report flow_steps=10 and generated_action_horizon=50")
    return result


def _validate_shape(task_ids: Iterable[int], episodes: int) -> tuple[int, ...]:
    ids = tuple(task_ids)
    if ids == FULL_TASK_IDS and episodes == 20:
        return ids
    if ids == FULL_TASK_IDS and episodes == 3:
        return ids
    if ids == (0,) and episodes == 3:
        return ids
    raise ValueError(
        "capture supports only full task IDs 0..9 x20, suite preflight "
        "task IDs 0..9 x3, or explicit smoke task ID 0 x3"
    )


def _safe_profile_path(root: Path, profile_path: str) -> str:
    relative = Path(_nonempty(profile_path, "profile_path"))
    if relative.is_absolute():
        raise ValueError("profile path escapes output root")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("profile path escapes output root") from exc
    return relative.as_posix()


def _validate_config_snapshot(snapshot: dict[str, Any], tokenizer_path: Path) -> None:
    for key, expected in OBJECT_IMAGE_PROTOCOL.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"config snapshot {key} must be {expected}")
    configured_tokenizer = snapshot.get("tokenizer")
    if not isinstance(configured_tokenizer, str) or not configured_tokenizer.strip():
        raise ValueError("config snapshot tokenizer must be a nonempty path")
    configured_path = Path(configured_tokenizer).expanduser().resolve()
    actual_path = Path(tokenizer_path).expanduser().resolve()
    if configured_path != actual_path:
        raise ValueError(
            "config snapshot tokenizer path does not match the --tokenizer directory"
        )


def capture_evidence(
    *,
    output_root: Path,
    config_path: Path,
    quantizer_manifest_path: Path,
    server_log_path: Path,
    source_policy_path: Path,
    quantized_policy_path: Path,
    mmproj_path: Path,
    tokenizer_path: Path,
    profile_path: str,
    task_ids: Iterable[int],
    episodes: int,
    server_command: str,
    client_command: str,
    arxiv_reference: str,
    arxiv_revision_date: str,
    build_identity: dict[str, str],
    repo_root: Path,
    repository_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate all identity inputs, then atomically populate fixed evidence files."""
    output_root = Path(output_root)
    ids = _validate_shape(task_ids, episodes)
    profile_relative = _safe_profile_path(output_root, profile_path)
    paths = {
        "config": Path(config_path),
        "quantizer": Path(quantizer_manifest_path),
        "startup_log": Path(server_log_path),
        "source": Path(source_policy_path),
        "quantized": Path(quantized_policy_path),
        "mmproj": Path(mmproj_path),
        "tokenizer": Path(tokenizer_path),
    }
    for key in ("config", "quantizer", "startup_log", "source", "quantized", "mmproj"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"required {key} path is missing: {paths[key]}")
    if not paths["tokenizer"].is_dir():
        raise FileNotFoundError(f"required tokenizer path is missing: {paths['tokenizer']}")

    startup = parse_startup_log(paths["startup_log"])
    resolved_repo_root = Path(repo_root).resolve()
    if _resolved_logged_path(startup["model_path"], resolved_repo_root) != paths[
        "quantized"
    ].resolve():
        raise ValueError(
            "startup model_path does not resolve to the hashed quantized_policy path"
        )
    if _resolved_logged_path(startup["mmproj_path"], resolved_repo_root) != paths[
        "mmproj"
    ].resolve():
        raise ValueError("startup mmproj_path does not resolve to the hashed mmproj path")
    quantizer = _json_object(paths["quantizer"], "quantizer manifest")
    required_quantizer = (
        "schema", "qtype", "quantization_scope", "source_sha256", "output_sha256",
        "quantized_tensor_count", "selected_quantized_bytes", "source_tensor_bytes",
        "output_tensor_bytes", "source_dtype_counts", "output_dtype_counts",
    )
    missing_quantizer = [key for key in required_quantizer if key not in quantizer]
    if missing_quantizer:
        raise ValueError(f"quantizer manifest is missing required fields: {', '.join(missing_quantizer)}")
    if quantizer["schema"] != "smolvla_q8_0_quantization.v1":
        raise ValueError("quantizer manifest schema is not SmolVLA Q8_0")
    source_hash = _sha256(paths["source"])
    quantized_hash = _sha256(paths["quantized"])
    if quantizer["source_sha256"] != source_hash:
        raise ValueError("quantizer source_sha256 does not match source policy")
    if quantizer["output_sha256"] != quantized_hash:
        raise ValueError("quantizer output_sha256 does not match quantized policy")
    if startup["resident_quantized_tensor_count"] != quantizer["quantized_tensor_count"]:
        raise ValueError("startup resident Q8 count disagrees with quantizer manifest")
    if startup["resident_quantized_bytes"] != quantizer["selected_quantized_bytes"]:
        raise ValueError("startup resident Q8 bytes disagree with quantizer manifest")

    try:
        config_snapshot = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"config snapshot is unreadable: {exc}") from exc
    if not isinstance(config_snapshot, dict):
        raise ValueError("config snapshot must parse to an object")
    _validate_config_snapshot(config_snapshot, paths["tokenizer"])

    required_build = (
        "type", "cmake_flags", "cuda_architecture", "compiler", "cuda", "driver", "gpu",
    )
    for key in required_build:
        if key not in build_identity:
            raise ValueError(f"build identity is missing required field build.{key}")
        _nonempty(build_identity[key], f"build.{key}")
    build = {key: build_identity[key] for key in required_build}
    build["os"] = platform.platform()
    build["python"] = platform.python_version()
    for key, value in build.items():
        _nonempty(value, f"build.{key}")

    repository = _validated_repository_identity(
        repository_identity
        if repository_identity is not None
        else _repository_identity(Path(repo_root))
    )
    hashes = {
        "source_policy": source_hash,
        "quantized_policy": quantized_hash,
        "mmproj": _sha256(paths["mmproj"]),
        "tokenizer": _tree_sha256(paths["tokenizer"]),
    }
    for key, digest in hashes.items():
        if not HEX64_RE.fullmatch(digest):
            raise ValueError(f"hashes.{key} is not a SHA-256 digest")

    config_name = "config_snapshot" + (paths["config"].suffix or ".yaml")
    targets = {
        "run": output_root / "run_manifest.json",
        "startup": output_root / "server_startup.json",
        "config": output_root / config_name,
        "quantizer": output_root / "quantizer_manifest.json",
    }
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(f"output root is not a directory: {output_root}")
    existing = [path for path in targets.values() if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite evidence artifact: {existing[0]}")

    provenance_quantizer = {
        "script": "scripts/quantize_smolvla_full_gguf.py",
        "qtype": quantizer["qtype"],
        "scope": quantizer["quantization_scope"],
        "selected_tensor_count": quantizer["quantized_tensor_count"],
        "selected_quantized_bytes": quantizer["selected_quantized_bytes"],
        "source_dtype_counts": quantizer["source_dtype_counts"],
        "output_dtype_counts": quantizer["output_dtype_counts"],
        "source_tensor_bytes": quantizer["source_tensor_bytes"],
        "output_tensor_bytes": quantizer["output_tensor_bytes"],
    }
    manifest = {
        "schema": "smolvla_libero_object_run.v1",
        "suite": "libero_object",
        "task_ids": list(ids),
        "episodes_per_task": episodes,
        "environment_seed": 1000,
        "action_noise_seed": 1000,
        "control_mode": "relative",
        "action_replay": 1,
        "flow_steps": 10,
        "generated_action_horizon": 50,
        "model_image_size": 512,
        "observation_width": 360,
        "observation_height": 360,
        "settling_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        "video_enabled": False,
        "artifacts": {
            "profile": profile_relative,
            "quantizer": "quantizer_manifest.json",
            "server_startup": "server_startup.json",
            "config_snapshot": config_name,
        },
        "provenance": {
            "arxiv_reference": _nonempty(arxiv_reference, "arxiv_reference"),
            "arxiv_revision_date": _nonempty(arxiv_revision_date, "arxiv_revision_date"),
            "repository": repository,
            "commands": {
                "server": _nonempty(server_command, "server_command"),
                "client": _nonempty(client_command, "client_command"),
            },
            "config": {
                "path": config_name,
                "sha256": _sha256(paths["config"]),
                "snapshot": config_snapshot,
            },
            "build": build,
            "hashes": hashes,
            "quantizer": provenance_quantizer,
        },
    }

    # All validation is complete before the first write.
    output_root.mkdir(parents=True, exist_ok=True)
    targets["startup"].write_text(
        json.dumps(startup, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    targets["config"].write_bytes(paths["config"].read_bytes())
    targets["quantizer"].write_bytes(paths["quantizer"].read_bytes())
    targets["run"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quantizer-manifest", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--quantized-policy", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--profile-path", default="profile.json")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(FULL_TASK_IDS))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--client-command", required=True)
    parser.add_argument("--arxiv-reference", required=True)
    parser.add_argument("--arxiv-revision-date", required=True)
    parser.add_argument("--build-type", required=True)
    parser.add_argument("--cmake-flags", required=True)
    parser.add_argument("--cuda-architecture", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--repo-remote")
    parser.add_argument("--repo-commit")
    parser.add_argument("--repo-branch")
    parser.add_argument("--repo-dirty-state")
    args = parser.parse_args()
    repository_values = {
        "remote": args.repo_remote,
        "commit": args.repo_commit,
        "branch": args.repo_branch,
        "dirty_state": args.repo_dirty_state,
    }
    supplied_repository_values = [
        key for key, value in repository_values.items() if value is not None
    ]
    if supplied_repository_values and len(supplied_repository_values) != len(
        repository_values
    ):
        parser.error(
            "repository identity overrides are all-or-nothing: pass --repo-remote, "
            "--repo-commit, --repo-branch, and --repo-dirty-state together"
        )
    manifest = capture_evidence(
        output_root=args.output_root,
        config_path=args.config,
        quantizer_manifest_path=args.quantizer_manifest,
        server_log_path=args.server_log,
        source_policy_path=args.source_policy,
        quantized_policy_path=args.quantized_policy,
        mmproj_path=args.mmproj,
        tokenizer_path=args.tokenizer,
        profile_path=args.profile_path,
        task_ids=args.task_ids,
        episodes=args.episodes,
        server_command=args.server_command,
        client_command=args.client_command,
        arxiv_reference=args.arxiv_reference,
        arxiv_revision_date=args.arxiv_revision_date,
        build_identity={
            "type": args.build_type,
            "cmake_flags": args.cmake_flags,
            "cuda_architecture": args.cuda_architecture,
            "compiler": args.compiler,
            "cuda": args.cuda_version,
            "driver": args.driver_version,
            "gpu": args.gpu,
        },
        repo_root=Path(__file__).resolve().parents[1],
        repository_identity=(
            repository_values if supplied_repository_values else None
        ),
    )
    print(
        f"captured {manifest['task_ids']} x{manifest['episodes_per_task']} evidence at "
        f"{args.output_root.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
