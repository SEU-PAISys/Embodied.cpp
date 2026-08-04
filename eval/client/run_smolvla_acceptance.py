#!/usr/bin/env python3
"""Run and report the reproducible SmolVLA LIBERO acceptance matrix.

The official protocol is four suites x ten tasks x ten episodes = 400
episodes per engine. Results are resumable at task granularity. The C++
inference server must already be listening before ``--run cpp`` is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK_IDS = list(range(10))
DEFAULT_EPISODES = 10
PUBLISHED_SMOLVLA = {
    "libero_spatial": 0.90,
    "libero_object": 0.96,
    "libero_goal": 0.92,
    "libero_10": 0.71,
}


def _wilson95(successes: int, episodes: int) -> list[float] | None:
    if episodes == 0:
        return None
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _cpp_task_complete(
    root: Path,
    suite: str,
    task_id: int,
    episodes: int,
    seed: int | None = None,
    noise_seed: int | None = None,
) -> bool:
    path = root / "smolvla" / suite / f"task_{task_id}" / "result.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    complete = (
        data.get("arch") == "smolvla"
        and data.get("suite") == suite
        and data.get("task_id") == task_id
        and data.get("episodes_requested") == episodes
        and data.get("episodes_counted") == episodes
        and data.get("skipped") == 0
        and data.get("n_action_steps") == 1
    )
    if seed is not None:
        complete = complete and data.get("seed") == seed
    if noise_seed is not None:
        complete = complete and data.get("noise_seed") == noise_seed
    return complete


def run_cpp(args: argparse.Namespace) -> None:
    for suite in SUITES:
        for task_id in args.task_ids:
            if not args.force and _cpp_task_complete(
                args.cpp_output, suite, task_id, args.episodes, args.seed, args.noise_seed
            ):
                print(f"skip complete C++ task: {suite}/task_{task_id}")
                continue
            command = [
                args.python, str(Path(__file__).with_name("run_sim_client_direct.py")),
                "--conf", str(args.cpp_config), "--task", suite,
                # Override the YAML ``task_ids: all`` default with a concrete
                # list. argparse applies the declared int converter to string
                # defaults, so leaving that YAML value in place while only
                # setting --task-id would fail before resolve_task_ids().
                "--task-ids", str(task_id),
                "--n-episodes", str(args.episodes),
                "--seed", str(args.seed),
                "--noise-seed", str(args.noise_seed),
                "--output-dir", str(args.cpp_output),
            ]
            _run(command)


def _official_task_complete(path: Path, episodes: int) -> bool:
    if not path.exists():
        return False
    try:
        values = _official_successes(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False
    return len(values) == episodes


def run_official(args: argparse.Namespace) -> None:
    if not args.lerobot_eval.is_file():
        raise FileNotFoundError(f"lerobot-eval not found: {args.lerobot_eval}")
    if not args.policy.is_dir():
        raise FileNotFoundError(f"SmolVLA policy directory not found: {args.policy}")
    if not (args.libero_root / "libero").is_dir():
        raise FileNotFoundError(
            f"LIBERO Python package not found under {args.libero_root}; "
            "pass --libero-root explicitly"
        )
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.libero_root),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "TOKENIZERS_PARALLELISM": "false",
        # Every task is a fresh lerobot-eval process. Force it to reuse the
        # already validated local cache instead of issuing 40 avoidable Hub
        # metadata requests, which would make a long benchmark network-fragile.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    for suite in SUITES:
        for task_id in args.task_ids:
            output = args.official_output / suite / f"task_{task_id}"
            result_path = output / "eval_info.json"
            if not args.force and _official_task_complete(result_path, args.episodes):
                print(f"skip complete official task: {suite}/task_{task_id}")
                continue
            command = [
                str(args.lerobot_eval), f"--output_dir={output}",
                "--env.type=libero", f"--env.task={suite}",
                f"--env.task_ids={[task_id]}", "--env.max_parallel_tasks=1",
                "--eval.batch_size=1", f"--eval.n_episodes={args.episodes}",
                f"--policy.path={args.policy}", "--policy.n_action_steps=1",
                "--policy.num_steps=10", f"--seed={args.seed}",
            ]
            _run(command, env)


def _official_successes(value: Any) -> list[bool]:
    found: list[bool] = []
    if isinstance(value, dict):
        successes = value.get("successes")
        if isinstance(successes, list):
            return [bool(item) for item in successes]
        per_episode = value.get("per_episode")
        if isinstance(per_episode, list):
            found.extend(bool(row["success"]) for row in per_episode if "success" in row)
        else:
            for child in value.values():
                found.extend(_official_successes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_official_successes(child))
    return found


def collect(args: argparse.Namespace) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    for suite in SUITES:
        cpp_rows = []
        for task_id in args.task_ids:
            path = args.cpp_output / "smolvla" / suite / f"task_{task_id}" / "result.json"
            if path.exists():
                cpp_rows.append(json.loads(path.read_text(encoding="utf-8")))
        cpp_n = sum(int(row.get("episodes_counted", 0)) for row in cpp_rows)
        cpp_s = sum(int(row.get("successes", 0)) for row in cpp_rows)

        official_values: list[bool] = []
        task_paths = [
            args.official_output / suite / f"task_{task_id}" / "eval_info.json"
            for task_id in args.task_ids
        ]
        existing_task_paths = [path for path in task_paths if path.exists()]
        if existing_task_paths:
            for path in existing_task_paths:
                official_values.extend(
                    _official_successes(json.loads(path.read_text(encoding="utf-8")))
                )
        else:
            # Backward compatibility with an older suite-level run.
            legacy_path = args.official_output / suite / "eval_info.json"
            if legacy_path.exists():
                official_values = _official_successes(
                    json.loads(legacy_path.read_text(encoding="utf-8"))
                )
        official_n = len(official_values)
        official_s = sum(official_values)
        cpp_rate = cpp_s / cpp_n if cpp_n else None
        official_rate = official_s / official_n if official_n else None
        expected = len(args.task_ids) * args.episodes
        complete = cpp_n == expected and official_n == expected
        delta = cpp_rate - official_rate if cpp_rate is not None and official_rate is not None else None
        suites[suite] = {
            "cpp": {
                "successes": cpp_s, "episodes": cpp_n, "success_rate": cpp_rate,
                "wilson_95ci": _wilson95(cpp_s, cpp_n),
            },
            "official": {
                "successes": official_s, "episodes": official_n,
                "success_rate": official_rate,
                "wilson_95ci": _wilson95(official_s, official_n),
            },
            "published_reference": PUBLISHED_SMOLVLA[suite],
            "cpp_minus_official": delta,
            "complete": complete,
            "passes_delta_tolerance": complete and delta is not None and delta >= -args.tolerance,
        }
    selected_complete = all(row["complete"] for row in suites.values())
    is_full_task_matrix = (
        len(args.task_ids) == len(TASK_IDS)
        and set(args.task_ids) == set(TASK_IDS)
    )
    full_complete = selected_complete and is_full_task_matrix
    object_complete = suites["libero_object"]["complete"] and is_full_task_matrix
    return {
        "protocol": {
            "suites": list(SUITES), "task_ids": args.task_ids,
            "tasks_per_suite": len(args.task_ids),
            "episodes_per_task": args.episodes,
            "episodes_per_engine": len(SUITES) * len(args.task_ids) * args.episodes,
            "seed": args.seed, "noise_seed": args.noise_seed,
            "relative_control": True,
            "n_action_steps": 1, "flow_steps": 10,
        },
        "criteria": {
            "selected_protocol_complete": selected_complete,
            "selected_cpp_within_official_tolerance": selected_complete and all(
                row["passes_delta_tolerance"] for row in suites.values()
            ),
            "full_libero_object": object_complete,
            "multi_episode_statistics": full_complete and args.episodes >= 10,
            "official_smolvla_comparison": full_complete,
            "full_libero_benchmark": full_complete,
            "cpp_within_official_tolerance": full_complete and all(
                row["passes_delta_tolerance"] for row in suites.values()
            ),
            "absolute_delta_tolerance": args.tolerance,
        },
        "suites": suites,
    }


def write_report(args: argparse.Namespace) -> None:
    report = collect(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = args.report.with_suffix(".md")
    lines = [
        "# SmolVLA LIBERO acceptance report", "",
        "| Suite | C++ | Official | Delta | Published reference | Complete |", "|---|---:|---:|---:|---:|:---:|",
    ]
    for suite, row in report["suites"].items():
        def rate(engine: str) -> str:
            item = row[engine]
            value = item["success_rate"]
            return "pending" if value is None else f'{value:.1%} ({item["successes"]}/{item["episodes"]})'
        delta = row["cpp_minus_official"]
        lines.append(
            f'| {suite} | {rate("cpp")} | {rate("official")} | '
            f'{"pending" if delta is None else f"{delta:+.1%}"} | '
            f'{row["published_reference"]:.1%} | {"yes" if row["complete"] else "no"} |'
        )
    lines.extend(["", "## Acceptance criteria", ""])
    for name, passed in report["criteria"].items():
        if isinstance(passed, bool):
            lines.append(f'- [{"x" if passed else " "}] {name}')
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {args.report.resolve()}")
    print(f"Markdown: {md.resolve()}")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("none", "cpp", "official", "all"), default="none")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--task-ids", nargs="+", type=int, default=TASK_IDS)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--noise-seed", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpp-config", type=Path, default=repo / "eval/conf/libero_smolvla_benchmark.yaml")
    parser.add_argument("--cpp-output", type=Path, default=repo / "outputs/smolvla_acceptance/cpp")
    parser.add_argument("--official-output", type=Path, default=repo / "outputs/smolvla_acceptance/official")
    parser.add_argument("--report", type=Path, default=repo / "outputs/smolvla_acceptance/report.json")
    parser.add_argument("--lerobot-eval", type=Path, default=Path("/root/smolvla-port/bin/lerobot-eval"))
    parser.add_argument("--policy", type=Path, default=Path("/root/checkpoints/smolvla_libero"))
    parser.add_argument("--libero-root", type=Path, default=repo / "eval/sim/libero/LIBERO")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.seed < 0 or args.noise_seed < 0:
        parser.error("--seed and --noise-seed must be non-negative")
    if not args.task_ids or any(task_id not in TASK_IDS for task_id in args.task_ids):
        parser.error("--task-ids must contain values from 0 through 9")
    args.task_ids = list(dict.fromkeys(args.task_ids))
    if args.run in ("cpp", "all"):
        run_cpp(args)
    if args.run in ("official", "all"):
        run_official(args)
    write_report(args)


if __name__ == "__main__":
    main()
