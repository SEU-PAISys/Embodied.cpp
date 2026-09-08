#!/usr/bin/env python3
"""Aggregate LIBERO success rates from outputs/ into a small durable summary.

The raw episode logs under ``outputs/`` are gitignored (they are videos and
per-episode text). This script distils them into a compact JSON + Markdown
pair that *can* be committed, so the reported numbers survive even if the
raw outputs are lost.

Usage:
    python scripts/aggregate_eval_summary.py [--outputs DIR] [--out-dir DIR]

Output:
    <out-dir>/eval_summary.json
    <out-dir>/eval_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

SUMMARY_NAME = "summary.txt"

# path patterns -> (model, variant, suite)
PATTERNS = [
    # xr0/libero2000/<variant>/<suite>/shardN/xr0/libero_<suite>/task_i/summary.txt
    re.compile(r"^xr0/libero2000/(?P<variant>[^/]+)/(?P<suite>[^/]+)/shard\d+/"
               r"(?P<model>[^/]+)/libero_(?P<s2>[^/]+)/task_(?P<task>\d+)/"),
    # turbovla/libero400/<variant>/<suite>/turbovla/libero_<suite>/task_i/summary.txt
    re.compile(r"^turbovla/libero400/(?P<variant>[^/]+)/(?P<suite>[^/]+)/"
               r"(?P<model>[^/]+)/libero_(?P<s2>[^/]+)/task_(?P<task>\d+)/"),
    # xvla_full_cpp{,_Q8_0,_Q4_K}/<suite>/xvla/libero_<suite>/task_i/summary.txt
    re.compile(r"^xvla_full_cpp(?:_(?P<variant>Q8_0|Q4_K))?/(?P<suite>[^/]+)/"
               r"(?P<model>[^/]+)/libero_(?P<s2>[^/]+)/task_(?P<task>\d+)/"),
]

# Legacy layout: <model>/libero_<suite>/task_i/summary.txt
# These are early smoke runs (1-5 episodes). They are kept in a separate
# "smoke" bucket so they can never overwrite a full sweep in the aggregate.
OLD_LAYOUT = re.compile(
    r"^(?P<model>xr0|turbovla|xvla)/libero_(?P<suite>[^/]+)/task_(?P<task>\d+)/")

# Public runner: <run-name>/<model>/libero_<suite>/task_i/summary.txt.
# A run name identifies an experiment, NOT a precision inferred from the path.
# Keep it separate from archived bf16/quantized/smoke results.
RUN_LAYOUT = re.compile(
    r"^(?:(?P<run>.+)/)?(?P<model>pi05|groot_n1|hy_vla|lingbot_va|smolvla|xr0|turbovla|xvla)/"
    r"libero_(?P<suite>[^/]+)/task_(?P<task>\d+)/summary\.txt$")

RATE_RE = re.compile(r"Success rate:\s*([\d.]+)%\s*\((\d+)/(\d+)\)")
SKIP_RE = re.compile(r"Skipped[^:]*:\s*(\d+)/(\d+)")
LAT_RE = re.compile(r"Average inference time per step:\s*([\d.]+)\s*ms")

SUITE_ORDER = ["spatial", "object", "goal", "10"]


def parse_summary(path: str):
    """Return (success, total, skipped, latency_ms) from a summary.txt."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    m = RATE_RE.search(text)
    if not m:
        return None
    rate = float(m.group(1))
    succ, total = int(m.group(2)), int(m.group(3))
    skipped = 0
    ms = None
    s = SKIP_RE.search(text)
    if s:
        skipped = int(s.group(1))
    lat = LAT_RE.search(text)
    if lat:
        ms = float(lat.group(1))
    return {"rate": rate, "success": succ, "total": total,
            "skipped": skipped, "latency_ms": ms}


def classify(rel: str):
    rel = rel.replace(os.sep, "/")
    for pat in PATTERNS:
        m = pat.search(rel)
        if not m:
            continue
        g = m.groupdict()
        variant = g.get("variant") or "bf16"
        suite = g.get("suite")
        # the older layout has no variant
        if "s2" in g and g.get("s2") and g["s2"] != suite:
            suite = g["s2"]
        return g["model"], variant, suite, int(g["task"])
    m = OLD_LAYOUT.search(rel)
    if m:
        return m.group("model"), "smoke", m.group("suite"), int(m.group("task"))
    m = RUN_LAYOUT.fullmatch(rel)
    if m:
        run = m.group("run") or ""
        # Sweep drivers optionally place one suite directory above the model.
        # It belongs to the same run, not a separate variant per suite.
        parts = run.split("/") if run else []
        if parts and parts[-1] in (m.group("suite"), "libero_" + m.group("suite")):
            run = "/".join(parts[:-1])
        return (m.group("model"), "run:" + (run or "unlabelled"),
                m.group("suite"), int(m.group("task")))
    return None


def aggregate(outputs_dir: str, *, strict=False):
    # buckets[model][variant][suite][task] = stats
    buckets = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    hits = 0
    for root, _dirs, files in os.walk(outputs_dir):
        if SUMMARY_NAME not in files:
            continue
        rel = os.path.relpath(os.path.join(root, SUMMARY_NAME), outputs_dir)
        info = classify(rel)
        if info is None:
            if strict:
                raise ValueError(f"unrecognized result path: {rel}")
            print(f"warning: unrecognized result path: {rel}", file=sys.stderr)
            continue
        model, variant, suite, task = info
        stats = parse_summary(os.path.join(root, SUMMARY_NAME))
        if stats is None:
            if strict:
                raise ValueError(f"unreadable result: {rel}")
            print(f"warning: unreadable result: {rel}", file=sys.stderr)
            continue
        if task in buckets[model][variant][suite]:
            raise ValueError(f"duplicate task result for {model}/{variant}/{suite}/task_{task}: {rel}")
        stats["source"] = rel.replace(os.sep, "/")
        buckets[model][variant][suite][task] = stats
        hits += 1
    return buckets, hits


def build(buckets):
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "outputs/ (gitignored raw episode logs)",
           "models": {}}
    for model in sorted(buckets):
        variants = {}
        for variant in sorted(buckets[model]):
            suites = {}
            tot_s = tot_n = 0
            for suite in sorted(buckets[model][variant],
                                key=lambda s: SUITE_ORDER.index(s)
                                if s in SUITE_ORDER else 99):
                tasks = buckets[model][variant][suite]
                s = sum(v["success"] for v in tasks.values())
                n = sum(v["total"] for v in tasks.values())
                lat = [v["latency_ms"] for v in tasks.values() if v["latency_ms"]]
                suites[suite] = {
                    "success": s,
                    "episodes": n,
                    "rate_pct": round(100.0 * s / n, 2) if n else None,
                    "per_task_rate_pct": [tasks[t]["rate"] for t in sorted(tasks)],
                    "mean_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
                    "task_ids": sorted(tasks),
                    "skipped": sum(v["skipped"] for v in tasks.values()),
                    "sources": [tasks[t]["source"] for t in sorted(tasks)],
                }
                tot_s += s
                tot_n += n
            variants[variant] = {
                "suites": suites,
                "overall_success": tot_s,
                "overall_episodes": tot_n,
                "overall_rate_pct": round(100.0 * tot_s / tot_n, 2) if tot_n else None,
            }
        out["models"][model] = variants
    return out


def to_markdown(data):
    lines = ["# LIBERO evaluation summary (generated)",
             "",
             f"- generated: `{data['generated_utc']}`",
             f"- source: {data['source']}",
             "- generated by: `scripts/aggregate_eval_summary.py`",
             "- `run:*` rows are named runs, not verified precision labels or full-suite acceptance.",
             "- latency is the mean of per-task amortized client step times, not model-call latency.",
             ""]
    for model, variants in data["models"].items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| variant | " + " | ".join(SUITE_ORDER) + " | overall |")
        lines.append("|---|" + "---|" * (len(SUITE_ORDER) + 1))
        for variant, v in variants.items():
            cells = []
            for s in SUITE_ORDER:
                if s in v["suites"]:
                    cells.append(f"{v['suites'][s]['rate_pct']}%")
                else:
                    cells.append("-")
            overall = v["overall_rate_pct"]
            cells.append(f"**{overall}%** ({v['overall_success']}/{v['overall_episodes']})")
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        lines.append("")
        for variant, v in variants.items():
            lines.append(f"### {model} / {variant} — per task")
            lines.append("")
            lines.append("| suite | per-task success rate (%) |")
            lines.append("|---|---|")
            for s in SUITE_ORDER:
                if s in v["suites"]:
                    pt = v["suites"][s]["per_task_rate_pct"]
                    lines.append(f"| {s} | " + " ".join(f"{x:g}" for x in pt) + " |")
            lines.append("")
    return "\n".join(lines)


def validate_full_matrix(buckets, *, suites=("spatial", "object", "goal", "10"),
                         tasks=tuple(range(10))):
    """Hard gate for a published full-matrix run: every (model, variant)
    bucket must contain exactly the expected unique (suite, task) set.

    Duplicates already raise in aggregate(); this adds missing tasks,
    unexpected tasks, and unexpected suites as hard failures so an
    incomplete or polluted sweep can never pass silently (the 394/400
    lesson: spatial_t3 duplicated, 10_t3 missing, line count looked fine).
    """
    problems = []
    if not buckets or not any(buckets.values()):
        problems.append("no result buckets")
    for model in sorted(buckets):
        for variant in sorted(buckets[model]):
            label = f"{model}/{variant}"
            found_suites = set(buckets[model][variant])
            for suite in sorted(found_suites - set(suites)):
                problems.append(f"{label}: unexpected suite {suite!r} "
                                f"(tasks {sorted(buckets[model][variant][suite])})")
            for suite in suites:
                found = buckets[model][variant].get(suite, {})
                unexpected = sorted(set(found) - set(tasks))
                missing = sorted(set(tasks) - set(found))
                if unexpected:
                    problems.append(f"{label}/{suite}: unexpected task ids {unexpected}")
                if missing:
                    problems.append(f"{label}/{suite}: missing task ids {missing}")
    if problems:
        raise ValueError(
            "full-matrix coverage validation FAILED "
            f"(expected suites {list(suites)} x tasks {list(tasks)}):\n  "
            + "\n  ".join(problems))
    return True


def validate_run_episodes(run_dir, *, arch="xvla", suites=("spatial", "object", "goal", "10"),
                          tasks=tuple(range(10)), episodes=10, require_no_skips=True,
                          task_paths=None):
    """Per-episode release gate for one run directory (runbook P3).

    Layout: <run_dir>/<suite>/<arch>/libero_<suite>/task_<t>/result.json (+summary.txt).
    Fails unless every task has exactly the planned episode ids, all terminal,
    counted/planned/skipped relations hold, and summary.txt agrees with
    result.json. Catches the "40 full-looking task dirs, zero valid episodes"
    and "1-of-10 episodes" failure modes that key-set validation cannot see.
    """
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    problems = []
    seen = set()
    for suite in suites:
        if task_paths is None:
            suite_dir = Path(run_dir) / suite / arch / f"libero_{suite}"
            if not suite_dir.is_dir():
                problems.append(f"{suite}: missing suite directory {suite_dir}")
                continue
            task_dirs = sorted(suite_dir.glob("task_*"))
        else:
            task_dirs = [Path(p) for p in task_paths.get(suite, [])]
        names = sorted(d.name for d in task_dirs)
        if names != sorted(f"task_{t}" for t in tasks):
            problems.append(f"{suite}: task dirs {names} != expected {list(tasks)}")
        for task_dir in task_dirs:
            key = (suite, task_dir.name)
            if key in seen:
                problems.append(f"{key}: duplicate task directory")
            seen.add(key)
            res = task_dir / "result.json"
            if not res.is_file():
                problems.append(f"{key}: result.json missing")
                continue
            try:
                data = json.loads(res.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{key}: result.json unreadable: {exc}")
                continue
            if not isinstance(data, dict):
                problems.append(f"{key}: result.json must be an object")
                continue
            eps = data.get("episodes", [])
            if not isinstance(eps, list) or any(not isinstance(e, dict) for e in eps):
                problems.append(f"{key}: episodes must be a list of objects")
                continue
            got_ids = [e.get("episode") for e in eps]
            if any(type(i) is not int for i in got_ids) or sorted(got_ids) != list(range(episodes)):
                problems.append(f"{key}: episode ids {got_ids} != 0..{episodes - 1}")
                continue
            for e in eps:
                if type(e.get("success")) is not bool or type(e.get("skipped")) is not bool:
                    problems.append(f"{key} ep{e.get('episode')}: success/skipped must be boolean")
                if e.get("success") is True and e.get("skipped") is True:
                    problems.append(f"{key} ep{e.get('episode')}: skipped episode cannot succeed")
            fields = ("episodes_counted", "episodes_requested", "skipped", "successes")
            if any(type(data.get(f)) is not int for f in fields):
                problems.append(f"{key}: counts must be integers")
                continue
            counted, requested, skipped, successes = (data[f] for f in fields)
            # Recompute from the per-episode records: a summary that claims
            # 10/10 while every episode failed must fail here.
            ep_success = sum(1 for e in data.get("episodes", [])
                             if e.get("success") is True)
            ep_skipped = sum(e.get("skipped") is True for e in eps)
            ep_counted = sum(type(e.get("success")) is bool and e.get("skipped") is False
                             for e in eps)
            if ep_counted != counted:
                problems.append(f"{key}: {ep_counted} counted episode records "
                                f"!= counted={counted}")
            if ep_skipped != skipped:
                problems.append(f"{key}: recomputed skipped={ep_skipped} != skipped={skipped}")
            if requested != episodes or counted + skipped != requested:
                problems.append(f"{key}: requested/count/skipped disagree with planned {episodes}")
            if ep_success != successes:
                problems.append(f"{key}: recomputed successes={ep_success} "
                                f"!= summary successes={successes}")
            if not (0 <= successes <= counted <= requested):
                problems.append(f"{key}: invalid relations success={successes} "
                                f"counted={counted} requested={requested}")
            if require_no_skips and (counted != requested or skipped != 0):
                problems.append(f"{key}: incomplete sweep counted={counted} "
                                f"requested={requested} skipped={skipped}")
            summary = task_dir / "summary.txt"
            if summary.is_file():
                text = summary.read_text(encoding="utf-8", errors="replace")
                m = RATE_RE.search(text)
                if not m:
                    problems.append(f"{key}: summary has no valid success rate")
                elif (int(m.group(2)) != successes or int(m.group(3)) != counted):
                    problems.append(f"{key}: summary {m.group(2)}/{m.group(3)} "
                                    f"contradicts result.json {successes}/{counted}")
                elif abs(float(m.group(1)) - 100 * successes / max(1, counted)) > 0.011:
                    problems.append(f"{key}: summary percentage contradicts counts")
                sm = SKIP_RE.search(text)
                if sm and (int(sm.group(1)), int(sm.group(2))) != (skipped, requested):
                    problems.append(f"{key}: summary skipped count contradicts result.json")
            else:
                problems.append(f"{key}: summary.txt missing")
    if problems:
        raise ValueError("per-episode coverage validation FAILED:\n  "
                         + "\n  ".join(problems[:40]))
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_outputs = os.path.join(here, os.pardir, "outputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.normpath(default_outputs))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--episodes-per-task", type=int, default=10,
                    help="planned episodes per task for --require-full-matrix (default: 10)")
    ap.add_argument("--require-full-matrix", action="store_true",
                    help="hard-fail unless every bucket covers exactly the "
                         "4 LIBERO suites x tasks 0..9 (duplicates, missing, "
                         "and unexpected entries are all failures)")
    args = ap.parse_args()
    if args.episodes_per_task <= 0:
        ap.error("--episodes-per-task must be positive")

    out_dir = args.out_dir or os.path.join(here, os.pardir, "docs", "results")
    out_dir = os.path.normpath(out_dir)
    buckets, hits = aggregate(args.outputs, strict=args.require_full_matrix)
    if not hits:
        raise SystemExit(f"no summary.txt matched under {args.outputs}")
    if args.require_full_matrix:
        validate_full_matrix(buckets)
        for arch, variants in buckets.items():
            for variant, suites in variants.items():
                paths = {suite: [Path(args.outputs) / stat["source"] for stat in tasks.values()]
                         for suite, tasks in suites.items()}
                validate_run_episodes(
                    args.outputs, arch=arch, episodes=args.episodes_per_task,
                    task_paths={suite: [p.parent for p in entries] for suite, entries in paths.items()},
                )
    data = build(buckets)
    os.makedirs(out_dir, exist_ok=True)

    jpath = os.path.join(out_dir, "eval_summary.json")
    mpath = os.path.join(out_dir, "eval_summary.md")
    with open(jpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_markdown(data))

    print(f"parsed {hits} summary.txt files")
    for model, variants in data["models"].items():
        for variant, v in variants.items():
            print(f"  {model:<10} {variant:<12} "
                  f"{v['overall_rate_pct']}%  "
                  f"({v['overall_success']}/{v['overall_episodes']})")
    print(f"wrote {jpath}")
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
