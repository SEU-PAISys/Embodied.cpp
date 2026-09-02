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
        return (m.group("model"), "run:" + (m.group("run") or "unlabelled"),
                m.group("suite"), int(m.group("task")))
    return None


def aggregate(outputs_dir: str):
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
            print(f"warning: unrecognized result path: {rel}", file=sys.stderr)
            continue
        model, variant, suite, task = info
        stats = parse_summary(os.path.join(root, SUMMARY_NAME))
        if stats is None:
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_outputs = os.path.join(here, os.pardir, "outputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.path.normpath(default_outputs))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(here, os.pardir, "docs", "results")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    buckets, hits = aggregate(args.outputs)
    if not hits:
        raise SystemExit(f"no summary.txt matched under {args.outputs}")
    data = build(buckets)

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
