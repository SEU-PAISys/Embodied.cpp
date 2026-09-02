"""Result aggregation and profiling checks; stdlib only, Python 3.10+."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "eval"))
import aggregate_eval_summary as summary
from client.libero_profile import LiberoSuiteProfiler


class ResultTests(unittest.TestCase):
    def test_current_and_archived_layouts_stay_separate(self):
        for arch in ("xr0", "turbovla", "xvla", "smolvla", "pi05", "groot_n1", "lingbot_va"):
            path = f"{arch}_libero_object/{arch}/libero_object/task_0/summary.txt"
            self.assertEqual(summary.classify(path), (arch, f"run:{arch}_libero_object", "object", 0))
        self.assertEqual(summary.classify("xr0/libero_object/task_0/summary.txt"),
                         ("xr0", "smoke", "object", 0))
        self.assertEqual(summary.classify("xr0/libero2000/bf16/object/shard0/xr0/libero_object/task_0/summary.txt"),
                         ("xr0", "bf16", "object", 0))
        self.assertEqual(summary.classify("xvla_full_cpp_Q8_0/object/xvla/libero_object/task_0/summary.txt"),
                         ("xvla", "Q8_0", "object", 0))

    def test_named_run_aggregates_all_suites_without_precision_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suite in summary.SUITE_ORDER:
                path = Path(tmp) / f"new-run/xvla/libero_{suite}/task_0/summary.txt"
                path.parent.mkdir(parents=True)
                path.write_text("Success rate: 50.00%  (1/2)\nSkipped: 0/2\n"
                                "Average inference time per step: 10 ms\n", encoding="utf-8")
            buckets, hits = summary.aggregate(tmp)
            data = summary.build(buckets)
            run = data["models"]["xvla"]["run:new-run"]
            self.assertEqual(hits, 4)
            self.assertEqual(run["overall_episodes"], 8)
            self.assertEqual(run["overall_success"], 4)
            self.assertEqual(run["suites"]["object"]["task_ids"], [0])
            self.assertEqual(len(run["suites"]["object"]["sources"]), 1)
            self.assertNotIn("bf16", data["models"]["xvla"])
            self.assertIn("not verified precision", summary.to_markdown(data))

    def test_duplicate_shards_are_rejected_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shard in (0, 1):
                path = Path(tmp) / f"xr0/libero2000/bf16/object/shard{shard}/xr0/libero_object/task_0/summary.txt"
                path.parent.mkdir(parents=True)
                path.write_text("Success rate: 50.00% (1/2)\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate task"):
                summary.aggregate(tmp)

    def test_unknown_results_warn_instead_of_silently_disappearing(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary.txt").write_text("Success rate: 100.00% (2/2)", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                _, hits = summary.aggregate(tmp)
            self.assertEqual(hits, 0)
            self.assertIn("unrecognized result path", stderr.getvalue())

    def test_profile_result_counts_requests_not_queued_actions(self):
        profiler = LiberoSuiteProfiler(
            output_path=Path("unused.json"), model_label="test", backbone_label="test",
            arch="xvla", suite="libero_object", replay_chunk_size=30, expected_episodes=1,
            server_address="tcp://localhost:5555", server_pid=1,
            vram_interval_s=0.25, warmup_requests=1,
        )
        for sequence in (1, 2):
            client = SimpleNamespace(get_last_inference_profile=lambda: {
                "sequence": sequence, "server_total_ms": sequence * 10,
                "model_chunk_size": 30,
            })
            for _ in range(30):
                profiler.capture_inference(client)
                profiler.record_step(1)
        profiler.episodes.append({"success": True, "skipped": False})
        result = profiler.result(complete=True)
        self.assertEqual(result["inf_ms"]["n"], 1)
        self.assertEqual(result["inf_ms"]["mean"], 20)
        self.assertEqual(result["step_ms"]["n"], 30)
        self.assertFalse(result["table_ready"])  # no GPU memory measurement


if __name__ == "__main__":
    unittest.main()
