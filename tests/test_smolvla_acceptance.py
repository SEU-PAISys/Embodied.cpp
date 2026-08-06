from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval" / "client"))

import run_smolvla_acceptance as acceptance


class AcceptanceTests(unittest.TestCase):
    def test_cpp_resume_requires_exact_complete_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "smolvla/libero_spatial/task_0/result.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({
                "arch": "smolvla", "suite": "libero_spatial", "task_id": 0,
                "episodes_requested": 3,
                "episodes_counted": 3,
                "skipped": 0, "n_action_steps": 1,
                "seed": 1000, "noise_seed": 1000,
            }))
            self.assertTrue(acceptance._cpp_task_complete(
                root, "libero_spatial", 0, 3, 1000, 1000
            ))
            self.assertFalse(acceptance._cpp_task_complete(root, "libero_spatial", 0, 10))
            self.assertFalse(acceptance._cpp_task_complete(
                root, "libero_spatial", 0, 3, 1001, 1000
            ))
            result.write_text("not json")
            self.assertFalse(acceptance._cpp_task_complete(root, "libero_spatial", 0, 3))

    def test_official_success_extraction_and_resume(self) -> None:
        payload = {"nested": {"per_episode": [
            {"success": True}, {"success": False}, {"success": True}
        ]}}
        self.assertEqual(acceptance._official_successes(payload), [True, False, True])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_info.json"
            path.write_text(json.dumps(payload))
            self.assertFalse(acceptance._official_task_complete(path, 3))
            (path.parent / "run_metadata.json").write_text(json.dumps(
                acceptance._official_metadata("libero_spatial", 0, 3, 1000, "policy")
            ))
            self.assertTrue(acceptance._official_task_complete(
                path, 3, "libero_spatial", 0, 1000, "policy"
            ))
            self.assertFalse(acceptance._official_task_complete(
                path, 10, "libero_spatial", 0, 1000, "policy"
            ))
            self.assertFalse(acceptance._official_task_complete(
                path, 3, "libero_object", 0, 1000, "policy"
            ))

    def test_minimal_report_never_claims_full_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpp = root / "cpp"
            official = root / "official"
            for suite in acceptance.SUITES:
                cpp_result = cpp / "smolvla" / suite / "task_0" / "result.json"
                cpp_result.parent.mkdir(parents=True)
                cpp_result.write_text(json.dumps({
                    "episodes_counted": 3, "successes": 2, "skipped": 0
                }))
                official_result = official / suite / "task_0" / "eval_info.json"
                official_result.parent.mkdir(parents=True)
                official_result.write_text(json.dumps({"successes": [True, True, False]}))
                (official_result.parent / "run_metadata.json").write_text(json.dumps(
                    acceptance._official_metadata(suite, 0, 3, 1000, "policy")
                ))
            args = argparse.Namespace(
                cpp_output=cpp, official_output=official, task_ids=[0], episodes=3,
                seed=1000, noise_seed=1000, tolerance=0.05, policy=Path("policy"),
            )
            report = acceptance.collect(args)
            self.assertTrue(report["criteria"]["selected_protocol_complete"])
            self.assertTrue(
                report["criteria"]["selected_cpp_within_official_tolerance"]
            )
            self.assertFalse(report["criteria"]["full_libero_object"])
            self.assertFalse(report["criteria"]["multi_episode_statistics"])
            self.assertFalse(report["criteria"]["official_smolvla_comparison"])
            self.assertFalse(report["criteria"]["full_libero_benchmark"])

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = acceptance._wilson95(7, 10)
        self.assertLessEqual(low, 0.7)
        self.assertGreaterEqual(high, 0.7)
        self.assertIsNone(acceptance._wilson95(0, 0))


if __name__ == "__main__":
    unittest.main()
