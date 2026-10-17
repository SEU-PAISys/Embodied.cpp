import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_smolvla_python_baseline",
    ROOT / "scripts" / "profile_smolvla_python_baseline.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PythonBaselineAggregationTests(unittest.TestCase):
    def _write_task(self, root: Path, task_id: int, request_samples=None):
        task = root / f"task_{task_id}"
        task.mkdir(parents=True)
        (task / "eval_info.json").write_text(
            json.dumps({"per_episode": [{"success": True}, {"success": False}]}),
            encoding="utf-8",
        )
        request_samples = request_samples or [100.0, 150.0, 200.0]
        (task / "python_profile.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "suite": "libero_object",
                    "task_id": task_id,
                    "episodes": 2,
                    "seed": 1000,
                    "noise_seed": 1000,
                    "generated_action_horizon": 50,
                    "max_action_dim": 32,
                    "warmup_requests_excluded": 1,
                    "timing_definition": MODULE.TIMING_DEFINITION,
                    "request_latency_ms": request_samples,
                    "generated_action_step_ms": [value / 50 for value in request_samples],
                    "device_used_mib": [1200.0, 1300.0, 1250.0],
                }
            ),
            encoding="utf-8",
        )

    def test_aggregate_recomputes_all_table_fields_from_raw_samples(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_task(root, 0)
            self._write_task(root, 1)
            result = MODULE.aggregate(root, task_ids=(0, 1), episodes=2, generated_horizon=50)
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["success_rate_percent"], 50.0)
        self.assertEqual(result["request_latency_ms"]["n"], 4)
        self.assertEqual(result["generated_action_step_ms"]["mean"], 3.5)
        self.assertEqual(result["vram_mib"]["max"], 1300.0)
        self.assertEqual(result["vram_source"], MODULE.VRAM_SOURCE)

    def test_aggregate_rejects_generated_latency_not_derived_from_request(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_task(root, 0)
            path = root / "task_0" / "python_profile.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["generated_action_step_ms"][1] += 0.1
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generated sample"):
                MODULE.aggregate(root, task_ids=(0,), episodes=2, generated_horizon=50)

    def test_aggregate_rejects_incomplete_episode_count(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_task(root, 0)
            path = root / "task_0" / "eval_info.json"
            path.write_text(json.dumps({"per_episode": [{"success": True}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "1 episodes, expected 2"):
                MODULE.aggregate(root, task_ids=(0,), episodes=2, generated_horizon=50)

    def test_aggregate_excludes_and_records_suspended_request(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_task(root, 0, request_samples=[100.0, 150.0, 6000.0])

            result = MODULE.aggregate(
                root,
                task_ids=(0,),
                episodes=2,
                generated_horizon=50,
                excluded_latency_samples={(0, 2): "user-requested host suspension"},
            )

        self.assertEqual(result["request_latency_ms"]["n"], 1)
        self.assertEqual(result["generated_action_step_ms"]["mean"], 3.0)
        self.assertEqual(
            result["latency_exclusions"],
            {
                "reason": "host process suspension",
                "count": 1,
                "samples": [
                    {
                        "task_id": 0,
                        "sample_index": 2,
                        "request_ms": 6000.0,
                        "reason": "user-requested host suspension",
                    }
                ],
            },
        )
        self.assertEqual(result["unfiltered_request_latency_ms"]["n"], 2)
        self.assertEqual(result["unfiltered_generated_action_step_ms"]["n"], 2)

    def test_aggregate_rejects_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_task(root, 0)
            path = root / "task_0" / "python_profile.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["noise_seed"] = 7
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noise_seed"):
                MODULE.aggregate(
                    root,
                    task_ids=(0,),
                    episodes=2,
                    generated_horizon=50,
                    expected_protocol={
                        "suite": "libero_object",
                        "seed": 1000,
                        "noise_seed": 1000,
                        "warmup_requests_excluded": 1,
                        "max_action_dim": 32,
                        "timing_definition": MODULE.TIMING_DEFINITION,
                    },
                )

    def test_normalize_final_info_converts_gymnasium_029_object_array(self):
        normalizer = getattr(MODULE, "_normalize_final_info", lambda info: info)
        legacy_info = {
            "final_info": MODULE.np.asarray(
                [{"is_success": True, "task_id": 0}], dtype=object
            ),
            "_final_info": MODULE.np.asarray([True]),
        }

        normalized = normalizer(legacy_info)

        self.assertIsInstance(normalized["final_info"], dict)
        self.assertEqual(normalized["final_info"]["is_success"].tolist(), [True])
        self.assertEqual(normalized["final_info"]["task_id"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
