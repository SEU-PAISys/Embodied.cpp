"""Strict SmolVLA Q8_0 LIBERO-Object result/profile/manifest tests."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from client.libero_profile import _distribution as _profile_distribution  # noqa: E402
from client.reproducibility import derive_episode_noise_seed  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "aggregate_smolvla_object_eval",
    ROOT / "scripts" / "aggregate_smolvla_object_eval.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SmolVLAObjectResultTests(unittest.TestCase):
    def test_aggregator_matches_real_profiler_percentiles_std_and_rounding(self):
        samples = [float(value) for value in range(1, 101)]
        profiler_latency = _profile_distribution(samples, digits=3)
        self.assertEqual(profiler_latency["p50"], 50.5)
        self.assertEqual(profiler_latency["p95"], 95.05)
        self.assertEqual(profiler_latency["p99"], 99.01)
        recomputed = MODULE._distribution(samples)
        MODULE._validate_optional_summary(
            profiler_latency, recomputed, Path("profile.json"), digits=3
        )
        profiler_vram = _profile_distribution(samples, digits=1)
        MODULE._validate_optional_summary(
            profiler_vram, recomputed, Path("profile.json"), digits=1
        )

    def _provenance(self, root: Path) -> dict:
        def digest(name: str) -> str:
            return hashlib.sha256((root / name).read_bytes()).hexdigest()

        return {
            "arxiv_reference": "arXiv:2608.00000",
            "arxiv_revision_date": "2026-08-09",
            "repository": {
                "remote": "https://example.invalid/embodied.cpp.git",
                "commit": "dcedfec5721768661bd81f4196a0325d55f814f9",
                "branch": "hy-vla-quant-exp",
                "dirty_state": "fixture",
            },
            "commands": {
                "server": "vla-server --model policy-q8.gguf --mmproj mmproj.gguf",
                "client": "run_sim_client_direct.py --conf object-q8.yaml",
            },
            "config": {
                "path": "config_snapshot.yaml",
                "sha256": digest("config_snapshot.yaml"),
                "snapshot": {
                    "suite": "libero_object",
                    "flow_steps": 10,
                    "observation_width": 360,
                    "observation_height": 360,
                    "image_size": 512,
                    "settling_gripper_action": -1.0,
                },
            },
            "build": {
                "type": "Release",
                "cmake_flags": "-DGGML_CUDA=ON",
                "cuda_architecture": "sm_86",
                "compiler": "gcc 13",
                "cuda": "12.8",
                "driver": "570.00",
                "os": "Ubuntu 24.04",
                "gpu": "fixture GPU",
                "python": "3.10.0",
            },
            "hashes": {
                "source_policy": digest("source-policy.gguf"),
                "quantized_policy": digest("policy-q8.gguf"),
                "mmproj": digest("mmproj.gguf"),
                "tokenizer": "e" * 64,
            },
            "quantizer": {
                "script": "scripts/quantize_smolvla_full_gguf.py",
                "qtype": "Q8_0",
                "scope": "main_model_full",
                "selected_tensor_count": 6,
                "selected_quantized_bytes": 400,
                "source_dtype_counts": {"F32": 6, "BF16": 1},
                "output_dtype_counts": {"Q8_0": 6, "F32": 6, "BF16": 1},
                "source_tensor_bytes": 1000,
                "output_tensor_bytes": 700,
            },
        }

    def _write_profile(self, root: Path, records: list[dict]) -> None:
        latency = [10.0 + index / 100.0 for index in range(100)]
        generated = [value / 50.0 for value in latency]
        steps = [5.0 + index / 100.0 for index in range(100)]
        vram = [1234, 1240]
        successes = sum(item["success"] for item in records if not item["skipped"])
        profile = {
            "schema_version": 2,
            "complete": True,
            "table_ready": True,
            "arch": "smolvla",
            "implementation": "cpp",
            "suite": "libero_object",
            "episodes": {
                "expected": len(records),
                "recorded": len(records),
                "evaluable": sum(not item["skipped"] for item in records),
                "skipped": sum(item["skipped"] for item in records),
                "successes": sum(item["success"] for item in records if not item["skipped"]),
                "success_rate_percent": 100.0 * successes / len(records),
                "wilson_95_percent": list(MODULE.wilson(successes, len(records))),
                "records": [
                    {
                        "task_id": item["task_id"],
                        "episode": item["episode"],
                        "success": item["success"],
                        "skipped": item["skipped"],
                        "environment_steps": item["environment_steps"],
                    }
                    for item in records
                ],
            },
            "n_a": 1,
            "model_chunk_sizes": [50],
            "step_ms": {
                "definition": "client request wall-clock after warmup",
                "warmup_requests_excluded": 5,
                **_profile_distribution(steps),
            },
            "inf_ms": {
                "definition": "server model forward after warmup",
                "warmup_requests_excluded": 5,
                **_profile_distribution(latency),
            },
            "generated_action_step_ms": {
                "definition": "server forward divided by generated model actions",
                **_profile_distribution(generated),
            },
            "model_step_ms": {
                "definition": "server forward amortized over one replayed action",
                **_profile_distribution(latency),
            },
            "vram_mib": {
                **_profile_distribution([float(value) for value in vram], digits=1),
                "samples": vram,
                "peak": max(vram),
                "source": "process_used_memory",
                "server_pid": 123,
                "gpu_uuids": ["GPU-fixture"],
                "sample_interval_s": 0.25,
                "definition": "target process used_memory reported by nvidia-smi",
            },
        }
        (root / "profile.json").write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8"
        )

    def _write_run(
        self,
        root: Path,
        *,
        task_ids: tuple[int, ...] = tuple(range(10)),
        episodes: int = 20,
    ) -> None:
        (root / "source-policy.gguf").write_bytes(b"source-policy")
        (root / "policy-q8.gguf").write_bytes(b"quantized-policy")
        (root / "mmproj.gguf").write_bytes(b"mmproj")
        (root / "config_snapshot.yaml").write_text(
            json.dumps({
                "suite": "libero_object",
                "flow_steps": 10,
                "observation_width": 360,
                "observation_height": 360,
                "image_size": 512,
                "settling_gripper_action": -1.0,
            }) + "\n",
            encoding="utf-8",
        )
        records = []
        for task_id in task_ids:
            task = root / "smolvla" / "libero_object" / f"task_{task_id}"
            task.mkdir(parents=True)
            task_records = []
            for episode in range(episodes):
                record = {
                    "task_id": task_id,
                    "episode": episode,
                    "noise_mode": "derived",
                    "noise_seed": derive_episode_noise_seed(
                        1000, "libero_object", task_id, episode
                    ),
                    "noise_checksum": f"{task_id:02x}{episode:02x}" + "a" * 12,
                    "noise_device": "cpu",
                    "noise_dtype": "float32",
                    "success": task_id == 0 and episode == 0,
                    "skipped": False,
                    "environment_steps": 10,
                    "average_step_ms": 5.0,
                }
                task_records.append(record)
                records.append(record)
            result = {
                "arch": "smolvla",
                "implementation": "cpp",
                "suite": "libero_object",
                "task_id": task_id,
                "episodes_requested": episodes,
                "episodes_counted": episodes,
                "successes": int(task_id == 0),
                "skipped": 0,
                "success_rate": (1.0 / episodes) if task_id == 0 else 0.0,
                "seed": 1000,
                "noise_seed": 1000,
                "derive_episode_noise": True,
                "n_action_steps": 1,
                "flow_steps": 10,
                "episodes": task_records,
                "observation_width": 360,
                "observation_height": 360,
                "image_size": 512,
                "max_state_dim": 32,
                "real_action_dim": 7,
                "max_length": 48,
                "image_keys": [
                    "observation.images.image",
                    "observation.images.image2",
                ],
                "prompt_policy": "smolvla_trailing_newline",
                "generated_action_horizon": 50,
                "control_mode": "relative",
                "num_steps_wait": 10,
                "settling_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            }
            (task / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )

        quantizer = {
            "schema": "smolvla_q8_0_quantization.v1",
            "qtype": "Q8_0",
            "quantization_scope": "main_model_full",
            "quantized_tensor_count": 6,
            "selected_quantized_bytes": 400,
            "source_tensor_bytes": 1000,
            "output_tensor_bytes": 700,
            "source_sha256": hashlib.sha256(b"source-policy").hexdigest(),
            "output_sha256": hashlib.sha256(b"quantized-policy").hexdigest(),
            "source_dtype_counts": {"F32": 6, "BF16": 1},
            "output_dtype_counts": {"Q8_0": 6, "F32": 6, "BF16": 1},
            "tensors": [
                {
                    "name": name,
                    "category": "fixture",
                    "selected": True,
                    "reason": "eligible-matrix",
                    "source_type": "F32",
                    "output_type": "Q8_0",
                    "source_bytes": 350 if index == 0 else 70,
                    "output_bytes": 150 if index == 0 else 50,
                }
                for index, name in enumerate((
                    "vlm.blk.0.attn_q.weight",
                    "aex.blk.0.ffn_down.weight",
                    "connector.weight",
                    "action_in_proj.weight",
                    "action_time_mlp_in.weight",
                    "action_time_mlp_out.weight",
                ))
            ] + [{
                "name": "vlm.embed_tokens.weight",
                "category": "embedding",
                "selected": False,
                "reason": "sensitive-category",
                "source_type": "F32",
                "output_type": "F32",
                "source_bytes": 300,
                "output_bytes": 300,
            }],
        }
        (root / "quantizer_manifest.json").write_text(
            json.dumps(quantizer, indent=2) + "\n", encoding="utf-8"
        )
        startup = {
            "schema": "smolvla_server_startup.v1",
            "backend": "CUDA",
            "storage_qtype": "Q8_0",
            "resident_qtype": "Q8_0",
            "resident_quantized_tensor_count": 6,
            "resident_quantized_bytes": 400,
            "resident_f32_bytes": 100,
            "resident_bf16_bytes": 0,
            "resident_weight_bytes": 500,
            "flow_steps": 10,
            "generated_action_horizon": 50,
            "model_path": "policy-q8.gguf",
            "mmproj_path": "mmproj.gguf",
        }
        (root / "server_startup.json").write_text(
            json.dumps(startup, indent=2) + "\n", encoding="utf-8"
        )
        self._write_profile(root, records)
        manifest = {
            "schema": "smolvla_libero_object_run.v1",
            "suite": "libero_object",
            "task_ids": list(task_ids),
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
                "profile": "profile.json",
                "quantizer": "quantizer_manifest.json",
                "server_startup": "server_startup.json",
                "config_snapshot": "config_snapshot.yaml",
            },
            "provenance": self._provenance(root),
        }
        (root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def test_valid_fixture_recounts_and_reports_raw_metrics(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root)
            data = MODULE.aggregate(root)
            self.assertEqual(data["totals"]["episodes_requested"], 200)
            self.assertEqual(data["totals"]["episodes_counted"], 200)
            self.assertEqual(data["totals"]["successes"], 1)
            self.assertEqual(data["totals"]["skipped"], 0)
            self.assertEqual(len(data["totals"]["wilson_95_pct"]), 2)
            self.assertEqual(data["raw_metrics"]["vram_source"], "process_used_memory")
            self.assertEqual(data["evidence_kind"], "full_suite")
            self.assertFalse(data["promotion_eligible"])
            self.assertEqual(data["protocol"]["observation_width"], 360)
            self.assertEqual(data["protocol"]["observation_height"], 360)
            self.assertEqual(data["protocol"]["model_image_size"], 512)
            self.assertGreater(
                data["quantizer"]["output_tensor_bytes"],
                data["startup"]["resident_weight_bytes"],
            )
            markdown = MODULE.to_markdown(data)
            self.assertIn("Q8_0", markdown)
            self.assertIn("Raw latency samples", markdown)
            self.assertIn("This is full 10x20 suite evidence.", markdown)
            self.assertIn("Table 3 promotion requires a matched baseline.", markdown)
            self.assertIn("Normalized Table 3 ratios: unavailable", markdown)

    def test_valid_task_zero_three_episode_smoke_is_not_promotion_evidence(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root, task_ids=(0,), episodes=3)
            data = MODULE.aggregate(root, task_ids=(0,), episodes=3)
            self.assertEqual(data["totals"]["episodes_counted"], 3)
            self.assertEqual(data["evidence_kind"], "smoke_preflight")
            self.assertFalse(data["promotion_eligible"])

    def test_valid_all_task_three_episode_suite_preflight_is_not_promotion_evidence(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root, task_ids=tuple(range(10)), episodes=3)
            data = MODULE.aggregate(root, task_ids=tuple(range(10)), episodes=3)
            self.assertEqual(data["evidence_kind"], "suite_preflight")
            self.assertEqual(data["totals"]["episodes_counted"], 30)
            self.assertFalse(data["promotion_eligible"])

    def test_accepts_profiler_three_decimal_success_rate_for_one_of_three(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root, task_ids=(0,), episodes=3)
            profile_path = root / "profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["episodes"]["success_rate_percent"] = 33.333
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            data = MODULE.aggregate(root, task_ids=(0,), episodes=3)
            self.assertEqual(data["totals"]["successes"], 1)
            self.assertIn("not promotion-eligible", MODULE.to_markdown(data))

    def test_rejects_profile_success_episode_reassignment_without_changing_total(self):
        def mutate(path):
            profile_path = path / "profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            records = profile["episodes"]["records"]
            first = next(record for record in records
                         if (record["task_id"], record["episode"]) == (0, 0))
            second = next(record for record in records
                          if (record["task_id"], record["episode"]) == (0, 1))
            first["success"], second["success"] = False, True
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "success.*episode"):
                MODULE.aggregate(root)

    def test_rejects_profile_environment_steps_mismatch_for_episode(self):
        def mutate(path):
            profile_path = path / "profile.json"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            record = next(
                record for record in profile["episodes"]["records"]
                if (record["task_id"], record["episode"]) == (0, 0)
            )
            record["environment_steps"] += 1
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "environment_steps.*episode"):
                MODULE.aggregate(root)

    def test_smoke_rejects_missing_episode(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root, task_ids=(0,), episodes=3)
            result_path = root / "smolvla" / "libero_object" / "task_0" / "result.json"
            data = json.loads(result_path.read_text())
            data["episodes"].pop()
            result_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "episode IDs"):
                MODULE.aggregate(root, task_ids=(0,), episodes=3)

    def test_smoke_rejects_missing_profile_samples(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root, task_ids=(0,), episodes=3)
            profile_path = root / "profile.json"
            data = json.loads(profile_path.read_text())
            data["generated_action_step_ms"]["samples"] = []
            profile_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generated action-step"):
                MODULE.aggregate(root, task_ids=(0,), episodes=3)

    def test_smoke_rejects_missing_noise_checksum(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            self._write_run(root, task_ids=(0,), episodes=3)
            result_path = root / "smolvla" / "libero_object" / "task_0" / "result.json"
            data = json.loads(result_path.read_text())
            del data["episodes"][0]["noise_checksum"]
            result_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noise_checksum"):
                MODULE.aggregate(root, task_ids=(0,), episodes=3)

    def test_rejects_wrong_resident_quantized_bytes(self):
        def mutate(path):
            startup_path = path / "server_startup.json"
            data = json.loads(startup_path.read_text())
            data["resident_quantized_bytes"] = 399
            data["resident_weight_bytes"] = 499
            startup_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "resident Q8_0 bytes"):
                MODULE.aggregate(root)

    def _mutated(self, mutate):
        scratch = tempfile.TemporaryDirectory()
        root = Path(scratch.name)
        self._write_run(root)
        mutate(root)
        return scratch, root

    def test_rejects_missing_task(self):
        scratch, root = self._mutated(
            lambda path: shutil.rmtree(path / "smolvla" / "libero_object" / "task_9")
        )
        with scratch:
            with self.assertRaisesRegex(ValueError, "task IDs"):
                MODULE.aggregate(root)

    def test_rejects_duplicate_task_id(self):
        def mutate(path):
            result_path = path / "smolvla" / "libero_object" / "task_0" / "result.json"
            data = json.loads(result_path.read_text())
            data["task_id"] = 1
            result_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "task IDs"):
                MODULE.aggregate(root)

    def test_rejects_duplicate_episode_id(self):
        def mutate(path):
            result_path = path / "smolvla" / "libero_object" / "task_0" / "result.json"
            data = json.loads(result_path.read_text())
            data["episodes"][1]["episode"] = 0
            result_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "episode IDs"):
                MODULE.aggregate(root)

    def test_rejects_missing_noise_checksum(self):
        def mutate(path):
            result_path = path / "smolvla" / "libero_object" / "task_0" / "result.json"
            data = json.loads(result_path.read_text())
            del data["episodes"][0]["noise_checksum"]
            result_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "noise_checksum"):
                MODULE.aggregate(root)

    def test_rejects_skip(self):
        def mutate(path):
            result_path = path / "smolvla" / "libero_object" / "task_2" / "result.json"
            data = json.loads(result_path.read_text())
            data["episodes"][0]["skipped"] = True
            result_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "skip"):
                MODULE.aggregate(root)

    def test_rejects_missing_raw_latency_samples(self):
        def mutate(path):
            profile_path = path / "profile.json"
            data = json.loads(profile_path.read_text())
            data["inf_ms"]["samples"] = []
            profile_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "raw server latency"):
                MODULE.aggregate(root)

    def test_rejects_generated_action_step_sample_not_matching_inf_over_chunk(self):
        def mutate(path):
            profile_path = path / "profile.json"
            data = json.loads(profile_path.read_text())
            generated = data["generated_action_step_ms"]["samples"]
            generated[0] += 0.01
            data["generated_action_step_ms"].update(_profile_distribution(generated))
            profile_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "generated_action_step_ms.samples\[0\].*inf_ms"):
                MODULE.aggregate(root)

    def test_rejects_generated_action_step_sample_count_mismatch(self):
        def mutate(path):
            profile_path = path / "profile.json"
            data = json.loads(profile_path.read_text())
            inf_samples = data["inf_ms"]["samples"]
            inf_samples.append(inf_samples[-1])
            data["inf_ms"].update(_profile_distribution(inf_samples))
            profile_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "inf_ms and generated_action_step_ms sample counts"):
                MODULE.aggregate(root)

    def test_rejects_missing_raw_vram_samples(self):
        def mutate(path):
            profile_path = path / "profile.json"
            data = json.loads(profile_path.read_text())
            data["vram_mib"]["samples"] = []
            profile_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "raw VRAM"):
                MODULE.aggregate(root)

    def test_rejects_table_not_ready(self):
        def mutate(path):
            profile_path = path / "profile.json"
            data = json.loads(profile_path.read_text())
            data["table_ready"] = False
            profile_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "table_ready"):
                MODULE.aggregate(root)

    def test_rejects_missing_provenance_hash(self):
        def mutate(path):
            manifest_path = path / "run_manifest.json"
            data = json.loads(manifest_path.read_text())
            del data["provenance"]["hashes"]["quantized_policy"]
            manifest_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "quantized_policy"):
                MODULE.aggregate(root)

    def test_rejects_non_q8_startup_residency(self):
        def mutate(path):
            startup_path = path / "server_startup.json"
            data = json.loads(startup_path.read_text())
            data["resident_qtype"] = "BF16"
            startup_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "resident_qtype"):
                MODULE.aggregate(root)

    def test_rejects_quantizer_policy_hash_disagreement(self):
        def mutate(path):
            quantizer_path = path / "quantizer_manifest.json"
            data = json.loads(quantizer_path.read_text())
            data["output_sha256"] = "f" * 64
            quantizer_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "output_sha256"):
                MODULE.aggregate(root)

    def test_rejects_config_snapshot_hash_disagreement(self):
        scratch, root = self._mutated(
            lambda path: (path / "config_snapshot.yaml").write_text(
                '{"suite":"tampered","flow_steps":10}\n', encoding="utf-8"
            )
        )
        with scratch:
            with self.assertRaisesRegex(ValueError, "config.*sha256"):
                MODULE.aggregate(root)

    def test_rejects_missing_or_regressed_observation_resolution_in_config_snapshot(self):
        cases = (
            ({"observation_height": 360, "image_size": 512}, "observation_width"),
            ({"observation_width": 256, "observation_height": 360, "image_size": 512}, "observation_width"),
            ({"observation_width": 360, "observation_height": 256, "image_size": 512}, "observation_height"),
            ({"observation_width": 360, "observation_height": 360, "image_size": 256}, "image_size"),
        )
        for snapshot, message in cases:
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                self._write_run(root)
                snapshot_path = root / "config_snapshot.yaml"
                snapshot_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
                manifest_path = root / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["provenance"]["config"]["snapshot"] = snapshot
                manifest["provenance"]["config"]["sha256"] = hashlib.sha256(
                    snapshot_path.read_bytes()
                ).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.aggregate(root)

    def test_rejects_protocol_or_noise_metadata_drift(self):
        cases = (
            ("observation_width", 256, "observation_width"),
            ("observation_height", 256, "observation_height"),
            ("max_state_dim", 31, "max_state_dim"),
            ("real_action_dim", 8, "real_action_dim"),
            ("max_length", 47, "max_length"),
            ("image_keys", ["wrong", "keys"], "image_keys"),
            ("prompt_policy", "no-newline", "prompt_policy"),
            ("generated_action_horizon", 49, "generated_action_horizon"),
            ("settling_action", [0.0] * 7, "settling_action"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                self._write_run(root)
                result_path = root / "smolvla" / "libero_object" / "task_0" / "result.json"
                data = json.loads(result_path.read_text())
                data[field] = value
                result_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.aggregate(root)

        for field, value in (
            ("noise_mode", "explicit"),
            ("noise_device", "cuda"),
            ("noise_dtype", "float64"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                self._write_run(root)
                result_path = root / "smolvla" / "libero_object" / "task_0" / "result.json"
                data = json.loads(result_path.read_text())
                data["episodes"][0][field] = value
                result_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    MODULE.aggregate(root)

    def test_rejects_run_manifest_generated_horizon_drift(self):
        def mutate(path):
            manifest_path = path / "run_manifest.json"
            data = json.loads(manifest_path.read_text())
            data["generated_action_horizon"] = 49
            manifest_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "generated_action_horizon"):
                MODULE.aggregate(root)

    def test_rejects_profile_sample_warmup_chunk_or_vram_identity_gaps(self):
        cases = (
            (("model_chunk_sizes",), [49], "model_chunk_sizes"),
            (("n_a",), 2, "n_a"),
            (("implementation",), "python", "implementation"),
            (("inf_ms", "samples"), [1.0] * 99, "100 post-warmup"),
            (("inf_ms", "std"), 999.0, "std"),
            (("generated_action_step_ms", "definition"), "", "definition"),
            (("model_step_ms", "definition"), "", "definition"),
            (("step_ms", "warmup_requests_excluded"), 4, "warmup"),
            (("vram_mib", "server_pid"), None, "server_pid"),
            (("vram_mib", "gpu_uuids"), [], "gpu_uuids"),
            (("vram_mib", "sample_interval_s"), 0, "sample_interval_s"),
        )
        for keys, value, message in cases:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as scratch:
                root = Path(scratch)
                self._write_run(root)
                profile_path = root / "profile.json"
                data = json.loads(profile_path.read_text())
                target = data
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                profile_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.aggregate(root)

    def test_rejects_non_finite_profile_sample(self):
        def mutate(path):
            profile_path = path / "profile.json"
            data = json.loads(profile_path.read_text())
            data["inf_ms"]["samples"][0] = "NaN"
            profile_path.write_text(json.dumps(data), encoding="utf-8")

        scratch, root = self._mutated(mutate)
        with scratch:
            with self.assertRaisesRegex(ValueError, "finite"):
                MODULE.aggregate(root)

    def test_rejects_negative_raw_server_latency_sample(self):
        for key in ("inf_ms", "generated_action_step_ms", "step_ms", "vram_mib"):
            def mutate(path, section=key):
                profile_path = path / "profile.json"
                data = json.loads(profile_path.read_text())
                data[section]["samples"][0] = -0.001
                profile_path.write_text(json.dumps(data), encoding="utf-8")

            scratch, root = self._mutated(mutate)
            with scratch, self.subTest(section=key):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    MODULE.aggregate(root)

    def test_direct_client_records_complete_smolvla_protocol_identity(self):
        source = (ROOT / "eval" / "client" / "run_sim_client_direct.py").read_text(
            encoding="utf-8"
        )
        for field in (
            '"max_state_dim": args.max_state_dim',
            '"real_action_dim": args.real_action_dim',
            '"max_length": args.max_length',
            '"image_keys": list(args.image_keys)',
            '"prompt_policy":',
            '"generated_action_horizon": args.generated_action_horizon',
        ):
            self.assertIn(field, source)
        config = (ROOT / "eval" / "conf" / "libero_smolvla_object_q8_eval.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("generated_action_horizon: 50", config)


if __name__ == "__main__":
    unittest.main()
