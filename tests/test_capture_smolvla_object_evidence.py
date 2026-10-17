"""Pre-rollout evidence capture tests for the SmolVLA Object gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CAPTURE = _load(
    "capture_smolvla_object_evidence",
    ROOT / "scripts" / "capture_smolvla_object_evidence.py",
)
VALIDATOR = _load(
    "aggregate_smolvla_object_eval_capture_tests",
    ROOT / "scripts" / "aggregate_smolvla_object_eval.py",
)


class CaptureEvidenceTests(unittest.TestCase):
    def test_capture_helper_is_not_gitignored(self):
        completed = subprocess.run(
            ["git", "check-ignore", "scripts/capture_smolvla_object_evidence.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout)

    def _inputs(self, base: Path, *, config_tokenizer: str | None = None) -> dict:
        source = base / "source.gguf"
        output = base / "policy-q8.gguf"
        mmproj = base / "mmproj.gguf"
        config = base / "object-q8.yaml"
        tokenizer = base / "tokenizer"
        tokenizer.mkdir(exist_ok=True)
        source.write_bytes(b"source-policy")
        output.write_bytes(b"quantized-policy")
        mmproj.write_bytes(b"mmproj")
        config.write_text(
            json.dumps({
                "suite": "libero_object",
                "tokenizer": (
                    config_tokenizer
                    if config_tokenizer is not None
                    else str(tokenizer.resolve())
                ),
                "flow_steps": 10,
                "observation_width": 360,
                "observation_height": 360,
                "image_size": 512,
                "settling_gripper_action": -1.0,
            }) + "\n",
            encoding="utf-8",
        )
        (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (tokenizer / "vocab.json").write_text('{"a": 0}\n', encoding="utf-8")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        quantizer = base / "source-quantizer-manifest.json"
        quantizer.write_text(
            json.dumps({
                "schema": "smolvla_q8_0_quantization.v1",
                "qtype": "Q8_0",
                "quantization_scope": "main_model_full",
                "source_sha256": sha(source),
                "output_sha256": sha(output),
                "quantized_tensor_count": 1,
                "selected_quantized_bytes": 400,
                "source_tensor_bytes": 1000,
                "output_tensor_bytes": 700,
                "source_dtype_counts": {"F32": 2},
                "output_dtype_counts": {"Q8_0": 1, "F32": 1},
                "tensors": [
                    {
                        "name": "vlm.blk.0.attn_q.weight",
                        "category": "vlm",
                        "selected": True,
                        "reason": "eligible-matrix",
                        "source_type": "F32",
                        "output_type": "Q8_0",
                        "source_bytes": 700,
                        "output_bytes": 400,
                    },
                    {
                        "name": "vlm.embed_tokens.weight",
                        "category": "embedding",
                        "selected": False,
                        "reason": "sensitive-category",
                        "source_type": "F32",
                        "output_type": "F32",
                        "source_bytes": 300,
                        "output_bytes": 300,
                    },
                ],
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        startup_log = base / "server.log"
        startup_log.write_text(
            "noise before startup\n"
            f"vla(smolvla): startup_evidence backend=CUDA model_path={output.resolve()} "
            f"mmproj_path={mmproj.resolve()} storage_qtype=Q8_0 resident_qtype=Q8_0 "
            "resident_quantized_tensors=1 resident_quantized_bytes=400 "
            "resident_f32_bytes=100 resident_bf16_bytes=0 resident_weight_bytes=500 "
            "resident_buffer_bytes=640 flow_steps=10 generated_action_horizon=50\n",
            encoding="utf-8",
        )
        return {
            "config_path": config,
            "quantizer_manifest_path": quantizer,
            "server_log_path": startup_log,
            "source_policy_path": source,
            "quantized_policy_path": output,
            "mmproj_path": mmproj,
            "tokenizer_path": tokenizer,
        }

    def _capture(self, base: Path, **overrides):
        config_tokenizer = overrides.pop("config_tokenizer", None)
        values = {
            "output_root": base / "evidence",
            **self._inputs(base, config_tokenizer=config_tokenizer),
            "profile_path": "profile.json",
            "task_ids": (0,),
            "episodes": 3,
            "server_command": "vla-server --model policy-q8.gguf --mmproj mmproj.gguf",
            "client_command": "run_sim_client_direct.py --task-ids 0 --n-episodes 3",
            "arxiv_reference": "arXiv:2608.00000",
            "arxiv_revision_date": "2026-08-09",
            "build_identity": {
                "type": "Release",
                "cmake_flags": "-DGGML_CUDA=ON",
                "cuda_architecture": "sm_86",
                "compiler": "gcc 13",
                "cuda": "12.8",
                "driver": "570.00",
                "gpu": "fixture GPU",
            },
            "repo_root": ROOT,
            "repository_identity": self._repository_override(),
        }
        values.update(overrides)
        return CAPTURE.capture_evidence(**values)

    @staticmethod
    def _repository_override() -> dict[str, str]:
        return {
            "remote": "https://github.com/SEU-PAISys/Embodied.cpp.git",
            "commit": "a" * 40,
            "branch": "hy-vla-quant-exp",
            "dirty_state": "dirty:fixture",
        }

    def test_valid_smoke_capture_is_non_claim_based_and_validator_compatible(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            manifest = self._capture(base)
            root = base / "evidence"
            self.assertEqual(manifest["task_ids"], [0])
            self.assertEqual(manifest["episodes_per_task"], 3)
            self.assertEqual(manifest["observation_width"], 360)
            self.assertEqual(manifest["observation_height"], 360)
            self.assertEqual(manifest["artifacts"]["profile"], "profile.json")
            self.assertTrue((root / "config_snapshot.yaml").is_file())
            self.assertTrue((root / "quantizer_manifest.json").is_file())
            startup = json.loads((root / "server_startup.json").read_text())
            self.assertEqual(startup["resident_quantized_bytes"], 400)
            self.assertEqual(startup["flow_steps"], 10)
            self.assertEqual(startup["generated_action_horizon"], 50)
            VALIDATOR._validate_run_manifest(root, (0,), 3)
            VALIDATOR._validate_quantizer_manifest(root / "quantizer_manifest.json")
            VALIDATOR._validate_startup(root / "server_startup.json")
            captured = {
                name: (root / name).read_bytes()
                for name in (
                    "run_manifest.json",
                    "server_startup.json",
                    "quantizer_manifest.json",
                    "config_snapshot.yaml",
                )
            }
            fixture_module = _load(
                "smolvla_object_quantized_capture_fixture",
                ROOT / "tests" / "test_smolvla_object_quantized.py",
            )
            fixture_module.SmolVLAObjectResultTests()._write_run(
                root, task_ids=(0,), episodes=3
            )
            for name, payload in captured.items():
                (root / name).write_bytes(payload)
            aggregate = VALIDATOR.aggregate(root, task_ids=(0,), episodes=3)
            self.assertEqual(aggregate["evidence_kind"], "smoke_preflight")

    def test_capture_accepts_relative_config_tokenizer_matching_absolute_input(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            relative_tokenizer = os.path.relpath(base / "tokenizer", Path.cwd())
            manifest = self._capture(base, config_tokenizer=relative_tokenizer)
            self.assertEqual(
                manifest["provenance"]["config"]["snapshot"]["tokenizer"],
                relative_tokenizer,
            )

    def test_capture_rejects_config_tokenizer_directory_mismatch_before_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            other_tokenizer = base / "other-tokenizer"
            other_tokenizer.mkdir()
            with self.assertRaisesRegex(ValueError, "config snapshot tokenizer"):
                self._capture(base, config_tokenizer=str(other_tokenizer.resolve()))
            self.assertFalse((base / "evidence" / "run_manifest.json").exists())

    def test_committed_object_config_pins_native_360_observations(self):
        config = yaml.safe_load(
            (ROOT / "eval" / "conf" / "libero_smolvla_object_q8_eval.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["observation_width"], 360)
        self.assertEqual(config["observation_height"], 360)
        self.assertEqual(config["image_size"], 512)
        self.assertEqual(config["settling_gripper_action"], -1.0)

    def test_capture_rejects_missing_or_regressed_native_observation_resolution(self):
        cases = (
            ({"observation_height": 360, "image_size": 512}, "observation_width"),
            ({"observation_width": 256, "observation_height": 360, "image_size": 512}, "observation_width"),
            ({"observation_width": 360, "observation_height": 256, "image_size": 512}, "observation_height"),
            ({"observation_width": 360, "observation_height": 360, "image_size": 256}, "image_size"),
            ({"observation_width": 360, "observation_height": 360, "image_size": 512}, "settling_gripper_action"),
            ({"observation_width": 360, "observation_height": 360, "image_size": 512,
              "settling_gripper_action": 0.0}, "settling_gripper_action"),
        )
        for snapshot, message in cases:
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as scratch:
                base = Path(scratch)
                bad_config = base / "bad-object-q8.yaml"
                bad_config.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self._capture(base, config_path=bad_config)
                self.assertFalse((base / "evidence" / "run_manifest.json").exists())

    def test_full_capture_uses_default_promotion_shape(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            manifest = self._capture(
                base, task_ids=tuple(range(10)), episodes=20
            )
            self.assertEqual(manifest["task_ids"], list(range(10)))
            self.assertEqual(manifest["episodes_per_task"], 20)
            VALIDATOR._validate_run_manifest(base / "evidence")

    def test_all_task_three_episode_capture_is_suite_preflight(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            manifest = self._capture(
                base, task_ids=tuple(range(10)), episodes=3
            )
            self.assertEqual(manifest["task_ids"], list(range(10)))
            self.assertEqual(manifest["episodes_per_task"], 3)
            VALIDATOR._validate_run_manifest(
                base / "evidence", task_ids=tuple(range(10)), episodes=3
            )

    def test_capture_rejects_malformed_startup_before_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            bad_log = base / "bad-server.log"
            bad_log.write_text(
                "vla(smolvla): startup_evidence backend=CUDA storage_qtype=Q8_0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "startup.*missing"):
                self._capture(base, server_log_path=bad_log)
            self.assertFalse((base / "evidence" / "run_manifest.json").exists())

    def test_capture_rejects_startup_model_or_mmproj_path_mismatch_before_writing(self):
        for field, wrong_name in (
            ("model_path", "other-policy.gguf"),
            ("mmproj_path", "other-mmproj.gguf"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as scratch:
                base = Path(scratch)
                inputs = self._inputs(base)
                log = inputs["server_log_path"]
                text = log.read_text(encoding="utf-8")
                actual = (
                    str(inputs["quantized_policy_path"].resolve())
                    if field == "model_path"
                    else str(inputs["mmproj_path"].resolve())
                )
                bad_log = base / f"bad-{field}.log"
                bad_log.write_text(
                    text.replace(actual, str((base / wrong_name).resolve())),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, field):
                    self._capture(base, server_log_path=bad_log)
                self.assertFalse((base / "evidence" / "run_manifest.json").exists())

    def test_explicit_repository_identity_bypasses_unusable_linked_worktree_git(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            manifest = self._capture(
                base,
                repo_root=base / "not-a-git-repository",
                repository_identity=self._repository_override(),
            )
            self.assertEqual(
                manifest["provenance"]["repository"], self._repository_override()
            )

    def test_explicit_repository_identity_is_all_or_nothing_and_strict(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            partial = self._repository_override()
            del partial["dirty_state"]
            with self.assertRaisesRegex(ValueError, "repository.*dirty_state"):
                self._capture(base, repository_identity=partial)
            self.assertFalse((base / "evidence" / "run_manifest.json").exists())
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            malformed = self._repository_override()
            malformed["commit"] = "not-a-commit"
            with self.assertRaisesRegex(ValueError, "repository.*commit"):
                self._capture(base, repository_identity=malformed)
            self.assertFalse((base / "evidence" / "run_manifest.json").exists())

    def test_cli_exposes_all_repository_identity_overrides(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "capture_smolvla_object_evidence.py"),
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--repo-remote",
            "--repo-commit",
            "--repo-branch",
            "--repo-dirty-state",
        ):
            self.assertIn(option, completed.stdout)

    def test_capture_rejects_missing_required_identity_before_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            with self.assertRaisesRegex(ValueError, "build.*gpu"):
                self._capture(base, build_identity={
                    "type": "Release",
                    "cmake_flags": "-DGGML_CUDA=ON",
                    "cuda_architecture": "sm_86",
                    "compiler": "gcc 13",
                    "cuda": "12.8",
                    "driver": "570.00",
                })
            self.assertFalse((base / "evidence" / "run_manifest.json").exists())

    def test_capture_refuses_overwrite_and_profile_path_escape(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            self._capture(base)
            with self.assertRaises(FileExistsError):
                self._capture(base)
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            with self.assertRaisesRegex(ValueError, "profile.*escapes"):
                self._capture(base, profile_path="../profile.json")


if __name__ == "__main__":
    unittest.main()
