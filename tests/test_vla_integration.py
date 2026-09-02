"""Shared VLA wiring regressions; CPU only, no tokenizer downloads or server.

Run in the eval environment:
    python -m unittest discover -s tests -p test_vla_integration.py -v
"""
from __future__ import annotations

from collections import deque
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from adapter.sim.libero import LIBERO_PARSER_REGISTRY
from client import run_sim_client_direct as runner
from client.libero_profile import LiberoSuiteProfiler
from client.run_libero_eval_xr0 import shared_arguments
from client.vla_cpp_client import ARCH_PRESETS, VlaCppClient


def fake_client(arch):
    """Exercise real preprocessing and response code with only transport mocked."""
    preset = ARCH_PRESETS[arch]
    client = VlaCppClient.__new__(VlaCppClient)
    client.arch = arch
    for key in ("image_size", "max_state_dim", "max_length", "n_action_steps"):
        setattr(client, key, preset[key])
    client.image_keys = ["observation.images.image", "observation.images.image2"]
    client.real_action_dim = 10 if arch == "xvla" else 7
    client.preset_chunk = preset.get("chunk", 12)
    client.use_server_tokenizer = preset.get("use_server_tokenizer", False)
    client._step = client._inference_sequence = 0
    client._initial_noise_seed = client._noise_rng = None
    client.noise_chunk_size = client.noise_action_dim = 0
    client._last_response = client._last_inference_profile = None
    client._action_queue = deque(maxlen=client.n_action_steps)
    client.tok = lambda *args, **kwargs: {"input_ids": np.array([[0, 10, 2]])}
    request = SimpleNamespace(
        images=SimpleNamespace(add=lambda: SimpleNamespace()),
        lang_tokens=[], attention_mask=[], state=[], noise=[],
        SerializeToString=lambda: b"request",
    )
    chunk_size, action_dim = {"xr0": (30, 32), "xvla": (30, 20),
                              "turbovla": (12, 7)}[arch]
    response = SimpleNamespace(
        error="", request_id=0, chunk_size=chunk_size, action_dim=action_dim,
        action_chunk=np.zeros(chunk_size * action_dim, dtype=np.float32),
        latency_ms_total=10, latency_ms_vision=2, latency_ms_inference=8,
        latency_ms_prefill=3, latency_ms_denoise=5,
        ParseFromString=lambda body: None,
    )
    client.pb = SimpleNamespace(
        PredictRequest=lambda: request, PredictResponse=lambda: response,
        Image=SimpleNamespace(RGB_U8=0, F32_RGB_01=1),
    )
    client.sock = MagicMock()
    obs = {key: np.zeros((3, 256, 256), dtype=np.float32) for key in client.image_keys}
    obs.update({"observation.state": np.zeros(client.max_state_dim, dtype=np.float32),
                "task": "pick up the bowl", "domain_id": 3,
                "action_noise": np.zeros((chunk_size, client.max_state_dim), dtype=np.float32)})
    return client, obs


class ConfigTests(unittest.TestCase):
    def test_all_architectures_have_profile_labels(self):
        self.assertEqual(set(runner.ARCH_CHOICES), set(runner.PROFILE_LABELS))

    def test_new_configs_and_profile_initialization(self):
        for arch in ("xr0", "turbovla", "xvla"):
            with self.subTest(arch=arch):
                argv = ["--conf", f"libero_{arch}_eval.yaml", "--task-id", "0"]
                args = runner.parse_args(argv)
                self.assertEqual((args.observation_width, args.observation_height), (256, 256))
                self.assertEqual(runner.resolve_task_ids(args), [0])
                self.assertTrue(args.profile_output)
                with patch.object(runner, "build_client"), \
                     patch.object(runner, "run_one_task") as rollout, \
                     patch.object(runner, "LiberoSuiteProfiler") as profiler, \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.main(argv)
                self.assertEqual(profiler.call_args.kwargs["arch"], arch)
                self.assertEqual(profiler.call_args.kwargs["model_label"], args.profile_model_label)
                rollout.assert_called_once()
                profiler.return_value.write.assert_called_once_with(complete=True)

    def test_render_overrides_and_old_default(self):
        self.assertEqual(runner.parse_args(["--arch", "pi05"]).observation_width, 360)
        args = runner.parse_args(["--conf", "libero_xr0_eval.yaml", "--observation-size", "320"])
        self.assertEqual((args.observation_width, args.observation_height), (320, 320))
        args = runner.parse_args(["--conf", "libero_xr0_eval.yaml", "--observation-width", "320"])
        self.assertEqual((args.observation_width, args.observation_height), (320, 256))
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "square.yaml"
            config.write_text("arch: xr0\nobservation_size: 256\n", encoding="utf-8")
            args = runner.parse_args(["--conf", str(config), "--observation-width", "320"])
            self.assertEqual((args.observation_width, args.observation_height), (320, 256))
        for flags in (["--arch", "xr0", "--observation-size", "360"],
                      ["--observation-size", "256", "--observation-width", "320"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(flags)

    def test_legacy_wrapper_uses_shared_config_and_protocol(self):
        args = runner.parse_args(shared_arguments([
            "--tokenizer", "local-snapshot", "--task", "libero_goal/task_3",
            "--host", "127.0.0.1", "--port", "5556", "--replan-steps", "5",
            "--num-steps-wait", "7",
        ]))
        self.assertEqual(args.task, "libero_goal")
        self.assertEqual(args.task_id, 3)
        self.assertEqual(args.n_action_steps, 5)
        self.assertEqual(args.num_steps_wait, 7)
        self.assertEqual(args.vla_addr, "tcp://127.0.0.1:5556")
        self.assertEqual(args.observation_width, 256)

    def test_standalone_tools_import_without_pythonpath(self):
        # Each child gets a clean import path; tests in this process otherwise
        # mask wrong sys.path setup in standalone entry points.
        for name in ("eval/client/bench_models.py", "eval/client/bench_xr0.py",
                     "scripts/parity_turbovla_cpp.py"):
            with self.subTest(script=name):
                code = "import runpy,sys; runpy.run_path(sys.argv[1], run_name='import_test')"
                result = subprocess.run([sys.executable, "-I", "-c", code, str(REPO / name)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


class ClientTests(unittest.TestCase):
    def test_success_profiles_and_queue_replay_for_all_three_models(self):
        for arch in ("xr0", "turbovla", "xvla"):
            with self.subTest(arch=arch):
                client, obs = fake_client(arch)
                profiler = LiberoSuiteProfiler(
                    output_path=Path("unused.json"), model_label=arch, backbone_label="test",
                    arch=arch, suite="libero_object", replay_chunk_size=client.n_action_steps,
                    expected_episodes=1, server_address="tcp://localhost:5555", server_pid=1,
                    vram_interval_s=0.25, warmup_requests=0,
                )
                action = client.get_action(obs)
                self.assertEqual(action.shape, (client.real_action_dim,))
                self.assertEqual(action.dtype, np.float32)
                profile = client.get_last_inference_profile()
                self.assertEqual(profile["sequence"], 1)
                self.assertEqual(profile["server_total_ms"], 10)
                self.assertEqual(profile["model_chunk_size"], client.preset_chunk)
                profiler.capture_inference(client)
                for _ in range(client.n_action_steps - 1):
                    client.get_action(obs)
                    profiler.capture_inference(client)
                self.assertEqual(len(profiler.server_requests), 1)
                client.sock.send.assert_called_once()
                client.get_action(obs)
                profiler.capture_inference(client)
                self.assertEqual(len(profiler.server_requests), 2)
                self.assertEqual(client.get_last_inference_profile()["sequence"], 2)
                client.reset()
                self.assertIsNone(client.get_last_inference_profile())

    def test_failed_responses_are_not_counted(self):
        client, obs = fake_client("xr0")
        response = client.pb.PredictResponse()
        response.error = "invalid domain"
        with self.assertRaisesRegex(RuntimeError, "invalid domain"):
            client._predict_chunk(obs)
        self.assertIsNone(client.get_last_inference_profile())
        response.error = ""
        response.action_chunk = []
        with self.assertRaises(ValueError):
            client._predict_chunk(obs)
        self.assertEqual(client._inference_sequence, 0)

    def test_xr0_rejects_invalid_camera_size(self):
        client, obs = fake_client("xr0")
        obs[client.image_keys[0]] = np.zeros((3, 360, 360), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            client._predict_chunk(obs)
        client.sock.send.assert_not_called()

    def test_typed_state_matches_each_models_compatibility_view(self):
        pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        obs = {"pixels": {"image": pixels, "image2": pixels},
               "robot_state": {"eef": {"pos": [1, 2, 3], "quat": [0, 0, 0, 1],
                                          "mat": np.eye(3)}, "gripper": {"qpos": [0, 0]}},
               "task_description": "pick"}
        for arch, dim in (("pi05", 8), ("xr0", 8), ("turbovla", 8), ("xvla", 20)):
            with self.subTest(arch=arch):
                parsed = LIBERO_PARSER_REGISTRY[arch]().parse_embodied_observation(obs)
                state = parsed.proprioception.data
                self.assertEqual(state.shape, (dim,))
                self.assertEqual(state.dtype, np.float32)
                np.testing.assert_array_equal(state, parsed.model_inputs["observation.state"])
                wrist = parsed.model_inputs["observation.images.image2"]
                expected = pixels if arch == "xvla" else pixels[::-1, ::-1]
                np.testing.assert_allclose(wrist, expected.transpose(2, 0, 1) / 255, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
