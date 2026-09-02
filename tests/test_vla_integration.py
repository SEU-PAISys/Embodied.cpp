"""Shared VLA wiring regressions; CPU only, no tokenizer downloads or server.

Run in the eval environment:
    python -m unittest discover -s tests -p test_vla_integration.py -v
"""
from __future__ import annotations

from collections import deque
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))

from adapter.sim.libero import LIBERO_PARSER_REGISTRY
from adapter.sim.libero import LIBEROSimAdapter
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
        # XR0 requires a square camera; a one-sided override that makes the
        # frame rectangular is rejected at the CLI (its runtime grid derives
        # from img_w alone and would overflow on rectangular input).
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--conf", "libero_xr0_eval.yaml", "--observation-width", "320"])
        # One-sided overrides still apply to models that allow rectangular renders.
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "rect.yaml"
            config.write_text("arch: pi05\nlibero_suite: object\ntask_ids: [0]\n"
                              "observation_size: 256\n", encoding="utf-8")
            args = runner.parse_args(["--conf", str(config), "--observation-width", "320"])
            self.assertEqual((args.observation_width, args.observation_height), (320, 256))
        for flags in (["--arch", "xr0", "--observation-size", "360"],
                      ["--conf", "libero_xr0_eval.yaml", "--observation-width", "320"],
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

    def test_cli_task_overrides_yaml_suite(self):
        # Explicit --task / --libero-suite on the CLI must override the YAML
        # libero_suite; without them the YAML default applies.
        args = runner.parse_args(["--conf", "libero_xr0_eval.yaml", "--task", "goal", "--task-id", "0"])
        self.assertEqual(args.task, "libero_goal")
        args = runner.parse_args(["--conf", "libero_xr0_eval.yaml", "--libero-suite", "goal", "--task-id", "0"])
        self.assertEqual(args.task, "libero_goal")
        args = runner.parse_args(["--conf", "libero_xr0_eval.yaml", "--task-id", "0"])
        self.assertEqual(args.task, "libero_object")

    def test_no_yaml_defaults_come_from_presets(self):
        # Every arch that has a preset must fall back to its own preset when
        # launched without a YAML config (regression: SmolVLA/GR00T used to
        # fall into a 512/1 catch-all and diverge from their presets).
        for arch, want_length, want_replay, extra in (
            ("pi05", 200, 10, []), ("smolvla", 48, 1, []), ("groot_n1", 1024, 8, []),
            ("xr0", 512, 10, ["--observation-size", "256"]),  # xr0 needs an explicit square camera
            ("turbovla", 64, 12, []), ("xvla", 50, 30, []),
        ):
            with self.subTest(arch=arch):
                args = runner.parse_args(
                    ["--arch", arch, "--task", "object", "--task-id", "0"] + extra)
                with patch.object(runner, "VlaCppClient") as vc, \
                     patch.object(runner, "LIBEROSimAdapter", return_value=MagicMock()):
                    runner.build_client(args)
                self.assertEqual(vc.call_args.kwargs["max_length"], want_length)
                self.assertEqual(vc.call_args.kwargs["n_action_steps"], want_replay)

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


class _RecordingProfiler:
    """Records the profiler contract without computing a report.

    The real LiberoSuiteProfiler.result() uses zip(strict=True) and needs
    Python 3.10+; these cases run on the 3.9 + torch CPU host as well, so
    they assert the call contract (one record_episode per episode) here.
    test_eval_results.py covers the real profiler's numbers on 3.10+.
    """

    def __init__(self):
        self.episodes = []

    def capture_inference(self, client):
        pass

    def record_step(self, ms):
        pass

    def record_episode(self, **kwargs):
        self.episodes.append(kwargs)


class AdapterCompatibilityTests(unittest.TestCase):
    """Every LIBERO parser registered by the shared adapter must keep the
    action interface the public rollout path calls (regression: GR00T's
    parse_action was dropped during the three-model integration)."""

    def test_all_registered_parsers_implement_parse_action(self):
        missing = [name for name, cls in LIBERO_PARSER_REGISTRY.items()
                   if not callable(getattr(cls, "parse_action", None))]
        self.assertEqual(missing, [])

    def test_groot_parse_action_first_action_and_queue_replay(self):
        client = MagicMock()
        client.get_arch.return_value = "groot_n1"
        client.has_queued_action.return_value = False
        # 7-D delta action; gripper channel is an open/close value.
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], dtype=np.float32)
        client.get_action.return_value = raw
        adapter = LIBEROSimAdapter(client)
        obs = {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8),
                          "image2": np.zeros((8, 8, 3), dtype=np.uint8)},
               "robot_state": {"eef": {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]},
                               "gripper": {"qpos": [0.0, 0.0]}},
               "task_description": "pick up the bowl"}
        action = adapter.get_action(obs)          # first action: full path
        self.assertEqual(action.shape, (7,))
        np.testing.assert_allclose(action[:6], raw[:6], atol=1e-6)
        self.assertEqual(action[6], -1.0)         # -sign(2*0.8 - 1) = -1
        client.get_action.assert_called_once()
        # Queue replay path must apply the same action conversion.
        client.has_queued_action.return_value = True
        client.get_action_from_queue.return_value = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9], dtype=np.float32)
        replayed = adapter.get_action(obs)
        self.assertEqual(replayed[6], 1.0)        # -sign(2*(-0.9) - 1) = 1
        self.assertEqual(client.get_action.call_count, 1)  # queue: no new inference


class EpisodeAccountingTests(unittest.TestCase):
    """Aborted (terminated mid-step) episodes must be recorded exactly once."""

    def _run_task(self, abort_flags, arch="xvla", real_profiler=False):
        """Run the real run_one_task against a fake env; flags say which abort."""
        client, _obs = fake_client(arch)
        workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workdir, True)
        argv = ["--arch", arch, "--task", "object", "--task-id", "0",
                "--n-episodes", str(len(abort_flags)), "--output-dir", workdir]
        if arch == "xr0":
            argv += ["--observation-size", "256"]
        args = runner.parse_args(argv)
        if real_profiler:
            profiler = LiberoSuiteProfiler(
                output_path=Path(workdir) / "profile.json", model_label=arch,
                backbone_label="test", arch=arch, suite="libero_object",
                replay_chunk_size=client.n_action_steps,
                expected_episodes=len(abort_flags),
                server_address="tcp://localhost:5555", server_pid=1,
                vram_interval_s=0.25, warmup_requests=0,
            )
        else:
            profiler = _RecordingProfiler()
        state = {"episode": -1}

        class FakeEnv:
            def reset(self, **kwargs):
                state["episode"] += 1
                return _obs, {"is_success": False}

            def step(self, action):
                if abort_flags[state["episode"]]:
                    raise ValueError("terminated episode: reset() before step()")
                return _obs, 1.0, True, False, {"is_success": True}

            def close(self):
                pass

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = lambda name, **kwargs: FakeEnv()
        fake_sim = types.ModuleType("sim")
        fake_libero = types.ModuleType("sim.libero")
        fake_sim.libero = fake_libero
        with patch.dict(sys.modules, {"gymnasium": fake_gym, "sim": fake_sim,
                                      "sim.libero": fake_libero}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_one_task(args, client, "libero_object", 0, profiler)
        summary = (Path(workdir) / arch / "libero_object" / "task_0" / "summary.txt").read_text(
            encoding="utf-8")
        return result, profiler, summary

    def test_all_episodes_aborted(self):
        result, profiler, summary = self._run_task([True, True])
        n = 2
        self.assertEqual(len(result["episodes"]), n, "every episode needs a detail record")
        self.assertEqual(len(profiler.episodes), n)
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["episodes_counted"], 0)
        self.assertIn("(0/0)", summary)
        self.assertTrue(all(e["skipped"] for e in result["episodes"]))

    def test_success_then_aborted(self):
        result, profiler, summary = self._run_task([False, True])
        self.assertEqual(len(result["episodes"]), 2)
        self.assertEqual(len(profiler.episodes), 2)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["episodes_counted"], 1)
        self.assertEqual(result["successes"], 1)
        self.assertIn("(1/1)", summary)
        self.assertEqual([e["skipped"] for e in result["episodes"]], [False, True])

    @unittest.skipUnless(sys.version_info >= (3, 10),
                         "LiberoSuiteProfiler.result() needs Python 3.10+ (zip strict)")
    def test_aborted_episodes_reach_the_real_profiler(self):
        _, profiler, _summary = self._run_task([False, True], real_profiler=True)
        self.assertEqual(len(profiler.episodes), 2)
        self.assertEqual([e["skipped"] for e in profiler.episodes], [False, True])

    def test_lingbot_aborted_stops_inference(self):
        # Regression: the LingBot branch used to only break out of its inner
        # frame loop on 'terminated episode', so the episode never finalized
        # and inference kept being requested. Each episode must request at
        # most one chunk and be recorded exactly once with skipped=True.
        client = MagicMock()
        client.image_keys = ["image", "image2"]
        client._last_response = SimpleNamespace(latency_ms_inference=0.0)
        client.predict_chunk.return_value = np.zeros((16, 7), dtype=np.float32)
        workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workdir, True)
        args = runner.parse_args([
            "--arch", "lingbot_va", "--task", "object", "--task-id", "0",
            "--n-episodes", "2", "--output-dir", workdir,
            "--lingbot-noise-mode", "zero",
        ])
        profiler = _RecordingProfiler()
        state = {"episode": -1, "aborts": 0}

        class FakeEnv:
            def reset(self, **kwargs):
                state["episode"] += 1
                return {"pixels": {}, "task_description": "pick"}, {"is_success": False}

            def step(self, action):
                # Both episodes abort on their first env step; a runaway loop
                # would keep hitting this, so fail fast past a sane bound.
                state["aborts"] += 1
                if state["aborts"] > 8:
                    raise RuntimeError("runaway episode loop after abort")
                raise ValueError("terminated episode: reset() before step()")

            def close(self):
                pass

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = lambda name, **kwargs: FakeEnv()
        fake_sim = types.ModuleType("sim")
        fake_libero = types.ModuleType("sim.libero")
        fake_sim.libero = fake_libero
        with patch.dict(sys.modules, {"gymnasium": fake_gym, "sim": fake_sim,
                                      "sim.libero": fake_libero}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_one_task(args, client, "libero_object", 0, profiler)
        self.assertEqual(client.predict_chunk.call_count, 2,
                         "one chunk per episode; no inference after an abort")
        self.assertEqual(len(result["episodes"]), 2)
        self.assertTrue(all(e["skipped"] for e in result["episodes"]))
        self.assertEqual(len(profiler.episodes), 2)
        self.assertTrue(all(e["skipped"] for e in profiler.episodes))
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["episodes_counted"], 0)


class BenchCliTests(unittest.TestCase):
    def test_bench_models_reports_total_and_unmeasured_n_a(self):
        # Run bench_models.main() against a fake client: output must include
        # the total latency and n/a for phases turbovla cannot measure (the
        # NameError regression: ARCH_PRESETS was used but not imported).
        spec = importlib.util.spec_from_file_location(
            "bench_models", str(REPO / "eval" / "client" / "bench_models.py"))
        bm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bm)
        fake = MagicMock()
        fake.preset_chunk, fake.max_state_dim = 12, 8
        fake._predict_chunk.return_value = np.zeros((12, 7), dtype=np.float32)
        fake._last_response = SimpleNamespace(
            latency_ms_total=10.0, latency_ms_inference=8.0,
            latency_ms_prefill=0.0, latency_ms_denoise=0.0, latency_ms_vision=0.0)
        out = io.StringIO()
        with patch.object(bm, "VlaCppClient", return_value=fake), \
             patch.object(sys, "argv", ["bench", "turbovla", "1"]), \
             contextlib.redirect_stdout(out):
            bm.main()
        text = out.getvalue()
        self.assertIn("ARCH=turbovla", text)
        self.assertIn("total", text)
        self.assertIn("mean=", text)                 # measured phases printed
        self.assertIn("n/a", text)                   # vision/prefill/denoise


class ParityCliTests(unittest.TestCase):
    """parity_turbovla_cpp must accept the final action without intermediate
    stage dumps (the stock runtime emits none), keep the over-tolerance
    failure, and only require stage files in explicit --stages mode."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "parity_turbovla_cpp", str(REPO / "scripts" / "parity_turbovla_cpp.py"))
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def _make_fixture(self, reference_delta: float):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        ref = np.linspace(0.0, 1.0, 12 * 7, dtype=np.float32).reshape(12, 7)
        np.save(tmp / "turbovla_parity_ref_env.npy", ref)
        actual = ref + reference_delta
        return tmp, ref, actual

    def test_passes_without_stage_dumps(self):
        tmp, ref, actual = self._make_fixture(1e-4)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.mod.check_parity(actual, tmp, atol=0.01)
        self.assertTrue(result)
        self.assertIn("PASS", out.getvalue())
        self.assertFalse((tmp / "turbovla_parity_ref_stages.npz").exists())

    def test_over_tolerance_fails(self):
        tmp, ref, actual = self._make_fixture(0.5)
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(SystemExit, "FAIL"):
            self.mod.check_parity(actual, tmp, atol=0.01)

    def test_stages_requested_but_missing_fails_explicitly(self):
        tmp, ref, actual = self._make_fixture(1e-4)
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(SystemExit, "not found"):
            self.mod.check_parity(actual, tmp, atol=0.01, compare_stages=True)

    def test_reference_with_nan_fails(self):
        tmp, ref, actual = self._make_fixture(0.0)
        ref = ref.copy()
        ref[0, 0] = np.nan
        np.save(tmp / "turbovla_parity_ref_env.npy", ref)
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(SystemExit, "reference contains non-finite"):
            self.mod.check_parity(actual, tmp, atol=0.01)

    def test_nan_atol_fails(self):
        tmp, ref, actual = self._make_fixture(0.0)
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(SystemExit, "atol must be finite"):
            self.mod.check_parity(actual, tmp, atol=float("nan"))

    def test_actual_with_nan_fails(self):
        tmp, ref, actual = self._make_fixture(0.0)
        actual[0, 0] = np.nan
        with contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(SystemExit, "FAIL"):
            self.mod.check_parity(actual, tmp, atol=0.01)


class SafetyScriptTests(unittest.TestCase):
    def test_aborts_when_temp_dir_cannot_be_created(self):
        # Point TMPDIR at an unusable path so mktemp -d fails: the script
        # must exit non-zero with a clear message and must not fall through
        # to file operations under an empty SAFE_TMP (which would otherwise
        # write/delete under the filesystem root).
        env = dict(os.environ, TMPDIR=os.path.join(tempfile.gettempdir(), "wb_nonexistent_xyz"))
        proc = subprocess.run(
            ["bash", str(REPO / "tests" / "check_setup_libero_safety.sh")],
            capture_output=True, env=env)
        # rc == 1 (not 0) proves the script aborted before running any of its
        # scenarios under an empty SAFE_TMP; a stderr message must be emitted.
        # (Message bytes are transcoded by MSYS on Windows pipes, so we only
        # assert non-empty rather than the exact text here.)
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue(proc.stderr.strip())


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

    def test_unmeasured_phases_report_none_others_keep_values(self):
        # TurboVLA declares vision/prefill/denoise unmeasured (fused single
        # graph): its profile must carry None so aggregators show N/A. Other
        # architectures keep their measured values untouched (even a real 0).
        for arch, field, want in (
            ("turbovla", "server_vision_ms", None),
            ("turbovla", "server_prefill_ms", None),
            ("turbovla", "server_inference_ms", 8.0),   # fused-graph execution
            ("xr0", "server_vision_ms", 2.0),
        ):
            with self.subTest(arch=arch, field=field):
                client, obs = fake_client(arch)
                client._predict_chunk(obs)
                profile = client.get_last_inference_profile()
                self.assertEqual(profile[field], want)

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
