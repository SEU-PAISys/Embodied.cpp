"""Reference-client preprocessing and replay, without model weights or CUDA."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("turbo_reference", ROOT / "scripts/rollout_turbovla_reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceTests(unittest.TestCase):
    def test_loader_does_not_autocast_fp32_vision(self):
        import parity_turbovla_reference as loader
        for precision in ("fp32", "bf16"):
            config = SimpleNamespace(text=SimpleNamespace(),
                                     vision=SimpleNamespace(image_size=256))
            config_type = MagicMock()
            config_type.from_mapping.return_value = config
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = model
            model.requires_grad_.return_value = model
            modules = {
                "transformers": SimpleNamespace(BertConfig=MagicMock(), BertModel=MagicMock()),
                "transformers.models.dinov3_vit": SimpleNamespace(
                    DINOv3ViTConfig=MagicMock(), DINOv3ViTModel=MagicMock()),
                "turbovla.models": SimpleNamespace(
                    text_encoder=SimpleNamespace(_load_pretrained_model=None),
                    vision_encoder=SimpleNamespace(_load_pretrained_model=None)),
                "turbovla.models.configuration": SimpleNamespace(TurboVLAConfig=config_type),
                "turbovla.models.turbovla": SimpleNamespace(build_turbovla=lambda _: model),
            }
            args = SimpleNamespace(official_root=Path("official"), checkpoint="weights",
                                   checkpoint_key="raw", bert_path=Path("bert"),
                                   precision=precision, device="cpu")
            with patch.dict(sys.modules, modules), \
                 patch.object(loader, "provide_eval_only_timm_shim"), \
                 patch.object(loader.torch, "load", return_value={"raw": {}, "model_config": {}}):
                _, actual = loader.load_reference_model(args)
            self.assertEqual(actual.vision.compute_precision, precision)

    def test_public_python_entry(self):
        from client import run_sim_client_direct as runner
        from scripts import rollout_turbovla_reference as reference
        args = runner.parse_args([
            "--arch", "turbovla", "--implementation", "python",
            "--turbovla-checkpoint", "weights.pth", "--turbovla-official-root", "official",
            "--turbovla-bert-path", "bert", "--turbovla-norm-gguf", "norm.gguf",
            "--turbovla-precision", "fp32", "--n-action-steps", "6",
        ])
        model, norm, client = MagicMock(), MagicMock(), MagicMock()
        with patch.object(reference, "load_reference_model", return_value=(model, None)) as load, \
             patch.object(reference, "load_norm_arrays", return_value=norm), \
             patch.object(reference, "TurboReferenceClient", return_value=client) as construct, \
             patch.object(runner, "LIBEROSimAdapter", side_effect=lambda value: value):
            self.assertIs(runner.build_client(args), client)
            self.assertEqual(load.call_args.args[0].precision, "fp32")
            self.assertEqual(load.call_args.args[0].checkpoint, Path("weights.pth"))
            construct.assert_called_once_with(model, norm, precision="fp32", n_action_steps=6)
            args.turbovla_checkpoint = None
            with self.assertRaisesRegex(ValueError, "--turbovla-checkpoint"):
                runner.build_client(args)

    def test_preprocessing_replay_and_reset(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = []

            def forward(self, instructions, samples, state):
                self.calls.append((instructions, samples["dinov3"].clone(), state.clone()))
                return torch.zeros((1, 12, 7), dtype=state.dtype)

        model = Model()
        client = module.TurboReferenceClient(model, (np.ones(8), np.ones(8)*2, -np.ones(7), np.ones(7)), "fp32")
        obs = {"observation.images.image":np.zeros((3,256,256),dtype=np.float32),
               "observation.images.image2":np.zeros((3,256,256),dtype=np.float32),
               "observation.state":np.ones(8,dtype=np.float32)*3, "task":"pick"}
        for _ in range(12):
            np.testing.assert_array_equal(client.get_action(obs), [0,0,0,0,0,0,1])
        self.assertEqual(len(model.calls),1)
        profile = client.get_last_inference_profile()
        self.assertEqual(profile["sequence"], 1)
        self.assertIsNone(profile["server_vision_ms"])
        self.assertGreaterEqual(profile["server_total_ms"], 0)
        profile["sequence"] = 999
        self.assertEqual(client.get_last_inference_profile()["sequence"], 1)
        instructions,pixels,state=model.calls[0]
        self.assertEqual(instructions,["pick"])
        self.assertEqual(tuple(pixels.shape),(1,2,3,256,256))
        np.testing.assert_allclose(pixels[0,0,:,0,0].numpy(), -np.array([.485,.456,.406])/np.array([.229,.224,.225]),rtol=1e-6)
        np.testing.assert_allclose(state.numpy(),np.ones((1,8))/(1+5e-7),rtol=1e-6)
        self.assertFalse(client.has_queued_action())
        client.get_action(obs)
        self.assertEqual(len(model.calls),2)
        self.assertEqual(client.get_last_inference_profile()["sequence"], 2)
        client.reset()
        self.assertIsNone(client.get_last_inference_profile())
        self.assertFalse(client.has_queued_action())
        obs["observation.images.image"] = np.zeros((3,128,128),dtype=np.float32)
        with self.assertRaises(ValueError):
            client.get_action(obs)
        self.assertIsNone(client.get_last_inference_profile())

    def test_invalid_replay_and_precision(self):
        for precision, steps in (("fp16", 12), ("fp32", 0), ("bf16", 13)):
            with self.assertRaises(ValueError):
                module.TurboReferenceClient(None, None, precision, steps)


if __name__ == "__main__":
    unittest.main()
