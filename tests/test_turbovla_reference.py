"""Reference-client preprocessing and replay, without model weights or CUDA."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("turbo_reference", ROOT / "scripts/rollout_turbovla_reference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReferenceTests(unittest.TestCase):
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
        instructions,pixels,state=model.calls[0]
        self.assertEqual(instructions,["pick"])
        self.assertEqual(tuple(pixels.shape),(1,2,3,256,256))
        np.testing.assert_allclose(pixels[0,0,:,0,0].numpy(), -np.array([.485,.456,.406])/np.array([.229,.224,.225]),rtol=1e-6)
        np.testing.assert_allclose(state.numpy(),np.ones((1,8))/(1+5e-7),rtol=1e-6)
        self.assertFalse(client.has_queued_action())
        client.get_action(obs)
        self.assertEqual(len(model.calls),2)
        client.reset()
        self.assertFalse(client.has_queued_action())
        obs["observation.images.image"] = np.zeros((3,128,128),dtype=np.float32)
        with self.assertRaises(ValueError):
            client.get_action(obs)


if __name__ == "__main__":
    unittest.main()
