from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import gguf  # noqa: F401 - required by the converter modules loaded below
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "converter tests require third_party/llama.cpp/gguf-py on PYTHONPATH"
    ) from exc

REPO = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


policy = load_script("convert_smolvla_to_gguf", "convert_smolvla_to_gguf.py")
mmproj = load_script("convert_smolvla_mmproj_to_gguf", "convert_smolvla_mmproj_to_gguf.py")


class FakeWriter:
    def __init__(self) -> None:
        self.tensors = []

    def add_tensor(self, name, data, **kwargs) -> None:
        self.tensors.append((name, data, kwargs))


class ConverterTests(unittest.TestCase):
    def test_policy_rejects_mismatched_vlm_and_expert_layers(self) -> None:
        with self.assertRaises(SystemExit):
            policy._validate_layer_counts(32, 31)

    def test_policy_accepts_matching_vlm_and_expert_layers(self) -> None:
        policy._validate_layer_counts(32, 32)

    def test_policy_tensor_preserves_f32(self) -> None:
        writer = FakeWriter()
        tensor = torch.tensor([[1.25, -2.5]], dtype=torch.float32)
        policy._add_one_tensor(writer, "x", tensor)
        name, data, kwargs = writer.tensors[0]
        self.assertEqual(name, "x")
        np.testing.assert_array_equal(data, tensor.numpy())
        self.assertEqual(kwargs["raw_dtype"], policy.gguf.GGMLQuantizationType.F32)

    def test_policy_tensor_exports_bf16_raw_shape(self) -> None:
        writer = FakeWriter()
        tensor = torch.ones((2, 3), dtype=torch.bfloat16)
        policy._add_one_tensor(writer, "x", tensor)
        _, data, kwargs = writer.tensors[0]
        self.assertEqual(data.dtype, np.uint16)
        self.assertEqual(kwargs["raw_shape"], [2, 3])
        self.assertEqual(kwargs["raw_dtype"], policy.gguf.GGMLQuantizationType.BF16)

    def test_policy_rejects_unexpected_source_dtype(self) -> None:
        with self.assertRaises(NotImplementedError):
            policy._add_one_tensor(FakeWriter(), "x", torch.ones(2, dtype=torch.float16))

    def test_mmproj_f32_override_uses_f32(self) -> None:
        writer = FakeWriter()
        mmproj._add_tensor(writer, "patch", torch.ones(2, dtype=torch.float32))
        _, data, kwargs = writer.tensors[0]
        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(kwargs["raw_dtype"], mmproj.gguf.GGMLQuantizationType.F32)


if __name__ == "__main__":
    unittest.main()
