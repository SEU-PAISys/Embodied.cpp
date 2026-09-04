"""Q8/Q4 checks need NumPy; Q6 also exercises the shared torch/GGML helper."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quantize_vla", ROOT / "scripts/quantize_vla_gguf.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
gguf = module.gguf


class QuantizeTests(unittest.TestCase):
    def fixture(self, path, dtype, nonfinite=False, width=32, arch="turbovla"):
        values = np.linspace(-1, 1, 128 * width, dtype=np.float32).reshape(128, width)
        if nonfinite:
            values[0, 0] = np.nan
        writer = gguf.GGUFWriter(str(path), arch=arch)
        writer.add_array("turbovla.pad_layout_instr", ["short", "long"])
        writer.add_key_value("turbovla.pad_layout_len", [11, 21], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
        writer.add_key_value("test.u32", [1, 4294967295], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        writer.add_tensor("vit.blk.0.q.w", gguf.quantize(values, dtype), raw_dtype=dtype)
        writer.add_tensor("text.token_emb", np.ones((32, 64), dtype=np.float32))
        if width == 256:
            writer.add_tensor("fusion.w", np.ones((128, 32), dtype=np.float32))
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        return values

    def test_metadata_shape_and_values_preserved(self):
        for dtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
            for outtype in ("q8_0", "q4_0"):
                with self.subTest(dtype=dtype, outtype=outtype), tempfile.TemporaryDirectory() as scratch:
                    source, output = Path(scratch)/"source.gguf", Path(scratch)/"output.gguf"
                    values = self.fixture(source, dtype)
                    self.assertEqual(module.quantize_file(source, output, outtype), 1)
                    before, after = gguf.GGUFReader(str(source)), gguf.GGUFReader(str(output))
                    for key in ("general.architecture", "turbovla.pad_layout_instr", "turbovla.pad_layout_len", "test.u32"):
                        self.assertEqual(before.fields[key].types, after.fields[key].types)
                        self.assertEqual(before.fields[key].contents(), after.fields[key].contents())
                    self.assertEqual(after.fields["general.file_type"].contents(), module.FILE_TYPES[outtype])
                    a, b = after.tensors
                    np.testing.assert_array_equal(a.shape, [32, 128])
                    self.assertEqual(a.tensor_type, module.QTYPES[outtype])
                    np.testing.assert_allclose(gguf.dequantize(a.data, a.tensor_type), values, atol=.08)
                    np.testing.assert_array_equal(b.data, before.tensors[1].data)
                    np.testing.assert_array_equal(b.shape, before.tensors[1].shape)

    def test_q6_native_codec_and_unaligned_matrix(self):
        lib = Path(os.environ.get("GGML_TEST_LIB", ROOT / "build/bin/libggml-base.so"))
        if not lib.is_file():
            self.skipTest("Q6 integration requires libggml-base; set GGML_TEST_LIB")
        for arch in ("turbovla", "xvla"):
            for dtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
                with self.subTest(arch=arch, dtype=dtype), tempfile.TemporaryDirectory() as scratch:
                    source, output = Path(scratch)/"source.gguf", Path(scratch)/"output.gguf"
                    values = self.fixture(source, dtype, width=256, arch=arch)
                    self.assertEqual(module.quantize_file(source, output, "q6_k", ggml_lib=lib), 1)
                    before, after = gguf.GGUFReader(str(source)), gguf.GGUFReader(str(output))
                    self.assertEqual(after.fields["general.architecture"].contents(), arch)
                    self.assertEqual(after.fields["general.file_type"].contents(), module.FILE_TYPES["q6_k"])
                    for key in ("turbovla.pad_layout_instr", "turbovla.pad_layout_len", "test.u32"):
                        self.assertEqual(before.fields[key].types, after.fields[key].types)
                        self.assertEqual(before.fields[key].contents(), after.fields[key].contents())
                    tensor = after.tensors[0]
                    self.assertEqual(tensor.tensor_type, module.QTYPES["q6_k"])
                    np.testing.assert_array_equal(tensor.shape, [256, 128])
                    np.testing.assert_allclose(gguf.dequantize(tensor.data, tensor.tensor_type), values, atol=.04)
                    for old, new in zip(before.tensors[1:], after.tensors[1:]):
                        self.assertEqual(new.tensor_type, old.tensor_type)
                        np.testing.assert_array_equal(new.data, old.data)

    def test_q6_missing_library_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            source, output = Path(scratch)/"source.gguf", Path(scratch)/"output.gguf"
            self.fixture(source, gguf.GGMLQuantizationType.F32, width=256)
            # The shared helper needs torch even though quantization runs on CPU.
            try:
                import torch
            except ImportError:
                self.skipTest("Q6 helper requires torch")
            with self.assertRaisesRegex(SystemExit, "missing.*build libggml-base"):
                module.quantize_file(source, output, "q6_k", ggml_lib=Path(scratch)/"absent.so")
            self.assertFalse(output.exists())

    def test_refuses_overwrite_and_invalid_source(self):
        with tempfile.TemporaryDirectory() as scratch:
            source, output = Path(scratch)/"source.gguf", Path(scratch)/"output.gguf"
            self.fixture(source, gguf.GGMLQuantizationType.F32, nonfinite=True)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                module.quantize_file(source, output, "q8_0")
            self.assertFalse(output.exists())
            with self.assertRaises(FileExistsError):
                module.quantize_file(source, source, "q8_0")
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                module.quantize_file(source, output, "q8_0")
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
