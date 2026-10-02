"""Tests for the SmolVLA Q8_0 quantizer and native-residency contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quantize_smolvla_full_gguf",
    ROOT / "scripts" / "quantize_smolvla_full_gguf.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
gguf = MODULE.gguf


class SmolVLAQuantizerTests(unittest.TestCase):
    _SELECTED = {
        "vlm.blk.0.attn_q.weight",
        "aex.blk.0.ffn_down.weight",
        "connector.weight",
        "action_in_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
    }

    def _add_bf16(self, writer, name: str, values: np.ndarray) -> None:
        bits = values.astype(np.float32, copy=False).view(np.uint32)
        writer.add_tensor(
            name,
            (bits >> np.uint32(16)).astype(np.uint16),
            raw_shape=list(values.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )

    def _fixture(self, path: Path) -> None:
        writer = gguf.GGUFWriter(str(path), arch="smolvla")
        writer.add_string("smolvla.architecture", "smolvla")
        writer.add_string("smolvla.state_norm_mode", "MEAN_STD")
        writer.add_key_value("test.u32", 7, gguf.GGUFValueType.UINT32)
        matrix = np.linspace(-1, 1, 16 * 256, dtype=np.float32).reshape(16, 256)
        writer.add_tensor("vlm.blk.0.attn_q.weight", matrix)
        self._add_bf16(writer, "aex.blk.0.ffn_down.weight", matrix)
        for name in (
            "connector.weight",
            "action_in_proj.weight",
            "action_time_mlp_in.weight",
            "action_time_mlp_out.weight",
            "vlm.embed_tokens.weight",
            "token_embd.weight",
            "state_proj.weight",
            "action_out_proj.weight",
        ):
            writer.add_tensor(name, np.ones((16, 256), dtype=np.float32))
        self._add_bf16(
            writer,
            "vlm.blk.0.attn_norm.weight",
            np.linspace(0.5, 1.5, 256, dtype=np.float32),
        )
        writer.add_tensor("action_in_proj.bias", np.ones(256, dtype=np.float32))
        writer.add_tensor("state_mean", np.ones(8, dtype=np.float32))
        writer.add_tensor(
            "vlm.blk.0.ffn_gate.weight", np.ones((8, 128), dtype=np.float32)
        )
        writer.add_tensor(
            "vlm.blk.0.attn_k.weight", np.ones((256, 18), dtype=np.float32)
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def _decisions(self, source: Path):
        reader = gguf.GGUFReader(str(source))
        try:
            return {
                tensor.name: MODULE._decide_tensor(
                    tensor,
                    MODULE.QTYPE_BY_NAME["Q8_0"],
                    min_elements=4096,
                )
                for tensor in reader.tensors
            }
        finally:
            del reader

    def test_q8_selects_each_compute_component_and_excludes_sensitive_classes(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source.gguf"
            self._fixture(source)
            decisions = self._decisions(source)
            selected = {name for name, decision in decisions.items() if decision.quantize}
            self.assertEqual(selected, self._SELECTED)
            self.assertEqual(
                MODULE.tensor_category("vlm.blk.0.attn_q.weight"), "vlm"
            )
            self.assertEqual(
                MODULE.tensor_category("aex.blk.0.ffn_down.weight"), "action_expert"
            )
            self.assertEqual(MODULE.tensor_category("connector.weight"), "connector")
            self.assertEqual(
                MODULE.tensor_category("action_time_mlp_out.weight"), "action_projection"
            )
            for name in (
                "vlm.embed_tokens.weight",
                "token_embd.weight",
                "vlm.blk.0.attn_norm.weight",
                "action_in_proj.bias",
                "state_mean",
                "state_proj.weight",
                "action_out_proj.weight",
                "vlm.blk.0.ffn_gate.weight",
                "vlm.blk.0.attn_k.weight",
            ):
                with self.subTest(name=name):
                    self.assertFalse(decisions[name].quantize)
            self.assertEqual(decisions["state_proj.weight"].reason, "sensitive-projection")
            self.assertEqual(decisions["action_out_proj.weight"].reason, "sensitive-projection")
            self.assertEqual(
                decisions["vlm.blk.0.attn_k.weight"].reason,
                "ne0-not-divisible-by-32",
            )
            self.assertEqual(
                decisions["vlm.blk.0.ffn_gate.weight"].reason, "too-small"
            )
            self.assertEqual(
                decisions["vlm.blk.0.attn_norm.weight"].reason, "sensitive-category"
            )
            self.assertEqual(decisions["token_embd.weight"].category, "embedding")

    def test_only_q8_is_supported(self):
        self.assertEqual(set(MODULE.QTYPE_BY_NAME), {"Q8_0", "q8_0"})
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source.gguf"
            self._fixture(source)
            with self.assertRaisesRegex(ValueError, "only Q8_0"):
                MODULE.quantize_file(source, Path(scratch) / "out.gguf", "F32")

    def test_refuses_overwrite_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source.gguf"
            self._fixture(source)
            with self.assertRaises(FileExistsError):
                MODULE.quantize_file(source, source, "Q8_0")
            output = Path(scratch) / "out.gguf"
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                MODULE.quantize_file(source, output, "Q8_0")
            self.assertEqual(output.read_bytes(), b"keep")

    def test_q8_metadata_manifest_and_tensor_types_are_deterministic(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source.gguf"
            output = Path(scratch) / "output.gguf"
            manifest = Path(scratch) / "manifest.json"
            self._fixture(source)
            count = MODULE.quantize_file(
                source, output, "Q8_0", manifest_path=manifest
            )
            self.assertEqual(count, len(self._SELECTED))
            before = gguf.GGUFReader(str(source))
            after = gguf.GGUFReader(str(output))
            self.assertEqual(after.fields["smolvla.quantized_by"].contents(),
                             "scripts/quantize_smolvla_full_gguf.py")
            self.assertEqual(after.fields["smolvla.quantization"].contents(), "Q8_0")
            self.assertEqual(after.fields["smolvla.quantization_scope"].contents(),
                             "main_model_full")
            self.assertEqual(after.fields["smolvla.quantized_tensor_count"].contents(), count)
            names = {tensor.name: tensor for tensor in after.tensors}
            for name in self._SELECTED:
                with self.subTest(name=name):
                    self.assertEqual(
                        names[name].tensor_type, gguf.GGMLQuantizationType.Q8_0
                    )
            for name in ("state_proj.weight", "action_out_proj.weight", "vlm.embed_tokens.weight"):
                with self.subTest(name=name):
                    self.assertEqual(
                        names[name].tensor_type,
                        next(t for t in before.tensors if t.name == name).tensor_type,
                    )
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["qtype"], "Q8_0")
            self.assertEqual(manifest_data["quantization_scope"], "main_model_full")
            self.assertEqual(manifest_data["quantized_tensor_count"], count)
            self.assertEqual(manifest_data["selected_quantized_bytes"], 26112)
            self.assertEqual(
                manifest_data["selected_quantized_bytes"],
                sum(
                    entry["output_bytes"]
                    for entry in manifest_data["tensors"]
                    if entry["selected"]
                ),
            )
            self.assertEqual(manifest_data["source_dtype_counts"]["BF16"], 2)
            self.assertEqual(manifest_data["output_dtype_counts"]["BF16"], 1)
            self.assertEqual(manifest_data["output_dtype_counts"]["Q8_0"], count)
            self.assertEqual(
                {entry["name"] for entry in manifest_data["tensors"] if entry["selected"]},
                self._SELECTED,
            )
            output2 = Path(scratch) / "output-2.gguf"
            manifest2 = Path(scratch) / "manifest-2.json"
            MODULE.quantize_file(
                source, output2, "Q8_0", manifest_path=manifest2
            )
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8")),
                json.loads(manifest2.read_text(encoding="utf-8")),
            )
            del names, after, before

    def test_unselected_bf16_tensor_preserves_layout_and_output_is_fully_readable(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source.gguf"
            output = Path(scratch) / "output.gguf"
            self._fixture(source)
            before = gguf.GGUFReader(str(source))
            try:
                original = next(
                    tensor
                    for tensor in before.tensors
                    if tensor.name == "vlm.blk.0.attn_norm.weight"
                )
                original_shape = original.shape.tolist()
                original_nbytes = original.n_bytes
                original_type = original.tensor_type
            finally:
                MODULE._close_reader(before)

            MODULE.quantize_file(source, output, "Q8_0")
            after = gguf.GGUFReader(str(output))
            try:
                copied = next(
                    tensor
                    for tensor in after.tensors
                    if tensor.name == "vlm.blk.0.attn_norm.weight"
                )
                self.assertEqual(copied.tensor_type, original_type)
                self.assertEqual(copied.shape.tolist(), original_shape)
                self.assertEqual(copied.n_bytes, original_nbytes)
                for tensor in after.tensors:
                    self.assertEqual(np.asarray(tensor.data).nbytes, tensor.n_bytes)
            finally:
                MODULE._close_reader(after)

    def test_dry_run_prints_inventory_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "source.gguf"
            output = Path(scratch) / "DRY-RUN-ONLY.gguf"
            self._fixture(source)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "quantize_smolvla_full_gguf.py"),
                    "--input", str(source),
                    "--output", str(output),
                    "--qtype", "Q8_0",
                    "--dry-run",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("quantized_tensors=6 /", completed.stdout)
            self.assertIn("tensor_bytes:", completed.stdout)
            self.assertFalse(output.exists())

    def test_smolvla_loader_preserves_quantized_source_type_for_matmul(self):
        source = (ROOT / "models" / "smolvla.cpp").read_text(encoding="utf-8")
        self.assertIn("const ggml_type src_type = g.tensor_type(name);", source)
        self.assertIn("if (ggml_is_quantized(src_type))", source)
        self.assertIn("target == t->type && ggml_is_quantized(target)", source)
        self.assertIn("resident_quantized_tensors", source)
        self.assertIn("resident_quantized_bytes", source)
        self.assertIn("flow_steps=%d", source)
        self.assertIn("generated_action_horizon=%lld", source)
        self.assertRegex(source, r"lw\.Wq\s*=\s*mk_mm")
        self.assertRegex(source, r"lw\.Wdown\s*=\s*mk_mm")
        self.assertRegex(source, r"W_state_proj\s*=\s*mk_f32")
        self.assertRegex(source, r"W_aout\s*=\s*mk_f32")

    def test_smolvla_loader_converts_unquantized_f16_tensors(self):
        source = (ROOT / "models" / "smolvla.cpp").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("GGML_TYPE_F16"), 2)
        self.assertGreaterEqual(source.count("ggml_fp16_to_fp32_row"), 2)
        self.assertRegex(
            source,
            r"t->type\s*==\s*GGML_TYPE_F16[\s\S]*?ggml_fp16_to_fp32_row",
        )
        self.assertRegex(
            source,
            r"is_f16\s*=\s*t->type\s*==\s*GGML_TYPE_F16"
            r"[\s\S]*?is_f16\)[\s\S]*?ggml_fp16_to_fp32_row",
        )

    def test_sensitive_matmul_projections_use_q8_or_f32_residency(self):
        source = (ROOT / "models" / "smolvla.cpp").read_text(encoding="utf-8")
        self.assertIn("auto mk_q8_or_f32", source)
        self.assertRegex(
            source,
            r"src_type\s*==\s*GGML_TYPE_Q8_0[\s\S]*?return mk\(name, src_type\)"
            r"[\s\S]*?return mk\(name, GGML_TYPE_F32\)",
        )
        self.assertRegex(source, r"W_ain\s*=\s*mk_q8_or_f32")
        self.assertRegex(source, r"W_tmlp_in\s*=\s*mk_q8_or_f32")
        self.assertRegex(source, r"W_tmlp_out\s*=\s*mk_q8_or_f32")
        self.assertRegex(source, r"W_connector\s*=\s*mk_q8_or_f32")


if __name__ == "__main__":
    unittest.main()
