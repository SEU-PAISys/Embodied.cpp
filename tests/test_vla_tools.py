"""Small executable guards for evidence-producing tools."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvidenceToolTests(unittest.TestCase):
    def test_latency_summary_rejects_invalid_samples(self):
        module = load_script("bench_vla_boundary")
        self.assertEqual(module.summarize([1, 2, 3])["mean"], 2)
        self.assertEqual(module.summarize([1, 2, 3])["p50"], 2)
        for values in ([], [np.nan], [np.inf], [-1], [[1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                module.summarize(values)

    def test_cpp_benchmark_calls_and_missing_process_memory(self):
        module = load_script("bench_vla_boundary")
        from client import vla_cpp_client

        class Client:
            calls, closed = 0, False

            def __init__(self, *args, **kwargs):
                pass

            def _predict_chunk(self, obs):
                self.__class__.calls += 1
                return np.ones((12, 7), np.float32)

            def get_last_inference_profile(self):
                return {"server_total_ms": 1., "server_vision_ms": None}

            def close(self):
                self.__class__.closed = True

        class Sampler:
            ident = None
            started_after_calls = None
            samples_mib, sources, gpu_uuids = [200], ["device_total_fallback"], ["test"]

            def __init__(self, *args):
                pass

            def start(self):
                self.__class__.started_after_calls = Client.calls
                self.ident = 1

            def stop(self):
                pass

        with tempfile.TemporaryDirectory() as scratch:
            fixture, output = Path(scratch)/"fixture.npz", Path(scratch)/"result.json"
            np.savez(fixture, images_chw=np.zeros((2, 3, 256, 256), np.float32),
                     state=np.zeros(8, np.float32), instruction=np.asarray("pick"))
            argv = ["bench", "--arch", "turbovla", "--backend", "cpp", "--fixture", str(fixture),
                    "--output", str(output), "--server-pid", "123", "--warmup", "2", "--n", "3", "--memory-requests", "2"]
            with patch("sys.argv", argv), patch.object(module, "VramSampler", Sampler), \
                 patch.object(vla_cpp_client, "VlaCppClient", Client):
                module.main()
            result = json.loads(output.read_text())
            self.assertEqual(Client.calls, 7)
            self.assertEqual(Sampler.started_after_calls, 5)
            self.assertTrue(Client.closed)
            self.assertEqual(len(result["samples_ms"]), 3)
            self.assertEqual(result["shape"], [12, 7])
            self.assertIsNone(result["sampled_process_peak_mib"])
            self.assertIsNone(result["server_samples"][0]["server_vision_ms"])
            self.assertEqual(np.load(output.with_suffix(".actions.npy")).shape, (12, 7))

    def test_xvla_converter_refuses_existing_output_before_reading_weights(self):
        module = load_script("convert_xvla_to_gguf")
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/"keep.gguf"
            output.write_bytes(b"do not replace")
            options = SimpleNamespace(output=output, hf_dir=Path(scratch)/"missing")
            with patch.object(module, "parse_args", return_value=options), \
                 self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                module.main()
            self.assertEqual(output.read_bytes(), b"do not replace")

    def test_xvla_converter_preserves_f32_and_domain_layout(self):
        module = load_script("convert_xvla_to_gguf")
        # Tiny shape-consistent converter fixture; no inference model required.
        config = dict(hidden_size=4, depth=0, num_heads=1, mlp_ratio=2,
            num_domains=2, len_soft_prompts=1, dim_time=2, max_len_seq=30,
            num_actions=30, action_mode="ee6d", florence_config=dict(projection_dim=4,
            text_config=dict(d_model=4, encoder_layers=0, encoder_attention_heads=1,
                             encoder_ffn_dim=8, vocab_size=4, max_position_embeddings=50),
            vision_config=dict(depths=[0]*4, dim_embed=[4]*4, num_heads=[1]*4,
                num_groups=[1]*4, patch_size=[1]*4, patch_stride=[1]*4,
                patch_padding=[0]*4, patch_prenorm=[False]*4, window_size=1)))
        tensors = {}

        def value(name, shape):
            tensors[name] = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000 + 1.0001)

        def pair(name, size=4):
            value(name+".weight", (size,))
            value(name+".bias", (size,))

        for stage in range(4):
            stem = f"vlm.vision_tower.convs.{stage}"
            value(stem+".proj.weight", (4, 3 if stage == 0 else 4, 1, 1))
            value(stem+".proj.bias", (4,))
            pair(stem+".norm")
        for name, shape in {
            "vlm.image_pos_embed.row_embeddings.weight": (50, 2),
            "vlm.image_pos_embed.column_embeddings.weight": (50, 2),
            "vlm.visual_temporal_embed.pos_idx_to_embed": (100, 4),
            "vlm.image_projection": (4, 4),
            "vlm.language_model.model.shared.weight": (4, 4),
            "vlm.language_model.model.encoder.embed_positions.weight": (52, 4),
            "transformer.vlm_proj.weight": (4, 4), "transformer.vlm_proj.bias": (4,),
            "transformer.aux_visual_proj.weight": (4, 4), "transformer.aux_visual_proj.bias": (4,),
            "transformer.pos_emb": (1, 30, 4),
            "transformer.action_encoder.fc.weight": (2, 42*4),
            "transformer.action_encoder.bias.weight": (2, 4),
            "transformer.action_decoder.fc.weight": (2, 4*20),
            "transformer.action_decoder.bias.weight": (2, 20),
            "transformer.soft_prompt_hub.weight": (2, 4),
        }.items():
            value(name, shape)
        for stem in ("vlm.image_proj_norm", "vlm.language_model.model.encoder.layernorm_embedding", "transformer.norm"):
            pair(stem)
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            (directory/"config.json").write_text(json.dumps(config))
            (directory/"model.safetensors").touch()
            output = directory/"model.gguf"
            with patch.object(module, "parse_args", return_value=SimpleNamespace(output=output, hf_dir=directory)), \
                 patch.object(module, "load_file", return_value=tensors):
                module.main()
            reader = module.gguf.GGUFReader(str(output))
            converted = {t.name: t.data for t in reader.tensors}
            np.testing.assert_array_equal(converted["act.vlm_proj.w"], tensors["transformer.vlm_proj.weight"])
            self.assertFalse(np.array_equal(converted["act.vlm_proj.w"], converted["act.vlm_proj.w"].astype(np.float16).astype(np.float32)))
            np.testing.assert_array_equal(converted["vproj.proj.w"], tensors["vlm.image_projection"].T)
            for prefix, source, n_in, n_out in (("act.aenc.fc", "action_encoder", 42, 4),
                                               ("act.adec.fc", "action_decoder", 4, 20)):
                expected = tensors[f"transformer.{source}.fc.weight"].reshape(2, n_in, n_out).transpose(0, 2, 1).reshape(2, -1)
                np.testing.assert_array_equal(converted[prefix], expected)


if __name__ == "__main__":
    unittest.main()
