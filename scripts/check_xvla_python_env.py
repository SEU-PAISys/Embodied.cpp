#!/usr/bin/env python3
"""Smoke check for the pinned X-VLA Python/LIBERO environment.

Verifies the exact version pins and offline model loading that the
X-VLA matched Python/C++ runs depend on. Exits non-zero on any mismatch.

Usage:
  python scripts/check_xvla_python_env.py [--model /home/xuling/yangzhixiao/xvla_hf]
"""
import argparse
import importlib
import sys

# Pins for the validated combination (server xr0_pyenv2 + numpy126_overlay +
# libero_uv site-packages on PYTHONPATH). The X-VLA denoise bit-exact check
# (scripts/verify_xvla_denoise_equiv.py) passed on exactly this combination.
PINS = {
    "numpy": "1.26.4",
    "torch": "2.5.1+cu124",
    "transformers": "4.51.3",
}


def version_of(name: str) -> str:
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/xuling/yangzhixiao/xvla_hf")
    args = ap.parse_args()

    failures = []

    def check(name: str, expect_prefix: str) -> str:
        got = version_of(name)
        ok = got.startswith(expect_prefix)
        print(f"  {name}: {got} (expect {expect_prefix}*) {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(name)
        return got

    print("[1/4] version pins")
    check("numpy", PINS["numpy"])
    check("torch", PINS["torch"])
    check("transformers", PINS["transformers"])

    print("[2/4] LIBERO simulation adapter import")
    try:
        importlib.import_module("adapter.sim.libero")
        print("  adapter.sim.libero: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"  adapter.sim.libero: FAIL ({exc})")
        failures.append("adapter.sim.libero")

    print("[3/4] offline model load (local_files_only)")
    try:
        from transformers import AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                          local_files_only=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  AutoModel.load: OK ({n_params / 1e6:.0f}M params)")
        del model
    except Exception as exc:  # noqa: BLE001
        print(f"  AutoModel.load: FAIL ({exc})")
        failures.append("model_load")

    print("[4/4] denoise equivalence artifact present")
    import os  # noqa: PLC0415

    ev = "/home/xuling/yangzhixiao/xvla_denoise_equiv.json"
    if os.path.exists(ev):
        print(f"  {ev}: OK")
    else:
        print(f"  {ev}: MISSING (run scripts/verify_xvla_denoise_equiv.py)")
        failures.append("denoise_evidence")

    if failures:
        print(f"ENV SMOKE: FAIL ({', '.join(failures)})")
        return 1
    print("ENV SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
