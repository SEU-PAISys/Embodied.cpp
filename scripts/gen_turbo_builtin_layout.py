"""Generate the builtin TurboVLA per-instruction padding layout C++ fragment.

Reads scripts/turbo_pad_layout.json (extracted from the official checkpoint
metadata) and writes scripts/turbo_builtin_pad_layout.inc, which is included
by models/turbovla.cpp as a fallback when a GGUF file lacks the
turbovla.pad_layout_* metadata arrays.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "turbo_pad_layout.json"), encoding="utf-8"))
instr, lens = d["instructions"], d["pad_lengths"]
assert len(instr) == 40 and len(lens) == 40, f"unexpected sizes {len(instr)}/{len(lens)}"

lines = []
for t, l in zip(instr, lens):
    esc = t.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'    {{"{esc}", {l}}},')

cpp = (
    "// Builtin fallback for GGUF files missing the per-instruction padding layout\n"
    "// (LIBERO all-4-suite instruction set, from the official checkpoint metadata).\n"
    "static const struct { const char * text; int64_t length; } kTurboBuiltinPadLayout[] = {\n"
    + "\n".join(lines)
    + "\n};\n"
    "static const size_t kTurboBuiltinPadLayoutN = "
    "sizeof(kTurboBuiltinPadLayout) / sizeof(kTurboBuiltinPadLayout[0]);\n"
)
out = os.path.join(HERE, "..", "models", "turbo_builtin_pad_layout.inc")
out = os.path.abspath(out)
open(out, "w", encoding="utf-8", newline="\n").write(cpp)
print(f"generated {out}: {len(lines)} entries")
