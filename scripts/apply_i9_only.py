#!/usr/bin/env python3
"""Apply I9 alone to a llama.cpp tree (docs/PROJECT.md 4j, build C).

    python probes/apply_i9_only.py <tree>

Removes the FAST_FP16 `__hmul2` branch from the MMVQ Q4_1 and Q5_1 dot products, so the FP32 branch
always runs. Nothing else changes. Every edit must match exactly once, so a partial patch is impossible.
"""

from __future__ import annotations

import sys
from pathlib import Path


def block(w: str) -> tuple[str, str]:
    fp32 = (f"    const float2 dm{w}f = __half22float2(dm{w});\n"
            "    const float2 ds8f = __half22float2(ds8);\n"
            f"    const float d{w}d8 = dm{w}f.x * ds8f.x;\n"
            f"    const float m{w}s8 = dm{w}f.y * ds8f.y;\n")
    old = ("#ifdef FAST_FP16_AVAILABLE\n"
           f"    const float2 tmp = __half22float2(__hmul2(dm{w}, ds8));\n"
           f"    const float d{w}d8 = tmp.x;\n"
           f"    const float m{w}s8 = tmp.y;\n"
           "#else\n" + fp32 + "#endif // FAST_FP16_AVAILABLE\n")
    return old, fp32


def main(root: Path) -> None:
    p = root / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh"
    text = p.read_text()
    for w in ("4", "5"):
        old, new = block(w)
        if text.count(old) != 1:
            raise SystemExit(f"Q{w}_1: expected exactly 1 match, found {text.count(old)}")
        text = text.replace(old, new)
    p.write_text(text)
    print("I9 applied:", p)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
