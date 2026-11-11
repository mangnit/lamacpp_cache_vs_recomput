#!/usr/bin/env python3
"""Score the current-llama.cpp run against docs/PROJECT.md section 4j (U1-U8).

    python probes/score_4j.py results/kaggle-master [/mnt/d/cs6886_work/runs/master/bundle.sha256]

Builds: A = 740251a (parent of #31247), B = 8fc0376 (#31247), C = B + I9 alone, D = cd4ed90 (master).
Each probe line records its build in "llama_cpp"; kv_paths and ppl_paths tags are "<build>:<P>_<format>".
Thresholds are exactly the frozen ones. A prediction is "pass" when its claim holds, "FALSIFIED" when its
falsification condition holds, and "partial" in between. Only U7 reads timing; nothing else does.
Anything not measured on a T4 (cc 7.5) is labelled PLUMBING and is not a result.

Section 4k (Llama-3.2-3B, the same builds) uses the same script; V1, V2, V3, V5 and V6 have the thresholds of
U1, U2, U3, U5 and U8 and are read off those lines; V4 and V7 print their own lines when L3B is present:

    python probes/score_4j.py results/kaggle-master-l3b/results BUNDLE.sha256 --ref results/kaggle-master/results
"""

from __future__ import annotations

import json
import statistics
import sys
from math import log
from pathlib import Path

BUILD_OF = {"740251a": "A", "8fc0376": "B", "8fc0376+i9": "C", "cd4ed90": "D"}
MODELS = ("Q15", "Q3B")
TIMING = {"decode_ms_median", "decode_ms_p90", "prefill_ms_median", "prefill_ms_p90"}
LABEL = {"tag", "llama_cpp", "model"}  # differ by build or path by construction, not numerics


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def verdict(ok: bool, falsified: bool) -> str:
    return "FALSIFIED" if falsified else "pass" if ok else "partial"


def overall(v: list[str]) -> str:
    if not v:
        return "NOT RUN"
    return "FALSIFIED" if "FALSIFIED" in v else "partial" if "partial" in v else "pass"


def ratio(x: float, y: float) -> float:
    return 1.0 if x == y else float("inf") if y == 0 else x / y


def iqr(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 4:
        return min(xs), max(xs)
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def zscore(r: dict) -> float:
    """dnll / se_chunk; NaN when the SE is zero (a one-chunk dry run)."""
    return r["dnll"] / r["dnll_se_chunk"] if r["dnll_se_chunk"] else float("nan")


def check_bundle(run: Path, manifest: Path) -> None:
    env = json.loads((run / "env.json").read_text())
    want = {}
    for line in manifest.read_text().splitlines():
        h, name = line.split()
        build, f = Path(name).name.split(".", 1)
        want[f"{build}/{f}"] = h
    got = env.get("bin_sha256", {})
    bad = sorted(k for k in want if got.get(k) != want[k])
    extra = sorted(set(got) - set(want))
    print(f"bundle: {len(want)} files expected, {len(got)} hashed by the job, mismatched {bad or 'none'}, unexpected {extra or 'none'}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("manifest", nargs="?", help="bundle.sha256 to check the job's bin_sha256 against")
    ap.add_argument("--ref", help="an earlier run on the same binaries: operator deltas are compared with it (4k)")
    args = ap.parse_args()
    run = Path(args.run)
    if args.manifest:
        check_bundle(run, Path(args.manifest))
    env = json.loads((run / "env.json").read_text()) if (run / "env.json").exists() else {}
    print("GPU:", env.get("start", {}).get("nvidia_smi", "?"))

    mm_rows = load(run / "matmul_paths.jsonl")
    devs = {(r["device"], r["cc"]) for r in mm_rows}
    print("matmul devices:", devs or "none")
    real = bool(devs) and all(str(cc) in ("75", "7.5") for _, cc in devs)
    if not real:
        print("*** PLUMBING: not a T4 run, nothing below is a result ***")

    # ---- U4: operator delta, K = N = 1536 ----
    mm = {(BUILD_OF.get(r["llama_cpp"], r["llama_cpp"]), r["type"]): r for r in mm_rows if r["K"] == 1536}
    if not mm:  # the tiny dry run uses K = 256
        mm = {(BUILD_OF.get(r["llama_cpp"], r["llama_cpp"]), r["type"]): r for r in mm_rows}
    print("\nU4 operator delta median: Q4_0/Q4_K B <= A/5 (F if B > A/2); Q4_1/Q5_1 B in [3e-4, 2e-3] and C <= 1e-6 "
          "(F if B < 1e-4 or C > 1e-5)")
    v = []
    for t in ("q4_0", "q4_K"):
        a, b = mm.get(("A", t)), mm.get(("B", t))
        if a and b:
            da, db = a["delta_median"], b["delta_median"]
            v.append(verdict(db <= da / 5, db > da / 2))
            print(f"  {t:5} A {da:.2e}  B {db:.2e}  (B/A {ratio(db, da):.3f})  {v[-1]}")
    for t in ("q4_1", "q5_1"):
        b, c = mm.get(("B", t)), mm.get(("C", t))
        if b and c:
            db, dc = b["delta_median"], c["delta_median"]
            v.append(verdict(3e-4 <= db <= 2e-3 and dc <= 1e-6, db < 1e-4 or dc > 1e-5))
            print(f"  {t:5} A {mm[('A', t)]['delta_median']:.2e}  B {db:.2e}  C {dc:.2e}  {v[-1]}")
    print("  U4:", overall(v))
    print("  recorded, every operator type (delta median A / B / C / D):")
    for t in sorted({k[1] for k in mm}):
        print(f"    {t:7}", "  ".join(f"{b} {mm[(b, t)]['delta_median']:.2e}" for b in "ABCD" if (b, t) in mm))

    # ---- end to end: kv_paths ----
    kv_all = load(run / "kv_paths.jsonl")
    kv = {}
    for r in kv_all:
        build, rest = r["tag"].split(":", 1)
        P, fmt = rest.split("_", 1)
        kv[(build, P, fmt, r["cmp"])] = r
    present = sorted({k[1] for k in kv})
    print(f"\nkv_paths: {len(kv_all)} lines, models {present}")

    def e2e(build: str, P: str, fmt: str) -> float | None:
        r = kv.get((build, P, fmt, "gen"))
        return r["maxabs_median"] if r else None

    def rel(build: str, P: str, fmt: str) -> float | None:
        x, f = e2e(build, P, fmt), e2e(build, P, "Q8_0")
        return None if x is None or f is None else ratio(x, f)

    # U1: F16 and Q8_0 identical between A and B in every non-timing field, every cmp
    v, diffs = [], []
    for (build, P, fmt, cmp), ra in kv.items():
        if build != "A" or fmt not in ("F16", "Q8_0"):
            continue
        rb = kv.get(("B", P, fmt, cmp))
        if rb is None:
            continue
        d = sorted(k for k in set(ra) | set(rb) if k not in TIMING | LABEL and ra.get(k) != rb.get(k))
        v.append(verdict(not d, bool(d)))
        if d:
            diffs.append(f"{P}_{fmt} {cmp}: {d}")
    print(f"\nU1 control, A vs B on F16 and Q8_0, every non-timing kv_paths field: {len(v)} comparisons")
    for d in diffs:
        print("   differs:", d)
    print("  U1:", overall(v))

    print("\ne2e = median max|dlogit| on generated positions; x = ratio to the same build's Q8_0")
    for P in present:
        print(f"  {P}: " + "  ".join(f"{b}[Q8_0 {e2e(b, P, 'Q8_0'):.3f} F16 {e2e(b, P, 'F16'):.3f}]"
                                     for b in "ABCD" if e2e(b, P, "Q8_0") is not None and e2e(b, P, "F16") is not None))
        for fmt in ("Q4_0", "Q4_1", "Q5_1", "Q4_K_M"):
            cells = []
            for b in "ABCD":
                x, r = e2e(b, P, fmt), rel(b, P, fmt)
                if x is not None and r is not None:
                    cells.append(f"{b} {x:.3f} ({r:.2f}x)")
            print(f"    {fmt:6} " + "  ".join(cells))

    v = []
    for P in present:
        for fmt in ("Q4_0", "Q4_1", "Q5_1"):
            r = rel("A", P, fmt)
            if r is not None:
                v.append(verdict(r >= 1.5, r <= 1.1))
    print("\nU2 A: Q4_0, Q4_1, Q5_1 >= 1.5x Q8_0 on both models (F if any <= 1.1x):", overall(v), f"({len(v)} cells)")

    v = []
    for P in present:
        for fmt in ("Q4_0", "Q4_1", "Q5_1", "Q4_K_M"):
            r = rel("B", P, fmt)
            if r is not None:
                v.append(verdict(r <= 1.25, r > 1.5))
    print("U3 B: Q4_0, Q4_1, Q5_1, Q4_K_M <= 1.25x Q8_0 on both models (F if any > 1.5x):", overall(v), f"({len(v)} cells)")

    v = []
    print("U5 C vs B, Q4_1 and Q5_1 e2e within +-10% (F if < 0.75x or > 1.25x):")
    for P in present:
        for fmt in ("Q4_1", "Q5_1"):
            b, c = e2e("B", P, fmt), e2e("C", P, fmt)
            if b is not None and c is not None:
                q = ratio(c, b)
                v.append(verdict(0.9 <= q <= 1.1, q < 0.75 or q > 1.25))
                print(f"    {P}_{fmt}: B {b:.3f}  C {c:.3f}  C/B {q:.3f}  {v[-1]}")
    print("  U5:", overall(v))

    v = []
    print("U8 D vs B, every format's e2e within +-10% (F if differs by > 25%):")
    for P in present:
        for fmt in ("F16", "Q8_0", "Q4_0", "Q4_1", "Q5_1", "Q4_K_M"):
            b, d = e2e("B", P, fmt), e2e("D", P, fmt)
            if b is not None and d is not None:
                q = ratio(d, b)
                v.append(verdict(0.9 <= q <= 1.1, abs(q - 1) > 0.25))
                print(f"    {P}_{fmt}: B {b:.3f}  D {d:.3f}  D/B {q:.3f}  {v[-1]}")
    print("  U8:", overall(v))

    if "L3B" in present:  # docs/PROJECT.md 4k, V4: the next turn after the history
        def rel_c(build: str, fmt: str, cmp: str) -> float | None:
            x, f = kv.get((build, "L3B", fmt, cmp)), kv.get((build, "L3B", "Q8_0", cmp))
            return None if x is None or f is None else ratio(x["maxabs_median"], f["maxabs_median"])
        v = []
        print("V4 suf_dec, L3B: Q4_1, Q5_1, Q4_K_M >= 1.5x Q8_0 on A and <= 1.25x on B (F if A <= 1.1x or B > 1.5x):")
        for fmt in ("Q4_1", "Q5_1", "Q4_K_M"):
            ra, rb = rel_c("A", fmt, "suf_dec"), rel_c("B", fmt, "suf_dec")
            if ra is not None and rb is not None:
                v.append(verdict(ra >= 1.5 and rb <= 1.25, ra <= 1.1 or rb > 1.5))
                print(f"    {fmt}: A {ra:.2f}x  B {rb:.2f}x  {v[-1]}")
        print("  V4:", overall(v))
        print("  recorded, x Q8_0 per cmp (A -> B):")
        for cmp in ("suf_dec", "suf_pmt"):
            for fmt in ("Q4_0", "Q4_1", "Q5_1", "Q4_K_M"):
                ra, rb = rel_c("A", fmt, cmp), rel_c("B", fmt, cmp)
                if ra is not None and rb is not None:
                    print(f"    {cmp:7} {fmt:6} {ra:.2f}x -> {rb:.2f}x")

    if args.ref:  # same binaries, same GPU model: does the operator probe repeat across sessions?
        ref = {(BUILD_OF.get(r["llama_cpp"], r["llama_cpp"]), r["type"]): r["delta_median"]
               for r in load(Path(args.ref) / "matmul_paths.jsonl") if r["K"] == 1536}
        worst = max((abs(ratio(r["delta_median"], ref[k]) - 1) for k, r in mm.items() if k in ref), default=None)
        same = sum(1 for k, r in mm.items() if k in ref and r["delta_median"] == ref[k])
        print(f"\nrecorded: operator delta vs {args.ref}: {same}/{len(mm)} identical, worst relative change "
              f"{worst if worst is None else f'{worst:.2e}'}")

    print("recorded: suf_dec and suf_pmt (median max|dlogit|, A / B / C / D)")
    for P in present:
        for fmt in ("F16", "Q8_0", "Q4_0", "Q4_1", "Q5_1", "Q4_K_M"):
            for cmp in ("suf_dec", "suf_pmt"):
                cells = [f"{b} {kv[(b, P, fmt, cmp)]['maxabs_median']:.3f}" for b in "ABCD" if (b, P, fmt, cmp) in kv]
                if cells:
                    print(f"    {P}_{fmt:6} {cmp:7} " + "  ".join(cells))

    # ---- U6: perplexity ----
    ppl = {}
    for r in load(run / "ppl_paths.jsonl"):
        build, rest = r["tag"].split(":", 1)
        P, fmt = rest.split("_", 1)
        ppl[(build, P, fmt)] = r
    print(f"\nU6 PPL ({len(ppl)} runs). z = dnll / se_chunk, dnll = decode - batch.")
    v = []
    a, b = ppl.get(("A", "Q15", "Q4_0")), ppl.get(("B", "Q15", "Q4_0"))
    if a and b:
        za, zb = zscore(a), zscore(b)
        v.append(verdict(za >= 2 and abs(zb) < 2, za < 1 or abs(zb) >= 2.5))
        print(f"  Q15 Q4_0 z: A {za:+.2f}  B {zb:+.2f}  (A z >= 2, B |z| < 2; F if A z < 1 or B |z| >= 2.5)  {v[-1]}")
    print("  NLL B / A per path (<= 1.0005; F if > 1.002):")
    for P in present:
        for fmt in ("Q4_0", "Q4_1", "Q5_1", "Q4_K"):
            ra, rb = ppl.get(("A", P, fmt)), ppl.get(("B", P, fmt))
            if not (ra and rb):
                continue
            cells = []
            for path in ("decode", "batch"):
                q = log(rb[f"ppl_{path}"]) / log(ra[f"ppl_{path}"])
                v.append(verdict(q <= 1.0005, q > 1.002))
                cells.append(f"{path} {ra[f'ppl_{path}']:.4f}->{rb[f'ppl_{path}']:.4f} ({q:.5f}) {v[-1]}")
            print(f"    {P}_{fmt:5} " + "  ".join(cells))
    print("  U6:", overall(v))
    if "L3B" in present:  # docs/PROJECT.md 4k, V7: no decode penalty on Llama, on A or B; NLL as U6
        vz = []
        for fmt in ("Q4_0", "Q4_1", "Q5_1", "Q4_K"):
            for bb in "AB":
                r = ppl.get((bb, "L3B", fmt))
                if r:
                    z = zscore(r)
                    vz.append(verdict(abs(z) < 2, abs(z) >= 3))
        vn = [x for x in v]  # the NLL cells above (only L3B is present in a 4k run)
        print(f"V7 L3B: |z| < 2 on A and B for Q4_0, Q4_1, Q5_1, Q4_K (F if any |z| >= 3): {overall(vz)};"
              f" NLL B/A as above: {overall(vn)}")
        print("  V7:", overall(vz + vn))
    print("  recorded z per build (A / B / C):")
    for P in present:
        for fmt in ("Q4_0", "Q4_1", "Q5_1", "Q4_K"):
            cells = [f"{bb} {zscore(ppl[(bb, P, fmt)]):+.2f}" for bb in "ABC" if (bb, P, fmt) in ppl]
            if cells:
                print(f"    {P}_{fmt:5} " + "  ".join(cells))

    # ---- U7: llama-bench, ABBA ----
    lb = load(run / "llama_bench.jsonl")
    print(f"\nU7 llama-bench tg (tokens/s), ABBA; |median C/B - 1| <= 3% (F if > 5%). {len(lb)} runs")
    v = []
    for fmt in ("Q4_1", "Q5_1"):
        ts = {"B": [], "C": []}
        for r in lb:
            if r["format"] != fmt or r["rc"] != 0:
                continue
            lines = [l for l in r["stdout"].splitlines() if l.startswith("{")]
            if lines:
                ts[r["build"]].append(json.loads(lines[-1])["avg_ts"])
        if ts["B"] and ts["C"]:
            mb, mc = statistics.median(ts["B"]), statistics.median(ts["C"])
            q = mc / mb - 1
            v.append(verdict(abs(q) <= 0.03, abs(q) > 0.05))
            (b1, b3), (c1, c3) = iqr(ts["B"]), iqr(ts["C"])
            print(f"  {fmt}: B median {mb:.1f} (IQR {b1:.1f}-{b3:.1f}, n={len(ts['B'])})  "
                  f"C median {mc:.1f} (IQR {c1:.1f}-{c3:.1f}, n={len(ts['C'])})  change {100 * q:+.2f}%  {v[-1]}")
    print("  U7:", overall(v))
    clk = [(r["build"], r["format"], r["clocks_start"], r["clocks_end"]) for r in lb]
    if clk:
        print("  clocks (start | end) per run:")
        for c in clk:
            print(f"    {c[0]} {c[1]}: {c[2]} | {c[3]}")
    if not real:
        print("\n*** PLUMBING: not a T4 run, nothing above is a result ***")


if __name__ == "__main__":
    main()
