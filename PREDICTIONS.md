# Frozen predictions and outcomes

Copied verbatim from the project log. Each block of predictions was frozen (timestamp from the system clock)
before the run it predicts; the outcomes below were written after scoring with `scripts/score_4j.py`. The
timestamps are self-reported: this repository's git history starts after both runs, so it cannot confirm them.
Each run's `env.json` records its start time from the Kaggle machine clock (`start.t`, Unix time): 2026-11-10 21:22:55 UTC (Qwen2.5) and 2026-11-11 07:03:16 UTC (Llama), after the freezes at 17:45:09 and 06:51:59 UTC.

Glossary for the IDs used below:
- **A, B, C, D:** the builds (`740251a`, `8fc0376`, `8fc0376` without the `__hmul2` branch, `cd4ed90`).
- **I9:** removing the FAST_FP16 `__hmul2` branch from the Q4_1/Q5_1 MMVQ dot products (`scripts/apply_i9_only.py`).
- **I10 / G:** the project's own earlier fix on b10434, a code sum stored as `half(d)*sum` (I10), plus I9 (G = I9 + I10).
- **M7:** the operator-level gap that the `__hmul2` branch leaves.
- **e2e / gen:** median max |Δlogit| on the generated positions; suf_dec and suf_pmt are the suffix comparisons.
- **Q15, Q3B, L3B:** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct.
- **Local paths** such as `/mnt/d/...` are on the author's machine; the files they name are in `results/`.

### 4j. Addendum, frozen 2026-11-10 23:15:09 IST (system clock), before any build or run on current llama.cpp

**Why.** Everything above is on b10434. llama.cpp PR #31247 (merged 2026-11-03) moved both Q8_1 quantizers to the
code sum. It stores `d*sum` with FP32 d, where I10 stores `half(d)*sum`, and it does not include I9. So G's numbers do
not carry over, and whether I9 still matters once the code sum is in was never tested.

**Setup.**
- **Hardware:** Kaggle T4 (sm_75), one session, account `ssmanoj`.
- **Builds (nvcc 12.6, no probe patch; `GGML_PROBE_FLAGS` does not exist in them):**
  - **A** = `740251a` (the parent of #31247).
  - **B** = `8fc0376` (#31247 merged). A and B differ only in `quantize.cu`.
  - **C** = B with the FAST_FP16 `__hmul2` branch removed from `vec_dot_q4_1_q8_1_impl` and `vec_dot_q5_1_q8_1_impl`,
    so the FP32 branch always runs (I9 alone, `probes/apply_i9_only.py`).
  - **D** = `cd4ed90` (master, 2026-11-10).
- **Models:** Qwen2.5-1.5B and Qwen2.5-3B. Quantized once, with A's tool; every build reads the same files.
- **Metric "e2e":** median max|Δlogit| on generated positions (`kv_paths`, cmp = gen), with its ratio to the same
  build's Q8_0 on the same model.
- IDs carry a **U** prefix.

| ID | Prediction (T4) | Falsified if |
|---|---|---|
| **U1 (control)** | F16 and Q8_0 do not use the sum: A and B match in every non-timing `kv_paths` field, on both models | any such field differs |
| **U2** | A still has the problem: Q4_0, Q4_1, Q5_1 e2e ≥ 1.5 × Q8_0 on both models | any ≤ 1.1 × |
| **U3** | #31247 closes it: on B, Q4_0, Q4_1, Q5_1, Q4_K_M e2e ≤ 1.25 × Q8_0 on both models | any > 1.5 × |
| **U4** | Operator δ median (`matmul_paths`, K = N = 1536): Q4_0 and Q4_K, B ≤ A / 5. Q4_1 and Q5_1: B stays in [3e-4, 2e-3] (M7 remains), C ≤ 1e-6 | Q4_0 or Q4_K with B > A / 2; Q4_1 or Q5_1 with B < 1e-4 or C > 1e-5 |
| **U5** | I9 does not matter end to end once the code sum is in: Q4_1, Q5_1 e2e on C within ±10% of B on both models | C < 0.75 × B or C > 1.25 × B on any |
| **U6** | Perplexity, 24 chunks. Qwen2.5-1.5B Q4_0, decode − batch: A has z ≥ 2, B has \|z\| < 2. Q4_0, Q4_1, Q5_1, Q4_K on both models and both paths: B's NLL ≤ 1.0005 × A's | A z < 1, or B \|z\| ≥ 2.5; or any path's NLL on B > 1.002 × A's |
| **U7** | Cost of I9 alone: `llama-bench -p 0 -n 128`, Qwen2.5-1.5B, Q4_1 and Q5_1, 5 ABBA-interleaved repeats per build: \|median change C vs B\| ≤ 3% | > 5% |
| **U8** | Master is the same on these paths: on D, every format's e2e is within ±10% of B on both models | any differs by > 25% |

*Recorded, not pass/fail:* suf_dec and suf_pmt; Q4_K_M on A; other operator formats; the Q8_0 vs F16 step; clocks and
throttle state per run. Latency is reported as median and IQR (rule 9).

### 4k. Addendum, frozen 2026-11-11 12:21:59 IST (system clock), before any run of the §4j builds on Llama-3.2-3B

**Why.** §4j covered only Qwen2.5. On b10434, Llama-3.2-3B showed the same decode-vs-batch inconsistency but no
perplexity penalty (§6, 2026-10-15 10:18). So whether #31247 behaves the same on a second model family is untested.

**Setup.** Exactly §4j: same bundle (`ssmanoj/cs6886-v2-master1`, builds A–D), Kaggle T4, account `ssmanoj`, one
session. Model: Llama-3.2-3B-Instruct F16 GGUF (bartowski mirror, as in the course runs), quantized once with A's
tool. `kv_paths` on F16, Q8_0, Q4_0, Q4_1, Q5_1, Q4_K_M for A–D; `ppl_paths` on Q4_0, Q4_1, Q5_1, Q4_K for A–C
(ctx 512, 24 chunks); `matmul_paths` as in §4j. No llama-bench (U7 already covers speed). e2e = median max|Δlogit|
as a multiple of the same build's Q8_0. Basis for the numbers: the b10434 Llama run (flags 0 gen: Q4_0 2.07×,
Q4_1 2.43×, Q5_1 2.69×, Q4_K_M 1.67×; suf_dec Q4_1 2.67×, Q5_1 2.70×, Q4_K_M 2.10×, Q4_0 1.13×). IDs carry a **V**
prefix.

| ID | Prediction (T4, Llama-3.2-3B) | Falsified if |
|---|---|---|
| **V1 (control)** | F16 and Q8_0: A and B match in every non-timing `kv_paths` field, every cmp | any such field differs |
| **V2** | A, gen: Q4_0, Q4_1, Q5_1 ≥ 1.5 × Q8_0 | any ≤ 1.1 × |
| **V3** | B, gen: Q4_0, Q4_1, Q5_1, Q4_K_M ≤ 1.25 × Q8_0 | any > 1.5 × |
| **V4** | suf_dec: Q4_1, Q5_1, Q4_K_M ≥ 1.5 × Q8_0 on A and ≤ 1.25 × on B | any on A ≤ 1.1 ×, or any on B > 1.5 × |
| **V5** | C vs B, gen: Q4_1, Q5_1 within ±10% | C < 0.75 × B or > 1.25 × B |
| **V6** | D vs B, gen: every format within ±10% | any differs by > 25% |
| **V7** | PPL decode − batch: \|z\| < 2 for Q4_0, Q4_1, Q5_1, Q4_K on both A and B; B's NLL ≤ 1.0005 × A's on both paths | any \|z\| ≥ 3, or any NLL on B > 1.002 × A's |

*Recorded, not pass/fail:* Q4_0 suf_dec; suf_pmt; Q4_K_M on A; C's perplexity; operator deltas compared with the
§4j session (same binaries); clocks.

## Outcomes (project log entries)

- **2026-11-11 02:40 IST (clock), §4j builds A–D done; MX330 plumbing passed (not a result).**
  - **Builds** (`runners/master_build.sh`, nvcc 12.6, sm_61;75, Release, shared libs, no probe patch). Each probe line
    carries its build in `llama_cpp`: `740251a`, `8fc0376`, `8fc0376+i9`, `cd4ed90`. 48 files, 1.3 GB, hashes in
    `/mnt/d/cs6886_work/runs/master/bundle.sha256`.
  - **Adjacency, from the object manifests:**
    - A→B rebuilt only `quantize.cu.o` and the three probe objects (the probes carry their tag).
    - B→C rebuilt only objects that include `vecdotq.cuh` (mmq instances, mmvq, fattn, ggml-cuda.cu, quantize.cu).
    - Across A, B and C, `libllama`, `libggml-base`, `libggml-cpu`, `libggml`, `llama-bench`, `quantize_pure` and
      `libllama-common` are byte-identical; only `libggml-cuda` and the probes differ.
  - **Plumbing** (`kaggle_job.py --plan master --local --tiny`, local test model `models/local-debug/t-F32.gguf`, MX330): 66 of 66 steps rc=0, the job
    hashed the same 48 files, `probes/score_4j.py` runs end to end. sm_61 has no FAST_FP16, so C = B there by
    construction. Numbers from this run are plumbing only and are not reported.
- **2026-11-11 03:53 IST (clock), §4j scored: U1–U8 all pass** (Tesla T4 cc 7.5, driver 580.178.04, Kaggle clock
  21:22–22:19 UTC; every step rc=0; `bin_sha256` matches all 48 files of `bundle.sha256`). Raw data:
  `results/kaggle-master/results/`. Scorer output: `/mnt/d/cs6886_work/runs/master/score_4j.txt`
  (`probes/score_4j.py results/kaggle-master/results /mnt/d/cs6886_work/runs/master/bundle.sha256`).
  - **U1 pass.** F16 and Q8_0: A and B identical in every non-timing `kv_paths` field (12 comparisons).
  - **U2 pass / U3 pass.** e2e as × same-build Q8_0 (Qwen2.5-1.5B; Qwen2.5-3B), A → B:
    Q4_0 2.57→1.01; 2.31→0.98. Q4_1 2.41→1.04; 2.48→1.02. Q5_1 2.33→0.98; 2.30→0.99. Q4_K_M 1.60→0.98; 1.62→1.02.
    Q8_0 floor 0.341 / 0.325, F16 0.110 / 0.110 (median max|Δlogit|, generated positions).
  - **U4 pass.** Operator δ median, K = N = 1536: Q4_0 2.02e-2 → 7.71e-4 (B/A 0.038); Q4_K 1.14e-2 → 8.13e-4 (0.071).
    Q4_1 B 7.02e-4, C 1.71e-7; Q5_1 B 7.14e-4, C 1.74e-7 (M7 remains on B; I9 removes it). Q4_1 and Q5_1 on A are
    already 6.96e-4 / 7.29e-4 at operator level, so this random-input probe does not show the A→B end-to-end change
    for those two formats. Recorded, not explained.
  - **U5 pass.** C/B e2e: Q4_1 0.975 / 0.994, Q5_1 1.001 / 1.003. I9 removes the operator gap but changes nothing
    end to end once the code sum is in. Recorded: suf_dec is slightly higher on C than B for Q4_1/Q5_1
    (e.g. 3B Q4_1 0.754 vs 0.709).
  - **U6 pass.** 1.5B Q4_0 decode − batch z: A +3.22, B +0.85 (3B: +2.98 → +0.16). B's NLL ≤ A's on every model,
    format and path (ratios 0.9968–1.0001); Q4_0 batch PPL is identical on A and B: on the MMA path Q4_0 runs the
    Q8_0 dot (`vec_dot_q8_0_q8_1_mma`, `mmq.cuh:765-769` at `740251a`), which reads only the scale, not the sum.
    So for Q4_0 the gain is all on decode.
  - **U7 pass.** llama-bench tg128 (tokens/s, 5 ABBA repeats each), C vs B: Q4_1 161.2 → 161.5 (+0.18%), Q5_1
    138.5 → 138.5 (−0.05%); IQRs ≤ 0.6 tok/s. SM clock 1545–1590 MHz of 1590 (ratio ≥ 0.97), 76–77 °C, ~45 W of
    70 W; throttle bits 0x0 at start, 0x4 (SW power cap) at the end of some runs on both builds alike.
  - **U8 pass.** D (master `cd4ed90`) equals B to three decimals on every e2e cell (D/B = 1.000) and on every
    operator δ. Nothing merged between #31247 and 2026-11-10 changed these paths on the T4.
  - **Reading.** #31247 closes the cache-hit vs recompute gap end to end on the T4 for these formats and models;
    what remains is the Q8_0 floor. I9 is free (U7) but has no end-to-end benefit (U5), so it has no upstream case on
    these models. Nothing posted, nothing committed.
- **2026-11-11 13:15 IST (clock), §4k scored: 5 pass, 2 partial, none falsified** (Tesla T4, Kaggle clock
  07:03–07:39 UTC, 36 GPU-min; every step rc=0; `bin_sha256` 48/48 match). Raw data:
  `results/kaggle-master-l3b/results/`; scorer output `/mnt/d/cs6886_work/runs/master/score_4k.txt`.
  - **V1 pass.** F16 and Q8_0: A and B identical in every non-timing field.
  - **V2 pass.** A, gen, × Q8_0: Q4_0 2.04, Q4_1 2.54, Q5_1 2.65 (Q4_K_M 1.67).
  - **V3 partial.** B, gen: Q4_0 1.04, Q4_1 1.13, **Q5_1 1.33** (above 1.25, below the 1.5 falsification line),
    Q4_K_M 1.00.
  - **V4 pass.** suf_dec A → B: Q4_1 2.71 → 1.09, Q5_1 2.66 → 1.12, Q4_K_M 2.03 → 1.00 (Q4_0 1.09 → 1.04).
  - **V5 partial.** C/B, gen: Q4_1 0.919, **Q5_1 0.754** (falsification line 0.75). With I9, Q5_1 is 1.00 × Q8_0 and
    Q4_1 1.04 ×. The whole distribution moves: Q5_1 p90 0.324 → 0.222 (Q8_0 0.224), max 1.31 → 0.44. suf_dec Q5_1
    1.12 → 1.03, Q4_1 1.09 → 1.09.
  - **V6 pass.** D equals B on every cell (D/B = 1.000).
  - **V7 pass.** z (A / B): Q4_0 +1.42 / +1.99, Q4_1 −1.69 / −0.04, Q5_1 +1.84 / +1.72, Q4_K −1.08 / +0.31. NLL B/A
    0.99826–1.00000. I9 leaves perplexity unchanged (C/B decode NLL 0.99967–1.00022; batch identical, so I9 does not
    touch the T4 batch path).
  - **Recorded.** Floor Q8_0 0.173 (gen) / 0.183 (suf_dec), F16 0.095 / 0.102, identical A–D. suf_pmt A → B: Q4_1
    2.50 → 1.04, Q5_1 2.39 → 1.01, Q4_K_M 1.92 → 0.99, Q4_0 1.04 → 1.04. Operator deltas: 44 of 44 identical to the
    §4j session (another T4 instance, same binaries).
  - **Reading.** #31247 closes most of the gap on Llama-3.2-3B too. What it leaves on the decoded tokens for Q5_1
    (and a little for Q4_1) is the FAST_FP16 `__hmul2` branch: I9 removes it. So U5's "I9 does not matter once the
    code sum is in" holds for Qwen2.5 only. I9 is the same change as PR #31019 (Cross2pro, closed 2026-10-23 by
    JohannesGaessler: no performance evidence, overflow "not a relevant use case"). U7 is the only speed evidence
    (T4, Qwen2.5-1.5B, ±0.2%). Nothing posted, nothing committed.
