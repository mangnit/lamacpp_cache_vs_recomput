# Cache hit vs recompute in llama.cpp CUDA, before and after #31247

Measurements for [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618).

They compare the same tokens processed two ways:
- **Decoded one at a time**, as when a cached prefix is reused.
- **Recomputed in one batch.**

The comparison is on llama.cpp before and after [#31247](https://github.com/ggml-org/llama.cpp/pull/31247),
which changed the per-block sum stored by the CUDA Q8_1 activation quantizers. All runs: Tesla T4 (sm_75), CUDA
12.6, Kaggle.

## Builds

| | Commit | What |
|---|---|---|
| **A** | [`740251a`](https://github.com/ggml-org/llama.cpp/commit/740251a920abf92b7744d9c0518ca157db0db579) | parent of #31247 ("before") |
| **B** | [`8fc0376`](https://github.com/ggml-org/llama.cpp/commit/8fc0376721e9662d498a5706d7ee8e7574ce99f4) | #31247 merged ("after") |
| **C** | `8fc0376` + `scripts/apply_i9_only.py` | B without the FAST_FP16 `__hmul2` branch in the Q4_1/Q5_1 MMVQ dot products |
| **D** | [`cd4ed90`](https://github.com/ggml-org/llama.cpp/commit/cd4ed902e174e54d7198105a720bb379a526309d) | master, 2026-11-10 |

A, B and D are unmodified upstream code.

## Results

**End to end:** median max |Δlogit| between the two paths, as a multiple of the same build's Q8_0, before (A) →
after (B).
- **Decoded:** the 48 tokens decoded one at a time (384 positions).
- **Suffix:** the 32 tokens that follow that history, i.e. the next turn after a cache hit vs after a recompute
  (256 positions).

| Qwen2.5 | Decoded, 1.5B | Decoded, 3B | Suffix, 1.5B | Suffix, 3B |
|---|---|---|---|---|
| Q4_0 | 2.57× → 1.01× | 2.31× → 0.98× | 1.50× → 1.13× | 1.17× → 1.25× |
| Q4_1 | 2.41× → 1.04× | 2.48× → 1.02× | 3.14× → 1.16× | 2.69× → 1.23× |
| Q5_1 | 2.33× → 0.98× | 2.30× → 0.99× | 2.41× → 1.02× | 2.53× → 1.08× |
| Q4_K_M | 1.60× → 0.98× | 1.62× → 1.02× | 1.85× → 1.06× | 2.20× → 1.08× |

| Llama-3.2-3B, decoded | A | B | C (no `__hmul2`) |
|---|---|---|---|
| Q4_0 | 2.04× | 1.04× | 1.04× |
| Q4_1 | 2.54× | 1.13× | 1.04× |
| Q5_1 | 2.65× | 1.33× | 1.00× |
| Q4_K_M | 1.67× | 1.00× | 1.00× |

Llama-3.2-3B suffix, A → B: Q4_0 1.09× → 1.04×, Q4_1 2.71× → 1.09×, Q5_1 2.66× → 1.12× (C: 1.03×),
Q4_K_M 2.03× → 1.00×.

- **Floor (Q8_0 and F16), unchanged by #31247**, median max |Δlogit|:

  | | Q8_0 decoded | Q8_0 suffix | F16 decoded | F16 suffix |
  |---|---|---|---|---|
  | Qwen2.5-1.5B | 0.341 | 0.401 | 0.110 | 0.141 |
  | Qwen2.5-3B | 0.325 | 0.577 | 0.110 | 0.211 |
  | Llama-3.2-3B | 0.173 | 0.183 | 0.095 | 0.102 |

  A and B are identical in every field for F16 and Q8_0.
- **Perplexity**, decode vs batch.
  - Setup: WikiText-2 test, ctx 512, 24 chunks, 6144 scored tokens. z = mean (decode − batch) NLL / its standard
    error over chunks.
  - Q4_0 z, A → B: Qwen2.5-1.5B +3.22 → +0.85; Qwen2.5-3B +2.98 → +0.16; Llama-3.2-3B +1.42 → +1.99 (both below 2).
  - B's NLL vs A's, every format, model and path: −0.32% to +0.007%.
  - Q4_0 batch perplexity is identical on A and B (10.717533, 9.360869).
- **Operator level** (`matmul_paths`, 1 token vs a batch of 64, relative L2, K = N = 1536, median of 16 tokens):

  | | A | B | C |
  |---|---|---|---|
  | Q4_0 | 2.02e-2 | 7.71e-4 | 7.71e-4 |
  | Q4_K | 1.14e-2 | 8.13e-4 | 8.13e-4 |
  | Q4_1 | 6.96e-4 | 7.02e-4 | 1.71e-7 |
  | Q5_1 | 7.29e-4 | 7.14e-4 | 1.74e-7 |
  | Q8_0 | 2.11e-4 | 2.11e-4 | 2.11e-4 |
  | F16 | 1.04e-3 | 1.04e-3 | 1.04e-3 |

  The two sessions (different T4 instances) give identical operator numbers.
- **Removing `__hmul2` (C vs B):**
  - End to end: no effect on Qwen2.5 (ratio 0.975–1.003). On Llama-3.2-3B, Q5_1 goes 1.33× → 1.00× and Q4_1 goes
    1.13× → 1.04× (decoded).
  - Perplexity unchanged (±0.03%).
  - Speed: llama-bench tg128, Qwen2.5-1.5B, 5 interleaved runs each: Q4_1 161.2 → 161.5 tok/s (+0.18%), Q5_1
    138.5 → 138.5 tok/s (−0.05%). One GPU only.
- **Master (D)** equals B on every logit and operator number.

## What #31247 changed

Both Q8_1 activation quantizers (`quantize_q8_1` for the decode path, MMVQ; `quantize_mmq_q8_1` for the batch path,
MMQ) used to store the sum of the original float values next to the scale. They now store scale × the integer sum
of the quantized codes.

- **Q4_0.** The decode path subtracts the offset using the stored sum
  ([vecdotq.cuh L136](https://github.com/ggml-org/llama.cpp/blob/740251a920abf92b7744d9c0518ca157db0db579/ggml/src/ggml-cuda/vecdotq.cuh#L136)).
  - On the T4, the batch path runs the Q8_0 dot product
    ([mmq.cuh L765-L769](https://github.com/ggml-org/llama.cpp/blob/8fc0376721e9662d498a5706d7ee8e7574ce99f4/ggml/src/ggml-cuda/mmq.cuh#L765-L769)).
  - That dot product reads only the scale
    ([mmq-vec-dot.cuh L261](https://github.com/ggml-org/llama.cpp/blob/740251a920abf92b7744d9c0518ca157db0db579/ggml/src/ggml-cuda/mmq-vec-dot.cuh#L261)).
- **Q4_K, the reverse.**
  - The decode path takes the min term from the sum of the codes
    ([vecdotq.cuh L522](https://github.com/ggml-org/llama.cpp/blob/740251a920abf92b7744d9c0518ca157db0db579/ggml/src/ggml-cuda/vecdotq.cuh#L522)).
  - The batch path takes it from the stored sum
    ([mmq-vec-dot.cuh L433](https://github.com/ggml-org/llama.cpp/blob/740251a920abf92b7744d9c0518ca157db0db579/ggml/src/ggml-cuda/mmq-vec-dot.cuh#L433)).
- **Q4_1 and Q5_1.** Both paths read the stored sum. The improvement is measured; the reason is not established.
- **`__hmul2` branch.** Still on master
  ([L158](https://github.com/ggml-org/llama.cpp/blob/cd4ed902e174e54d7198105a720bb379a526309d/ggml/src/ggml-cuda/vecdotq.cuh#L158),
  [L229](https://github.com/ggml-org/llama.cpp/blob/cd4ed902e174e54d7198105a720bb379a526309d/ggml/src/ggml-cuda/vecdotq.cuh#L229)).
  [#31019](https://github.com/ggml-org/llama.cpp/pull/31019) proposed the same removal, for FP16 overflow.

## Method

- **Models.**
  - Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct, from Qwen's FP16 GGUFs.
  - Llama-3.2-3B-Instruct, from bartowski's F16 GGUF.
  - Each was quantized once with A's `quantize_pure`, so every build reads the same files. Every tensor gets one
    type, including `output.weight`; Q4_K_M is the stock mix.
- **`kv_paths`** (end to end).
  - 8 fixed prompts (up to 160 tokens each) are processed as one batch. Then 48 greedy tokens are fed one at a
    time, then a 32-token suffix as one batch.
  - The same tokens are also processed as one batch. Both paths see the same tokens (teacher-forced).
  - Next-token logits are compared at the same positions. F16 KV cache, flash attention auto.
- **`ppl_paths`.** The second half of each 512-token chunk is scored token by token and as a batch.
- **`matmul_paths`.** The same weight and activation column go through `mul_mat` with 1 column and with 64.
  Inputs are seeded random, with fixed outlier channels.
- **Pre-registration.** The pass/fail thresholds were frozen before each run: [PREDICTIONS.md](PREDICTIONS.md).
  - Qwen2.5: 8 of 8 passed.
  - Llama-3.2-3B: 5 passed, 2 partial (both the Q5_1 `__hmul2` effect), none falsified.

## Reproduce

```bash
./build.sh                                   # A, B, C, D into work/build-*   (CUDA_ARCH=75 by default)
W=work
# quantize once with A's tool; model-F16.gguf is the downloaded FP16 file
for f in Q8_0 Q4_0 Q4_1 Q5_1 Q4_K; do $W/build-A/quantize_pure model-F16.gguf model-$f.gguf $f; done
$W/build-A/quantize_pure model-F16.gguf model-Q4_K_M.gguf Q4_K mix
mkdir -p out
for b in A B C D; do
  $W/build-$b/matmul_paths --K 1536 --N 1536 --B 64 --tokens 16 --out out/matmul_paths.jsonl
  for f in F16 Q8_0 Q4_0 Q4_1 Q5_1 Q4_K_M; do
    $W/build-$b/kv_paths --model model-$f.gguf --tag $b:Q15_$f --out out/kv_paths.jsonl
  done
done
for b in A B C; do for f in Q4_0 Q4_1 Q5_1 Q4_K; do
  $W/build-$b/ppl_paths --model model-$f.gguf --text wiki.test.raw --tag $b:Q15_$f --ctx 512 --chunks 24 \
    --out out/ppl_paths.jsonl
done; done
python3 scripts/score_4j.py out
```

- **Tags** are `<build>:<model>_<format>`, with model `Q15`, `Q3B` or `L3B`.
- **The scorer** applies the frozen T4 thresholds. On any other GPU it prints a "PLUMBING" warning, but the
  numbers still print.
- **WikiText-2:** `wiki.test.raw` from
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`.
- **What ran on Kaggle:** `scripts/kaggle_job.py --plan master` (Qwen2.5) and `--plan master --master-models L3B`.
  It ran these exact steps, plus the llama-bench comparison (`-ngl 99 -p 0 -n 128 -r 1`, order B C C B B C C B B
  C).
- **How the binaries were built for the runs:**
  - They were built for sm_61 and sm_75 with nvcc 12.6.
  - B was built as A plus `8fc0376`'s `quantize.cu`, the only file in which the two commits differ.
  - `build.sh` builds B from the `8fc0376` tarball instead, which is the same source.
  - `results/bundle.sha256` lists the binaries the runs used; each run's `env.json` records the same hashes.
- **Status of this section:** the commands above are the job's steps written out. They were not run outside the
  job.

## Files

- **`probes/`:** the probes, byte-for-byte as used. Their comments mention `docs/PROJECT.md` and
  `GGML_PROBE_FLAGS`, which belong to the earlier study; with these builds the flag does nothing and records `0`.
- **`scripts/`:**
  - `apply_i9_only.py` makes build C.
  - `score_4j.py` scores the results (`U*`/`V*` are the IDs in PREDICTIONS.md).
  - `kaggle_job.py` is the job that produced the results.
- **`results/kaggle-master/`:** Qwen2.5 raw JSONL, the job log, `env.json` (GPU, clocks, binary hashes) and the
  scorer output `score.txt`.
- **`results/kaggle-master-l3b/`:** the same for Llama-3.2-3B.

## Not tested

- Other backends (ROCm, Vulkan, Metal, CPU).
- Ampere and newer GPUs.
- Models beyond these three.
- Quantized KV cache.
- Speculative decoding.
- Text-level (episode) divergence.
- The speed of removing `__hmul2` on any GPU but the T4.
- The third `__hmul2`, in `vec_dot_q8_1_q8_1_impl`, which #31019 also removed.

## Note

The probe programs, scripts and analysis were written with an AI assistant (Claude, Anthropic) and checked by the
author.

## License

MIT, see [LICENSE](LICENSE).
