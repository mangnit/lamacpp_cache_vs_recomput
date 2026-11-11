#!/usr/bin/env python3
"""Night-1 GPU job: operator and end-to-end path probes on whatever GPU Kaggle grants.

Runs on Kaggle as a script kernel, with the probe binaries (built locally with nvcc
12.6 for sm_60 and sm_75) attached as a dataset. Also runs locally for a dry run:

    python probes/kaggle_job.py --local --bin build/probes-sm61 --model models/tiny/stories15M.gguf \
        --out results/local_dryrun --formats Q8_0,Q4_0 --tiny

Everything written goes to <out>/: env.json (GPU, driver, clocks, library versions),
matmul_paths.jsonl, kv_paths.jsonl and a log. Raw data only; analysis is separate.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HF_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-fp16.gguf"
CUBLAS_PINS = ["nvidia-cublas-cu12==12.6.0.22", "nvidia-cuda-runtime-cu12==12.6.37"]

#: Pure formats (one tensor type everywhere) + the two stock mixtures the paper used.
FORMATS = ["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q2_K"]
MIXES = ["Q4_K", "Q3_K"]
#: GGML_PROBE_FLAGS settings (docs/PROJECT.md): 0 baseline, 1 I1, 2 I2, 4 I4, 8 I5, 15 all numeric, 16 I6.
MATMUL_FLAGS = ["0", "1", "2", "4", "8", "7", "15", "16"]
#: 16 = I6 (decode also uses MMQ): the path-invariance mitigation; its latency is in decode_ms.
KV_FLAGS = ["0", "4", "15", "16", "31"]


def log(out: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "job.log", "a") as f:
        f.write(line + "\n")


def sh(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    return r


def gpu_snapshot() -> dict:
    q = "name,driver_version,compute_cap,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)
    return {"nvidia_smi": r.stdout.strip().splitlines(), "t": time.time()}


def cublas_env(out: Path, local: bool) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # T4x2: never split a model across two GPUs
    if local:
        return env
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *CUBLAS_PINS], capture_output=True, text=True)
    log(out, f"pip {' '.join(CUBLAS_PINS)} -> rc={r.returncode} {r.stderr[-300:]}")
    import site
    libs = []
    for sp in site.getsitepackages():
        libs += glob.glob(os.path.join(sp, "nvidia", "cublas", "lib")) + glob.glob(os.path.join(sp, "nvidia", "cuda_runtime", "lib"))
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return env


#: docs/PROJECT.md 4g: the two follow-up turns appended to every MT-Bench episode, fixed before any run.
EPISODE_FOLLOWUPS = (
    "Thank you. Now summarize everything you have said so far in three sentences.",
    "Finally, list three questions a curious reader might ask next, each with a one-line answer.",
)
MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
QWEN7B = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-"
QWEN7B_PARTS = {"Q8_0": ("q8_0", 3), "Q4_K_M": ("q4_k_m", 2), "Q4_0": ("q4_0", 2)}


def http_json(url: str, body: dict | None = None, timeout: float = 900) -> dict:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from None


def load_episodes(work: Path, limit: int) -> list[dict]:
    """The 80 MT-Bench questions in ascending id; each episode = its 2 turns + EPISODE_FOLLOWUPS."""
    p = work / "mt_bench_question.jsonl"
    if not p.exists():
        sh(["wget", "-q", "-O", str(p), MT_BENCH_URL])
    qs = sorted((json.loads(l) for l in p.read_text().splitlines() if l.strip()), key=lambda q: q["question_id"])
    return [{"id": q["question_id"], "turns": list(q["turns"]) + list(EPISODE_FOLLOWUPS)} for q in qs[:limit]]


def run_episodes(server: Path, model: Path, tag: str, flags: str, arm: str, episodes: list[dict], out: Path,
                 env: dict, n_predict: int, extra: list[str] | None = None) -> None:
    """One pass (docs/PROJECT.md 4g): a fresh llama-server with one slot and the server-level prompt cache off,
    every episode in order, one arm. cache_prompt=true reuses the slot's KV (prompt + generated tokens) for the
    common prefix of the next request; false re-evaluates the whole prompt. Greedy decoding."""
    port = 18080
    logp = out.parent / "server_logs" / f"{tag}_{flags}_{arm}.log"  # outside results/, not returned
    logp.parent.mkdir(exist_ok=True)
    with open(logp, "w") as logf:
        proc = subprocess.Popen([str(server), "-m", str(model), "-ngl", "999", "-np", "1", "-c", "16384",
                                 "--cache-ram", "0", "--host", "127.0.0.1", "--port", str(port), "--seed", "42"] + (extra or []),
                                env=dict(env, GGML_PROBE_FLAGS=flags), stdout=logf, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        try:
            t0 = time.time()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited rc={proc.returncode}: {logp.read_text()[-600:]}")
                try:
                    if http_json(base + "/health", timeout=5).get("status") == "ok":
                        break
                except Exception:
                    pass
                if time.time() - t0 > 900:
                    raise RuntimeError("llama-server did not become healthy in 900 s")
                time.sleep(2)
            with open(out / "episodes.jsonl", "a") as fo:
                for ep in episodes:
                    messages = []
                    for k, user in enumerate(ep["turns"]):
                        messages.append({"role": "user", "content": user})
                        prompt = http_json(base + "/apply-template", {"messages": messages})["prompt"]
                        r = http_json(base + "/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                                                             "seed": 42, "cache_prompt": arm == "cache",
                                                             "return_tokens": True})
                        t = r.get("timings", {})
                        fo.write(json.dumps({"probe": "episodes", "llama_cpp": "b10434", "tag": tag, "model": model.name,
                                             "probe_flags": flags, "arm": arm, "episode": ep["id"], "request": k,
                                             "tokens": r.get("tokens", []), "predicted_n": t.get("predicted_n"),
                                             "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                                             "tokens_cached": r.get("tokens_cached"), "stop_type": r.get("stop_type"),
                                             "predicted_ms": t.get("predicted_ms")}) + "\n")
                        messages.append({"role": "assistant", "content": r.get("content", "")})
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()


#: docs/PROJECT.md 4j: four llama.cpp builds without the probe patch. Bundle files are flat, "<build>.<name>", so a
#: change in how Kaggle mounts a dataset cannot mix the builds up.
MASTER_BUILDS = ("A", "B", "C", "D")
QWEN3B_F16 = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-fp16-"
MASTER_MODELS = (("Q15", [HF_URL]), ("Q3B", [f"{QWEN3B_F16}{i:05d}-of-00002.gguf" for i in (1, 2)]))
#: docs/PROJECT.md 4k: the same builds on a second model family (the F16 the course runs used).
L3B_F16 = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf"
MASTER_URLS = dict(MASTER_MODELS) | {"L3B": [L3B_F16]}
MASTER_KV = ("F16", "Q8_0", "Q4_0", "Q4_1", "Q5_1", "Q4_K_M")
MASTER_PPL = ("Q4_0", "Q4_1", "Q5_1", "Q4_K")


def run_master(a, out: Path) -> None:
    """docs/PROJECT.md 4j. The GGUFs are quantized once, with build A's tool, so the code is the only variable.
    Most informative first, so a time-out loses least."""
    import hashlib
    import zipfile
    src = Path(a.bin) if a.bin else None
    if src is None:
        hits = glob.glob("/kaggle/input/**/A.matmul_paths", recursive=True)
        if not hits:
            raise SystemExit("probe binaries not found under /kaggle/input")
        src = Path(hits[0]).parent
    root = Path("/tmp/cs6886_bin") if not a.local else out / "bin"
    for f in src.iterdir():
        build, name = f.name.split(".", 1)
        (root / build).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, root / build / name); os.chmod(root / build / name, 0o755)
    base = cublas_env(out, a.local)

    def env_of(build: str) -> dict:
        return dict(base, LD_LIBRARY_PATH=":".join([str(root / build), base.get("LD_LIBRARY_PATH", "")]))

    def exe(build: str, name: str) -> str:
        return str(root / build / name)

    def clocks() -> str:
        r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                            "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip()

    envinfo = {"start": gpu_snapshot(), "python": sys.version, "local": a.local,
               "bin_sha256": {f"{d.name}/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest()
                              for d in sorted(root.iterdir()) for f in sorted(d.iterdir())}}
    (out / "env.json").write_text(json.dumps(envinfo, indent=2))
    log(out, f"plan=master builds={MASTER_BUILDS} GPU: {envinfo['start']['nvidia_smi']}")

    work = Path("/tmp/cs6886_models") if not a.local else out / "models"
    work.mkdir(parents=True, exist_ok=True)
    K = "256" if a.tiny else "1536"
    for build in MASTER_BUILDS:
        t0 = time.time()
        r = sh([exe(build, "matmul_paths"), "--K", K, "--N", K, "--B", "64", "--tokens", "16",
                "--out", str(out / "matmul_paths.jsonl")], env=env_of(build), check=False)
        log(out, f"matmul_paths {build} rc={r.returncode} {time.time()-t0:.0f}s clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")

    wiki = work / "wiki.test.raw"
    if not wiki.exists():
        if a.local:
            shutil.copy2("models/text/wikitext-2-raw/wiki.test.raw", wiki)
        else:
            z = work / "wikitext-2-raw-v1.zip"
            sh(["wget", "-q", "-O", str(z), "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"])
            with zipfile.ZipFile(z) as zf, open(wiki, "wb") as fo:
                fo.write(zf.read("wikitext-2-raw/wiki.test.raw"))

    selected = [(P, MASTER_URLS[P]) for P in a.master_models.split(",")]
    for P, urls in (selected[:1] if a.local else selected):
        if a.model:
            f16 = Path(a.model)
        else:
            t0 = time.time()
            for u in urls:
                sh(["wget", "-q", "-O", str(work / u.rsplit("/", 1)[1]), u])
            f16 = work / urls[0].rsplit("/", 1)[1]  # llama.cpp finds the remaining split parts itself
            log(out, f"downloaded {P} ({len(urls)} part(s)) in {time.time()-t0:.0f}s")
        models = {"F16": f16}
        for fmt in sorted((set(MASTER_KV) | set(MASTER_PPL)) - {"F16"}):
            q = work / f"{P}-{fmt}.gguf"
            r = sh([exe("A", "quantize_pure"), str(f16), str(q), "Q4_K" if fmt == "Q4_K_M" else fmt] + (["mix"] if fmt == "Q4_K_M" else []),
                   env=env_of("A"), check=False)
            log(out, f"quantize {P} {fmt} rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")
            models[fmt] = q
        for build in MASTER_BUILDS:
            for fmt in MASTER_KV:
                t0 = time.time()
                args = [exe(build, "kv_paths"), "--model", str(models[fmt]), "--tag", f"{build}:{P}_{fmt}", "--out", str(out / "kv_paths.jsonl")]
                r = sh(args + (["--P", "64", "--G", "16", "--S", "16", "--prompts", "3"] if a.tiny else []), env=env_of(build), check=False)
                log(out, f"kv_paths {build} {P}_{fmt} rc={r.returncode} {time.time()-t0:.0f}s clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")
        for build in ("A", "B", "C"):
            for fmt in MASTER_PPL:
                t0 = time.time()
                args = [exe(build, "ppl_paths"), "--model", str(models[fmt]), "--text", str(wiki), "--tag", f"{build}:{P}_{fmt}",
                        "--out", str(out / "ppl_paths.jsonl")] + (["--ctx", "128", "--chunks", "1"] if a.tiny else ["--ctx", "512", "--chunks", "24"])
                r = sh(args, env=env_of(build), check=False)
                log(out, f"ppl_paths {build} {P}_{fmt} rc={r.returncode} {time.time()-t0:.0f}s clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")
        if P == "Q15":  # U7: the cost of I9 alone, ABBA so that drift cancels
            for fmt in ("Q4_1", "Q5_1"):
                for rep, build in enumerate(("B", "C", "C", "B", "B", "C", "C", "B", "B", "C")):
                    c0 = clocks()
                    r = sh([exe(build, "llama-bench"), "-m", str(models[fmt]), "-ngl", "99", "-p", "0", "-n", "16" if a.tiny else "128",
                            "-r", "1", "-o", "jsonl"], env=env_of(build), check=False)
                    with open(out / "llama_bench.jsonl", "a") as fo:
                        fo.write(json.dumps({"probe": "llama_bench", "build": build, "format": fmt, "rep": rep, "rc": r.returncode,
                                             "clocks_start": c0, "clocks_end": clocks(), "stdout": r.stdout.strip()[-3000:]}) + "\n")
                    log(out, f"llama-bench {build} {fmt} rep={rep} rc={r.returncode} clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")
        for fmt, q in models.items():
            if fmt != "F16":
                q.unlink(missing_ok=True)
        if not a.model:
            for u in urls:
                (work / u.rsplit("/", 1)[1]).unlink(missing_ok=True)
    envinfo["end"] = gpu_snapshot()
    (out / "env.json").write_text(json.dumps(envinfo, indent=2))
    log(out, "done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--bin", default=None, help="directory holding matmul_paths, kv_paths, quantize_pure")
    ap.add_argument("--model", default=None, help="local F16/F32 GGUF instead of downloading Qwen2.5-1.5B")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--mixes", default=",".join(MIXES))
    ap.add_argument("--tiny", action="store_true", help="small shapes for a dry run")
    ap.add_argument("--skip-7b", action="store_true")
    ap.add_argument("--plan", default="night1", choices=["night1", "night2", "gen", "p100", "bisect", "i8", "explain", "final", "close",
                             "episodes", "episodes7b", "costbf", "master"],
                    help="night2: I7/M6 test; gen: decode-vs-batch PPL on another model (--src-url); "
                         "p100: operator+gate+1.5B grid for a P100 (M0)")
    ap.add_argument("--src-url", default="", help="comma-separated URL(s) of an F16 GGUF (split parts in order)")
    ap.add_argument("--ep-limit", type=int, default=80, help="episodes per pass (4g: all 80 MT-Bench questions)")
    ap.add_argument("--ep-predict", type=int, default=384, help="max generated tokens per response (4g: 384)")
    ap.add_argument("--ep-repeat", action="store_true", help="4g H5: repeat the Q4_K_M upstream cache pass")
    ap.add_argument("--ep-server-args", default="", help="extra llama-server arguments (local plumbing tests only)")
    ap.add_argument("--no-f16", action="store_true", help="skip the F16 reference runs (the model is only quantized from it)")
    ap.add_argument("--ppl-formats", default="", help="only these formats get PPL runs, e.g. Q4_0,Q5_0 (default: all)")
    ap.add_argument("--tag-prefix", default="", help="prefix for result tags, e.g. Q3B_")
    ap.add_argument("--master-models", default="Q15,Q3B", help="plan master: models to run, from Q15, Q3B, L3B (4k: L3B)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.plan == "master":
        run_master(a, out)
        return
    global MATMUL_FLAGS, KV_FLAGS
    import random
    rng = random.Random(6886)
    PPL_FLAGS = ["0", "4", "16"]
    PPL_CHUNKS = 8
    if a.plan == "gen":
        MATMUL_FLAGS = []
        KV_FLAGS = ["0", "4"]
        PPL_FLAGS = ["0", "4"]
        PPL_CHUNKS = 24
        a.skip_7b = True
        if a.formats == ",".join(FORMATS):
            a.formats = "Q8_0,Q4_0,Q5_0,Q4_1,Q5_1,Q4_K,Q5_K,Q6_K"
        if a.mixes == ",".join(MIXES):
            a.mixes = "Q4_K"
    if a.plan == "p100":
        MATMUL_FLAGS = ["0", "4", "15", "16"]
        KV_FLAGS = ["0", "4", "16"]
        PPL_FLAGS = ["0", "4", "16"]
        a.skip_7b = True
        if a.formats == ",".join(FORMATS):
            a.formats = "Q8_0,Q4_0,Q5_0,Q4_1,Q4_K,Q6_K"
        if a.mixes == ",".join(MIXES):
            a.mixes = ""
    if a.plan == "bisect":
        MATMUL_FLAGS = ["0", "47"]  # adds the K = 8960 ffn_down shape below, which night 3 did not probe
        a.skip_7b = True
    if a.plan == "final":
        MATMUL_FLAGS = ["0", "4", "128", "256", "384"]
        a.skip_7b = True
    if a.plan == "explain":
        MATMUL_FLAGS = []
        a.skip_7b = True
    if a.plan in ("episodes", "episodes7b"):
        # docs/PROJECT.md 4g Part 1: llama-server cache_prompt on vs off, upstream vs G, MT-Bench episodes.
        MATMUL_FLAGS = []
        a.skip_7b = True
        if a.formats == ",".join(FORMATS):
            # 4g: episodes7b uses the official Qwen2.5-7B GGUFs, which have no F16 (QWEN7B_PARTS)
            a.formats = "Q8_0,Q4_K_M,Q4_0" if a.plan == "episodes7b" else "F16,Q8_0,Q4_K_M,Q4_0"
    if a.plan == "costbf":
        # docs/PROJECT.md 4g Parts 2-3: ABBA cost of G (K1-K3) and the BF16-precision emulation I11 (BF1-BF2).
        MATMUL_FLAGS = ["0", "512", "384", "896"]
        a.skip_7b = True
    if a.plan == "close":
        # docs/PROJECT.md 4f (E1-E4): upstream vs the global code-sum fix G = 384 (I9 + I10) on one model
        # (--src-url, default Qwen2.5-1.5B); Q4_1/Q5_1 also at 128 (I9 alone, the M8 test).
        MATMUL_FLAGS = []
        KV_FLAGS = ["0", "384"]
        PPL_FLAGS = ["0", "384"]
        PPL_CHUNKS = 24
        a.skip_7b = True
        if a.formats == ",".join(FORMATS):
            a.formats = "Q8_0,Q4_0,Q5_0,Q4_1,Q5_1,Q4_K,Q5_K"
        if a.mixes == ",".join(MIXES):
            a.mixes = "Q4_K"
    if a.plan == "i8":
        # Frozen predictions B1-B6 (docs/PROJECT.md 4b): I8 = bit 64, MMQ code-sum for Q4_K/Q5_K.
        MATMUL_FLAGS = ["0", "4", "64", "47"]
        KV_FLAGS = ["0", "64"]
        PPL_FLAGS = ["0", "64"]
        a.skip_7b = True
        if a.formats == ",".join(FORMATS):
            a.formats = "Q4_K,Q5_K,Q8_0"
        if a.mixes == ",".join(MIXES):
            a.mixes = "Q4_K"
    if a.plan == "night2":
        # bit 32 = I7 (Turing only): MMVQ reproduces MMQ-MMA's FP16 sub-scale products for
        # Q4_K/Q5_K (implies I4 for those types). 47 = I7 + all numeric (1|2|4|8).
        MATMUL_FLAGS = ["0", "4", "32", "15", "47"]
        KV_FLAGS = ["0", "4", "32", "47"]
        PPL_FLAGS = ["0", "4", "32", "47"]
        if a.formats == ",".join(FORMATS):
            a.formats = "Q4_K,Q5_K,Q4_0,Q5_0,Q4_1,Q5_1,Q8_0"
        if a.mixes == ",".join(MIXES):
            a.mixes = "Q4_K"
    log(out, f"plan={a.plan} matmul_flags={MATMUL_FLAGS} kv_flags={KV_FLAGS} ppl_flags={PPL_FLAGS} "
             f"ppl_chunks={PPL_CHUNKS} formats={a.formats} mixes={a.mixes} src={a.src_url or 'qwen2.5-1.5b'}")
    P = a.tag_prefix
    work = Path("/tmp/cs6886_models") if not a.local else out / "models"
    work.mkdir(parents=True, exist_ok=True)

    # Binaries: dataset files are read-only and may lose the exec bit.
    if a.bin:
        src = Path(a.bin)
    else:  # the dataset mount path has changed across Kaggle versions: find it
        hits = glob.glob("/kaggle/input/**/matmul_paths", recursive=True)
        if not hits:
            raise SystemExit("probe binaries not found under /kaggle/input")
        src = Path(hits[0]).parent
    # Outside /kaggle/working, so ~290 MB of binaries are not returned as job output.
    bindir = Path("/tmp/cs6886_bin") if not a.local else src
    if not a.local:
        bindir.mkdir(exist_ok=True)
        for f in src.iterdir():  # executables and the shared ggml/llama libraries
            shutil.copy2(f, bindir / f.name); os.chmod(bindir / f.name, 0o755)
    env = cublas_env(out, a.local)
    env["LD_LIBRARY_PATH"] = ":".join([str(bindir), env.get("LD_LIBRARY_PATH", "")])

    import hashlib
    envinfo = {"start": gpu_snapshot(), "python": sys.version, "local": a.local,
               # proves which build actually ran (night 2 silently mounted a stale bundle)
               "bin_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(bindir.iterdir()) if f.is_file()}}
    for lib in ("libcublas.so.12", "libcudart.so.12"):
        r = subprocess.run(["bash", "-c", f"ldd {bindir}/matmul_paths | grep {lib}"], env=env, capture_output=True, text=True)
        envinfo[lib] = r.stdout.strip()
    (out / "env.json").write_text(json.dumps(envinfo, indent=2))
    log(out, f"GPU: {envinfo['start']['nvidia_smi']}")

    # 1. Operator probe, every flag setting, two shapes (attention/MLP-in and MLP-down).
    shapes = [("1536", "1536")] if not a.tiny else [("256", "256")]
    if not a.tiny and a.plan in ("night1", "bisect", "i8"):
        shapes.append(("8960", "1536"))
    for K, N in shapes:
        for fl in MATMUL_FLAGS:
            e = dict(env, GGML_PROBE_FLAGS=fl)
            r = sh([str(bindir / "matmul_paths"), "--K", K, "--N", N, "--B", "64", "--tokens", "16",
                    "--out", str(out / "matmul_paths.jsonl")], env=e, check=False)
            log(out, f"matmul_paths K={K} N={N} flags={fl} rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

    # 1b. Gate transfer function (mechanism M5): tiny input perturbations through one path.
    if (bindir / "gate_probe").exists() and a.plan in ("night1", "p100", "i8"):
        for K, N in shapes[:1]:
            for gfl in (["0", "64"] if a.plan == "i8" else ["0"]):
                r = sh([str(bindir / "gate_probe"), "--K", K, "--N", N, "--types", "f16,q8_0,q4_0,q4_1,q5_0,q4_K,q5_K,q6_K",
                        "--out", str(out / "gate.jsonl")], env=dict(env, GGML_PROBE_FLAGS=gfl), check=False)
                log(out, f"gate_probe K={K} N={N} flags={gfl} rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")

    # 2. End-to-end probe on real models.
    if a.model:
        f16 = Path(a.model)
    elif a.src_url:
        urls = [u for u in a.src_url.split(",") if u]
        t0 = time.time()
        for u in urls:
            dst = work / u.rsplit("/", 1)[1]
            if not dst.exists():
                sh(["wget", "-q", "-O", str(dst), u])
        f16 = work / urls[0].rsplit("/", 1)[1]  # llama.cpp finds the remaining split parts itself
        log(out, f"downloaded {len(urls)} part(s) of {f16.name} in {time.time()-t0:.0f}s")
    elif a.plan == "episodes7b":
        f16 = None  # official quantized Qwen2.5-7B files, downloaded per format below
    else:
        f16 = work / "qwen2.5-1.5b-instruct-fp16.gguf"
        if not f16.exists():
            t0 = time.time()
            sh(["wget", "-q", "-O", str(f16), HF_URL])
            log(out, f"downloaded {f16.name} {f16.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s")

    def clocks() -> str:
        r = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                            "--format=csv,noheader"], capture_output=True, text=True)
        return r.stdout.strip()

    def order(flags: list[str]) -> list[str]:
        if a.plan == "night1":
            return list(flags)
        f = list(flags); rng.shuffle(f); return f  # randomized so drift is not confounded with flags

    def run_kv(model: Path, tag: str, flags: list[str], extra_env: dict | None = None) -> None:
        for fl in order(flags):
            e = dict(env, GGML_PROBE_FLAGS=fl, **(extra_env or {}))
            args = [str(bindir / "kv_paths"), "--model", str(model), "--tag", tag, "--out", str(out / "kv_paths.jsonl")]
            if a.tiny:
                args += ["--P", "64", "--G", "16", "--S", "16", "--prompts", "3"]
            t0 = time.time()
            r = sh(args, env=e, check=False)
            log(out, f"kv_paths {tag} flags={fl} rc={r.returncode} {time.time()-t0:.0f}s clocks[{clocks()}] {r.stderr[-400:] if r.returncode else ''}")

    wiki = work / "wiki.test.raw"
    if not wiki.exists():
        import zipfile
        if a.local and Path("models/text/wikitext-2-raw/wiki.test.raw").exists():
            shutil.copy2("models/text/wikitext-2-raw/wiki.test.raw", wiki)
        else:
            z = work / "wikitext-2-raw-v1.zip"
            sh(["wget", "-q", "-O", str(z), "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"])
            with zipfile.ZipFile(z) as zf, open(wiki, "wb") as fo:
                fo.write(zf.read("wikitext-2-raw/wiki.test.raw"))

    def run_ppl(model: Path, tag: str, flags: list[str], extra_env: dict | None = None) -> None:
        if not (bindir / "ppl_paths").exists():
            return
        for fl in order(flags):
            args = [str(bindir / "ppl_paths"), "--model", str(model), "--text", str(wiki), "--tag", tag,
                    "--out", str(out / "ppl_paths.jsonl")] + (["--ctx", "128", "--chunks", "1"] if a.tiny else ["--ctx", "512", "--chunks", str(PPL_CHUNKS)])
            t0 = time.time()
            r = sh(args, env=dict(env, GGML_PROBE_FLAGS=fl, **(extra_env or {})), check=False)
            log(out, f"ppl_paths {tag} flags={fl} rc={r.returncode} {time.time()-t0:.0f}s clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")

    if a.plan in ("episodes", "episodes7b"):
        episodes = load_episodes(work, a.ep_limit)
        server = bindir / "llama-server"
        log(out, f"episodes: {len(episodes)} x {len(episodes[0]['turns'])} requests, n_predict={a.ep_predict}, "
                 f"follow-ups={EPISODE_FOLLOWUPS!r}")

        def passes(model: Path, tag: str, gflags: list[str]) -> None:
            for fl in gflags:
                for arm in ("cache", "recompute"):
                    t0, c0 = time.time(), clocks()
                    try:
                        run_episodes(server, model, tag, fl, arm, episodes, out, env, a.ep_predict, a.ep_server_args.split())
                        log(out, f"episodes {tag} flags={fl} arm={arm} ok {time.time()-t0:.0f}s clocks[{c0}] -> [{clocks()}]")
                    except Exception as e:  # keep going: one failed pass must not lose the others
                        log(out, f"episodes {tag} flags={fl} arm={arm} FAILED {time.time()-t0:.0f}s: {str(e)[:600]}")
            if a.ep_repeat and tag.endswith("Q4_K_M"):  # H5: run to run, same arm again
                t0 = time.time()
                run_episodes(server, model, tag + "_rep", "0", "cache", episodes, out, env, a.ep_predict, a.ep_server_args.split())
                log(out, f"episodes {tag}_rep flags=0 arm=cache ok {time.time()-t0:.0f}s")

        for fmt in [f for f in a.formats.split(",") if f]:
            gflags = ["0", "384"] if fmt in ("Q4_K_M", "Q4_0") else ["0"]
            if a.plan == "episodes7b":
                stem, n = QWEN7B_PARTS[fmt]
                parts = [f"{stem}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
                t0 = time.time()
                for part in parts:
                    sh(["wget", "-q", "-O", str(work / part), f"{QWEN7B}{part}"])
                log(out, f"downloaded Qwen2.5-7B {fmt} ({n} parts) in {time.time()-t0:.0f}s")
                passes(work / parts[0], P + fmt, gflags)
                for part in parts:
                    (work / part).unlink()
            else:
                if fmt == "F16":
                    q = f16
                else:
                    q = work / f"model-{fmt}.gguf"
                    base = "Q4_K" if fmt == "Q4_K_M" else fmt
                    sh([str(bindir / "quantize_pure"), str(f16), str(q), base] + (["mix"] if fmt == "Q4_K_M" else []), env=env)
                passes(q, P + fmt, gflags)
                if fmt != "F16":
                    q.unlink()
        envinfo["end"] = gpu_snapshot()
        (out / "env.json").write_text(json.dumps(envinfo, indent=2))
        log(out, "done")
        return
    if a.plan == "costbf":
        def quantb(fmt: str) -> Path:
            q = work / f"model-{fmt}.gguf"
            if not q.exists():
                sh([str(bindir / "quantize_pure"), str(f16), str(q), fmt], env=env)
            return q
        # BF2 (and recorded rows): end to end with the BF16-precision emulation I11 (512) and G + I11 (896).
        for fmt, fls in (("Q8_0", ("0", "512")), ("Q6_K", ("0", "512")), ("Q4_0", ("0", "512", "384", "896"))):
            for fl in fls:
                r = sh([str(bindir / "kv_paths"), "--model", str(quantb(fmt)), "--tag", "BF_" + fmt, "--out", str(out / "kv_paths.jsonl")],
                       env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                log(out, f"kv_paths BF_{fmt} flags={fl} rc={r.returncode} clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")
        # K1-K3: the cost of G, ABBA (drift cancels); Q4_0 adds I4 in a balanced three-way order for K2.
        orders = {"Q4_0": ("0", "384", "4", "4", "384", "0", "0", "384", "4", "4", "384", "0", "0", "384", "4"),
                  "Q5_0": ("0", "384", "384", "0", "0", "384", "384", "0", "0", "384"),
                  "Q4_1": ("0", "384", "384", "0", "0", "384", "384", "0", "0", "384"),
                  "Q4_K": ("0", "384", "384", "0", "0", "384", "384", "0", "0", "384")}
        for fmt, order in orders.items():
            q = quantb(fmt)
            for i, fl in enumerate(order):
                r = sh([str(bindir / "kv_paths"), "--model", str(q), "--tag", f"LAT_{fmt}_r{i}", "--out", str(out / "latency.jsonl")],
                       env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                log(out, f"latency {fmt} rep={i} flags={fl} rc={r.returncode} clocks[{clocks()}]")
        for f in work.glob("model-*.gguf"):
            f.unlink()
        envinfo["end"] = gpu_snapshot()
        (out / "env.json").write_text(json.dumps(envinfo, indent=2))
        log(out, "done")
        return
    if a.plan == "final":
        # docs/PROJECT.md 4d: C3 (M8 test on Q4_1/Q5_1) and C4 (costs, ABBA-interleaved repeats).
        def quant(fmt: str) -> Path:
            q = work / f"model-{fmt}.gguf"
            if not q.exists():
                sh([str(bindir / "quantize_pure"), str(f16), str(q), fmt], env=env, check=False)
            return q
        for fmt in ("Q4_1", "Q5_1"):
            q = quant(fmt)
            for fl in ("0", "128", "384"):
                r = sh([str(bindir / "kv_paths"), "--model", str(q), "--tag", fmt, "--out", str(out / "kv_paths.jsonl")],
                       env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                log(out, f"kv_paths {fmt} flags={fl} rc={r.returncode} clocks[{clocks()}] {r.stderr[-300:] if r.returncode else ''}")
        for fmt, alt in (("Q4_K", "64"), ("Q4_1", "256"), ("Q4_0", "4"), ("Q5_0", "4")):
            q = quant(fmt)
            for i, fl in enumerate(("0", alt, alt, "0", "0", alt, alt, "0", "0", alt)):  # ABBA: drift cancels
                r = sh([str(bindir / "kv_paths"), "--model", str(q), "--tag", f"LAT_{fmt}_r{i}", "--out", str(out / "latency.jsonl")],
                       env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                log(out, f"latency {fmt} rep={i} flags={fl} rc={r.returncode} clocks[{clocks()}]")
        for f in work.glob("model-*.gguf"):
            f.unlink()
        envinfo["end"] = gpu_snapshot()
        (out / "env.json").write_text(json.dumps(envinfo, indent=2))
        log(out, "done")
        return
    if a.plan == "explain":
        # docs/PROJECT.md 4c (P1-P4), capped: real-data replay + propagation, nothing else.
        for fmt in ("Q4_0", "Q5_0", "Q8_0"):
            q = work / f"model-{fmt}.gguf"
            sh([str(bindir / "quantize_pure"), str(f16), str(q), fmt], env=env, check=False)
            for fl in ("0", "4"):
                t0 = time.time()
                r = sh([str(bindir / "real_replay"), "--model", str(q), "--text", str(wiki), "--tokens", "64", "--rows", "256",
                        "--tok-sample", "4", "--clip", "0,5,4,3.5,3", "--tag", P + fmt, "--out", str(out / "real_replay.jsonl")],
                       env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                log(out, f"real_replay {fmt} flags={fl} rc={r.returncode} {time.time()-t0:.0f}s {r.stderr[-300:] if r.returncode else ''}")
                if fmt == "Q4_0":
                    for pr in ("0", "1", "2"):
                        r = sh([str(bindir / "kv_bisect"), "--model", str(q), "--filter", "l_out", "--prompt", pr, "--tag", P + fmt,
                                "--out", str(out / "kv_bisect.jsonl")], env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                        log(out, f"kv_bisect {fmt} flags={fl} prompt={pr} rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")
            q.unlink()
        envinfo["end"] = gpu_snapshot()
        (out / "env.json").write_text(json.dumps(envinfo, indent=2))
        log(out, "done")
        return
    if a.plan in ("bisect", "i8"):
        # Op-by-op decode-vs-recompute bisection for one token (open item: K-quant e2e residual).
        for fmt in (("Q4_K", "Q4_0", "Q8_0") if a.plan == "bisect" else ("Q4_K", "Q5_K", "Q4_0")):
            q = work / f"model-{fmt}.gguf"
            sh([str(bindir / "quantize_pure"), str(f16), str(q), fmt], env=env, check=False)
            for fl in (("0", "4", "47") if a.plan == "bisect" else ("0", "64")):
                for pr in ("0", "1", "2"):
                    r = sh([str(bindir / "kv_bisect"), "--model", str(q), "--filter", "", "--prompt", pr, "--tag", fmt,
                            "--out", str(out / "kv_bisect.jsonl")], env=dict(env, GGML_PROBE_FLAGS=fl), check=False)
                    log(out, f"kv_bisect {fmt} flags={fl} prompt={pr} rc={r.returncode} {r.stderr[-300:] if r.returncode else ''}")
            q.unlink()
        if a.plan == "bisect":
            envinfo["end"] = gpu_snapshot()
            (out / "env.json").write_text(json.dumps(envinfo, indent=2))
            log(out, "done")
            return
    if not a.no_f16:  # on the 2 GB MX330 the F16 model spills into shared memory and takes hours (4i run, 2026-10-29)
        run_kv(f16, P + "F16", ["0"])
        run_ppl(f16, P + "F16", ["0"])
        if a.plan != "night1":
            run_ppl(f16, P + "F16_cublas_f32", ["0"], {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"})
        run_kv(f16, P + "F16_cublas_f32", ["0"], {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"})
    jobs = [(fmt, False) for fmt in a.formats.split(",") if fmt] + [(fmt, True) for fmt in a.mixes.split(",") if fmt]
    for fmt, mix in jobs:
        tag = P + (f"{fmt}_M" if mix else fmt)
        q = work / f"model-{tag}.gguf"
        t0 = time.time()
        r = sh([str(bindir / "quantize_pure"), str(f16), str(q), fmt] + (["mix"] if mix else []), env=env, check=False)
        if r.returncode != 0:
            log(out, f"quantize {tag} failed: {r.stderr[-400:]}")
            continue
        log(out, f"quantized {tag} in {time.time()-t0:.0f}s ({q.stat().st_size/1e9:.2f} GB)")
        i9 = ["128"] if a.plan == "close" and fmt in ("Q4_1", "Q5_1") and not mix else []
        run_kv(q, tag, KV_FLAGS + i9)
        if not a.ppl_formats or (f"{fmt}_M" if mix else fmt) in a.ppl_formats.split(","):
            run_ppl(q, tag, PPL_FLAGS + i9)
        q.unlink()
    # 3. The paper's own model (Qwen2.5-7B, official GGUF quantizations). F16 7B does not fit a
    #    16 GB T4, so only the two quantized files the paper used. Last, so a time-out loses least.
    if not a.local and not a.skip_7b:
        base = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
        models7b = (("7B_Q4_K_M", ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"]),
                    ("7B_Q8_0", ["qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf", "qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf",
                                 "qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf"]))
        for tag, parts in (models7b if a.plan == "night1" else models7b[:1]):
            t0 = time.time()
            try:
                for p in parts:
                    sh(["wget", "-q", "-O", str(work / p), base + p])
            except RuntimeError as e:
                log(out, f"download {tag} failed: {str(e)[:300]}")
                continue
            log(out, f"downloaded {tag} in {time.time()-t0:.0f}s")
            run_kv(work / parts[0], tag, ["0", "4", "15", "16"] if a.plan == "night1" else KV_FLAGS)
            run_ppl(work / parts[0], tag, ["0", "4", "16"] if a.plan == "night1" else PPL_FLAGS)
            for p in parts:
                (work / p).unlink(missing_ok=True)
    envinfo["end"] = gpu_snapshot()
    (out / "env.json").write_text(json.dumps(envinfo, indent=2))
    log(out, "done")


if __name__ == "__main__":
    main()
