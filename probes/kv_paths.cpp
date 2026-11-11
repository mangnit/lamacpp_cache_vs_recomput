// End-to-end path probe: the same tokens, the same positions, two execution histories.
//
//   A  "cache history"  prompt P as one batch, then G greedy tokens fed ONE AT A TIME
//                       (decode path: ne11 = 1 -> MMVQ), then a suffix S as one batch.
//   B  "recompute"      P + gen + S as ONE batch (prompt path: MMQ / cuBLAS).
//   C  prompt-only      P as one batch, then S as one batch.
//   D  recompute        P + S as one batch.
//
// Comparisons, all on next-token logits at identical positions with identical tokens
// (teacher-forced, so no sampling noise):
//   gen      A vs B at the G generated positions  -> decode path vs prompt path
//   suf_dec  A vs B at the S suffix positions     -> suffix after a decode-built prefix
//   suf_pmt  C vs D at the S suffix positions     -> suffix after a prompt-built prefix
// Prediction E2 (docs/PROJECT.md): for M4 formats suf_dec >> suf_pmt.
//
// The KV cache stays F16 (llama.cpp default). Interventions come from GGML_PROBE_FLAGS.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * PROMPTS[] = {
    "The Industrial Revolution began in Great Britain in the late eighteenth century and spread to Western Europe and North America. "
    "It replaced hand production with machines, introduced new chemical and iron production processes, and increased the use of steam power. "
    "Historians continue to debate its causes, which include the availability of coal, the growth of trade, and institutional changes in property rights.",
    "def merge_sort(xs):\n    if len(xs) <= 1:\n        return xs\n    mid = len(xs) // 2\n    left = merge_sort(xs[:mid])\n    right = merge_sort(xs[mid:])\n"
    "    out = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n"
    "        else:\n            out.append(right[j]); j += 1\n    return out + left[i:] + right[j:]\n\n# The function above",
    "A patient presents with a three-day history of fever, productive cough and pleuritic chest pain. On examination the respiratory rate is 24 per minute, "
    "oxygen saturation is 93 percent on room air, and there are bronchial breath sounds over the right lower zone. The chest radiograph shows consolidation.",
    "Let f(x) = x^3 - 6x^2 + 11x - 6. To find the roots, first try small integers: f(1) = 1 - 6 + 11 - 6 = 0, so x = 1 is a root. Dividing f(x) by (x - 1) gives "
    "x^2 - 5x + 6, which factors as (x - 2)(x - 3). Therefore the roots are",
    "Photosynthesis converts light energy into chemical energy stored in glucose. In the light-dependent reactions, water is split and oxygen is released, while ATP "
    "and NADPH are produced. The Calvin cycle then uses these molecules to fix carbon dioxide into three-carbon sugars inside the stroma of the chloroplast.",
    "The committee met on Tuesday to review the quarterly budget. Expenses for travel exceeded the forecast by twelve percent, mainly because of two unplanned "
    "conferences, while equipment spending was below target. The chair proposed moving part of the equipment allocation to cover the travel overrun.",
    "In distributed systems, consensus protocols such as Paxos and Raft allow a group of machines to agree on a sequence of values despite failures. Raft separates "
    "leader election, log replication and safety, which makes it easier to understand. A leader accepts client requests and replicates them to followers.",
    "Once upon a time, in a village at the edge of a great forest, there lived an old clockmaker who repaired every clock in the valley except his own. "
    "Each evening the children gathered at his window to watch him work, and each evening he told them a story about the clock that never kept time.",
};
static const char * SUFFIX =
    " Now answer the following carefully. Summarise the passage above in two sentences, then list three specific facts it states, and finally explain which"
    " fact you consider most important and why.";

struct Cmp { std::vector<double> maxabs, rel; int flips = 0, n = 0, bit_equal = 0; std::vector<double> margin; };

static void compare(Cmp & c, const float * a, const float * b, int nv) {
    double ma = 0, num = 0, den = 0; int ia = 0, ib = 0; bool eq = true;
    for (int i = 0; i < nv; ++i) {
        const double d = (double) a[i] - b[i];
        ma = std::max(ma, std::fabs(d)); num += d*d; den += (double) b[i]*b[i];
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
        eq = eq && a[i] == b[i];
    }
    float second = -INFINITY;
    for (int i = 0; i < nv; ++i) if (i != ib) second = std::max(second, b[i]);
    c.maxabs.push_back(ma); c.rel.push_back(std::sqrt(num/std::max(den, 1e-300)));
    c.flips += ia != ib; c.n += 1; c.bit_equal += eq; c.margin.push_back(b[ib] - second);
}

static double q(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[(size_t) std::min<double>(v.size() - 1, std::floor(p*(v.size() - 1) + 0.5))];
}

// Decode `toks` at positions [p0, p0+n) as one llama_decode call, keeping logits for
// every token whose index is flagged in `want`. Returns logits rows in order.
static std::vector<std::vector<float>> run(llama_context * ctx, const std::vector<llama_token> & toks, int p0,
                                           const std::vector<bool> & want, int nv) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = p0 + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = want[i];
    }
    b.n_tokens = (int) toks.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "llama_decode failed\n"); exit(4); }
    std::vector<std::vector<float>> out;
    for (size_t i = 0; i < toks.size(); ++i)
        if (want[i]) { const float * l = llama_get_logits_ith(ctx, (int) i); out.emplace_back(l, l + nv); }
    llama_batch_free(b);
    return out;
}

static int argmax(const std::vector<float> & v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

int main(int argc, char ** argv) {
    std::string model_path, out = "-", tag = "";
    int P = 160, G = 48, S = 32, fa = -1, nprompts = 8;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model_path = next();
        else if (k == "--out") out = next();
        else if (k == "--tag") tag = next();
        else if (k == "--P") P = std::stoi(next());
        else if (k == "--G") G = std::stoi(next());
        else if (k == "--S") S = std::stoi(next());
        else if (k == "--fa") fa = std::stoi(next());
        else if (k == "--prompts") nprompts = std::stoi(next());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.flash_attn_type = (llama_flash_attn_type) fa;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<llama_token> suf(256);
    suf.resize(llama_tokenize(vocab, SUFFIX, (int) strlen(SUFFIX), suf.data(), (int) suf.size(), false, false));
    if ((int) suf.size() > S) suf.resize(S);
    S = (int) suf.size();

    const char * flags = getenv("GGML_PROBE_FLAGS");
    FILE * fo = out == "-" ? stdout : fopen(out.c_str(), "a");
    Cmp gen, suf_dec, suf_pmt;
    std::vector<double> decode_ms;  // wall time of each single-token llama_decode in run A
    std::vector<double> prefill_ms; // wall time of run B's one-batch llama_decode (prompt path)

    for (int pi = 0; pi < std::min(nprompts, (int) (sizeof(PROMPTS)/sizeof(PROMPTS[0]))); ++pi) {
        std::vector<llama_token> pt(1024);
        pt.resize(llama_tokenize(vocab, PROMPTS[pi], (int) strlen(PROMPTS[pi]), pt.data(), (int) pt.size(), true, false));
        if ((int) pt.size() > P) pt.resize(P);
        const int np = (int) pt.size();

        // A: prompt batch, then greedy decode one token at a time, then suffix batch.
        llama_memory_clear(mem, true);
        std::vector<bool> last(np, false); last[np - 1] = true;
        std::vector<std::vector<float>> a_gen;
        std::vector<llama_token> gtok;
        llama_token t = (llama_token) argmax(run(ctx, pt, 0, last, nv)[0]);
        for (int g = 0; g < G; ++g) {
            gtok.push_back(t);
            const auto t0 = std::chrono::steady_clock::now();
            auto l = run(ctx, {t}, np + g, {true}, nv);
            decode_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            a_gen.push_back(l[0]);
            t = (llama_token) argmax(l[0]);
        }
        auto a_suf = run(ctx, suf, np + G, std::vector<bool>(S, true), nv);

        // B: everything in one batch.
        llama_memory_clear(mem, true);
        std::vector<llama_token> all = pt; all.insert(all.end(), gtok.begin(), gtok.end()); all.insert(all.end(), suf.begin(), suf.end());
        std::vector<bool> want(all.size(), false);
        for (int g = 0; g < G; ++g) want[np + g] = true;
        for (int s = 0; s < S; ++s) want[np + G + s] = true;
        const auto tb0 = std::chrono::steady_clock::now();
        auto b_all = run(ctx, all, 0, want, nv);
        prefill_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb0).count());
        for (int g = 0; g < G; ++g) compare(gen, a_gen[g].data(), b_all[g].data(), nv);
        if (getenv("KV_PATHS_TRACE")) {  // per-position max|dlogit| for the gen positions of this prompt
            fprintf(stderr, "trace prompt %d gen:", pi);
            for (int g = G - (int) a_gen.size(); g < G; ++g) fprintf(stderr, " %.1e", gen.maxabs[gen.maxabs.size() - G + g]);
            fprintf(stderr, "\n");
        }
        for (int s = 0; s < S; ++s) compare(suf_dec, a_suf[s].data(), b_all[G + s].data(), nv);

        // C: prompt batch then suffix batch.  D: prompt + suffix in one batch.
        llama_memory_clear(mem, true);
        run(ctx, pt, 0, last, nv);
        auto c_suf = run(ctx, suf, np, std::vector<bool>(S, true), nv);
        llama_memory_clear(mem, true);
        std::vector<llama_token> ps = pt; ps.insert(ps.end(), suf.begin(), suf.end());
        std::vector<bool> want2(ps.size(), false);
        for (int s = 0; s < S; ++s) want2[np + s] = true;
        auto d_suf = run(ctx, ps, 0, want2, nv);
        for (int s = 0; s < S; ++s) compare(suf_pmt, c_suf[s].data(), d_suf[s].data(), nv);
        fprintf(stderr, "prompt %d: P=%d G=%d S=%d done\n", pi, np, G, S);
    }

    auto emit = [&](const char * name, const Cmp & c) {
        fprintf(fo, "{\"probe\":\"kv_paths\",\"llama_cpp\":\"" PROBE_LLAMA_TAG "\",\"tag\":\"%s\",\"model\":\"%s\",\"probe_flags\":\"%s\",\"fa\":%d,"
                    "\"cmp\":\"%s\",\"positions\":%d,\"maxabs_median\":%.4e,\"maxabs_p90\":%.4e,\"maxabs_max\":%.4e,"
                    "\"rel_median\":%.4e,\"argmax_flips\":%d,\"bitwise_equal_positions\":%d,\"ref_margin_median\":%.4f,\"decode_ms_median\":%.4f,\"decode_ms_p90\":%.4f,\"prefill_ms_median\":%.4f,\"prefill_ms_p90\":%.4f}\n",
                tag.c_str(), model_path.c_str(), flags ? flags : "0", fa, name, c.n, q(c.maxabs, .5), q(c.maxabs, .9), q(c.maxabs, 1.0),
                q(c.rel, .5), c.flips, c.bit_equal, q(c.margin, .5), q(decode_ms, .5), q(decode_ms, .9), q(prefill_ms, .5), q(prefill_ms, .9));
    };
    emit("gen", gen); emit("suf_dec", suf_dec); emit("suf_pmt", suf_pmt);
    fflush(fo);
    if (fo != stdout) fclose(fo);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
