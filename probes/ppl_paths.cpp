// Perplexity through the prompt path vs through the decode path, same text, same model.
//
// Standard quantization benchmarks (llama-perplexity) score text in batches, i.e. through
// MMQ. Generation runs one token at a time through MMVQ. If the two paths differ in
// accuracy (docs/PROJECT.md: the M4 side finding), the benchmark never sees the decode
// path. For each chunk of `ctx` tokens, the second half is scored two ways:
//   batch   all ctx tokens in one llama_decode call (prompt path)
//   decode  first half as one batch, then each scored token fed alone (decode path)
// Output: mean NLL / perplexity for both, and the mean |delta log p| per scored token.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double logprob(const float * l, int nv, int tok) {
    float mx = l[0];
    for (int i = 1; i < nv; ++i) mx = std::max(mx, l[i]);
    double s = 0;
    for (int i = 0; i < nv; ++i) s += std::exp((double) l[i] - mx);
    return (double) l[tok] - mx - std::log(s);
}

static void decode(llama_context * ctx, const llama_token * t, int n, int p0, bool all_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) { b.token[i] = t[i]; b.pos[i] = p0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = all_logits || i == n - 1; }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(4); }
    llama_batch_free(b);
}

int main(int argc, char ** argv) {
    std::string model_path, text_path, out = "-", tag;
    int n_ctx = 512, chunks = 4;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--model") model_path = argv[++i];
        else if (k == "--text") text_path = argv[++i];
        else if (k == "--out") out = argv[++i];
        else if (k == "--tag") tag = argv[++i];
        else if (k == "--ctx") n_ctx = atoi(argv[++i]);
        else if (k == "--chunks") chunks = atoi(argv[++i]);
    }
    std::ifstream f(text_path); std::stringstream ss; ss << f.rdbuf(); const std::string text = ss.str();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(text.size() + 16);
    toks.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.n_seq_max = 1; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    double nll_b = 0, nll_d = 0, absd = 0; int n = 0, flips = 0;
    std::vector<double> dtok;    // per scored token: logp_decode - logp_batch
    std::vector<double> dchunk;  // per chunk: mean(nll_decode - nll_batch)
    const int half = n_ctx / 2;
    for (int c = 0; c < chunks && (size_t) (c + 1)*n_ctx <= toks.size(); ++c) {
        const llama_token * t = toks.data() + (size_t) c*n_ctx;
        llama_memory_clear(mem, true);
        decode(ctx, t, n_ctx, 0, true);
        std::vector<double> lb; std::vector<int> ab;
        for (int i = half - 1; i < n_ctx - 1; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            lb.push_back(logprob(l, nv, t[i + 1]));
            ab.push_back((int) (std::max_element(l, l + nv) - l));
        }
        llama_memory_clear(mem, true);
        decode(ctx, t, half, 0, false);  // prompt path for the context half
        double chunk_d = 0; int chunk_n = 0;
        for (int i = half - 1, k = 0; i < n_ctx - 1; ++i, ++k) {
            if (i >= half) decode(ctx, t + i, 1, i, true);  // decode path, one token at a time
            const float * l = llama_get_logits_ith(ctx, -1);
            const double ld = logprob(l, nv, t[i + 1]);
            nll_b -= lb[k]; nll_d -= ld; absd += std::fabs(ld - lb[k]); ++n;
            dtok.push_back(ld - lb[k]); chunk_d += lb[k] - ld; ++chunk_n;
            flips += ab[k] != (int) (std::max_element(l, l + nv) - l);
        }
        dchunk.push_back(chunk_d / chunk_n);
        fprintf(stderr, "chunk %d done\n", c);
    }
    // Uncertainty on dNLL = mean(nll_decode - nll_batch): token-level SE (tokens treated as
    // independent, optimistic) and chunk-level SE (chunks independent, conservative).
    double m = 0, v = 0, mc = 0, vc = 0;
    for (double d : dtok) m += -d; m /= dtok.size();
    for (double d : dtok) v += (-d - m)*(-d - m); v /= std::max<size_t>(1, dtok.size() - 1);
    for (double d : dchunk) mc += d; mc /= dchunk.size();
    for (double d : dchunk) vc += (d - mc)*(d - mc); vc /= std::max<size_t>(1, dchunk.size() - 1);
    const double se_tok = std::sqrt(v / dtok.size()), se_chunk = std::sqrt(vc / dchunk.size());
    const char * flags = getenv("GGML_PROBE_FLAGS");
    FILE * fo = out == "-" ? stdout : fopen(out.c_str(), "a");
    fprintf(fo, "{\"probe\":\"ppl_paths\",\"llama_cpp\":\"" PROBE_LLAMA_TAG "\",\"tag\":\"%s\",\"model\":\"%s\",\"probe_flags\":\"%s\",\"ctx\":%d,"
                "\"scored_tokens\":%d,\"ppl_batch\":%.6f,\"ppl_decode\":%.6f,\"mean_abs_dlogp\":%.4e,\"argmax_flips\":%d,"
                "\"dnll\":%.6e,\"dnll_se_token\":%.3e,\"dnll_se_chunk\":%.3e,\"chunks\":%d}\n",
            tag.c_str(), model_path.c_str(), flags ? flags : "0", n_ctx, n, std::exp(nll_b/n), std::exp(nll_d/n), absd/n, flips, m, se_tok, se_chunk, (int) dchunk.size());
    if (fo != stdout) fclose(fo);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
