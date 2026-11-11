// Operator-level path probe: does a token's matmul output depend on how many other
// tokens share its batch?
//
// For each weight type the same weight W and the same activation column x_j are
// pushed through ggml's CUDA mul_mat twice:
//   alone   ne11 = 1   -> MMVQ (quantized) / MMVF (F16)       = the decode path
//   batch   ne11 = B   -> MMQ (Turing+, sm_61) / cuBLAS (P100) = the prompt / recompute path
// and the discrepancy delta_j = ||y_alone - y_batch|| / ||y_batch|| is reported,
// together with each path's error against an FP64 reference built from the
// dequantized weights. A batch-size sweep on token 0 locates the step where the
// dispatcher switches kernels, so the path attribution is measured, not assumed.
//
// Interventions are selected by GGML_PROBE_FLAGS (see apply_probe_patch.py); the flag
// value is recorded in every output line. Output: one JSON object per weight type.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Args {
    std::vector<std::string> types = {"f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1",
                                      "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"};
    int K = 1536, N = 1536, B = 64, tokens = 16, reps = 30;
    unsigned seed = 6886;
    std::string out = "-";
};

static ggml_type parse_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * name = ggml_type_name((ggml_type) t);
        if (name && s == name) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(2);
}

// One mul_mat on the CUDA backend: y [N x nb] = W [N x K] * X [K x nb].
static std::vector<float> run_mm(ggml_backend_t be, ggml_type type, const std::vector<uint8_t> & wq,
                                 int K, int N, const float * x, int nb, double * ms = nullptr, int reps = 0) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, type, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, nb);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(W, wq.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(X, x, 0, ggml_nbytes(X));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(3); }
    if (ms && reps > 0) {
        std::vector<double> t;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        *ms = t[t.size()/2];
    }
    std::vector<float> y((size_t) N*nb);
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return y;
}

static double rel_l2(const float * a, const float * b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { const double d = (double) a[i] - b[i]; num += d*d; den += (double) b[i]*b[i]; }
    return std::sqrt(num / std::max(den, 1e-300));
}
static double rel_l2_ref(const float * a, const double * r, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { const double d = (double) a[i] - r[i]; num += d*d; den += r[i]*r[i]; }
    return std::sqrt(num / std::max(den, 1e-300));
}
static double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[(size_t) std::min<double>(v.size() - 1, std::floor(q*(v.size() - 1) + 0.5))];
}

int main(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (k == "--types") {
            a.types.clear(); std::string s = next(); size_t p = 0;
            while (p != std::string::npos) { size_t q = s.find(',', p); a.types.push_back(s.substr(p, q == std::string::npos ? q : q - p)); p = q == std::string::npos ? q : q + 1; }
        } else if (k == "--K") a.K = std::stoi(next());
        else if (k == "--N") a.N = std::stoi(next());
        else if (k == "--B") a.B = std::stoi(next());
        else if (k == "--tokens") a.tokens = std::stoi(next());
        else if (k == "--reps") a.reps = std::stoi(next());
        else if (k == "--seed") a.seed = (unsigned) std::stoul(next());
        else if (k == "--out") a.out = next();
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    FILE * fo = a.out == "-" ? stdout : fopen(a.out.c_str(), "a");

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 1; }
    char desc[256]; ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    const char * flags = getenv("GGML_PROBE_FLAGS");

    // Activations shaped like post-RMSNorm hidden states: unit-variance Gaussian with a
    // fixed 1% of channels carrying 20x outliers, and a per-token scale in [0.5, 2].
    std::mt19937 rng(a.seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<int> outlier(a.K, 0);
    for (int k = 0; k < a.K; ++k) outlier[k] = u01(rng) < 0.01f;
    std::vector<float> X((size_t) a.K*a.B);
    for (int j = 0; j < a.B; ++j) {
        const float s = 0.5f + 1.5f*u01(rng);
        for (int k = 0; k < a.K; ++k) X[(size_t) j*a.K + k] = s*g(rng)*(outlier[k] ? 20.0f : 1.0f);
    }
    std::vector<float> Wf((size_t) a.N*a.K);
    for (auto & w : Wf) w = 0.02f*g(rng);

    for (const std::string & ts : a.types) {
        const ggml_type type = parse_type(ts);
        if (ggml_is_quantized(type) && ggml_quantize_requires_imatrix(type)) { fprintf(stderr, "skip %s (needs imatrix)\n", ts.c_str()); continue; }
        std::vector<uint8_t> wq(ggml_row_size(type, a.K)*a.N);
        if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(Wf.data(), (ggml_fp16_t *) wq.data(), (int64_t) a.N*a.K);
        else ggml_quantize_chunk(type, Wf.data(), wq.data(), 0, a.N, a.K, nullptr);

        // FP64 reference from the dequantized weights and the exact activations.
        std::vector<float> Wd((size_t) a.N*a.K);
        const size_t rs = ggml_row_size(type, a.K);
        for (int n = 0; n < a.N; ++n) {
            if (type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) (wq.data() + n*rs), Wd.data() + (size_t) n*a.K, a.K);
            else ggml_get_type_traits(type)->to_float(wq.data() + n*rs, Wd.data() + (size_t) n*a.K, a.K);
        }
        std::vector<double> ref((size_t) a.N*a.tokens);
        for (int j = 0; j < a.tokens; ++j)
            for (int n = 0; n < a.N; ++n) {
                double s = 0; const float * w = Wd.data() + (size_t) n*a.K; const float * x = X.data() + (size_t) j*a.K;
                for (int k = 0; k < a.K; ++k) s += (double) w[k]*x[k];
                ref[(size_t) j*a.N + n] = s;
            }

        double ms_batch = 0, ms_alone = 0;
        const std::vector<float> yb = run_mm(be, type, wq, a.K, a.N, X.data(), a.B, &ms_batch, a.reps);
        std::vector<double> delta, e_alone, e_batch; long eq = 0, tot = 0;
        for (int j = 0; j < a.tokens; ++j) {
            const std::vector<float> y1 = run_mm(be, type, wq, a.K, a.N, X.data() + (size_t) j*a.K, 1,
                                                 j == 0 ? &ms_alone : nullptr, j == 0 ? a.reps : 0);
            const float * ybj = yb.data() + (size_t) j*a.N;
            delta.push_back(rel_l2(y1.data(), ybj, a.N));
            e_alone.push_back(rel_l2_ref(y1.data(), ref.data() + (size_t) j*a.N, a.N));
            e_batch.push_back(rel_l2_ref(ybj, ref.data() + (size_t) j*a.N, a.N));
            for (int n = 0; n < a.N; ++n) { eq += memcmp(&y1[n], &ybj[n], 4) == 0; ++tot; }
        }
        // Batch-size sweep for token 0: where does the kernel switch?
        std::string sweep;
        for (int nb : {1, 2, 4, 7, 8, 9, 10, 12, 16, 32, a.B}) {
            const std::vector<float> y = run_mm(be, type, wq, a.K, a.N, X.data(), nb);
            char tmp[96]; snprintf(tmp, sizeof(tmp), "%s\"%d\":%.3e", sweep.empty() ? "" : ",", nb, rel_l2(y.data(), yb.data(), a.N));
            sweep += tmp;
        }
        fprintf(fo, "{\"probe\":\"matmul_paths\",\"llama_cpp\":\"" PROBE_LLAMA_TAG "\",\"device\":\"%s\",\"cc\":%d%d,"
                    "\"cudart\":%d,\"probe_flags\":\"%s\",\"type\":\"%s\",\"K\":%d,\"N\":%d,\"B\":%d,\"tokens\":%d,\"seed\":%u,"
                    "\"delta_median\":%.4e,\"delta_p90\":%.4e,\"delta_max\":%.4e,\"bitwise_equal_frac\":%.6f,"
                    "\"err_alone_vs_fp64_median\":%.4e,\"err_batch_vs_fp64_median\":%.4e,"
                    "\"ms_alone_median\":%.4f,\"ms_batch_median\":%.4f,\"reps\":%d,\"delta_vs_batchB_by_nb\":{%s}}\n",
                desc, prop.major, prop.minor, CUDART_VERSION, flags ? flags : "0", ts.c_str(), a.K, a.N, a.B, a.tokens, a.seed,
                quantile(delta, 0.5), quantile(delta, 0.9), quantile(delta, 1.0), (double) eq/tot,
                quantile(e_alone, 0.5), quantile(e_batch, 0.5), ms_alone, ms_batch, a.reps, sweep.c_str());
        fflush(fo);
    }
    ggml_backend_free(be);
    if (fo != stdout) fclose(fo);
    return 0;
}
