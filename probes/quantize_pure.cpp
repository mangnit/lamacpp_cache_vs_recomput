// Quantize one GGUF to a named format with this exact llama.cpp version.
//
//   quantize_pure IN.gguf OUT.gguf FORMAT [mix]
//
// FORMAT is one of F16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K Q4_K Q5_K Q6_K. By default
// every tensor, including output.weight, gets that one type ("pure"), so a format
// comparison is not confounded by llama.cpp's k-quant mixtures. Pass "mix" to get the
// stock mixture (e.g. Q4_K -> Q4_K_M) to match the published Q4_K_M / Q3_K_M models.

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s IN OUT FORMAT [mix]\n", argv[0]); return 2; }
    const bool mix = argc > 4 && std::string(argv[4]) == "mix";
    const std::map<std::string, llama_ftype> pure = {
        {"F16", LLAMA_FTYPE_MOSTLY_F16},   {"Q8_0", LLAMA_FTYPE_MOSTLY_Q8_0}, {"Q4_0", LLAMA_FTYPE_MOSTLY_Q4_0},
        {"Q4_1", LLAMA_FTYPE_MOSTLY_Q4_1}, {"Q5_0", LLAMA_FTYPE_MOSTLY_Q5_0}, {"Q5_1", LLAMA_FTYPE_MOSTLY_Q5_1},
        {"Q2_K", LLAMA_FTYPE_MOSTLY_Q2_K}, {"Q3_K", LLAMA_FTYPE_MOSTLY_Q3_K_S}, {"Q4_K", LLAMA_FTYPE_MOSTLY_Q4_K_S},
        {"Q5_K", LLAMA_FTYPE_MOSTLY_Q5_K_S}, {"Q6_K", LLAMA_FTYPE_MOSTLY_Q6_K},
    };
    const std::map<std::string, llama_ftype> mixes = {
        {"Q4_K", LLAMA_FTYPE_MOSTLY_Q4_K_M}, {"Q3_K", LLAMA_FTYPE_MOSTLY_Q3_K_M},
    };
    const auto & table = mix ? mixes : pure;
    auto it = table.find(argv[3]);
    if (it == table.end()) { fprintf(stderr, "unknown format %s (mix=%d)\n", argv[3], mix); return 2; }

    llama_backend_init();
    llama_model_quantize_params p = llama_model_quantize_default_params();
    p.ftype = it->second;
    p.pure = !mix;
    p.quantize_output_tensor = true;
    const uint32_t rc = llama_model_quantize(argv[1], argv[2], &p);
    fprintf(stderr, "quantize %s -> %s as %s%s: rc=%u\n", argv[1], argv[2], argv[3], mix ? " (mix)" : " (pure)", rc);
    return rc == 0 ? 0 : 1;
}
