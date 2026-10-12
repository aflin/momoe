// Momoe: an expert read in place from the GGUF (ExpertJobMulti::up / ::down, ExpertSource::roles) computes the same
// bits as the assembled blob.  The GGUF's layout is built here - each role one tensor of every expert, the down tensor
// before the gate tensor and the up tensor after it, as in Ornith-1.5's file - and every expert is run both ways on
// the pool's native path.  CPU only, no model.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

int check(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_type, d_type, n_embd, n_ff, f, err)) {
        std::printf("skip %d/%d: %s\n", gu_type, d_type, err.c_str());
        return 0;
    }
    constexpr int E = 5;   // experts in each role tensor
    const size_t gb = f.up_off, db = f.bytes - f.down_off;
    // the file: [down x E][gap][gate x E][gap][up x E], the gaps odd so no role is aligned like its neighbour
    const size_t gap = 96;
    std::vector<uint8_t> file(E * db + gap + E * gb + gap + E * gb);
    uint8_t* down0 = file.data();
    uint8_t* gate0 = down0 + E * db + gap;
    uint8_t* up0 = gate0 + E * gb + gap;
    std::vector<float> w((size_t) n_embd * n_ff);
    for (int e = 0; e < E; ++e) {
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::sin((float) i * 0.13f + 0.7f * (float) e);
        ggml_quantize_chunk((ggml_type) gu_type, w.data(), gate0 + e * gb, 0, n_ff, n_embd, nullptr);
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::cos((float) i * 0.11f + 0.5f * (float) e);
        ggml_quantize_chunk((ggml_type) gu_type, w.data(), up0 + e * gb, 0, n_ff, n_embd, nullptr);
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::sin((float) i * 0.07f + 0.3f * (float) e);
        ggml_quantize_chunk((ggml_type) d_type, w.data(), down0 + e * db, 0, n_embd, n_ff, nullptr);
    }
    // the assembled blobs, as FileExpertSource::copy_from_files writes them
    std::vector<uint8_t> blobs((size_t) E * f.bytes);
    for (int e = 0; e < E; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * f.bytes;
        std::memcpy(b, gate0 + e * gb, gb);
        std::memcpy(b + f.up_off, up0 + e * gb, gb);
        std::memcpy(b + f.down_off, down0 + e * db, db);
    }
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact(cpu::MAXT);
    std::vector<float> x((size_t) n_embd);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int64_t i = 0; i < n_embd; ++i) x[(size_t) i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
        cpu::act_quant_any(x.data(), (int) n_embd, qact[(size_t) t]);
    }
    int checks = 0;
    for (int nt : {1, 2, 3, cpu::MAXT}) {
        std::vector<float> ref((size_t) E * cpu::MAXT * n_embd), got(ref.size());
        auto jobs_for = [&](bool in_place, std::vector<float>& out) {
            std::vector<cpu::ExpertJobMulti> jobs(E);
            for (int e = 0; e < E; ++e) {
                cpu::ExpertJobMulti& j = jobs[(size_t) e];
                if (in_place) {
                    j.blob = gate0 + e * gb;
                    j.up = up0 + e * gb;
                    j.down = down0 + e * db;
                } else {
                    j.blob = blobs.data() + (size_t) e * f.bytes;
                }
                j.nt = nt;
                for (int t = 0; t < nt; ++t) {
                    j.nact[t] = act[(size_t) t].data();
                    j.act[t] = &qact[(size_t) t];
                    j.out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * (size_t) n_embd;
                }
            }
            return jobs;
        };
        for (int workers : {1, 3}) {
            cpu::ExpertPool pool(workers, false);
            std::fill(ref.begin(), ref.end(), 0.0f);
            std::fill(got.begin(), got.end(), std::numeric_limits<float>::quiet_NaN());
            auto a = jobs_for(false, ref);
            pool.run_split_multi_native(f, a.data(), E);
            auto b = jobs_for(true, got);
            for (auto& j : b) for (int t = nt; t < cpu::MAXT; ++t)   // the slots no token uses stay as the reference's
                std::fill(got.data() + ((size_t) (&j - b.data()) * cpu::MAXT + t) * (size_t) n_embd,
                          got.data() + ((size_t) (&j - b.data()) * cpu::MAXT + t + 1) * (size_t) n_embd, 0.0f);
            pool.run_split_multi_native(f, b.data(), E);
            for (float v : ref) if (!std::isfinite(v)) { std::fprintf(stderr, "non-finite reference\n"); return -1; }
            if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
                std::fprintf(stderr, "mismatch: types %d/%d, nt %d, workers %d\n", gu_type, d_type, nt, workers);
                return -1;
            }
            ++checks;
        }
    }
    std::printf("types %d/%d (%lld x %lld): %d bitwise comparisons passed\n", gu_type, d_type, (long long) n_embd,
                (long long) n_ff, checks);
    return checks;
}

}  // namespace

int main() {
    if (!cpu::native_experts_available()) { std::printf("built without native experts\n"); return 0; }
    // Qwen3.6 / Ornith geometry (2048 x 512): IQ4_XS everywhere (Ornith's IQ4_XS file), Q4_0, IQ3_S gate/up with
    // IQ4_NL down (the i-quant packs' mix), Q8_0 down
    const int pairs[][2] = {{GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS}, {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0},
                            {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL}, {GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0}};
    int total = 0;
    for (const auto& p : pairs) {
        const int n = check(p[0], p[1], 2048, 512);
        if (n < 0) return 1;
        total += n;
    }
    std::printf("pool roles: %d bitwise comparisons passed\n", total);
    return total > 0 ? 0 : 1;
}
