// momoe-metal: Momoe's engine for Apple Silicon (Metal), qwen35moe (Qwen3.6-35B-A3B, Ornith-1.5, KAT-Coder) and
// qwen4exp (Qwen3.8-Flash-Next).
//
// The dense weights (attention, delta net, shared expert, embeddings, head: ~1.6 GiB for Ornith IQ4_XS) are loaded
// whole.  The routed experts (~16.4 GiB) are not: they live in a pool of slots, one tensor each for the gate, up
// and down weights, and a slot is filled from the GGUF the first time a token routes to that expert.  After each
// layer's router has chosen its experts, the scheduler's eval callback stops there, the chosen experts are made
// resident (read from the file straight into the pool: on Apple Silicon the pool is memory the GPU reads, so the
// read lands where the GPU uses it), and the expert matmuls run against the pool through a slot-number tensor.
//
// The graph follows llama.cpp's qwen35moe and qwen4exp (src/models/qwen35moe.cpp, qwen4exp.cpp, delta-net-base.cpp;
// MIT) op for op; qwen4exp's sparse attention (QSA) is not done yet: its attention layers attend to every position,
// which is what QSA selects up to ~2,048 positions.
//
//   momoe-metal -m MODEL.gguf --tokens 1,2,3 [-n 64] [--slots N | --pool-gib G] [--ctx 32768]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(MOMOE_CUDA)
#include "ggml-cuda.h"
#include "ggml-momoe-mailbox.h"
#include <cuda_runtime.h>
#else
#include "ggml-metal.h"
#endif
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <array>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The process ends with _exit, not exit: exit runs ggml's static destructors, and the Metal device's asserts that
// every residency set was released first (the mapped model, the pool and the MTP layer's buffers still hold theirs,
// and exit from a reader thread would race them anyway): "GGML_ASSERT([rsets->data count] == 0)", an abort after
// the run's results
[[noreturn]] static void quit(int code) {
    fflush(stdout);
    fflush(stderr);
    _exit(code);
}

[[noreturn]] static void die(const char * fmt, const char * a = "") {
    fprintf(stderr, "momoe-metal: ");
    fprintf(stderr, fmt, a);
    fprintf(stderr, "\n");
    quit(1);
}

// Apple Silicon: every backend buffer is host memory, written and read in place. CUDA (momoe-cuda): device memory,
// reached through ggml_backend_tensor_set / _get; the model files are mapped, and an expert is copied from the mapping.
#if defined(MOMOE_CUDA)
static constexpr bool kHostMem = false;
#else
static constexpr bool kHostMem = true;
#endif

static void tset(ggml_tensor * t, const void * src, size_t off, size_t n) {
    if (kHostMem) memcpy((char *) t->data + off, src, n);
    else ggml_backend_tensor_set(t, src, off, n);
}

static void tget(const ggml_tensor * t, void * dst, size_t off, size_t n) {
    if (kHostMem) memcpy(dst, (const char *) t->data + off, n);
    else ggml_backend_tensor_get(t, dst, off, n);
}

// ------------------------------------------------------------------------------------------------ the model
struct Layer {
    bool recurrent = false;
    // nemotron_h_moe (Nemotron 3.5 Bolt / Super): every block is one of a Mamba-2 mixer ('m'), attention ('a') or a
    // mixture of experts ('e'), each with its own pre-norm (attn_norm) and residual
    char kind = 0;
    int next_moe = -1;                                       // nemo: the next MoE block (its experts are predicted here)
    int n_exp = 0;                                           // this layer's routed experts (Bolt: 240..496 per layer)
    ggml_tensor *ssm_in = nullptr, *conv1d_b = nullptr, *ssm_d = nullptr;               // Mamba-2
    ggml_tensor *exp_probs_b = nullptr, *latent_down = nullptr, *latent_up = nullptr;    // latent MoE, sigmoid router
    ggml_tensor *cpu_dn = nullptr;                           // --hybrid: [latent, k, HYB_MAX_T] the CPU's experts' output
    // both kinds
    ggml_tensor *attn_norm = nullptr, *post_norm = nullptr;
    // full attention
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    // gated delta net
    ggml_tensor *wqkv = nullptr, *wgate = nullptr, *conv1d = nullptr, *ssm_dt = nullptr, *ssm_a = nullptr,
                *ssm_beta = nullptr, *ssm_alpha = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
    // MoE
    ggml_tensor *gate_inp = nullptr, *gate_inp_sh = nullptr, *gate_sh = nullptr, *up_sh = nullptr, *down_sh = nullptr;
    // the routed experts in the file: offset of expert 0 and the bytes of one expert, per gate / up / down
    size_t exp_off[3] = {0, 0, 0}, exp_bytes[3] = {0, 0, 0};
    ggml_type exp_type[3] = {GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT};
    int64_t exp_ne[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    // state
    ggml_tensor *k_cache = nullptr, *v_cache = nullptr;      // full attention: [256, n_kv_head, n_ctx] f16
    ggml_tensor *conv_state = nullptr, *ssm_state = nullptr; // delta net
    ggml_tensor *ssm_snap = nullptr, *conv_snap = nullptr;   // verify windows: the delta net's per-token states
    ggml_tensor *exps[3] = {nullptr, nullptr, nullptr};      // the MTP layer: its experts whole (gate / up / down)
    ggml_tensor *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;   // MTP only
    ggml_tensor *fc_e = nullptr, *fc_h = nullptr;            // qwen4exp's MTP: eh_proj as its two halves
    ggml_tensor *res_mask = nullptr, *res_keep = nullptr;    // --nostall: [n_expert] f32, per expert in the pool:
                                                             // 0 / -inf (added before the top-k), 1 / 0 (its weight)
    ggml_tensor *slot_map = nullptr;                         // [1, n_expert] i32: expert -> slot, for this layer's
                                                             // chosen experts (looked up on the GPU: a real dependency)
    int grp = 0;                                             // the pool group of this layer's experts (their formats)
    int dev = 0;                                             // momoe-cuda --layer-split: the card it runs on
    int exp_fd[3] = {-1, -1, -1};                            // the file (shard) each expert tensor is in (a split
    const char * exp_map[3] = {};                            // model may put a layer's up and down in two); momoe-cuda: mapped
    std::vector<int32_t> map_host;                           // momoe-cuda: slot_map's host copy (written, then set)
    int32_t * slots_host() { return kHostMem ? (int32_t *) slot_map->data : map_host.data(); }
    ggml_tensor *hslot_map = nullptr;                        // --helper-tier: expert -> the helper card's slot (its
    std::vector<int32_t> hmap_host;                          // slot of zeros for the others), on that card
    void sync_slots() { if (!kHostMem) ggml_backend_tensor_set(slot_map, map_host.data(), 0, ggml_nbytes(slot_map)); }
    // qwen4exp: hyper-connections (before the token mixer and before the MoE) and the n-gram embedding (PLE, one layer)
    ggml_tensor *hc_attn_norm = nullptr, *hc_attn_down = nullptr, *hc_attn_up = nullptr, *hc_attn_inject = nullptr;
    ggml_tensor *hc_ffn_norm = nullptr, *hc_ffn_down = nullptr, *hc_ffn_up = nullptr, *hc_ffn_inject = nullptr;
    ggml_tensor *ple_key = nullptr, *ple_value = nullptr, *ple_norm_key = nullptr, *ple_norm_query = nullptr,
                *ple_norm_conv = nullptr, *ple_conv1d = nullptr;
    ggml_tensor *ple_state = nullptr;                        // the PLE convolution's history [(kernel-1)*ngram, hc*n_embd]
    // qwen4exp's sparse attention (QSA): the indexer, its raw keys per position and pooled keys per block of positions
    ggml_tensor *idx_q = nullptr, *idx_k = nullptr, *idx_q_norm = nullptr, *idx_k_norm = nullptr;
    ggml_tensor *idx_raw = nullptr, *idx_pool = nullptr;     // [idx_dim, n_ctx], [idx_dim, n_ctx / kpool + 64] f16
    // glm5-next (GLM-5.3-Flash): KDA blocks ('k': q / k / v through their own short convolutions, a per-channel decay
    // gate f_a -> f_b, an output gate g_a -> g_b) and MLA blocks ('a': the query and the key compressed, the cache holds
    // the 512-wide latent) whose sparse attention a k-pool indexer selects; mHC around both and around the FFN
    ggml_tensor *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr, *ssm_f_a = nullptr, *ssm_f_b = nullptr,
                *ssm_g_a = nullptr, *ssm_g_b = nullptr;
    ggml_tensor *wq_a = nullptr, *q_a_norm = nullptr, *wq_b = nullptr, *wkv_a = nullptr, *kv_a_norm = nullptr,
                *wk_b = nullptr, *wv_b = nullptr;
    ggml_tensor *idx_k_norm_b = nullptr, *idx_proj = nullptr, *idx_ape = nullptr, *idx_gate = nullptr;
    ggml_tensor *hc_attn_fn = nullptr, *hc_attn_base = nullptr, *hc_attn_scale = nullptr;
    ggml_tensor *hc_ffn_fn = nullptr, *hc_ffn_base = nullptr, *hc_ffn_scale = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;   // + the dense FFN
    ggml_tensor *lat_cache = nullptr;                        // MLA: [kv_lora, n_ctx] f16 (the key, and the value)
    // flint: per-head attention sinks; a sliding-window block (the window's mask) and whether it ropes
    ggml_tensor *sinks = nullptr;
    bool swa = false, rope = true;
    // gemma4: the block's own head size and KV heads (sliding-window blocks 256 x 8, full ones 512 x 2); the norms
    // around its dense FFN and MoE (the FFN's output, the MLP's, the experts' output and input); the router's input
    // scale, each expert's output scale and the block's output scale
    int hd = 0, n_kv = 0;
    ggml_tensor *post_ffw = nullptr, *post_ffw1 = nullptr, *post_ffw2 = nullptr, *pre_ffw2 = nullptr;
    ggml_tensor *gate_inp_s = nullptr, *down_s = nullptr, *out_scale = nullptr;
    // deepseek41 (DeepSeek-V4.1-Flash): one 512-wide KV row (the key is the value; RoPE on its last n_rot) in wkv_a /
    // kv_a_norm, the query through wq_a -> wq_b, the output through o_groups LoRAs (wo_a, then wo); per-head sinks.  A
    // block with a compress ratio attends to the window's ring (k_cache, swa) and to compressed rows of the KV source
    // block at or before it: a source compresses ratio positions into one row (comp_kv weighted by the softmax of
    // comp_gate, comp_norm), stored with its index key (idx_k, idx_k_norm) in comp_cache [hd + idx_dim, n_ctx / ratio
    // + 1] (the last row a dummy target), the unfinished group's rows in cst_kv / cst_sc [hd, ratio]; an index source
    // scores the rows (idx_q, idx_proj) and keeps the top idx_top_k.  Engram blocks: hashed rows (the Engram file, read
    // on the CPU) through eg_wkv, gated by q * k (eg_q, eg_k)
    ggml_tensor *wo_a = nullptr, *comp_kv = nullptr, *comp_gate = nullptr, *comp_norm = nullptr;
    ggml_tensor *eg_q = nullptr, *eg_k = nullptr, *eg_wkv = nullptr;
    ggml_tensor *comp_cache = nullptr, *cst_kv = nullptr, *cst_sc = nullptr;
    int ratio = 0, kv_src = -1, topk_src = -1, eg = -1;
};

struct Model {
    int n_embd, n_layer, n_head, n_head_kv, head_dim, n_rot, n_expert, n_used, n_ff_exp, n_vocab;
    int ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_dt_rank = 0, ssm_inner = 0, full_attn_interval = 0;
    int sections[4];
    float rms_eps, rope_base;
    bool qwen4 = false;                                      // qwen4exp (Flash-Next): hyper-connections, PLE, sigmoid
                                                             // gate in the delta net
    bool nemo = false;                                       // nemotron_h_moe: Mamba-2, NoPE attention, latent MoE
    int latent = 0;                                          // nemo: the experts' width (moe_latent_size)
    float exp_scale = 1.0f;                                  // nemo: the routed weights' scale (expert_weights_scale)
    int n_exp_all = 0;                                       // routed experts in the model, all layers
    std::vector<int32_t> eos_ids;                            // the tokens that end a reply (<|im_end|>, <|endoftext|>,
                                                             // the file's eos), from the vocabulary
    int32_t im_start = -1;                                   // <|im_start|>: the conversation checkpoints go before it
    int32_t img_pad = -1;                                    // a picture's cells in the prompt (<|image_pad|>)
    int part0() const { return nemo || gemma4 ? 1 : 0; }     // the first expert part: gate (0) or, without a gate, up
                                                             // (gemma4: gate and up in one, ffn_gate_up_exps)
    int hc = 1, hc_lr = 0;                                   // hyper-connection streams, low rank
    ggml_tensor *hc_head_norm = nullptr, *hc_head_down = nullptr, *hc_head_up = nullptr;   // the final mixer (output norm)
    ggml_tensor *mtp_head_norm = nullptr, *mtp_head_down = nullptr, *mtp_head_up = nullptr;   // the MTP layer's own
    int h_dim() const { return qwen4 ? n_embd * hc : n_embd; }   // the hidden state the MTP layer reads (qwen4exp: all streams)
    // PLE: n-gram hash rows of a table read on the CPU (qwen4exp: 28.8 GB in IQ3_XXS's second file)
    int ple_layer = -1, ple_ngram = 0, ple_per_gram = 0, ple_heads = 0, ple_dim = 0, ple_kernel = 0, ple_eos = 0;
    std::vector<uint64_t> ple_mult, ple_off, ple_vocab;
    int ple_fd = -1;
    size_t ple_table_off = 0, ple_row_bytes = 0;
    int64_t ple_rows = 0;
    ggml_type ple_type = GGML_TYPE_COUNT;
    // QSA: a block of kpool positions is scored by its pooled indexer key; each position attends to the top_k / kpool
    // best blocks and its own unfinished block
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0, kpool = 0;
    // glm5-next (see Layer): the first n_dense blocks have a dense FFN, the others MoE (a sigmoid router, SwiGLU
    // experts clamped at swiglu_clamp); the k-pool selection with the unfinished block (kpool_tail)
    bool glm5 = false;
    // lfm2moe (LFM2.5-24B-A2B, LFM2-8B-A1B): short-convolution blocks ('c': in_proj -> b | c | x, a depthwise
    // convolution over b * x whose history is the conv state, c * it, out_proj) and GQA attention blocks ('a': q / k
    // norms, NEOX RoPE); the first n_dense blocks have a dense FFN, the others a sigmoid-routed MoE (its bias only
    // chooses, the chosen weights normalized), no shared expert
    bool lfm = false;
    // flint (Recursive's Flint 40B-A2.5B): lfm2moe's MoE and dense lead blocks (lfm is set too) with every block GQA
    // attention: q / k norms, attention sinks, a sliding window (n_swa: the token and the 512 before it) with NEOX RoPE
    // on 35 blocks, full attention without RoPE on 14; a shared expert; the down projection's input scaled per expert
    // and token by a power of two (llama.cpp's build_amax_src1_f16: the 8-bit activation blocks' f16 sums stay finite)
    bool flint = false;
    int n_swa = 0;
    // flint's sliding-window blocks keep only a ring of swa_rows positions (the window and the largest batch, as
    // llama.cpp's iswa cache): position p in row p % swa_rows; the full blocks keep every position
    int swa_rows = 0;
    // gemma4 (Gemma 4 26B-A4B): every block attention, five sliding-window blocks (n_swa 1024, RoPE base rope_base_swa)
    // to one full block (rope_base, proportional RoPE: rope_freqs), q / k norms and an unweighted v norm, attention
    // scale 1; a GELU dense FFN beside a softmax-routed MoE of GELU experts; the embeddings scaled by sqrt(n_embd), the
    // head tied to them and its logits soft-capped (softcap)
    bool gemma4 = false;
    // bailingmoe3 (Ling-3.0-flash and its fine-tunes, Sante): glm5-next's KDA blocks without the gate LoRAs (f_a and g_a
    // straight to the inner width; A is exp(A_log)) and MLA blocks with RoPE on n_rot of the key (the latent and that
    // part in the cache), the query without a LoRA, a sigmoid gate per head; no mHC, no indexer.  The MoE as glm5-next's
    // with DeepSeek's group selection (n_groups groups, the n_group_used best by their two best biased scores)
    bool ling = false;
    int n_groups = 0, n_group_used = 0;
    float rope_base_swa = 0, softcap = 0;
    ggml_tensor *rope_freqs = nullptr;
    int n_dense = 0, kda_hd = 0, q_lora = 0, kv_lora = 0, mla_k = 0, mla_v = 0, hc_iters = 0;
    float norm_eps = 1e-6f, kda_lb = -INFINITY, hc_eps = 1e-6f;
    bool kpool_tail = true;
    std::vector<float> clamp_exp, clamp_sh;                  // per block
    // deepseek41 (see Layer): mHC (glm5-next's) whose pre-mix lags one sublayer, no head mixer; the KV width ds_hd, the
    // output groups and their rank; RoPE: the window-only blocks at rope_base, the compressed ones at ds_comp_base with
    // YaRN (ds_freq_scale, ds_ext, ds_attn_factor, ds_beta_*, ds_ctx_orig); the candidate blocks (cand_src picks the
    // cand_blocks best blocks of cand_bs rows, the index sources after it score inside them); the router sqrt(softplus)
    bool ds41 = false;
    int ds_hd = 0, o_groups = 0, o_lora = 0, ds_ctx_orig = 0, cand_src = -1, cand_bs = 0, cand_blocks = 0;
    float ds_comp_base = 0, ds_freq_scale = 1, ds_ext = 0, ds_attn_factor = 1, ds_beta_fast = 0, ds_beta_slow = 0;
    // Engram (deepseek41): per module (block), the n-gram hash (eg_mult[eg_ngram], eg_primes / eg_offs[eg_heads], back
    // to back) into its own table of eg_dim-wide rows in the Engram file; tokens mapped through eg_map (pad: eg_pad)
    int eg_ngram = 0, eg_per = 0, eg_heads = 0, eg_dim = 0;
    int64_t eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<uint64_t> eg_mult, eg_primes, eg_offs;
    std::vector<int> eg_fd;
    std::vector<size_t> eg_off, eg_row;
    std::vector<int64_t> eg_rows;
    ggml_type eg_type = GGML_TYPE_COUNT;
    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *output = nullptr;
    std::vector<Layer> layers;
    Layer mtp;                                               // the draft layer (blk.n_layer), when loaded
    bool has_mtp = false;
    ggml_tensor * dhead = nullptr;                           // the draft head: the output head's rows for a token
    std::vector<int32_t> dvocab;                             // subset (--draft-vocab; Momoe's data/draft_vocab*.bin)
};

// ------------------------------------------------------------------------------------------------ the expert pool
// One slot holds one (layer, expert): its gate, up and down rows.  The layers whose experts share formats (gate / up /
// down types and shapes) share a group of slots: one group for most files, ~10 for a file quantized per layer (Strata's
// Flash-Next files).  Each group gets the same share of its experts.  Slots are taken least used first (use counts
// halved every 64 tokens, recency breaking ties); the experts of the layer being computed are pinned (a batch of
// tokens can route to more experts than one token).
struct Group {
    ggml_type type[3];
    int64_t ne[3][2];
    size_t bytes = 0;                                        // one expert: gate + up + down
    int n_layers = 0, n_slots = 0;
    int n_all = 0;                                           // the experts of its layers
    ggml_tensor *t[3] = {nullptr, nullptr, nullptr};          // gate / up / down: [ne0, ne1, n_slots]
    ggml_tensor *stage[3] = {nullptr, nullptr, nullptr};      // [ne0, ne1, n_expert]: one layer's experts, a big batch
    size_t stage_pad[3] = {0, 0, 0};                         // after each expert's part there: zeros (see stage_pads)
    std::vector<int> owner;                                  // slot -> layer * n_expert + expert, -1
    std::vector<uint64_t> used;                              // slot -> last use (a clock)
    std::unique_ptr<std::atomic<int>[]> left;                // slot -> parts still being read
    std::vector<uint8_t> by_prefetch;                        // slot -> filled by a prediction, not used yet
    int zero = -1;                                           // --nostall: a slot of zeros, never taken
    int dev = 0;                                             // momoe-cuda --layer-split: the card its slots are on
};

struct Pool {
    std::vector<Group> g;
    int stage_zeroed[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  // a card's staging buffer: the group whose pads are zeros
    int n_slots = 0;                                         // all groups
    size_t all_bytes = 0, pool_bytes = 0;                    // every expert; the pool's slots
    ggml_backend_buffer_t buf = nullptr;                     // the slots' memory (rebuilt by fit_pool)
    std::vector<ggml_backend_buffer_t> more;                 // --layer-split: the later cards' slots
    std::vector<double> frac;                                // --layer-split: each card's share of its experts
    char * host = nullptr, * dev = nullptr;                  // a host-mapped pool (unified memory, --direct-io): its
                                                             // address for the CPU and the GPU's (readers write it)
    std::vector<int> slot_of;                                // layer * n_expert + expert -> slot in its group, -1
    uint64_t clock = 0;
    // counters
    uint64_t hits = 0, misses = 0, bytes_read = 0, calls = 0;
    uint64_t prefetched = 0, prefetch_hits = 0, late = 0;    // late: a prefetched expert not read yet when needed
    uint64_t stage_from_pool = 0;                            // momoe-cuda: staged experts copied from the pool, not read
    double read_s = 0, cb_s = 0;
    bool lfu = true;                                         // the least used (counts halved every
    std::vector<uint32_t> freq;                              // 64 tokens) is replaced, recency breaks ties
    uint64_t uses = 0;
};

static void read_at(int fd, void * dst, size_t size, size_t off);

// momoe-cuda --direct-io (a unified-memory machine such as the DGX Spark, whose memory cannot hold the model): experts
// are read from the SSD with O_DIRECT, past the page cache, instead of copied from the mapped file (paging a mapped
// file in reached 0.1-0.25 of the SSD's 11-13 GB/s there: llama.cpp, GLM-5.3-Flash). g_direct_fd: the model file
// opened again with O_DIRECT, by the descriptor of its ordinary opening
static bool g_direct = false;
static std::vector<int> g_direct_fd;
// momoe-cuda --layer-split: the card each of the engine's buffers is on (the readers copy to it on its own stream)
static std::vector<std::pair<ggml_backend_buffer_t, int>> g_buf_dev;
static int buf_dev(ggml_backend_buffer_t b) {
    for (auto & x : g_buf_dev)
        if (x.first == b) return x.second;
    return 0;
}
static const int MAX_CARDS = 8;

// read `need` bytes at `off` (a multiple of 4096) with O_DIRECT into `dst` (4096-aligned): `size` (a multiple of 4096,
// >= need) is asked for, the end of the file may cut it short
static void read_direct(int fd, char * dst, size_t size, size_t off, size_t need) {
    size_t got = 0;
    while (got < need) {
        const ssize_t r = pread(fd, dst + got, size - got, (off_t) (off + got));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) {
            const std::string why = std::to_string(off + got) + ": " + (r < 0 ? strerror(errno) : "end of file");
            die("a direct read failed at %s", why.c_str());
        }
        got += (size_t) r;
    }
}

// a few threads that read expert parts from the model file in the background (the SSD gives about twice the bandwidth
// with 4-8 reads in flight as with one: measured on the M4); each job counts down its slot's outstanding parts
#include <deque>
struct Readers {
    // src: a copy; t: momoe-cuda, src (the mapped file) copied to bytes t_off.. of the device tensor t
    struct Job { int fd; void * dst; size_t size, off; std::atomic<int> * left; const void * src = nullptr;
                 ggml_tensor * t = nullptr; size_t t_off = 0; };
    std::vector<std::thread> th;
    std::deque<Job> q;
    std::mutex mu;
    std::condition_variable cv;
    bool quit = false;
    int device = 0;                                          // momoe-cuda: the GPU the copies go to
    void start(int n) {
        for (int i = 0; i < n; ++i) th.emplace_back([this] { work(); });
    }
#if defined(MOMOE_CUDA)
    // momoe-cuda: each reader copies an expert from the mapped file into its own pinned buffer, then to the device on
    // its own stream (a copy from pageable memory: ~0.8 GB/s for these 1-2 MB pieces; from pinned: ~20 GB/s)
    struct Pinned { char * buf = nullptr; size_t cap = 0; cudaStream_t sts[MAX_CARDS] = {}; };
    // the copies are waited for by polling with short sleeps: cudaStreamSynchronize spins on a core, and 16 readers
    // spinning took those cores from the CPU's experts (measured: 24 instead of 11 ms of a token)
    static void sync(cudaStream_t st) {
        const auto t0 = std::chrono::steady_clock::now();
        bool told = false;
        cudaError_t r;
        while ((r = cudaStreamQuery(st)) == cudaErrorNotReady) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
            if (!told && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
                told = true;
                fprintf(stderr, "momoe-cuda: a reader's copy to the GPU has not finished after 2 s\n");
            }
        }
        if (r != cudaSuccess) fprintf(stderr, "momoe-cuda: a reader's copy failed: %s\n", cudaGetErrorString(r));
    }
    // (--direct-io: from the file at `off` instead of from `src`)
    static void to_device(Pinned & p, ggml_tensor * t, size_t t_off, const void * src, size_t n, int fd, size_t off) {
        const size_t piece = (size_t) 8 << 20, A = 4096, half = piece + 2 * A;   // (a direct read's aligned window)
        if (!p.buf) {
            p.cap = 2 * half;
            if (cudaMallocHost((void **) &p.buf, p.cap) != cudaSuccess) die("cannot allocate pinned memory");
        }
        const int d = g_buf_dev.empty() ? -1 : buf_dev(t->buffer);   // (--layer-split: the card the tensor is on)
        if (d >= 0) cudaSetDevice(d);
        cudaStream_t & st = p.sts[std::max(d, 0)];
        if (!st && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) die("cannot create a stream");
        char * dst = (char *) t->data + t_off;
        size_t k = 0;
        for (size_t done = 0; done < n; done += piece, ++k) {   // two halves: the next is filled while one copies
            const size_t len = std::min(piece, n - done);
            char * b = p.buf + (k & 1) * half;
            if (k >= 2) sync(st);                            // (wait before reusing a half)
            const char * from = b;
            if (g_direct) {
                const size_t at = off + done, a0 = at & ~(A - 1), lead = at - a0;
                read_direct(g_direct_fd[fd], b, (lead + len + A - 1) & ~(A - 1), a0, lead + len);
                from = b + lead;
            } else memcpy(b, (const char *) src + done, len);
            if (cudaMemcpyAsync(dst + done, from, len, cudaMemcpyHostToDevice, st) != cudaSuccess) die("a copy to the GPU failed");
        }
        sync(st);
    }
#endif
    // --direct-io into host memory (a host-mapped pool): 4 KiB-aligned windows into this thread's aligned buffer, copied
    // to `dst` by the CPU
    static void to_host_direct(void * dst, int fd, size_t off, size_t n) {
        const size_t piece = (size_t) 8 << 20, A = 4096;
        static thread_local char * buf = nullptr;
        if (!buf && posix_memalign((void **) &buf, A, piece + 2 * A) != 0) die("cannot allocate a reader's buffer");
        for (size_t done = 0; done < n; done += piece) {
            const size_t len = std::min(piece, n - done), at = off + done, a0 = at & ~(A - 1), lead = at - a0;
            read_direct(g_direct_fd[fd], buf, (lead + len + A - 1) & ~(A - 1), a0, lead + len);
            memcpy((char *) dst + done, buf + lead, len);
        }
    }
    void work() {
#if defined(MOMOE_CUDA)
        cudaSetDevice(device);
        Pinned pin;
#endif
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return quit || !q.empty(); });
                if (quit) return;
                j = q.front();
                q.pop_front();
            }
#if defined(MOMOE_CUDA)
            if (j.t) to_device(pin, j.t, j.t_off, j.src, j.size, j.fd, j.off);
            else
#endif
            if (j.src) memcpy(j.dst, j.src, j.size);
            else if (g_direct) to_host_direct(j.dst, j.fd, j.off, j.size);
            else read_at(j.fd, j.dst, j.size, j.off);
            j.left->fetch_sub(1, std::memory_order_release);
        }
    }
    void submit(std::vector<Job> & js, bool front = false) {   // front: the GPU is waiting for these
        if (js.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (front) q.insert(q.begin(), js.begin(), js.end());
            else q.insert(q.end(), js.begin(), js.end());
        }
        cv.notify_all();
        js.clear();
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto & t : th) t.join();
    }
};

// Metal's batched expert matmul (32+ tokens: a prompt chunk) takes at most 1,024 experts per tensor
// (ggml_metal_op_mul_mat_id: ne02 <= threads per threadgroup), the pool has more: such a batch reads a layer's experts
// from a staging copy of one layer (its experts in their own order), filled at the pause.  So does a batch whose
// experts could crowd a small pool (a group keeps at least 16 tokens' worth of slots)
static const int STAGE_MIN_TOKENS = 32;
static bool staged(int n_used, int n_slots, int T) { return T >= STAGE_MIN_TOKENS || 3 * n_used * T > n_slots; }

// momoe-cuda --hybrid: the routed experts the GPU's pool does not hold are computed on the CPU, straight from the
// mapped model file, while the GPU computes the ones it holds (llama.cpp's and Strata's split): a missing expert costs
// a read from RAM (~1.7 MB at tens of GB/s over all cores) instead of a copy over PCIe the GPU waits for. The
// arithmetic is ggml-cpu's own (the activation quantized to the weight type's vec_dot_type, one vec_dot per row).
struct CpuExperts {
    struct Miss { int x; std::vector<std::pair<int, int>> uses; };   // an expert; its (token, top-k position) pairs
    int n_threads = 0;
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t job = 0;                                        // bumped per job; workers run each job once
    bool quit = false;
    std::atomic<int> left{0};                                // workers still in the current job
    std::atomic<int> next{0}, bar_n{0}, bar_gen{0};
    // the job: one layer's missing experts
    const char * up = nullptr, * down = nullptr;             // the layer's up / down tensors in the mapping
    size_t up_bytes = 0, down_bytes = 0, up_row = 0, down_row = 0;
    ggml_type up_type = GGML_TYPE_COUNT, down_type = GGML_TYPE_COUNT;
    const char * gate = nullptr;                             // qwen35moe: the gate tensor too, h = silu(gate . x) * (up . x)
    size_t gate_bytes = 0, gate_row = 0;                     // (null: Nemotron's h = relu(up . x)^2)
    ggml_type gate_type = GGML_TYPE_COUNT;
    bool gelu = false;                                       // gemma4: h = gelu(gate . x) * (up . x), gate and up one
    float limit = 0;                                         // ling: SwiGLU clamped (ggml_swiglu_clamp: the gate at most
                                                             // limit, up within +-limit) where the block's limit is set
    int up_row0 = 0;                                         // tensor: its up rows from up_row0 (n_ff), its gate rows first
    int n_in = 0, n_ff = 0, k = 0, T = 0;                    // latent width, expert width, top-k, tokens
    std::vector<Miss> miss;
    std::vector<char> act;                                   // [T] x latent in up's vec_dot_type
    size_t act_row = 0, h_row = 0;
    std::vector<float> h;                                    // [miss, use] x n_ff
    std::vector<char> hq;                                    // the same in down's vec_dot_type
    std::vector<float> out;                                  // [T][k][latent]: the GPU adds it to its own part
    float * outp = nullptr;                                  // where run() writes: out.data(), or a mailbox
    std::vector<int> use_off;                                // first use of each miss in h
    int n_uses = 0;
    double busy_s = 0;
    uint64_t experts = 0;
    double t_launch = 0, prof[5] = {0, 0, 0, 0, 0};          // MOMOE_CPU_PROF: wake, up, to down's format, down, the
    std::atomic<int> started{0};                             // last thread's finish (thread 0's view)

    void start(int n) {
        n_threads = n;
        for (int i = 0; i < n; ++i) th.emplace_back([this, i] { work(i); });
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto & t : th) t.join();
    }
    void barrier() {
        const int g = bar_gen.load(std::memory_order_acquire);
        if (bar_n.fetch_add(1, std::memory_order_acq_rel) + 1 == n_threads) {
            bar_n.store(0, std::memory_order_relaxed);
            next.store(0, std::memory_order_relaxed);
            bar_gen.fetch_add(1, std::memory_order_acq_rel);
        } else {
            while (bar_gen.load(std::memory_order_acquire) == g) { }
        }
    }
    void work(int tid) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || job != seen; });
                if (quit) return;
                seen = job;
            }
            run(tid);
            left.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    void run(int tid) {
        static const bool pr = getenv("MOMOE_CPU_PROF") != nullptr;
        double p0 = 0;
        if (pr && started.fetch_add(1) == n_threads - 1) prof[0] += now_s() - t_launch;   // the last one awake
        if (pr && tid == 0) p0 = now_s();
        const auto * tu = ggml_get_type_traits_cpu(up_type), * td = ggml_get_type_traits_cpu(down_type);
        const int RB = 32;                                   // rows per work unit
        // up: h = relu(up . act)^2 (Nemotron) or silu(gate . act) * (up . act) (qwen35moe), every (expert, row block)
        // (gate and up share their type and so their activation format: llama.cpp's qwen35moe files do)
        const auto * tg = gate ? ggml_get_type_traits_cpu(gate_type) : nullptr;
        const int ub = (n_ff + RB - 1) / RB, nu = (int) miss.size() * ub;
        for (int u; (u = next.fetch_add(1)) < nu;) {
            const Miss & M = miss[u / ub];
            const int r0 = (u % ub) * RB, r1 = std::min(n_ff, r0 + RB);
            const char * w = up + (size_t) M.x * up_bytes;
            const char * wg = gate ? gate + (size_t) M.x * gate_bytes : nullptr;
            const int base = use_off[u / ub];
            for (int r = r0; r < r1; ++r)
                for (size_t j = 0; j < M.uses.size(); ++j) {
                    const char * a = act.data() + (size_t) M.uses[j].first * act_row;
                    float v;
                    tu->vec_dot(n_in, &v, 0, w + (size_t) (up_row0 + r) * up_row, 0, a, 0, 1);
                    if (wg) {
                        float g;
                        tg->vec_dot(n_in, &g, 0, wg + (size_t) r * gate_row, 0, a, 0, 1);
                        if (limit > 0) {
                            g = std::min(g, limit);
                            v = std::clamp(v, -limit, limit);
                        }
                        // (gelu: ggml's tanh form, as ggml_geglu_split)
                        const float a2 = gelu ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * g * (1.0f + 0.044715f * g * g))) : 0;
                        h[(size_t) (base + j) * n_ff + r] = gelu ? a2 * v : g / (1.0f + expf(-g)) * v;
                    } else {
                        v = v > 0 ? v : 0;
                        h[(size_t) (base + j) * n_ff + r] = v * v;
                    }
                }
        }
        barrier();
        if (pr && tid == 0) { const double t = now_s(); prof[1] += t - p0; p0 = t; }
        for (int u; (u = next.fetch_add(1)) < n_uses;)      // h in down's activation format
            ggml_get_type_traits_cpu(td->vec_dot_type)->from_float(h.data() + (size_t) u * n_ff, hq.data() + (size_t) u * h_row, n_ff);
        barrier();
        if (pr && tid == 0) { const double t = now_s(); prof[2] += t - p0; p0 = t; }
        const int db = (n_in + RB - 1) / RB, nd = (int) miss.size() * db;
        for (int u; (u = next.fetch_add(1)) < nd;) {
            const Miss & M = miss[u / db];
            const int r0 = (u % db) * RB, r1 = std::min(n_in, r0 + RB);
            const char * w = down + (size_t) M.x * down_bytes;
            const int base = use_off[u / db];
            for (size_t j = 0; j < M.uses.size(); ++j) {
                float * o = outp + ((size_t) M.uses[j].first * k + M.uses[j].second) * n_in;
                const char * a = hq.data() + (size_t) (base + j) * h_row;
                for (int r = r0; r < r1; ++r) td->vec_dot(n_ff, o + r, 0, w + (size_t) r * down_row, 0, a, 0, 1);
            }
        }
        if (pr && tid == 0) prof[3] += now_s() - p0;
    }
    // the layer's job: x_lat [T][n_in] floats; returns at once (wait() joins)
    void launch(const float * x_lat) {
        const auto * tu = ggml_get_type_traits_cpu(up_type), * td = ggml_get_type_traits_cpu(down_type);
        act_row = ggml_row_size(tu->vec_dot_type, n_in);
        h_row = ggml_row_size(td->vec_dot_type, n_ff);
        act.resize(act_row * T);
        for (int t = 0; t < T; ++t)
            ggml_get_type_traits_cpu(tu->vec_dot_type)->from_float(x_lat + (size_t) t * n_in, act.data() + t * act_row, n_in);
        use_off.resize(miss.size());
        n_uses = 0;
        for (size_t i = 0; i < miss.size(); ++i) { use_off[i] = n_uses; n_uses += (int) miss[i].uses.size(); }
        h.assign((size_t) n_uses * n_ff, 0.0f);
        hq.resize(h_row * n_uses);
        experts += miss.size();
        next.store(0);
        bar_n.store(0);
        started.store(0);
        t_launch = now_s();
        left.store(n_threads);
        { std::lock_guard<std::mutex> lk(mu); ++job; }
        cv.notify_all();
    }
    void wait() {
        while (left.load(std::memory_order_acquire) > 0) { }
        static const bool pr = getenv("MOMOE_CPU_PROF") != nullptr;
        if (pr) prof[4] += now_s() - t_launch;
    }
};

static const int HYB_MAX_T = 16;                             // --hybrid: graphs of up to this many tokens (decode)

// the CPU's experts: a layer's parts in the mapped file (gate, up, down; their formats and rows)
static void cpu_parts(const Model & m, const Layer & L, CpuExperts & C) {
    C.n_ff = (int) L.exp_ne[1][1];
    C.up_type = L.exp_type[1];
    C.down_type = L.exp_type[2];
    C.up = L.exp_map[1] + L.exp_off[1];
    C.down = L.exp_map[2] + L.exp_off[2];
    C.up_bytes = L.exp_bytes[1];
    C.down_bytes = L.exp_bytes[2];
    C.up_row = ggml_row_size(C.up_type, L.exp_ne[1][0]);
    C.down_row = ggml_row_size(C.down_type, L.exp_ne[2][0]);
    C.gate = m.part0() == 0 ? L.exp_map[0] + L.exp_off[0] : nullptr;   // qwen35moe: SwiGLU
    C.gate_type = L.exp_type[0];
    C.gate_bytes = L.exp_bytes[0];
    C.gate_row = C.gate ? ggml_row_size(C.gate_type, L.exp_ne[0][0]) : 0;
    C.gelu = m.gemma4;
    C.limit = m.ling ? m.clamp_exp[&L - m.layers.data()] : 0.0f;
    C.up_row0 = 0;
    if (m.gemma4) {                                          // gate | up in one tensor: GEGLU
        C.n_ff /= 2;
        C.gate = C.up;
        C.gate_type = C.up_type;
        C.gate_bytes = C.up_bytes;
        C.gate_row = C.up_row;
        C.up_row0 = C.n_ff;
    }
}

// Nemotron; qwen35moe unless MOMOE_QWEN_HYBRID=0.  For qwen35moe the CPU path is one of two the engine tries on the first
// decode tokens and keeps the faster of, measuring again while it runs (forward): on the RTX 4070 Ti + Ryzen 9 5950X it
// won with Ornith-1.5-122B UD-IQ3_XXS (13% of the experts in the pool: 16.4 tok/s, the copies 10.6) and with
// Ornith-1.5-35B IQ4_XS (52%: 14.1 ms per token, the copies 15.6)
static bool hybrid_model(const Model & m) {
    static const char * q = getenv("MOMOE_QWEN_HYBRID");
    return m.nemo || (!(q && q[0] == '0') && !m.qwen4 && !m.glm5 && !m.ds41 && (!m.lfm || m.flint));   // (flint: SwiGLU experts as qwen35moe's)
}


#if defined(MOMOE_CUDA)
// momoe-cuda --hybrid with mailboxes (default; MOMOE_NO_MAILBOX=1: the stops of the scheduler's callback instead): a
// token runs as one graph. Each MoE layer's SEND op posts its latent input and chosen experts to page-locked memory the
// GPU maps; a worker thread follows the layers, computes the experts the pool lacks on the CPU and posts their outputs,
// which the layer's WAIT op (spinning on the flag) adds to the GPU's. The slot table changes only between graphs: an
// expert being read into the pool is published at the next token boundary (ggml-momoe-mailbox.h).
struct MbLayer {
    volatile uint32_t * sent = nullptr, * done = nullptr;    // host addresses
    float * x = nullptr, * out = nullptr;                    // [T][latent], [T][k][latent]
    int32_t * ids = nullptr;                                 // [T][k]
    momoe_mailbox_op send{}, wait{};                         // the ops' userdata (device addresses)
};
struct Mailbox {
    char * host = nullptr;
    char * dev = nullptr;
    volatile uint32_t * epoch = nullptr, * timeout = nullptr;
    std::vector<MbLayer> layer;                              // by layer index (MoE layers only)
    uint32_t cur = 0;                                        // this graph's epoch
    // the worker
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t job = 0, done_job = 0;
    bool quit = false;
    int T = 0;
    std::vector<int> moe;                                    // the MoE layers, in order
    std::vector<int> graph;                                  // the ones this graph posts (mailbox layers), in order
    std::vector<std::pair<int, int>> want;                   // (layer, expert): missed and admitted, to read in
};
#endif

struct Engine {
    Model m;
    Pool pool;
    std::vector<int> fds;                                    // the model's files (shards)
    std::vector<int32_t> tok_at;                             // qwen4exp: the token at each position (the PLE n-grams)
    Readers readers;
    Readers demand;                                          // MOMOE_DEMAND_READERS: the reads a layer waits for, on
                                                             // readers of their own (never behind predictions)
    Readers fillers;                                         // momoe-cuda --hybrid: the background fills (few threads:
                                                             // many took the RAM bandwidth the CPU's experts need)
    int n_ctx = 32768, max_batch = 512;
    int device = 0;                                          // momoe-cuda: --device N (CUDA's numbering)
    int cur_layer = -1;                                      // (debug names: the block being built)
    bool uma = false;                                        // momoe-cuda: an integrated GPU (the DGX Spark): its memory is the RAM
    bool reserve_only = false;                               // forward(): build the graph, reserve its buffers, stop
    bool hybrid = false;                                     // momoe-cuda: missing experts computed on the CPU
    bool hyb_active = false;                                 // qwen35moe: this graph takes the CPU path (Nemotron: always)
    bool hyb_done = false;                                   // qwen35moe: the trial has chosen (or MOMOE_QWEN_HYBRID=1 did)
    bool predict_copy = true;                                // the copy path's read-ahead (the CPU path runs without it)
    float hyb_frac = 0;                                      // qwen35moe's CPU path: the share of a layer's missing experts
                                                             // copied in instead (the most used first), the rest on the CPU
    bool hyb_forced = false;                                 // MOMOE_QWEN_HYBRID=1 or MOMOE_HYB_FRAC: no trials
    int hyb_trial = 0;                                       // the running trial's decode calls so far (forward)
    int hyb_trials = 0;                                      // the trials done
    double hyb_s[2] = {0, 0};                                // its measured time and tokens per candidate
    int hyb_tok[2] = {0, 0};
    const char * hyb_why = "at load";                        // why the running trial runs
    double hyb_ms = 0;                                       // the chosen path's pace in its trial (ms per token)
    double hyb_ws = 0;                                       // the watch: the chosen path's time and tokens in this window,
    int hyb_wt = 0, hyb_off = 0, hyb_since = 0;              // windows in a row off its pace, tokens since the trial
    CpuExperts cpu_x;
    std::vector<ggml_tensor *> xl;                           // this graph's latent MoE inputs, by layer (hybrid)
    std::vector<float> xl_host;
#if defined(MOMOE_CUDA)
    bool mailbox = false;                                    // --hybrid through mailboxes (default)
    bool rmb = false;                                        // resident mailboxes (rmb_init: the GPU waits, the CPU
                                                             // makes the experts resident and hands back their slots)
    Mailbox mb;
    bool mb_graph = false;                                   // this graph uses them
    bool mb_stale = false;                                   // a graph without them has moved experts since
                                                             // (the slot tables are rebuilt before the next one uses them)
    std::vector<ggml_tensor *> force_gpu;                    // ops the scheduler must place on the GPU
    struct Pending { int il, x, slot; };
    std::vector<Pending> mb_pending;                         // experts being read into the pool (published when read)
    uint64_t mb_fills = 0, mb_graphs = 0;
#endif
    uint64_t hyb_hits = 0, hyb_miss = 0, hyb_copy = 0;
    double t_cpu_wait = 0;
    bool load_mtp = false;                                   // --mtp K: draft K tokens with the MTP layer
    bool preloaded = false;                                  // every expert in its own slot (preload)
    bool pool_asked = false;                                 // --pool-gib or --slots given (fit_pool_metal's margin)
    std::vector<ggml_backend_t> sched_backends;              // e.sched's (a reused graph's own scheduler)
    std::string draft_vocab;                                 // --draft-vocab: the draft head's token subset
    std::string mtp_gguf;                                    // --mtp-gguf: qwen4exp's MTP layer (not in its GGUFs)
    std::string engram;                                      // --engram: deepseek41's Engram file (else found by its name)
    // deepseek41's DSpark drafter (--dspark FILE, llama.cpp's dflash graph_dsv41 + its DSpark heads): three blocks as the
    // model's (window-only attention, mHC, a small MoE whose experts stay whole), each with a window ring (its Layer's
    // k_cache) that the model's forward fills from its blocks' inputs (`targets`, through fc and enc_norm); a draft is
    // a block of `block` positions [the last token, the mask token...], every position read at once, the Markov head
    // (mk_w1, mk_w2) biasing each one's logits by the token before it and the confidence head (conf) cutting the block
    struct Dspark {
        std::string path;
        bool on = false;
        std::vector<Layer> layers;
        std::vector<int> targets;
        int n_expert = 0, n_used = 0, block = 0;
        int32_t mask_tok = 0;
        float scale = 1.0f, conf_min = 0.65f;
        std::vector<float> clamp_exp, clamp_sh;
        ggml_tensor *fc = nullptr, *enc_norm = nullptr, *out_norm = nullptr, *mk_w1 = nullptr, *mk_w2 = nullptr, *conf = nullptr;
        ggml_context * c = nullptr;                          // the draft graph, kept: its shapes never change
        ggml_cgraph * gf = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ggml_tensor *tok = nullptr, *pos = nullptr, *idx = nullptr, *mask = nullptr;
        std::vector<ggml_tensor *> best, cf;                 // each position's token and confidence
        ggml_tensor *on_cpu = nullptr, *on_gpu = nullptr;
        double t = 0;
        uint64_t calls = 0;
    } ds;
    double t_mtp = 0;                                        // time in the MTP layer
    int max_window = 8;                                      // verify windows up to this many tokens
    ggml_backend_t metal = nullptr, cpu = nullptr;
    // momoe-cuda --layer-split auto|K[,K2..] (the PC engine's flag, the server adds it for a config's "gpu": [a, b]): the
    // layers in contiguous ranges, one per visible card (CUDA's numbering, the config's order), each range's weights,
    // state and experts on its card; the token embedding on the first, the head and the MTP layer on the last
    std::vector<ggml_backend_t> gpus;                        // gpus[0] == metal
    std::string split;                                       // the --layer-split value ("": one card)
    // momoe-cuda --helper-tier (the PC engine's switch; "gpu": [main, helper]): the second visible card holds a second
    // pool under the main one - the experts the main pool evicts, computed there when a token chooses them
    bool helper = false;
    ggml_backend_t helper_gpu = nullptr;
    Pool hpool;
    double helper_gib = 0, helper_reserve_mib = 512;         // --helper-gib, --helper-reserve-mib
    uint64_t helper_hits = 0, helper_down = 0;               // experts computed there; evictions copied down to it
    ggml_backend_sched_t sched = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<int32_t> ids_host;                           // scratch for the callback
    std::vector<int32_t> pred_host;                          // a pause's predictions, copied before the GPU goes on
    std::vector<std::string> saved;                          // the recurrent state before a verify window
    int batch_tokens = 0;                                    // tokens in the graph being computed
    bool predict = true;                                     // read the next layer's predicted experts ahead
    bool predict2 = true;                                    // and the layer after it
    bool pause = true;                                       // pause points in the command buffer (else: the
                                                             // scheduler's callback, a stop per layer)
    std::vector<ggml_tensor *> topk;                         // this graph's chosen-expert tensors, by layer
    // --nostall sub|drop: a token's graph runs without pauses. Each layer routes among the experts in the pool
    // (sub: the best ones there; drop: its own choice, the missing ones weighted 0) and reports the experts it wanted;
    // between graphs the CPU reads those (and the predictions) into the pool, and publishes them once read. Changes the
    // output; the prompt (and any graph a pool cannot serve this way) takes the exact path with pauses.
    int nostall = 0;                                         // 1 sub, 2 drop
    int ns_every = 0;                                        // --nostall-every N: layers 0, N, 2N... keep their
                                                             // pause (exact), and publish what was read since for
                                                             // the layers after them (0: no pauses at all)
    bool ns_graph = false;                                   // this graph
    uint64_t ns_tokens = 0, ns_wanted = 0, ns_missing = 0;
    double t_build = 0, t_compute = 0, t_wait = 0;
    // pictures (--serve's GENI): each a run of n image cells from s on, its rows the encoder's embeddings (a grid of nx
    // by ny, row i at x = i % nx, y = i / nx) in img_data from off; they stay until the next request (decode's positions)
    struct Image { int s, n, nx, ny; size_t off; };
    std::vector<Image> imgs;
    std::vector<float> img_data;
    bool img_now = false;                                    // this graph's tokens include image cells
};

// the backend of card d (one card, and Metal: the GPU backend)
static ggml_backend_t card(const Engine & e, int d) {
    return d > 0 && d < (int) e.gpus.size() ? e.gpus[d] : e.metal;
}

static int n_cards(const Engine & e) { return std::max(1, (int) e.gpus.size()); }

static int gguf_int(gguf_context * g, const std::string & key) {
    int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) die("GGUF key %s missing", key.c_str());
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(g, i);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, i);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(g, i);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, i);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(g, i);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(g, i);
        default: die("GGUF key %s: not an integer", key.c_str());
    }
}

static float gguf_float(gguf_context * g, const std::string & key) {
    int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) die("GGUF key %s missing", key.c_str());
    return gguf_get_val_f32(g, i);
}

// a per-block float given as an array or as one value; `dflt` when the file has neither
static std::vector<float> gguf_per_block_f(gguf_context * g, const std::string & key, int n, float dflt) {
    const int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) return std::vector<float>(n, dflt);
    if (gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY) return std::vector<float>(n, gguf_get_val_f32(g, i));
    if (gguf_get_arr_type(g, i) != GGUF_TYPE_FLOAT32) die("GGUF key %s: not a float array", key.c_str());
    const float * d = (const float *) gguf_get_arr_data(g, i);
    std::vector<float> v(d, d + gguf_get_arr_n(g, i));
    if ((int) v.size() < n) die("GGUF key %s: shorter than the model", key.c_str());
    return v;
}

// read `size` bytes at `off` of the model file into `dst`
static void read_at(int fd, void * dst, size_t size, size_t off) {
    char * p = (char *) dst;
    while (size) {
        ssize_t n = pread(fd, p, std::min(size, (size_t) 1 << 30), (off_t) off);
        if (n <= 0) die("reading the model file failed");
        p += n; off += n; size -= n;
    }
}

// a GGUF integer array, widened
static std::vector<uint64_t> gguf_uarr(gguf_context * g, const std::string & key) {
    int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) die("GGUF key %s missing", key.c_str());
    const size_t n = gguf_get_arr_n(g, i);
    const void * d = gguf_get_arr_data(g, i);
    std::vector<uint64_t> v(n);
    for (size_t j = 0; j < n; ++j) {
        switch (gguf_get_arr_type(g, i)) {
            case GGUF_TYPE_UINT32: v[j] = ((const uint32_t *) d)[j]; break;
            case GGUF_TYPE_INT32:  v[j] = (uint64_t) ((const int32_t *) d)[j]; break;
            case GGUF_TYPE_UINT64: v[j] = ((const uint64_t *) d)[j]; break;
            case GGUF_TYPE_INT64:  v[j] = (uint64_t) ((const int64_t *) d)[j]; break;
            default: die("GGUF key %s: not an integer array", key.c_str());
        }
    }
    return v;
}

// a bool array (flint's sliding_window_pattern, rope_pattern)
static std::vector<bool> gguf_barr(gguf_context * g, const std::string & key) {
    int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) die("GGUF key %s missing", key.c_str());
    if (gguf_get_arr_type(g, i) != GGUF_TYPE_BOOL) die("GGUF key %s: not a bool array", key.c_str());
    const size_t n = gguf_get_arr_n(g, i);
    const int8_t * d = (const int8_t *) gguf_get_arr_data(g, i);
    std::vector<bool> v(n);
    for (size_t j = 0; j < n; ++j) v[j] = d[j] != 0;
    return v;
}

// a per-block value given as an array (Bolt) or as one value for every block (Super)
static std::vector<uint64_t> gguf_per_block(gguf_context * g, const std::string & key, int n) {
    int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) die("GGUF key %s missing", key.c_str());
    if (gguf_get_kv_type(g, i) == GGUF_TYPE_ARRAY) return gguf_uarr(g, key);
    return std::vector<uint64_t>(n, (uint64_t) gguf_int(g, key));
}

// the files of a split model: NAME-00001-of-0000N.gguf -> every part (one file: itself)
static std::vector<std::string> shard_paths(const char * path, int n) {
    std::vector<std::string> r = {path};
    if (n <= 1) return r;
    std::string s = path;
    char tag[32];
    snprintf(tag, sizeof tag, "-00001-of-%05d.gguf", n);
    const size_t at = s.rfind(tag);
    if (at == std::string::npos || at + strlen(tag) != s.size()) die("%s: a split model is opened by its first file (-00001-of-...)", path);
    for (int i = 2; i <= n; ++i) {
        snprintf(tag, sizeof tag, "-%05d-of-%05d.gguf", i, n);
        r.push_back(s.substr(0, at) + tag);
    }
    return r;
}

// qwen4exp's MTP layer, from its own file (llama.cpp's GGUFs leave it out): the checkpoint's mtp.* tensors as
// tools/mtp_pack.py writes them (--experts q4_0: a format Metal computes). The norms are stored zero-centred (the
// checkpoint scales by 1 + w): +1 here, as llama.cpp's converter does for the model's own norms. The fused gate_up
// experts split into gate and up (gate first, as the converter splits them).
static void load_mtp_qwen4(Engine & e) {
    Model & m = e.m;
    const char * path = e.mtp_gguf.c_str();
    if (e.mtp_gguf.empty()) die("--mtp with this model needs --mtp-gguf FILE (its MTP layer: tools/mtp_pack.py --experts q4_0)");
    ggml_context * meta = nullptr;
    gguf_init_params gp = {/*no_alloc*/ true, &meta};
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) die("cannot read %s as GGUF", path);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    const size_t data_off = gguf_get_data_offset(g);
    Layer & L = m.mtp;
    const std::string l0 = "mtp.layers.0.";
    struct Want { std::string name; ggml_tensor ** to; int how; };   // how: 0 as stored, 1 a norm (+1), 2 gate, 3 up
    const std::vector<Want> want = {
        {"mtp.pre_fc_norm_embedding.weight", &L.enorm, 1}, {"mtp.pre_fc_norm_hidden.weight", &L.hnorm, 1},
        {"mtp.fc_embedding.weight", &L.fc_e, 0}, {"mtp.fc_hidden.weight", &L.fc_h, 0},
        {l0 + "attn_hyper_connection.hc_norm.weight", &L.hc_attn_norm, 1},
        {l0 + "attn_hyper_connection.input_mix_weight_down.weight", &L.hc_attn_down, 0},
        {l0 + "attn_hyper_connection.input_mix_weight_up.weight", &L.hc_attn_up, 0},
        {l0 + "attn_hyper_connection.block_inject_weight.weight", &L.hc_attn_inject, 0},
        {l0 + "mlp_hyper_connection.hc_norm.weight", &L.hc_ffn_norm, 1},
        {l0 + "mlp_hyper_connection.input_mix_weight_down.weight", &L.hc_ffn_down, 0},
        {l0 + "mlp_hyper_connection.input_mix_weight_up.weight", &L.hc_ffn_up, 0},
        {l0 + "mlp_hyper_connection.block_inject_weight.weight", &L.hc_ffn_inject, 0},
        {l0 + "self_attn.q_proj.weight", &L.wq, 0}, {l0 + "self_attn.k_proj.weight", &L.wk, 0},
        {l0 + "self_attn.v_proj.weight", &L.wv, 0}, {l0 + "self_attn.o_proj.weight", &L.wo, 0},
        {l0 + "self_attn.q_norm.weight", &L.q_norm, 1}, {l0 + "self_attn.k_norm.weight", &L.k_norm, 1},
        {l0 + "mlp.gate.weight", &L.gate_inp, 0}, {l0 + "mlp.shared_expert_gate.weight", &L.gate_inp_sh, 0},
        {l0 + "mlp.shared_expert.gate_proj.weight", &L.gate_sh, 0},
        {l0 + "mlp.shared_expert.up_proj.weight", &L.up_sh, 0},
        {l0 + "mlp.shared_expert.down_proj.weight", &L.down_sh, 0},
        {l0 + "mlp.experts.gate_up_proj", &L.exps[0], 2}, {l0 + "mlp.experts.gate_up_proj", &L.exps[1], 3},
        {l0 + "mlp.experts.down_proj", &L.exps[2], 0},
        {"mtp.hyper_connection_mixer.hc_norm.weight", &m.mtp_head_norm, 1},
        {"mtp.hyper_connection_mixer.input_mix_weight_down.weight", &m.mtp_head_down, 0},
        {"mtp.hyper_connection_mixer.input_mix_weight_up.weight", &m.mtp_head_up, 0},
    };
    ggml_init_params ip = {ggml_tensor_overhead() * 64, nullptr, true};
    ggml_context * cm = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, size_t>> src;
    for (const Want & w : want) {
        const int64_t id = gguf_find_tensor(g, w.name.c_str());
        if (id < 0) die("the MTP file has no %s", w.name.c_str());
        ggml_tensor * s = ggml_get_tensor(meta, w.name.c_str());
        src.push_back({s, data_off + gguf_get_tensor_offset(g, id)});
        if (w.how == 1 && s->type != GGML_TYPE_F32) die("%s: a norm not stored as F32", w.name.c_str());
        ggml_tensor * t = w.how == 1 ? ggml_new_tensor_1d(cm, GGML_TYPE_F32, ggml_nelements(s))
                        : w.how >= 2 ? ggml_new_tensor_3d(cm, s->type, s->ne[0], s->ne[1] / 2, s->ne[2])
                                     : ggml_dup_tensor(cm, s);
        ggml_set_name(t, (w.name + (w.how == 3 ? ".up" : "")).c_str());
        *w.to = t;
    }
    e.bufs.push_back(ggml_backend_alloc_ctx_tensors(cm, card(e, n_cards(e) - 1)));   // (--layer-split: the last card)
    size_t bytes = 0;
    std::vector<char> raw;
    for (size_t i = 0; i < want.size(); ++i) {
        ggml_tensor * s = src[i].first, * t = *want[i].to;
        bytes += ggml_nbytes(t);
        raw.resize(ggml_nbytes(s));
        read_at(fd, raw.data(), raw.size(), src[i].second);
        if (want[i].how == 0) {
            tset(t, raw.data(), 0, raw.size());
            continue;
        }
        std::vector<char> d(ggml_nbytes(t));
        if (want[i].how == 1) {
            const float * r = (const float *) raw.data();
            for (int64_t j = 0; j < ggml_nelements(s); ++j) ((float *) d.data())[j] = r[j] + 1.0f;
        } else {                                             // each expert's first (gate) or second (up) half of rows
            const size_t half = s->nb[1] * (s->ne[1] / 2);
            for (int64_t x = 0; x < s->ne[2]; ++x)
                memcpy(d.data() + x * half, raw.data() + x * s->nb[2] + (want[i].how == 3 ? half : 0), half);
        }
        tset(t, d.data(), 0, d.size());
    }
    close(fd);
    gguf_free(g);
    ggml_free(meta);
    m.has_mtp = true;
    fprintf(stderr, "momoe-metal: MTP layer from %s: %.2f GiB\n", path, bytes / 1073741824.0);
}

// deepseek41's DSpark drafter (see Engine::Dspark), from its own GGUF (llama.cpp's dflash arch): its blocks' tensors by
// the model's names (the experts whole), the encoder (fc, enc.output_norm), its output norm and its two heads; the
// token embedding and the output head are the model's.  BF16 tensors (the Markov head) as F32
static void load_dspark(Engine & e) {
    Model & m = e.m;
    auto & D = e.ds;
    const char * path = D.path.c_str();
    if (!m.ds41) die("--dspark: only for DeepSeek-V4.1 models");
    ggml_context * meta = nullptr;
    gguf_init_params gp = {/*no_alloc*/ true, &meta};
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) die("cannot read %s as GGUF", path);
    if (gguf_get_val_str(g, gguf_find_key(g, "general.architecture")) != std::string("dflash")) die("--dspark: %s is not a DFlash / DSpark file", path);
    auto k = [&](const char * s) { return std::string("dflash.") + s; };
    auto has = [&](const char * s) { return gguf_find_key(g, k(s).c_str()) >= 0; };
    const int n_layer = gguf_int(g, k("block_count"));
    if (!has("hyper_connection.single_pass") || !gguf_get_val_bool(g, gguf_find_key(g, k("hyper_connection.single_pass").c_str())))
        die("--dspark: only the single-pass mHC drafter (DeepSeek-V4.1's)");
    if (gguf_int(g, k("embedding_length")) != m.n_embd || gguf_int(g, k("attention.key_length")) != m.ds_hd ||
        gguf_int(g, k("attention.head_count")) != m.n_head || gguf_int(g, k("attention.sliding_window")) != m.n_swa ||
        gguf_int(g, k("hyper_connection.count")) != m.hc)
        die("--dspark: the drafter's shapes are not this model's");
    for (uint64_t r : gguf_uarr(g, k("attention.compress_ratios")))
        if (r) die("--dspark: a drafter block with compression (not done)");
    for (uint64_t t : gguf_uarr(g, k("target_layers"))) {
        if ((int) t >= m.n_layer) die("--dspark: a target block past the model's");
        D.targets.push_back((int) t);
    }
    D.block = gguf_int(g, k("block_size"));
    D.n_expert = gguf_int(g, k("expert_count"));
    D.n_used = gguf_int(g, k("expert_used_count"));
    D.scale = has("expert_weights_scale") ? gguf_float(g, k("expert_weights_scale")) : 1.0f;
    if (gguf_int(g, k("expert_gating_func")) != 4) die("--dspark: only the sqrt(softplus) router");
    D.clamp_exp = gguf_per_block_f(g, k("swiglu_clamp_exp"), n_layer, 0.0f);
    D.clamp_sh = has("swiglu_clamp_shexp") ? gguf_per_block_f(g, k("swiglu_clamp_shexp"), n_layer, 0.0f) : D.clamp_exp;
    if (has("sample_from_anchor") && !gguf_get_val_bool(g, gguf_find_key(g, k("sample_from_anchor").c_str())))
        die("--dspark: only the anchor-first block (sample_from_anchor)");
    const int64_t mi = gguf_find_key(g, "tokenizer.ggml.mask_token_id");
    if (mi < 0) die("--dspark: the file names no mask token");
    D.mask_tok = (int32_t) gguf_get_val_u32(g, mi);
    if (D.block < 2 || D.targets.empty() || D.n_used < 1) die("--dspark: an empty drafter");
    D.layers.resize(n_layer);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    const size_t data_off = gguf_get_data_offset(g);
    ggml_init_params ip = {ggml_tensor_overhead() * 256, nullptr, true};
    ggml_context * cw = ggml_init(ip);
    struct ToRead { ggml_tensor * t; ggml_tensor * s; size_t off; };
    std::vector<ToRead> rd;
    for (int i = 0; i < (int) gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * s = ggml_get_tensor(meta, name);
        ggml_tensor * t = s->type == GGML_TYPE_BF16 ? ggml_new_tensor(cw, GGML_TYPE_F32, ggml_n_dims(s), s->ne) : ggml_dup_tensor(cw, s);
        if (strstr(name, "attn_output_a.weight") && ggml_n_dims(s) == 2) {   // (stored [d, rank * groups]: the model's
            if (s->ne[1] % m.o_lora) die("--dspark: attn_output_a's rows are not whole groups");   // [d, rank, groups])
            t = ggml_new_tensor_3d(cw, s->type, s->ne[0], m.o_lora, s->ne[1] / m.o_lora);
        }
        ggml_set_name(t, (std::string("dspark.") + name).c_str());
        int il = -1;
        char rest[128] = {0};
        ggml_tensor ** to = nullptr;
        if (sscanf(name, "blk.%d.%127s", &il, rest) == 2 && il >= 0 && il < n_layer) {
            Layer & L = D.layers[il];
            const std::map<std::string, ggml_tensor **> slot = {
                {"attn_norm.weight", &L.attn_norm}, {"attn_sinks.weight", &L.sinks}, {"attn_q_a.weight", &L.wq_a},
                {"attn_q_a_norm.weight", &L.q_a_norm}, {"attn_q_b.weight", &L.wq_b}, {"attn_kv.weight", &L.wkv_a},
                {"attn_kv_a_norm.weight", &L.kv_a_norm}, {"attn_output_a.weight", &L.wo_a}, {"attn_output_b.weight", &L.wo},
                {"hc_attn_fn.weight", &L.hc_attn_fn}, {"hc_attn_base.weight", &L.hc_attn_base},
                {"hc_attn_scale.weight", &L.hc_attn_scale}, {"hc_ffn_fn.weight", &L.hc_ffn_fn},
                {"hc_ffn_base.weight", &L.hc_ffn_base}, {"hc_ffn_scale.weight", &L.hc_ffn_scale},
                {"ffn_norm.weight", &L.ffn_norm}, {"ffn_gate_inp.weight", &L.gate_inp}, {"exp_probs_b.bias", &L.exp_probs_b},
                {"ffn_gate_shexp.weight", &L.gate_sh}, {"ffn_up_shexp.weight", &L.up_sh}, {"ffn_down_shexp.weight", &L.down_sh},
                {"ffn_gate_exps.weight", &L.exps[0]}, {"ffn_up_exps.weight", &L.exps[1]}, {"ffn_down_exps.weight", &L.exps[2]}};
            auto it = slot.find(rest);
            if (it != slot.end()) to = it->second;
        } else {
            const std::map<std::string, ggml_tensor **> slot = {
                {"fc.weight", &D.fc}, {"enc.output_norm.weight", &D.enc_norm}, {"output_norm.weight", &D.out_norm},
                {"markov_w1.weight", &D.mk_w1}, {"markov_w2.weight", &D.mk_w2}, {"conf_proj.weight", &D.conf}};
            auto it = slot.find(name);
            if (it != slot.end()) to = it->second;
        }
        if (!to) { fprintf(stderr, "momoe-metal: DSpark tensor %s not used\n", name); continue; }
        *to = t;
        rd.push_back({t, s, data_off + gguf_get_tensor_offset(g, i)});
    }
    if (!D.fc || !D.enc_norm || !D.out_norm || !D.mk_w1 || !D.mk_w2 || !D.conf) die("--dspark: the drafter's encoder or heads are missing");
    for (const Layer & L : D.layers)
        if (!L.wq_a || !L.wq_b || !L.wkv_a || !L.kv_a_norm || !L.wo_a || !L.wo || !L.sinks || !L.hc_attn_fn || !L.hc_ffn_fn ||
            !L.gate_inp || !L.exp_probs_b || !L.exps[0] || !L.exps[1] || !L.exps[2] || !L.gate_sh || !L.down_sh)
            die("--dspark: a drafter block's tensors are missing");
    if (D.fc->ne[0] != (int64_t) D.targets.size() * m.n_embd) die("--dspark: fc does not take the target blocks' features");
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, card(e, n_cards(e) - 1));
    if (!bw) die("cannot allocate the DSpark drafter");
    e.bufs.push_back(bw);
    size_t bytes = 0;
    std::vector<char> raw;
    for (const ToRead & r : rd) {
        raw.resize(ggml_nbytes(r.s));
        read_at(fd, raw.data(), raw.size(), r.off);
        bytes += ggml_nbytes(r.t);
        if (r.s->type == GGML_TYPE_BF16) {
            std::vector<float> f(ggml_nelements(r.s));
            ggml_get_type_traits(GGML_TYPE_BF16)->to_float(raw.data(), f.data(), (int64_t) f.size());
            tset(r.t, f.data(), 0, f.size() * sizeof(float));
        } else {
            tset(r.t, raw.data(), 0, raw.size());
        }
    }
    close(fd);
    // the window rings, as the model's (m.swa_rows positions)
    ggml_context * cs = ggml_init(ip);
    for (Layer & L : D.layers) {
        L.k_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.ds_hd, m.swa_rows);
        L.swa = true;
    }
    ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors(cs, card(e, n_cards(e) - 1));
    if (!bs) die("cannot allocate the DSpark drafter's rings");
    ggml_backend_buffer_clear(bs, 0);
    e.bufs.push_back(bs);
    gguf_free(g);
    ggml_free(meta);
    D.on = true;
    fprintf(stderr, "momoe-metal: DSpark drafter from %s: %.2f GiB, %d blocks of %d experts (%d used), drafts of up to %d "
            "tokens (confidence at least %.2f), its input from blocks", path, bytes / 1073741824.0, n_layer, D.n_expert,
            D.n_used, D.block, D.conf_min);
    for (int t : D.targets) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");
}

#if defined(MOMOE_CUDA)
static bool uma_small();
// lfm2moe on a small unified-memory machine (the Jetson) whose pool cannot hold every expert: the pool in mapped host
// memory and resident mailboxes, a token one graph with no stops (measured on the Orin Nano Super, same tokens: 8B-A1B
// UD-Q4_K_XL 17.1 -> 19.2 tok/s, 24B-A2B UD-IQ3_XXS 6.2 -> 6.8; the host pool with the stops: 16.3).  MOMOE_NO_RMB=1: off
static bool lfm_rmb(const Engine & e, int n_slots) {
    return e.m.lfm && e.uma && !g_direct && uma_small() && n_slots < e.m.n_exp_all && !getenv("MOMOE_NO_RMB");
}
#endif

// the pool's slots: each group gets `frac` of its experts (at least a few tokens' worth), every slot empty
// a buffer of card d, known to the readers (--layer-split: they copy to it on that card's stream)
static void card_buf(ggml_backend_buffer_t b, int d) {
    if (b) g_buf_dev.push_back({b, d});
}

static void free_card_buf(ggml_backend_buffer_t b) {
    for (size_t i = 0; i < g_buf_dev.size(); ++i)
        if (g_buf_dev[i].first == b) { g_buf_dev.erase(g_buf_dev.begin() + i); break; }
    ggml_backend_buffer_free(b);
}

static void make_pool(Engine & e, double frac) {
    Model & m = e.m;
    Pool & P = e.pool;
    if (P.buf) { free_card_buf(P.buf); P.buf = nullptr; }
    for (auto b : P.more) free_card_buf(b);
    P.more.clear();
    ggml_init_params ip = {ggml_tensor_overhead() * 64 * (P.g.size() + 1), nullptr, true};
    const int nc = n_cards(e);
    std::vector<ggml_context *> cps(nc);
    for (auto & c : cps) c = ggml_init(ip);
    ggml_context * cp = cps[0];
    P.n_slots = 0;
    P.pool_bytes = 0;
    const int min_tok = P.g.size() > 4 ? 4 : 16;             // a file with many formats (Bolt: 17): 4 tokens' worth
    for (auto & G : P.g) {
        const int n_all = G.n_all;
        const double f = (int) P.frac.size() > G.dev ? P.frac[G.dev] : frac;   // (--layer-split: its card's share)
        G.n_slots = std::min(std::max((int) (f * n_all), min_tok * m.n_used), n_all);
        for (int w = m.part0(); w < 3; ++w)
            G.t[w] = ggml_new_tensor_3d(cps[G.dev], G.type[w], G.ne[w][0], G.ne[w][1], G.n_slots);
        if (G.t[0]) ggml_set_name(G.t[0], "pool_gate");
        ggml_set_name(G.t[1], "pool_up");
        ggml_set_name(G.t[2], "pool_down");
        G.owner.assign(G.n_slots, -1);
        G.used.assign(G.n_slots, 0);
        G.left.reset(new std::atomic<int>[G.n_slots]);
        for (int i = 0; i < G.n_slots; ++i) G.left[i] = 0;
        G.by_prefetch.assign(G.n_slots, 0);
        P.n_slots += G.n_slots;
        P.pool_bytes += G.bytes * G.n_slots;
    }
    P.host = P.dev = nullptr;
#if defined(MOMOE_CUDA)
    if (e.uma && (getenv("MOMOE_POOL_HOST") || lfm_rmb(e, P.n_slots))) {   // (no --direct-io: from the file cache)
        // unified memory: the pool in pinned host memory the GPU maps.  The readers write the slots with the CPU (no
        // copy through the GPU, no stream to wait for); a copy through the GPU could not finish while a resident
        // mailbox's wait held the GPU (measured: the readers' copies stalled behind it)
        size_t total = 0;
        for (auto & G : P.g)
            for (int w = m.part0(); w < 3; ++w) total += (ggml_nbytes(G.t[w]) + 4095) / 4096 * 4096;
        void * h = nullptr;
        P.buf = ggml_backend_cuda_buffer_from_host(e.device, total + 4096, &h);
        if (!P.buf) die("cannot allocate the expert pool (mapped host memory)");
        ggml_tallocr ta = ggml_tallocr_new(P.buf);
        for (auto & G : P.g)
            for (int w = m.part0(); w < 3; ++w) ggml_tallocr_alloc(&ta, G.t[w]);
        P.host = (char *) h;
        P.dev = (char *) ggml_backend_buffer_get_base(P.buf);
        for (auto & G : P.g)                                 // zeros past the last slot's last row (as the staging copy)
            for (int w = m.part0(); w < 3; ++w) {
                const size_t n = ggml_nbytes(G.t[w]), a = ggml_backend_buffer_get_alloc_size(P.buf, G.t[w]);
                if (a > n) memset(P.host + ((char *) G.t[w]->data - P.dev) + n, 0, a - n);
            }
    } else
#endif
    if (nc == 1) {
        P.buf = ggml_backend_alloc_ctx_tensors(cp, e.metal);
        if (!P.buf) die("cannot allocate the expert pool");
    } else {                                                 // --layer-split: each card's groups in its own buffer
        for (int d = 0; d < nc; ++d) {
            if (!ggml_get_first_tensor(cps[d])) continue;    // (a card whose layers have no experts)
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(cps[d], card(e, d));
            if (!b) die("cannot allocate the expert pool on card %s", std::to_string(d).c_str());
            card_buf(b, d);
            if (!P.buf) P.buf = b;
            else P.more.push_back(b);
        }
    }
    if (e.hybrid || e.helper) {                              // a slot of zeros per group: a missing expert's slot (the
        for (auto & G : P.g) {                               // GPU adds nothing for it; the CPU or the helper computes it)
            if (G.n_slots < 2) die("the expert pool is too small for --hybrid");
            G.zero = G.n_slots - 1;
            G.used[G.zero] = UINT64_MAX;                     // never taken
            for (int w = m.part0(); w < 3; ++w) {
                std::vector<char> z(G.t[w]->nb[2], 0);
                tset(G.t[w], z.data(), (size_t) G.zero * G.t[w]->nb[2], z.size());
            }
        }
    }
    const int all = m.n_layer * m.n_expert;                  // (layer, expert) keys; nemo: some unused
    P.slot_of.assign(all, -1);
    P.freq.assign(all, 0);
    fprintf(stderr, "momoe-metal: expert pool %d slots = %.2f GiB of %.2f GiB (%d experts in the model, %.0f%%), %zu "
            "format group%s\n", P.n_slots, P.pool_bytes / 1073741824.0, P.all_bytes / 1073741824.0, m.n_exp_all,
            100.0 * P.pool_bytes / P.all_bytes, P.g.size(), P.g.size() == 1 ? "" : "s");
}

// --layer-split: which card each layer runs on.  "auto": in proportion to each card's free VRAM (every layer counted
// alike); else the first layer of each later card ("18", or "16,32" for three cards), as the PC engine reads it
static void assign_cards(Engine & e) {
#if defined(MOMOE_CUDA)
    Model & m = e.m;
    if (e.split.empty()) return;
    int nc = 0;
    if (cudaGetDeviceCount(&nc) != cudaSuccess || nc < 2) {
        fprintf(stderr, "momoe-cuda: --layer-split %s with %d card(s) visible: one card\n", e.split.c_str(), nc);
        e.split.clear();
        return;
    }
    nc = std::min(nc, MAX_CARDS);
    if (e.device != 0) die("--layer-split runs on the visible cards in order: no --device");
    std::vector<int> first(nc, 0);                           // first[d]: card d's first layer
    if (e.split == "auto") {
        std::vector<double> fr(nc);
        double all = 0;
        for (int d = 0; d < nc; ++d) {
            size_t f = 0, t = 0;
            cudaSetDevice(d);
            cudaMemGetInfo(&f, &t);
            fr[d] = (double) f;
            all += fr[d];
        }
        double acc = 0;
        for (int d = 1; d < nc; ++d) {
            acc += fr[d - 1];
            first[d] = std::min(m.n_layer - (nc - d), std::max(first[d - 1] + 1, (int) std::lround(m.n_layer * acc / all)));
        }
    } else {
        std::string v = e.split;
        int d = 1;
        for (size_t at = 0; at <= v.size() && d < nc; ++d) {
            const size_t comma = v.find(',', at);
            const int k = atoi(v.substr(at, comma == std::string::npos ? std::string::npos : comma - at).c_str());
            if (k <= first[d - 1] || k >= m.n_layer) die("--layer-split %s: each card's first layer must rise, below the "
                                                          "layer count", v.c_str());
            first[d] = k;
            if (comma == std::string::npos) { ++d; break; }
            at = comma + 1;
        }
        nc = d;                                              // (fewer numbers than cards: the cards they name)
    }
    for (int il = 0; il < m.n_layer; ++il) {
        int d = 0;
        while (d + 1 < nc && il >= first[d + 1]) ++d;
        m.layers[il].dev = d;
    }
    e.gpus.assign(nc, nullptr);                              // (gpus[0]: the GPU backend, set at its start)
    std::string where;
    for (int d = 0; d < nc; ++d) {
        if (d > 0) {
            e.gpus[d] = ggml_backend_cuda_init(d);
            if (!e.gpus[d]) die("--layer-split: no CUDA device %s", std::to_string(d).c_str());
        }
        char desc[256];
        ggml_backend_cuda_get_device_description(d, desc, sizeof desc);
        const int last = d + 1 < nc ? first[d + 1] - 1 : m.n_layer - 1;
        where += (d ? "; " : "") + std::string("layers ") + std::to_string(first[d]) + "-" + std::to_string(last) + " on " + desc;
    }
    fprintf(stderr, "momoe-cuda: layer split (%s): %s\n", e.split.c_str(), where.c_str());
#else
    (void) e;
#endif
}

static void load(Engine & e, const char * path, double pool_gib, int slots_arg) {
    Model & m = e.m;
    ggml_context * meta = nullptr;
    gguf_init_params gp = {/*no_alloc*/ true, &meta};
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) die("cannot read %s as GGUF", path);
    std::string arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
    if (arch != "qwen35moe" && arch != "qwen4exp" && arch != "nemotron_h_moe" && arch != "glm5-next" && arch != "lfm2moe"
        && arch != "flint" && arch != "gemma4" && arch != "bailingmoe3" && arch != "deepseek41")
        die("%s: only qwen35moe, qwen4exp, nemotron_h_moe, glm5-next, lfm2moe, flint, gemma4, bailingmoe3 and deepseek41 "
            "models", arch.c_str());
    m.qwen4 = arch == "qwen4exp";
    m.nemo = arch == "nemotron_h_moe";
    m.glm5 = arch == "glm5-next";
    m.flint = arch == "flint";
    m.lfm = arch == "lfm2moe" || m.flint;
    m.gemma4 = arch == "gemma4";
    m.ling = arch == "bailingmoe3";
    m.ds41 = arch == "deepseek41";
    {                                                        // the control tokens the serve loop needs, by name
        const int64_t ti = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (ti < 0) die("the file has no vocabulary");
        std::vector<int32_t> glm_ends;
        int32_t glm_asst = -1;
        const size_t nt = gguf_get_arr_n(g, ti);
        for (size_t i = 0; i < nt; ++i) {
            const char * s = gguf_get_arr_str(g, ti, i);
            if (!strcmp(s, "<|im_start|>")) m.im_start = (int32_t) i;
            else if (!strcmp(s, "<|image_pad|>")) m.img_pad = (int32_t) i;
            else if (!strcmp(s, "<image>") && m.nemo) m.img_pad = (int32_t) i;   // (nemo: <img> <image>.. </img>)
            else if (!strcmp(s, "<|image|>") && (m.gemma4 || m.glm5)) m.img_pad = (int32_t) i;   // (gemma4: <|image>
                                                             // <|image|>.. <image|>; glm5: <|begin_of_image|> ..)
            else if (!strcmp(s, "<|im_end|>") || !strcmp(s, "<|endoftext|>")) m.eos_ids.push_back((int32_t) i);
            // GLM-5: a reply ends where the next turn would start (its eos_token_id list); the checkpoint goes in
            // front of the reply's own <|assistant|> as before <|im_start|>
            else if (!strcmp(s, "<|user|>") || !strcmp(s, "<|observation|>")) glm_ends.push_back((int32_t) i);
            else if (!strcmp(s, "<|assistant|>")) glm_asst = (int32_t) i;
            else if (!strcmp(s, "<\uff5cAssistant\uff5c>") && m.ds41) m.im_start = (int32_t) i;   // (deepseek41: the
                                                             // checkpoint in front of the reply's own <|Assistant|>)
            else if (!strcmp(s, "<role>") && m.ling) m.im_start = (int32_t) i;   // (ling: <role>ASSISTANT</role>; a
                                                             // reply ends at <|role_end|>, the file's eos)
            else if (!strcmp(s, "<|turn>") && m.gemma4) m.im_start = (int32_t) i;   // (gemma4: a turn starts there;
            else if ((!strcmp(s, "<eos>") || !strcmp(s, "<|tool_response>")) && m.gemma4)   // it ends at <turn|> (the file's
                m.eos_ids.push_back((int32_t) i);            // eos), <eos> or, after a tool call, <|tool_response>)
        }
        if (m.glm5 || m.flint) {                             // (flint: llama.cpp ends its replies there too)
            m.eos_ids.insert(m.eos_ids.end(), glm_ends.begin(), glm_ends.end());
            if (m.im_start < 0) m.im_start = glm_asst;
        }
        const int64_t ei = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
        if (ei >= 0) {
            const int32_t eos = (int32_t) gguf_get_val_u32(g, ei);
            if (std::find(m.eos_ids.begin(), m.eos_ids.end(), eos) == m.eos_ids.end()) m.eos_ids.push_back(eos);
        }
    }
    auto k = [&](const char * s) { return arch + "." + std::string(s); };
    auto has = [&](const char * s) { return gguf_find_key(g, k(s).c_str()) >= 0; };
    m.n_embd = gguf_int(g, k("embedding_length"));
    int nextn = has("nextn_predict_layers") ? gguf_int(g, k("nextn_predict_layers")) : 0;
    m.n_layer = gguf_int(g, k("block_count")) - nextn;     // the MTP layer is not run yet
    m.n_head = gguf_int(g, k("attention.head_count"));
    if (m.ds41) {                                            // llama.cpp's deepseek4.cpp / deepseek41.cpp load_arch_hparams,
        m.layers.resize(m.n_layer);                          // with the IST-DAS sets' key names (llama.cpp's beside them)
        m.n_head_kv = 1;
        m.ds_hd = gguf_int(g, k("attention.key_length"));
        m.head_dim = m.ds_hd;
        m.n_rot = gguf_int(g, k("rope.dimension_count"));
        m.q_lora = gguf_int(g, k("attention.q_lora_rank"));
        m.n_swa = gguf_int(g, k("attention.sliding_window"));
        m.o_groups = gguf_int(g, k("attention.output_group_count"));
        m.o_lora = gguf_int(g, k("attention.output_lora_rank"));
        m.idx_heads = gguf_int(g, k("attention.indexer.head_count"));
        m.idx_dim = gguf_int(g, k("attention.indexer.key_length"));
        m.idx_top_k = gguf_int(g, k("attention.indexer.top_k"));
        m.n_expert = gguf_int(g, k("expert_count"));
        m.n_used = gguf_int(g, k("expert_used_count"));
        m.n_ff_exp = gguf_int(g, k("expert_feed_forward_length"));
        m.exp_scale = has("expert_weights_scale") ? gguf_float(g, k("expert_weights_scale")) : 1.0f;
        if (gguf_int(g, k("expert_gating_func")) != 4) die("deepseek41: only the sqrt(softplus) router");
        if (has("expert_weights_norm") && !gguf_get_val_bool(g, gguf_find_key(g, k("expert_weights_norm").c_str())))
            die("deepseek41: only normalized expert weights");
        if (has("expert_shared_count") && gguf_int(g, k("expert_shared_count")) != 1) die("deepseek41: one shared expert only");
        if (has("hash_layer_count") && gguf_int(g, k("hash_layer_count")) != 0) die("deepseek41: hash-routed blocks are not done");
        m.rms_eps = gguf_float(g, k("attention.layer_norm_rms_epsilon"));
        m.rope_base = gguf_float(g, k("rope.freq_base"));
        m.ds_comp_base = gguf_float(g, k("attention.compress_rope_freq_base"));
        m.ds_freq_scale = has("rope.scaling.factor") ? 1.0f / gguf_float(g, k("rope.scaling.factor")) : 1.0f;
        m.ds_ext = m.ds_freq_scale != 1.0f ? 1.0f : 0.0f;    // (YaRN: llama.cpp's ext_factor; its attention factor
        m.ds_attn_factor = m.ds_ext != 0.0f ? 1.0f / (1.0f + 0.1f * logf(1.0f / m.ds_freq_scale)) : 1.0f;   // cancels mscale)
        m.ds_ctx_orig = has("rope.scaling.original_context_length") ? gguf_int(g, k("rope.scaling.original_context_length")) : 0;
        m.ds_beta_fast = has("rope.scaling.yarn_beta_fast") ? gguf_float(g, k("rope.scaling.yarn_beta_fast")) : 32.0f;
        m.ds_beta_slow = has("rope.scaling.yarn_beta_slow") ? gguf_float(g, k("rope.scaling.yarn_beta_slow")) : 1.0f;
        m.hc = gguf_int(g, k("hyper_connection.count"));
        if (m.hc != 4) die("deepseek41: mHC with 4 streams only (as llama.cpp)");
        m.hc_iters = gguf_int(g, k("hyper_connection.sinkhorn_iterations"));
        m.hc_eps = gguf_float(g, k("hyper_connection.epsilon"));
        m.clamp_exp = gguf_per_block_f(g, k("swiglu_clamp_exp"), m.n_layer, 0.0f);
        m.clamp_sh = has("swiglu_clamp_shexp") ? gguf_per_block_f(g, k("swiglu_clamp_shexp"), m.n_layer, 0.0f) : m.clamp_exp;
        // each compressed block reads the KV / top-k of the latest source block at or before it (deepseek41.cpp)
        const std::vector<uint64_t> ratios = gguf_uarr(g, k("attention.compress_ratios"));
        if ((int) ratios.size() < m.n_layer) die("deepseek41: compress_ratios is shorter than the model");
        auto list = [&](const char * a, const char * b) {
            return gguf_uarr(g, k(has(a) ? a : b));
        };
        const std::vector<uint64_t> kv_src = list("kv_source_layer_ids", "attention.kv_source_layers");
        const std::vector<uint64_t> idx_src = list("index_source_layer_ids", "attention.indexer.source_layers");
        int last_kv = -1, last_idx = -1;
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            L.ratio = (int) ratios[il];
            L.swa = true;                                    // (every block: the window's ring)
            const bool is_kv = std::find(kv_src.begin(), kv_src.end(), (uint64_t) il) != kv_src.end();
            const bool is_idx = std::find(idx_src.begin(), idx_src.end(), (uint64_t) il) != idx_src.end();
            if ((is_kv || is_idx) && L.ratio == 0) die("deepseek41: a source block without compression");
            if (is_kv) last_kv = il;
            if (is_idx) last_idx = il;
            if (L.ratio == 0) continue;
            if (L.ratio > 2 || last_kv < 0 || last_idx < last_kv || m.layers[last_kv].ratio != L.ratio ||
                m.layers[last_idx].ratio != L.ratio) die("deepseek41: block %s's ratio has no matching kv / index source", std::to_string(il).c_str());
            L.kv_src = last_kv;
            L.topk_src = last_idx;
        }
        auto opt = [&](const char * a, const char * b, int d) {
            return has(a) ? gguf_int(g, k(a)) : has(b) ? gguf_int(g, k(b)) : d;
        };
        m.cand_src = opt("attention.candidate_source_layer_id", "attention.indexer.candidate_source_layer", -1);
        if (m.cand_src >= 0) {
            m.cand_bs = opt("attention.candidate_block_size", "attention.indexer.candidate_block_size", 0);
            m.cand_blocks = opt("attention.candidate_topk_blocks", "attention.indexer.candidate_top_k_blocks", 0);
            if (m.cand_src >= m.n_layer || m.layers[m.cand_src].topk_src != m.cand_src || m.cand_bs <= 0 || m.cand_blocks <= 0)
                die("deepseek41: invalid candidate selection");
        }
        if (has("engram.layer_ids")) {                       // (the hash constants: the first file's integer tensors)
            const std::vector<uint64_t> el = gguf_uarr(g, k("engram.layer_ids"));
            for (size_t j = 0; j < el.size(); ++j) {
                if ((int) el[j] >= m.n_layer || (j && el[j] <= el[j - 1])) die("deepseek41: the Engram blocks are out of order");
                m.layers[el[j]].eg = (int) j;
            }
            m.eg_ngram = gguf_int(g, k("engram_max_ngram_size"));
            m.eg_per = gguf_int(g, k("engram_n_heads"));
            m.eg_dim = gguf_int(g, k("engram_head_dim"));
            m.eg_pad = gguf_int(g, k("engram_pad_token_id"));
            m.eg_heads = (m.eg_ngram - 1) * m.eg_per;
            if (e.engram.empty() && has("engram.file")) {    // the IST-DAS sets name it: beside the model, above it or in
                const std::string f = gguf_get_val_str(g, gguf_find_key(g, k("engram.file").c_str()));   // setup's shared
                std::string dir = path;                      // folder (models/deepseek-v4.1-flash-shared)
                dir = dir.find('/') == std::string::npos ? "." : dir.substr(0, dir.rfind('/'));
                for (const std::string & d : {dir, dir + "/..", dir + "/../deepseek-v4.1-flash-shared"}) {
                    struct stat st;
                    if (stat((d + "/" + f).c_str(), &st) == 0) { e.engram = d + "/" + f; break; }
                }
                if (e.engram.empty()) die("deepseek41: the Engram file %s is not beside the model (--engram FILE)", f.c_str());
            }
            if (e.engram.empty()) die("deepseek41: this file names no Engram file: --engram FILE");
            m.eg_fd.assign(el.size(), -1);
            m.eg_off.assign(el.size(), 0);
            m.eg_row.assign(el.size(), 0);
            m.eg_rows.assign(el.size(), 0);
        } else if (has("ple.layers")) die("deepseek41: llama.cpp's Engram keys (ple.*) are not read yet: use the IST-DAS files");
        m.full_attn_interval = 0;
        for (int j = 0; j < 4; ++j) m.sections[j] = 0;
    } else if (m.glm5) {                                     // llama.cpp's glm5-next.cpp load_arch_hparams
        const std::vector<uint64_t> kv = gguf_uarr(g, k("attention.head_count_kv"));
        if ((int) kv.size() < m.n_layer) die("the per-block arrays are shorter than the model");
        m.layers.resize(m.n_layer);
        for (int il = 0; il < m.n_layer; ++il) {
            m.layers[il].kind = kv[il] > 0 ? 'a' : 'k';
            m.layers[il].recurrent = kv[il] == 0;
        }
        m.n_head_kv = 1;
        m.head_dim = 0;
        m.n_rot = 0;
        m.n_dense = has("leading_dense_block_count") ? gguf_int(g, k("leading_dense_block_count")) : 0;
        m.n_expert = gguf_int(g, k("expert_count"));
        m.n_used = gguf_int(g, k("expert_used_count"));
        m.n_ff_exp = gguf_int(g, k("expert_feed_forward_length"));
        m.exp_scale = has("expert_weights_scale") ? gguf_float(g, k("expert_weights_scale")) : 1.0f;
        if (has("expert_gating_func") && gguf_int(g, k("expert_gating_func")) != 2) die("glm5-next: only the sigmoid router");
        if (has("expert_weights_norm") && !gguf_get_val_bool(g, gguf_find_key(g, k("expert_weights_norm").c_str())))
            die("glm5-next: only normalized expert weights");
        m.rms_eps = gguf_float(g, k("attention.layer_norm_rms_epsilon"));
        if (has("attention.layer_norm_epsilon")) m.norm_eps = gguf_float(g, k("attention.layer_norm_epsilon"));
        m.ssm_conv = gguf_int(g, k("ssm.conv_kernel"));
        m.kda_hd = gguf_int(g, k("kda.head_dim"));
        if (has("kda.gate_lower_bound")) m.kda_lb = gguf_float(g, k("kda.gate_lower_bound"));
        m.q_lora = gguf_int(g, k("attention.q_lora_rank"));
        m.kv_lora = gguf_int(g, k("attention.kv_lora_rank"));
        m.mla_k = gguf_int(g, k("attention.key_length_mla"));
        m.mla_v = gguf_int(g, k("attention.value_length_mla"));
        if (gguf_int(g, k("rope.dimension_count")) != 0) die("glm5-next: only the NoPE MLA");
        m.idx_heads = gguf_int(g, k("attention.indexer.head_count"));
        m.idx_dim = gguf_int(g, k("attention.indexer.key_length"));
        m.idx_top_k = gguf_int(g, k("attention.indexer.top_k"));
        m.kpool = gguf_int(g, k("attention.indexer.kpool"));
        if (m.kpool < 2 || m.idx_top_k % m.kpool) die("the k-pool indexer needs a pool above 1 that divides top_k");
        // (the unfinished pool is selected unless the file says otherwise: llama.cpp's default, the reference's)
        m.kpool_tail = true;
        if (has("attention.indexer.kpool_select_tail"))
            m.kpool_tail = gguf_get_val_bool(g, gguf_find_key(g, k("attention.indexer.kpool_select_tail").c_str()));
        m.hc = gguf_int(g, k("hyper_connection.count"));
        if (m.hc != 4) die("glm5-next: mHC with 4 streams only (as llama.cpp)");
        m.hc_iters = gguf_int(g, k("hyper_connection.sinkhorn_iterations"));
        m.hc_eps = gguf_float(g, k("hyper_connection.epsilon"));
        m.clamp_exp = gguf_per_block_f(g, k("swiglu_clamp_exp"), m.n_layer + nextn, 0.0f);   // (+ the MTP block)
        m.clamp_sh = has("swiglu_clamp_shexp") ? gguf_per_block_f(g, k("swiglu_clamp_shexp"), m.n_layer + nextn, 0.0f)
                                               : m.clamp_exp;
        m.ssm_inner = m.kda_hd * m.n_head;
        m.full_attn_interval = 0;
        m.rope_base = 0;
        for (int j = 0; j < 4; ++j) m.sections[j] = 0;
    } else if (m.ling) {                                     // llama.cpp's bailingmoe3.cpp load_arch_hparams
        const std::vector<uint64_t> kv = gguf_uarr(g, k("attention.head_count_kv"));
        if ((int) kv.size() < m.n_layer) die("the per-block arrays are shorter than the model");
        m.layers.resize(m.n_layer);
        for (int il = 0; il < m.n_layer; ++il) {
            m.layers[il].kind = kv[il] > 0 ? 'a' : 'k';
            m.layers[il].recurrent = kv[il] == 0;
        }
        m.n_head_kv = 1;
        m.head_dim = 0;
        m.n_dense = has("leading_dense_block_count") ? gguf_int(g, k("leading_dense_block_count")) : 0;
        m.n_expert = gguf_int(g, k("expert_count"));
        m.n_used = gguf_int(g, k("expert_used_count"));
        m.n_ff_exp = gguf_int(g, k("expert_feed_forward_length"));
        m.exp_scale = has("expert_weights_scale") ? gguf_float(g, k("expert_weights_scale")) : 1.0f;
        if (has("expert_gating_func") && gguf_int(g, k("expert_gating_func")) != 2) die("bailingmoe3: only the sigmoid router");
        if (has("expert_weights_norm") && !gguf_get_val_bool(g, gguf_find_key(g, k("expert_weights_norm").c_str())))
            die("bailingmoe3: only normalized expert weights");
        m.n_groups = has("expert_group_count") ? gguf_int(g, k("expert_group_count")) : 0;
        m.n_group_used = has("expert_group_used_count") ? gguf_int(g, k("expert_group_used_count")) : 0;
        if (m.n_groups > 1 && (m.n_expert % m.n_groups || m.n_group_used < 1 || m.n_group_used > m.n_groups))
            die("bailingmoe3: the expert groups do not divide the experts");
        if (has("expert_shared_count") && gguf_int(g, k("expert_shared_count")) != 1) die("bailingmoe3: one shared expert only");
        m.rms_eps = gguf_float(g, k("attention.layer_norm_rms_epsilon"));
        m.ssm_conv = gguf_int(g, k("ssm.conv_kernel"));
        m.kda_hd = gguf_int(g, k("kda.head_dim"));
        m.kda_lb = gguf_float(g, k("kda.gate_lower_bound"));
        if (has("kda.safe_gate") && !gguf_get_val_bool(g, gguf_find_key(g, k("kda.safe_gate").c_str())))
            die("bailingmoe3: only the safe KDA gate");
        m.q_lora = has("attention.q_lora_rank") ? gguf_int(g, k("attention.q_lora_rank")) : 0;
        if (m.q_lora) die("bailingmoe3: only the query without a LoRA (as Ling-3.0-flash)");
        m.kv_lora = gguf_int(g, k("attention.kv_lora_rank"));
        m.mla_k = gguf_int(g, k("attention.key_length_mla"));
        m.mla_v = gguf_int(g, k("attention.value_length_mla"));
        m.n_rot = gguf_int(g, k("rope.dimension_count"));
        m.rope_base = gguf_float(g, k("rope.freq_base"));
        if (has("rope.dimension_sections")) die("bailingmoe3: the VL file's M-RoPE is not done yet");
        m.clamp_exp = gguf_per_block_f(g, k("swiglu_clamp_exp"), m.n_layer + nextn, 0.0f);   // (+ the MTP block)
        m.clamp_sh = has("swiglu_clamp_shexp") ? gguf_per_block_f(g, k("swiglu_clamp_shexp"), m.n_layer + nextn, 0.0f)
                                               : m.clamp_exp;
        m.ssm_inner = m.kda_hd * m.n_head;
        m.full_attn_interval = 0;
        for (int j = 0; j < 4; ++j) m.sections[j] = 0;
    } else if (m.lfm) {                                      // llama.cpp's lfm2moe.cpp load_arch_hparams
        const std::vector<uint64_t> kv = gguf_per_block(g, k("attention.head_count_kv"), m.n_layer);
        if ((int) kv.size() < m.n_layer) die("the per-block arrays are shorter than the model");
        m.layers.resize(m.n_layer);
        m.n_head_kv = 0;
        std::vector<bool> swa, rope;
        if (m.flint) {                                       // llama.cpp's flint.cpp load_arch_hparams
            swa = gguf_barr(g, k("attention.sliding_window_pattern"));
            rope = has("attention.rope_pattern") ? gguf_barr(g, k("attention.rope_pattern")) : swa;
            if ((int) swa.size() < m.n_layer || (int) rope.size() < m.n_layer) die("the per-block arrays are shorter than the model");
            m.n_swa = gguf_int(g, k("attention.sliding_window"));
            if (m.n_swa < 1) die("flint: a sliding window of %s tokens", std::to_string(m.n_swa).c_str());
        }
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            L.kind = kv[il] > 0 ? 'a' : 'c';
            L.recurrent = kv[il] == 0;
            if (m.flint) {
                if (L.kind != 'a') die("flint: a block without attention");
                L.swa = swa[il];
                L.rope = rope[il];
            }
            if (L.kind == 'a') {
                if (m.n_head_kv && m.n_head_kv != (int) kv[il]) die("attention blocks with different KV head counts");
                m.n_head_kv = (int) kv[il];
            }
        }
        m.head_dim = has("attention.key_length") ? gguf_int(g, k("attention.key_length")) : m.n_embd / m.n_head;
        m.n_rot = m.head_dim;
        m.n_dense = has("leading_dense_block_count") ? gguf_int(g, k("leading_dense_block_count")) : 0;
        m.n_expert = gguf_int(g, k("expert_count"));
        m.n_used = gguf_int(g, k("expert_used_count"));
        m.n_ff_exp = gguf_int(g, k("expert_feed_forward_length"));
        if (gguf_int(g, k("expert_gating_func")) != 2) die("%s: only the sigmoid router", arch.c_str());
        // (llama.cpp scales the normalized weights only by a scale other than 0 and 1)
        m.exp_scale = has("expert_weights_scale") ? gguf_float(g, k("expert_weights_scale")) : 1.0f;
        if (m.exp_scale == 0.0f) m.exp_scale = 1.0f;
        if (!m.flint) {
            m.ssm_conv = gguf_int(g, k("shortconv.l_cache"));   // the convolution's width; its history: l_cache - 1
            if (m.ssm_conv < 2) die("lfm2moe: a short convolution of width 1");
        }
        if (m.flint && has("expert_weights_norm") && !gguf_get_val_bool(g, gguf_find_key(g, k("expert_weights_norm").c_str())))
            die("flint: only normalized expert weights");
        m.rms_eps = gguf_float(g, k("attention.layer_norm_rms_epsilon"));
        m.rope_base = gguf_float(g, k("rope.freq_base"));
        m.full_attn_interval = 0;
        for (int j = 0; j < 4; ++j) m.sections[j] = 0;
    } else if (m.gemma4) {                                   // llama.cpp's gemma4.cpp load_arch_hparams
        const std::vector<uint64_t> kv = gguf_per_block(g, k("attention.head_count_kv"), m.n_layer);
        const std::vector<bool> swa = gguf_barr(g, k("attention.sliding_window_pattern"));
        if ((int) kv.size() < m.n_layer || (int) swa.size() < m.n_layer) die("the per-block arrays are shorter than the model");
        if (has("attention.shared_kv_layers") && gguf_int(g, k("attention.shared_kv_layers")) != 0)
            die("gemma4: blocks that share another block's KV cache (E2B, E4B) are not supported");
        if (has("embedding_length_per_layer_input") && gguf_int(g, k("embedding_length_per_layer_input")) != 0)
            die("gemma4: per-layer embeddings (E2B, E4B) are not supported");
        const int hd_full = gguf_int(g, k("attention.key_length")), hd_swa = gguf_int(g, k("attention.key_length_swa"));
        if (hd_full != gguf_int(g, k("attention.value_length")) || hd_swa != gguf_int(g, k("attention.value_length_swa")))
            die("gemma4: keys and values of different sizes");
        m.layers.resize(m.n_layer);
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            L.kind = 'a';
            L.swa = swa[il];
            L.hd = L.swa ? hd_swa : hd_full;
            L.n_kv = (int) kv[il];
        }
        m.n_head_kv = (int) kv[0];
        m.head_dim = hd_full;
        m.n_rot = hd_full;
        m.n_swa = gguf_int(g, k("attention.sliding_window"));
        if (m.n_swa < 1) die("gemma4: a sliding window of %s tokens", std::to_string(m.n_swa).c_str());
        m.n_expert = gguf_int(g, k("expert_count"));
        m.n_used = gguf_int(g, k("expert_used_count"));
        m.n_ff_exp = gguf_int(g, k("expert_feed_forward_length"));
        m.rms_eps = gguf_float(g, k("attention.layer_norm_rms_epsilon"));
        m.rope_base = gguf_float(g, k("rope.freq_base"));
        m.rope_base_swa = has("rope.freq_base_swa") ? gguf_float(g, k("rope.freq_base_swa")) : m.rope_base;
        m.softcap = has("final_logit_softcapping") ? gguf_float(g, k("final_logit_softcapping")) : 0.0f;
        m.full_attn_interval = 0;
        for (int j = 0; j < 4; ++j) m.sections[j] = 0;
    } else if (m.nemo) {                                            // per-block arrays: which block is which, expert sizes
        const std::vector<uint64_t> kv = gguf_uarr(g, k("attention.head_count_kv"));
        // (an MoE block is one with a feed-forward size: Super gives the expert sizes as one value, these per block)
        const std::vector<uint64_t> ff = gguf_per_block(g, k("feed_forward_length"), m.n_layer);
        const std::vector<uint64_t> used = gguf_per_block(g, k("expert_used_count"), m.n_layer);
        if ((int) kv.size() < m.n_layer || (int) ff.size() < m.n_layer || (int) used.size() < m.n_layer)
            die("the per-block arrays are shorter than the model");
        m.layers.resize(m.n_layer);
        m.n_head_kv = 0;
        m.n_used = 0;
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            L.kind = kv[il] > 0 ? 'a' : ff[il] > 0 ? 'e' : 'm';
            if (L.kind == 'a') {
                if (m.n_head_kv && m.n_head_kv != (int) kv[il]) die("attention blocks with different KV head counts");
                m.n_head_kv = (int) kv[il];
            }
            if (L.kind == 'e') {
                if (m.n_used && m.n_used != (int) used[il]) die("MoE blocks with different expert counts per token");
                m.n_used = (int) used[il];
            }
        }
        for (int il = m.n_layer - 1, nx = -1; il >= 0; --il) {
            m.layers[il].next_moe = nx;
            if (m.layers[il].kind == 'e') nx = il;
        }
        m.head_dim = gguf_int(g, k("attention.key_length"));
        m.n_rot = 0;
        m.n_expert = gguf_int(g, k("expert_count"));       // the most any block has
        m.n_ff_exp = 0;
        // the experts' width: the latent (Bolt, Super), else the hidden state itself (Lightning 30B-A3B: no latent
        // projections)
        m.latent = has("moe_latent_size") ? gguf_int(g, k("moe_latent_size")) : m.n_embd;
        m.exp_scale = has("expert_weights_scale") ? gguf_float(g, k("expert_weights_scale")) : 1.0f;
        m.ssm_conv = gguf_int(g, k("ssm.conv_kernel"));
        m.ssm_state = gguf_int(g, k("ssm.state_size"));
        m.ssm_groups = gguf_int(g, k("ssm.group_count"));
        m.ssm_dt_rank = gguf_int(g, k("ssm.time_step_rank"));
        m.ssm_inner = gguf_int(g, k("ssm.inner_size"));
        m.full_attn_interval = 0;
        m.rms_eps = gguf_float(g, k(has("attention.layer_norm_rms_epsilon") ? "attention.layer_norm_rms_epsilon"
                                                                            : "attention.layer_norm_epsilon"));   // (Super: as llama.cpp)
        m.rope_base = 0;
        for (int j = 0; j < 4; ++j) m.sections[j] = 0;
    } else {
    m.n_head_kv = gguf_int(g, k("attention.head_count_kv"));
    m.head_dim = gguf_int(g, k("attention.key_length"));
    }   // (the indented block below is qwen35moe / qwen4exp only)
    if (!m.nemo && !m.glm5 && !m.lfm && !m.gemma4 && !m.ling && !m.ds41) {
    m.n_rot = gguf_int(g, k("rope.dimension_count"));
    m.n_expert = gguf_int(g, k("expert_count"));
    m.n_used = gguf_int(g, k("expert_used_count"));
    m.n_ff_exp = gguf_int(g, k("expert_feed_forward_length"));
    m.ssm_conv = gguf_int(g, k("ssm.conv_kernel"));
    m.ssm_state = gguf_int(g, k("ssm.state_size"));
    m.ssm_groups = gguf_int(g, k("ssm.group_count"));
    m.ssm_dt_rank = gguf_int(g, k("ssm.time_step_rank"));
    m.ssm_inner = gguf_int(g, k("ssm.inner_size"));
    m.full_attn_interval = has("full_attention_interval") ? gguf_int(g, k("full_attention_interval")) : 4;
    m.rms_eps = gguf_float(g, k("attention.layer_norm_rms_epsilon"));
    m.rope_base = gguf_float(g, k("rope.freq_base"));
    {
        int64_t i = gguf_find_key(g, k("rope.dimension_sections").c_str());
        const int32_t * s = (const int32_t *) gguf_get_arr_data(g, i);
        for (int j = 0; j < 4; ++j) m.sections[j] = j < (int) gguf_get_arr_n(g, i) ? s[j] : 0;
    }
    if (m.qwen4) {
        m.hc = gguf_int(g, k("hyper_connection.count"));
        m.hc_lr = gguf_int(g, k("hyper_connection.low_rank"));
        if (has("attention.compress_ratios")) {
            for (uint64_t r : gguf_uarr(g, k("attention.compress_ratios"))) {
                if (r == 0) continue;
                if (m.kpool && (int) r != m.kpool) die("QSA layers with different compress ratios");
                m.kpool = (int) r;
            }
        }
        if (m.kpool) {
            m.idx_heads = gguf_int(g, k("attention.indexer.head_count"));
            m.idx_dim = gguf_int(g, k("attention.indexer.key_length"));
            m.idx_top_k = gguf_int(g, k("attention.indexer.top_k"));
            if (m.kpool < 2 || m.idx_top_k % m.kpool) die("QSA needs a compress ratio above 1 that divides top_k");
        }
        if (has("ple.layers")) {
            std::vector<uint64_t> pl = gguf_uarr(g, k("ple.layers"));
            if (pl.size() != 1) die("only one PLE layer is supported");
            m.ple_layer = (int) pl[0];
            m.ple_ngram = gguf_int(g, k("ple.ngram_size"));
            m.ple_per_gram = gguf_int(g, k("ple.heads_per_ngram"));
            m.ple_kernel = gguf_int(g, k("ple.conv_kernel"));
            m.ple_eos = gguf_int(g, k("ple.eos_token_id"));
            m.ple_dim = gguf_int(g, k("embedding_length_per_layer_input"));
            m.ple_heads = (m.ple_ngram - 1) * m.ple_per_gram;
            m.ple_mult = gguf_uarr(g, k("ple.layer_multipliers"));
            m.ple_off = gguf_uarr(g, k("ple.head_offsets"));
            m.ple_vocab = gguf_uarr(g, k("ple.head_vocab_sizes"));
            if ((int) m.ple_mult.size() < m.ple_ngram || (int) m.ple_off.size() < m.ple_heads ||
                (int) m.ple_vocab.size() < m.ple_heads) die("the PLE arrays are too short");
        }
    }
    m.layers.resize(m.n_layer);
    for (int il = 0; il < m.n_layer; ++il) m.layers[il].recurrent = (il + 1) % m.full_attn_interval != 0;
    }

    assign_cards(e);
    // the dense weights: every tensor but the routed experts, the MTP layer and the PLE table, in one Metal buffer
    const int n_split = gguf_find_key(g, "split.count") >= 0 ? gguf_int(g, "split.count") : 1;
    const std::vector<std::string> paths = shard_paths(path, n_split);
    ggml_init_params ip = {ggml_tensor_overhead() * 4096, nullptr, true};
    ggml_context * cw = ggml_init(ip);
    // --layer-split: a context per card - a layer's weights on its card, the token embedding on the first, the head,
    // the output norm and the MTP layer on the last
    const int nc = n_cards(e);
    std::vector<ggml_context *> cws(nc, cw);
    for (int d = 1; d < nc; ++d) cws[d] = ggml_init(ip);
    auto cw_of = [&](int il, const char * name) -> ggml_context * {
        if (nc == 1) return cw;
        if (il >= 0 && il < m.n_layer) return cws[m.layers[il].dev];
        if (il < 0 && !strcmp(name, "token_embd.weight")) return cws[0];
        return cws[nc - 1];
    };
    // momoe-cuda: the token embeddings stay in RAM (a few rows a token, read on the CPU), as llama.cpp keeps them:
    // their VRAM goes to the expert pool. Only when the head has its own weights (output.weight in this file)
    ggml_context * ccpu = ggml_init(ip);
    const bool embd_cpu = !kHostMem && gguf_find_tensor(g, "output.weight") >= 0 && !getenv("MOMOE_EMBD_GPU");
    struct ToRead { ggml_tensor * t; int fd; size_t off; bool bf16 = false; };
    // momoe-metal on a GPU without bfloat kernels (macOS before 14: an M1 on Big Sur): BF16 weights (flint's routers)
    // loaded as F32 (exact: BF16 is F32's top half); its scheduler would want the CPU for them, from a Metal buffer
    bool bf16_f32 = false;
    if (kHostMem) {
        ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        ggml_init_params tp = {ggml_tensor_overhead() * 4, nullptr, true};
        ggml_context * tc = ggml_init(tp);
        ggml_tensor * op = ggml_mul_mat(tc, ggml_new_tensor_2d(tc, GGML_TYPE_BF16, 64, 64),
                                        ggml_new_tensor_2d(tc, GGML_TYPE_F32, 64, 1));
        bf16_f32 = gpu && !ggml_backend_dev_supports_op(gpu, op);
        ggml_free(tc);
    }
    std::vector<ToRead> to_read;
    for (size_t sh = 0; sh < paths.size(); ++sh) {
        if (sh > 0) {
            gguf_free(g);
            meta = nullptr;
            g = gguf_init_from_file(paths[sh].c_str(), gp);
            if (!g) die("cannot read %s as GGUF", paths[sh].c_str());
        }
        const int fd = open(paths[sh].c_str(), O_RDONLY);
        if (fd < 0) die("cannot open %s", paths[sh].c_str());
        e.fds.push_back(fd);
        if (g_direct) {
#if defined(O_DIRECT)
            const int dfd = open(paths[sh].c_str(), O_RDONLY | O_DIRECT);
#else
            const int dfd = -1;
            errno = ENOTSUP;
#endif
            if (dfd < 0) die("cannot open %s for direct reads", (paths[sh] + ": " + strerror(errno)).c_str());
            if ((int) g_direct_fd.size() <= fd) g_direct_fd.resize(fd + 1, -1);
            g_direct_fd[fd] = dfd;
        }
        const char * fmap = nullptr;                         // momoe-cuda: experts are copied from the mapping
        if (!kHostMem) {
            struct stat st;
            if (fstat(fd, &st) != 0) die("cannot stat %s", paths[sh].c_str());
            void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (p == MAP_FAILED) die("cannot map %s", paths[sh].c_str());
            fmap = (const char *) p;
        }
        const size_t data_off = gguf_get_data_offset(g);
        const int n_t = (int) gguf_get_n_tensors(g);
        for (int i = 0; i < n_t; ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * src = ggml_get_tensor(meta, name);
            int il = -1;
            char rest[128] = {0};
            if (sscanf(name, "blk.%d.%127s", &il, rest) == 2 && il > m.n_layer) continue;
            if (il == m.n_layer && (!e.load_mtp || m.nemo)) continue;   // the MTP layer, loaded only for drafting
            const size_t off = data_off + gguf_get_tensor_offset(g, i);
            std::string r = rest;
            if (!strcmp(name, "per_layer_token_embd.weight")) {   // the PLE table: rows read on the CPU
                m.ple_fd = fd;
                m.ple_table_off = off;
                m.ple_row_bytes = src->nb[1];
                m.ple_rows = src->ne[1];
                m.ple_type = src->type;
                continue;
            }
            if (il == m.n_layer) {                             // the MTP layer: everything whole, its experts too
                if (m.qwen4) continue;                         // (qwen4exp's: not yet)
                ggml_tensor * t = ggml_dup_tensor(cw_of(il, name), src);
                ggml_set_name(t, name);
                to_read.push_back({t, fd, off});
                Layer & L = m.mtp;
                m.has_mtp = true;
                std::map<std::string, ggml_tensor **> slot = {
                    {"attn_norm.weight", &L.attn_norm}, {"post_attention_norm.weight", &L.post_norm},
                    {"attn_q.weight", &L.wq}, {"attn_k.weight", &L.wk}, {"attn_v.weight", &L.wv},
                    {"attn_output.weight", &L.wo}, {"attn_q_norm.weight", &L.q_norm}, {"attn_k_norm.weight", &L.k_norm},
                    {"ffn_gate_inp.weight", &L.gate_inp}, {"ffn_gate_inp_shexp.weight", &L.gate_inp_sh},
                    {"ffn_gate_shexp.weight", &L.gate_sh}, {"ffn_up_shexp.weight", &L.up_sh},
                    {"ffn_down_shexp.weight", &L.down_sh}, {"ffn_gate_exps.weight", &L.exps[0]},
                    {"ffn_up_exps.weight", &L.exps[1]}, {"ffn_down_exps.weight", &L.exps[2]},
                    {"nextn.eh_proj.weight", &L.eh_proj}, {"nextn.enorm.weight", &L.enorm},
                    {"nextn.hnorm.weight", &L.hnorm}, {"nextn.shared_head_norm.weight", &L.head_norm},
                    // glm5-next's MTP block: MLA + its indexer, the sigmoid router's bias
                    {"ffn_norm.weight", &L.ffn_norm}, {"exp_probs_b.bias", &L.exp_probs_b},
                    {"attn_q_a.weight", &L.wq_a}, {"attn_q_a_norm.weight", &L.q_a_norm}, {"attn_q_b.weight", &L.wq_b},
                    {"attn_kv_a_mqa.weight", &L.wkv_a}, {"attn_kv_a_norm.weight", &L.kv_a_norm},
                    {"attn_k_b.weight", &L.wk_b}, {"attn_v_b.weight", &L.wv_b}, {"indexer.attn_k.weight", &L.idx_k},
                    {"indexer.attn_q_b.weight", &L.idx_q}, {"indexer.k_norm.weight", &L.idx_k_norm},
                    {"indexer.k_norm.bias", &L.idx_k_norm_b}, {"indexer.proj.weight", &L.idx_proj},
                    {"indexer_compressor_ape.weight", &L.idx_ape}, {"indexer_compressor_gate.weight", &L.idx_gate},
                    // bailingmoe3's: its MLA block's gate per head, the shared head norm
                    {"attn_gate.weight", &L.wgate}, {"layer_output_norm.weight", &L.head_norm}};
                auto it = slot.find(r);
                if (it == slot.end()) fprintf(stderr, "momoe-metal: MTP tensor %s not used\n", name);
                else *it->second = t;
                continue;
            }
            if (il >= 0 && (r == "ffn_gate_exps.weight" || r == "ffn_up_exps.weight" || r == "ffn_down_exps.weight" ||
                            r == "ffn_gate_up_exps.weight")) {   // (gemma4: gate and up in one, the up part)
                int w = r == "ffn_gate_exps.weight" ? 0 : r == "ffn_down_exps.weight" ? 2 : 1;
                Layer & L = m.layers[il];
                L.exp_off[w] = off;
                L.exp_bytes[w] = src->nb[2];
                L.exp_type[w] = src->type;
                L.exp_ne[w][0] = src->ne[0];
                L.exp_ne[w][1] = src->ne[1];
                L.n_exp = (int) src->ne[2];
                L.exp_fd[w] = fd;
                L.exp_map[w] = fmap;
                continue;
            }
            if (il < 0 && m.ds41 && !strncmp(name, "engram_", 7)) {   // deepseek41: the Engram's hash constants (host)
                const size_t nb = ggml_nbytes(src), n = (size_t) ggml_nelements(src);
                std::vector<char> raw(nb);
                read_at(fd, raw.data(), nb, off);
                if (!strcmp(name, "engram_token_map.weight") && src->type == GGML_TYPE_I32) {
                    m.eg_map.assign((const int32_t *) raw.data(), (const int32_t *) raw.data() + n);
                } else if (src->type == GGML_TYPE_I64) {
                    std::vector<uint64_t> v((const uint64_t *) raw.data(), (const uint64_t *) raw.data() + n);
                    if (!strcmp(name, "engram_multipliers.weight")) m.eg_mult = v;
                    else if (!strcmp(name, "engram_primes.weight")) m.eg_primes = v;
                    else if (!strcmp(name, "engram_offsets.weight")) m.eg_offs = v;
                    else fprintf(stderr, "momoe-metal: tensor %s not used\n", name);
                } else fprintf(stderr, "momoe-metal: tensor %s not used\n", name);
                continue;
            }
            ggml_context * tctx = embd_cpu && !strcmp(name, "token_embd.weight") ? ccpu : cw_of(il, name);
            // (deepseek41's Engram gate weights as F32: their product is taken in the graph)
            const bool as_f32 = (bf16_f32 || (m.ds41 && (r == "engram_q.weight" || r == "engram_k.weight"))) &&
                                src->type == GGML_TYPE_BF16;
            ggml_tensor * t = as_f32 ? ggml_new_tensor(tctx, GGML_TYPE_F32, ggml_n_dims(src), src->ne) : ggml_dup_tensor(tctx, src);
            ggml_set_name(t, name);
            to_read.push_back({t, fd, off, as_f32});
            if (il < 0) {
                if (!strcmp(name, "token_embd.weight")) m.tok_embd = t;
                else if (!strcmp(name, "output_norm.weight")) m.out_norm = t;
                else if (!strcmp(name, "token_embd_norm.weight") && m.lfm) m.out_norm = t;   // (lfm2moe's output norm)
                else if (!strcmp(name, "output.weight")) m.output = t;
                else if (!strcmp(name, "rope_freqs.weight") && m.gemma4) m.rope_freqs = t;   // (its full blocks' RoPE)
                else if (!strcmp(name, "output_hc_norm.weight")) m.hc_head_norm = t;
                else if (!strcmp(name, "output_hc_down.weight")) m.hc_head_down = t;
                else if (!strcmp(name, "output_hc_up.weight")) m.hc_head_up = t;
                else fprintf(stderr, "momoe-metal: tensor %s not used\n", name);
                continue;
            }
            Layer & L = m.layers[il];
            std::map<std::string, ggml_tensor **> slot = {
                {"attn_norm.weight", &L.attn_norm}, {"post_attention_norm.weight", &L.post_norm},
                {"attn_q.weight", &L.wq}, {"attn_k.weight", &L.wk}, {"attn_v.weight", &L.wv},
                {"attn_output.weight", &L.wo}, {"attn_q_norm.weight", &L.q_norm}, {"attn_k_norm.weight", &L.k_norm},
                {"attn_qkv.weight", &L.wqkv}, {"attn_gate.weight", &L.wgate}, {"ssm_conv1d.weight", &L.conv1d},
                {"ssm_dt.bias", &L.ssm_dt}, {"ssm_a", &L.ssm_a}, {"ssm_beta.weight", &L.ssm_beta},
                {"ssm_alpha.weight", &L.ssm_alpha}, {"ssm_norm.weight", &L.ssm_norm}, {"ssm_out.weight", &L.ssm_out},
                {"ffn_gate_inp.weight", &L.gate_inp}, {"ffn_gate_inp_shexp.weight", &L.gate_inp_sh},
                {"ffn_gate_shexp.weight", &L.gate_sh}, {"ffn_up_shexp.weight", &L.up_sh},
                {"ffn_down_shexp.weight", &L.down_sh},
                {"hc_attn_norm.weight", &L.hc_attn_norm}, {"hc_attn_down.weight", &L.hc_attn_down},
                {"hc_attn_up.weight", &L.hc_attn_up}, {"hc_attn_inject.weight", &L.hc_attn_inject},
                {"hc_ffn_norm.weight", &L.hc_ffn_norm}, {"hc_ffn_down.weight", &L.hc_ffn_down},
                {"hc_ffn_up.weight", &L.hc_ffn_up}, {"hc_ffn_inject.weight", &L.hc_ffn_inject},
                {"ple_key.weight", &L.ple_key}, {"ple_value.weight", &L.ple_value},
                {"ple_norm_key.weight", &L.ple_norm_key}, {"ple_norm_query.weight", &L.ple_norm_query},
                {"ple_norm_conv.weight", &L.ple_norm_conv}, {"ple_conv1d.weight", &L.ple_conv1d},
                {"indexer.q_proj.weight", &L.idx_q}, {"indexer.k_proj.weight", &L.idx_k},
                {"indexer.q_norm.weight", &L.idx_q_norm}, {"indexer.k_norm.weight", &L.idx_k_norm},
                {"ssm_in.weight", &L.ssm_in}, {"ssm_conv1d.bias", &L.conv1d_b}, {"ssm_d", &L.ssm_d},
                {"exp_probs_b.bias", &L.exp_probs_b}, {"ffn_latent_down.weight", &L.latent_down},
                {"ffn_latent_up.weight", &L.latent_up},
                // glm5-next
                {"ssm_conv1d_q.weight", &L.conv_q}, {"ssm_conv1d_k.weight", &L.conv_k}, {"ssm_conv1d_v.weight", &L.conv_v},
                {"ssm_f_a.weight", &L.ssm_f_a}, {"ssm_f_b.weight", &L.ssm_f_b}, {"ssm_g_a.weight", &L.ssm_g_a},
                {"ssm_g_b.weight", &L.ssm_g_b}, {"attn_q_a.weight", &L.wq_a}, {"attn_q_a_norm.weight", &L.q_a_norm},
                {"attn_q_b.weight", &L.wq_b}, {"attn_kv_a_mqa.weight", &L.wkv_a}, {"attn_kv_a_norm.weight", &L.kv_a_norm},
                {"attn_k_b.weight", &L.wk_b}, {"attn_v_b.weight", &L.wv_b}, {"indexer.attn_k.weight", &L.idx_k},
                {"indexer.attn_q_b.weight", &L.idx_q}, {"indexer.k_norm.weight", &L.idx_k_norm},
                {"indexer.k_norm.bias", &L.idx_k_norm_b}, {"indexer.proj.weight", &L.idx_proj},
                {"indexer_compressor_ape.weight", &L.idx_ape}, {"indexer_compressor_gate.weight", &L.idx_gate},
                {"hc_attn_fn.weight", &L.hc_attn_fn}, {"hc_attn_base.weight", &L.hc_attn_base},
                {"hc_attn_scale.weight", &L.hc_attn_scale}, {"hc_ffn_fn.weight", &L.hc_ffn_fn},
                {"hc_ffn_base.weight", &L.hc_ffn_base}, {"hc_ffn_scale.weight", &L.hc_ffn_scale},
                {"ffn_norm.weight", &L.ffn_norm}, {"ffn_gate.weight", &L.ffn_gate}, {"ffn_up.weight", &L.ffn_up},
                {"ffn_down.weight", &L.ffn_down},
                // lfm2moe: the short convolution (its kernel, in_proj, out_proj)
                {"shortconv.conv.weight", &L.conv1d}, {"shortconv.in_proj.weight", &L.ssm_in},
                {"shortconv.out_proj.weight", &L.ssm_out},
                {"attn_sinks.weight", &L.sinks},                     // flint
                // gemma4
                {"post_ffw_norm.weight", &L.post_ffw}, {"post_ffw_norm_1.weight", &L.post_ffw1},
                {"post_ffw_norm_2.weight", &L.post_ffw2}, {"pre_ffw_norm_2.weight", &L.pre_ffw2},
                {"ffn_gate_inp.scale", &L.gate_inp_s}, {"ffn_down_exps.scale", &L.down_s},
                {"layer_output_scale.weight", &L.out_scale},
                // deepseek41 (its attention's names; the indexer's and the KV norm's as glm5-next's)
                {"attn_kv.weight", &L.wkv_a}, {"attn_output_a.weight", &L.wo_a}, {"attn_output_b.weight", &L.wo},
                {"attn_compressor_kv.weight", &L.comp_kv}, {"attn_compressor_gate.weight", &L.comp_gate},
                {"attn_compressor_norm.weight", &L.comp_norm}, {"engram_q.weight", &L.eg_q}, {"engram_k.weight", &L.eg_k},
                {"engram_wkv.weight", &L.eg_wkv}};
            auto it = slot.find(r);
            if (it == slot.end()) fprintf(stderr, "momoe-metal: tensor %s not used\n", name);
            else *it->second = t;
        }
    }
    if (!m.output) m.output = m.tok_embd;
    m.n_vocab = (int) m.tok_embd->ne[1];
    for (size_t il = 0; il < m.layers.size(); ++il) {        // (glm5-next: the first n_dense blocks have a dense FFN)
        const Layer & L = m.layers[il];
        if (L.exp_fd[1] < 0 && (!m.nemo || L.kind == 'e') && !((m.glm5 || m.lfm || m.ling) && (int) il < m.n_dense))
            die("a layer has no routed experts");
    }
    if (m.qwen4 && !m.hc_head_norm) die("the hyper-connection head (output_hc_*) is missing");
    if (m.ds41 && m.eg_heads) {                              // the Engram tables: one per Engram block, read on the CPU
        const size_t ne = m.eg_fd.size();
        if (m.eg_mult.size() != ne * m.eg_ngram || m.eg_primes.size() != ne * m.eg_heads || m.eg_offs.size() != ne * m.eg_heads ||
            m.eg_map.empty() || m.eg_pad >= (int64_t) m.eg_map.size())
            die("deepseek41: the Engram's hash constants (engram_* tensors) are missing or of the wrong size");
        ggml_context * em = nullptr;
        gguf_init_params ep = {true, &em};
        gguf_context * eg = gguf_init_from_file(e.engram.c_str(), ep);
        if (!eg) die("cannot read %s as GGUF", e.engram.c_str());
        const int fd = open(e.engram.c_str(), O_RDONLY);
        if (fd < 0) die("cannot open %s", e.engram.c_str());
        e.fds.push_back(fd);
        for (int il = 0; il < m.n_layer; ++il) {
            const int j = m.layers[il].eg;
            if (j < 0) continue;
            const std::string name = "blk." + std::to_string(il) + ".engram_embd.weight";
            const int64_t ti = gguf_find_tensor(eg, name.c_str());
            if (ti < 0) die("deepseek41: the Engram file has no %s", name.c_str());
            const ggml_tensor * t = ggml_get_tensor(em, name.c_str());
            if (t->ne[0] != m.eg_dim || !ggml_get_type_traits(t->type)->to_float) die("deepseek41: %s: not a table of rows the CPU can read, eg_dim wide", name.c_str());
            m.eg_fd[j] = fd;
            m.eg_off[j] = gguf_get_data_offset(eg) + gguf_get_tensor_offset(eg, ti);
            m.eg_row[j] = t->nb[1];
            m.eg_rows[j] = t->ne[1];
            m.eg_type = t->type;
            for (int h = 0; h < m.eg_heads; ++h)
                if (m.eg_offs[(size_t) j * m.eg_heads + h] + m.eg_primes[(size_t) j * m.eg_heads + h] > (uint64_t) t->ne[1])
                    die("deepseek41: an Engram head reaches past its table");
        }
        gguf_free(eg);
        ggml_free(em);
        fprintf(stderr, "momoe-metal: Engram: %s (%zu tables, rows read on the CPU)\n", e.engram.c_str(), ne);
    }
    if (m.ds41) {
        for (int il = 0; il < m.n_layer; ++il) {
            const Layer & L = m.layers[il];
            if (!L.wkv_a || !L.kv_a_norm || !L.wo_a || !L.wo || !L.sinks || !L.wq_a || !L.wq_b || !L.hc_attn_fn || !L.hc_ffn_fn)
                die("deepseek41: block %s's attention or mHC tensors are missing", std::to_string(il).c_str());
            if (L.kv_src == il && (!L.comp_kv || !L.comp_norm || !L.idx_k || !L.idx_k_norm || (L.ratio > 1 && !L.comp_gate)))
                die("deepseek41: block %s's compressor is missing", std::to_string(il).c_str());
            if (L.topk_src == il && (!L.idx_q || !L.idx_proj)) die("deepseek41: block %s's indexer is missing", std::to_string(il).c_str());
            if (L.eg >= 0 && (!L.eg_q || !L.eg_k || !L.eg_wkv)) die("deepseek41: block %s's Engram tensors are missing", std::to_string(il).c_str());
        }
    }
    if (m.ple_layer >= 0) {
        if (m.ple_fd < 0) die("the PLE table (per_layer_token_embd.weight) is missing: are all the model's files there?");
        if (!ggml_get_type_traits(m.ple_type)->to_float) die("the PLE table's format cannot be read on the CPU");
        if (!m.layers[m.ple_layer].recurrent) die("the PLE layer is not a delta net layer");
    }

#if defined(MOMOE_CUDA)
    e.metal = ggml_backend_cuda_init(e.device);              // (the GPU backend; "metal" by history)
    if (!e.metal) die("no CUDA device %s", std::to_string(e.device).c_str());
    {
        char desc[256];
        ggml_backend_cuda_get_device_description(e.device, desc, sizeof desc);
        fprintf(stderr, "momoe-cuda: CUDA device %d: %s\n", e.device, desc);
    }
#else
    e.metal = ggml_backend_metal_init();
    if (!e.metal) die("no Metal device");
#endif
    e.cpu = ggml_backend_cpu_init();
    if (!e.gpus.empty()) e.gpus[0] = e.metal;
    for (int d = 0; d < nc; ++d) {
        if (d > 0 && !ggml_get_first_tensor(cws[d])) continue;
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cws[d], card(e, d));
        if (nc > 1) {
            if (!bw) die("cannot allocate the dense weights on card %s", std::to_string(d).c_str());
            card_buf(bw, d);
        }
        e.bufs.push_back(bw);
    }
    if (embd_cpu) e.bufs.push_back(ggml_backend_alloc_ctx_tensors_from_buft(ccpu, ggml_backend_cpu_buffer_type()));
    size_t dense = 0;
    std::vector<char> tmp;
    size_t n_bf16 = 0;
    for (auto & r : to_read) {
        if (r.bf16) {                                        // (see bf16_f32)
            const int64_t n = ggml_nelements(r.t);
            std::vector<ggml_bf16_t> raw(n);
            read_at(r.fd, raw.data(), n * sizeof(ggml_bf16_t), r.off);
            std::vector<float> f(n);
            ggml_bf16_to_fp32_row(raw.data(), f.data(), n);
            if (kHostMem) memcpy(r.t->data, f.data(), n * sizeof(float));
            else ggml_backend_tensor_set(r.t, f.data(), 0, n * sizeof(float));
            dense += ggml_nbytes(r.t);
            ++n_bf16;
            continue;
        }
        if (kHostMem) read_at(r.fd, r.t->data, ggml_nbytes(r.t), r.off);   // a Metal buffer is host memory
        else {
            tmp.resize(ggml_nbytes(r.t));
            read_at(r.fd, tmp.data(), tmp.size(), r.off);
            ggml_backend_tensor_set(r.t, tmp.data(), 0, tmp.size());
        }
        dense += ggml_nbytes(r.t);
    }
    fprintf(stderr, "momoe-metal: dense weights %.2f GiB loaded\n", dense / 1073741824.0);
    if (n_bf16 && bf16_f32) fprintf(stderr, "momoe-metal: %zu BF16 weights loaded as F32 (this GPU has no bfloat kernels: macOS before 14)\n", n_bf16);
    if (m.qwen4 && e.load_mtp) load_mtp_qwen4(e);
    if (m.has_mtp && !e.draft_vocab.empty()) {               // the draft head: the output head's rows for the subset
        FILE * f = fopen(e.draft_vocab.c_str(), "rb");
        if (!f) die("cannot open %s", e.draft_vocab.c_str());
        fseek(f, 0, SEEK_END);
        const long n = ftell(f) / 4;
        fseek(f, 0, SEEK_SET);
        m.dvocab.resize(n);
        if (fread(m.dvocab.data(), 4, n, f) != (size_t) n) die("cannot read %s", e.draft_vocab.c_str());
        fclose(f);
        for (int32_t id : m.dvocab)                          // (the shipped subsets are Qwen's token ids)
            if (id < 0 || id >= m.output->ne[1])
                die("--draft-vocab %s: a token is not in this model's vocabulary (a subset of another model's)",
                    e.draft_vocab.c_str());
        ggml_context * cd = ggml_init(ip);
        m.dhead = ggml_new_tensor_2d(cd, m.output->type, m.n_embd, n);
        e.bufs.push_back(ggml_backend_alloc_ctx_tensors(cd, card(e, nc - 1)));
        const size_t row = m.output->nb[1];
        std::vector<char> r(row);
        for (long i = 0; i < n; ++i) {
            tget(m.output, r.data(), (size_t) m.dvocab[i] * row, row);
            tset(m.dhead, r.data(), (size_t) i * row, row);
        }
        fprintf(stderr, "momoe-metal: draft head over %ld tokens (%.0f MiB)\n", n, (double) n * row / 1048576.0);
    }

    // the expert pool: a group per expert format, each the same share of its experts
    Pool & P = e.pool;
    size_t all_bytes = 0;
    const int p0 = m.part0();                                // nemo: no gate part
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        if (L.exp_fd[1] < 0) continue;                       // nemo: a Mamba-2 or attention block
        int gi = 0;
        for (; gi < (int) P.g.size(); ++gi) {
            Group & G = P.g[gi];
            bool same = true;
            for (int w = 0; w < 3; ++w)
                same &= G.type[w] == L.exp_type[w] && G.ne[w][0] == L.exp_ne[w][0] && G.ne[w][1] == L.exp_ne[w][1];
            if (same && G.dev == L.dev) break;
        }
        if (gi == (int) P.g.size()) {
            P.g.emplace_back();
            Group & G = P.g.back();
            for (int w = 0; w < 3; ++w) {
                G.type[w] = L.exp_type[w];
                G.ne[w][0] = L.exp_ne[w][0];
                G.ne[w][1] = L.exp_ne[w][1];
                G.bytes += L.exp_bytes[w];
            }
            G.dev = L.dev;
        }
        L.grp = gi;
        ++P.g[gi].n_layers;
        P.g[gi].n_all += L.n_exp;
        m.n_exp_all += L.n_exp;
        all_bytes += P.g[gi].bytes * L.n_exp;
    }
    P.all_bytes = all_bytes;
    // the share of the experts that fits: --slots N (of all), --pool-gib G, or all of them (momoe-cuda without either:
    // the smallest pool now, sized to the free VRAM by fit_pool once the compute buffers are reserved)
    const double frac = slots_arg > 0 ? (double) slots_arg / m.n_exp_all : pool_gib > 0 ? pool_gib * 1073741824.0 / all_bytes
                      : kHostMem ? 1.0 : 0.0;
    make_pool(e, frac);
    // one staging buffer for every group (one layer at a time): each group's staging tensors at its start (--layer-split:
    // one a card, for its groups)
    for (int d = 0; d < nc; ++d) {
        size_t stage_bytes = 0;
        auto stride = [&](const Group & G, int w) {          // an expert's part and a row for its pad (CUDA's kernels
            const size_t row = ggml_row_size(G.type[w], G.ne[w][0]);   // take the stride in whole blocks)
            return row * (G.ne[w][1] + (G.ne[w][0] % 512 ? 1 : 0));
        };
        for (auto & G : P.g)
            if (G.dev == d) {
                size_t b = 0;
                for (int w = p0; w < 3; ++w) b += stride(G, w) * m.n_expert;
                stage_bytes = std::max(stage_bytes, b);
            }
        if (!stage_bytes) continue;
        ggml_backend_buffer_t sb = ggml_backend_alloc_buffer(card(e, d), stage_bytes + 4096 * 3);
        if (!sb) die("cannot allocate the staging copy");
        if (nc > 1) card_buf(sb, d);
        e.bufs.push_back(sb);
        ggml_context * cg = ggml_init(ip);
        for (auto & G : P.g) {
            if (G.dev != d) continue;
            char * at = (char *) ggml_backend_buffer_get_base(sb);
            for (int w = p0; w < 3; ++w) {
                G.stage[w] = ggml_new_tensor_3d(cg, G.type[w], G.ne[w][0], G.ne[w][1], m.n_expert);
                G.stage[w]->nb[2] = stride(G, w);            // (each expert's pad: zeros, see stage_pads)
                G.stage_pad[w] = stride(G, w) - (size_t) G.stage[w]->nb[1] * G.ne[w][1];
                if (ggml_backend_tensor_alloc(sb, G.stage[w], at) != GGML_STATUS_SUCCESS) die("cannot place the staging copy");
                at += GGML_PAD(G.stage[w]->nb[2] * m.n_expert, 4096);
            }
        }
    }
    P.lfu = !(getenv("MOMOE_POLICY") && !strcmp(getenv("MOMOE_POLICY"), "lru"));   // lfu by default (measured faster)

    // state: the KV cache of the attention layers, the delta net's conv and recurrent states, the slot numbers
    // (--layer-split: on the layer's card)
    std::vector<ggml_context *> css(nc);
    for (auto & c : css) c = ggml_init(ip);
    ggml_context * cs = css[nc - 1];                         // (the MTP layer's: the last card)
    const int conv_ch = m.ssm_inner + 2 * m.ssm_groups * m.ssm_state;
    const int head_v = m.ssm_dt_rank ? m.ssm_inner / m.ssm_dt_rank : 0;   // (lfm2moe has no SSM)
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        ggml_context * cs = css[L.dev];
        if (m.ds41) {                                        // the window's ring (written at p % swa_rows, as flint's);
            if (!m.swa_rows) {                               // a KV source's compressed rows and its unfinished group
                const int t_max = std::max(e.max_batch, e.max_window + 1);
                m.swa_rows = std::min(e.n_ctx, (m.n_swa + t_max - 1 + 255) / 256 * 256);
            }
            L.k_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.ds_hd, m.swa_rows);
            if (L.kv_src == il) {
                L.comp_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.ds_hd + m.idx_dim, e.n_ctx / L.ratio + 1);
                if (L.ratio > 1) {
                    L.cst_kv = ggml_new_tensor_2d(cs, GGML_TYPE_F32, m.ds_hd, L.ratio);
                    L.cst_sc = ggml_new_tensor_2d(cs, GGML_TYPE_F32, m.ds_hd, L.ratio);
                }
            }
        } else if ((m.glm5 || m.ling) && L.kind == 'k') {           // KDA: q, k, v's convolution histories; the state
            L.conv_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) 3 * (m.ssm_conv - 1) * m.ssm_inner);
            L.ssm_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) m.kda_hd * m.kda_hd * m.n_head);
        } else if (m.ling) {                                 // MLA: the latent and the key's RoPE part
            L.lat_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.kv_lora + m.n_rot, e.n_ctx);
        } else if (m.glm5) {                                 // MLA: the latent; the indexer's key | gate rows, pooled keys
            L.lat_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.kv_lora, e.n_ctx);
            L.idx_raw = ggml_new_tensor_2d(cs, GGML_TYPE_F16, 2 * m.idx_dim, e.n_ctx);
            L.idx_pool = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.idx_dim, e.n_ctx / m.kpool + 64);
        } else if (L.kind == 'm') {                          // Mamba-2: [d_state, head_dim, n_head] per sequence
            L.conv_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1) * conv_ch);
            L.ssm_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) m.ssm_state * head_v * m.ssm_dt_rank);
        } else if (m.lfm && L.kind == 'c') {                 // the short convolution's history: [l_cache - 1, n_embd]
            L.conv_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1) * m.n_embd);
        } else if (L.kind == 'e') {
            if (e.hybrid) L.cpu_dn = ggml_new_tensor_3d(cs, GGML_TYPE_F32, m.latent, m.n_used, HYB_MAX_T);
        } else if (L.recurrent) {
            L.conv_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1) * conv_ch);
            L.ssm_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) head_v * head_v * m.ssm_dt_rank);
        } else {
            int rows = e.n_ctx;
            if (L.swa) {                                     // (flint) a batch writes at most max(max_batch, a window)
                const int t_max = std::max(e.max_batch, e.max_window + 1);   // rows: none it still reads is overwritten
                m.swa_rows = std::min(e.n_ctx, (m.n_swa + t_max - 1 + 255) / 256 * 256);
                if (getenv("MOMOE_FLINT_NO_RING")) m.swa_rows = e.n_ctx;   // a test: every position kept
                rows = m.swa_rows;
            }
            const int hd = L.hd ? L.hd : m.head_dim, nkv = L.n_kv ? L.n_kv : m.n_head_kv;   // (gemma4: per block)
            L.k_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, hd, nkv, rows);
            L.v_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, hd, nkv, rows);
        }
        if (L.idx_k && !m.glm5 && !m.ds41) {
            L.idx_raw = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.idx_dim, e.n_ctx);
            L.idx_pool = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.idx_dim, e.n_ctx / m.kpool + 64);
        }
        if (il == m.ple_layer)
            L.ple_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ple_kernel - 1) * m.ple_ngram * m.hc * m.n_embd);
        L.slot_map = ggml_new_tensor_2d(cs, GGML_TYPE_I32, 1, m.n_expert);
        L.res_mask = ggml_new_tensor_2d(cs, GGML_TYPE_F32, m.n_expert, 1);
        L.res_keep = ggml_new_tensor_2d(cs, GGML_TYPE_F32, 1, m.n_expert);
        if (e.hybrid && hybrid_model(m) && !m.nemo && L.exp_fd[1] >= 0)   // qwen35moe --hybrid: the CPU's
            L.cpu_dn = ggml_new_tensor_3d(cs, GGML_TYPE_F32, m.n_embd, m.n_used, HYB_MAX_T);   // experts' outputs
        if (L.recurrent && e.load_mtp) {                     // a verify window's per-token states (rollback)
            L.ssm_snap = ggml_new_tensor_2d(cs, GGML_TYPE_F32, (int64_t) head_v * head_v * m.ssm_dt_rank, e.max_window);
            L.conv_snap = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1 + e.max_window) * conv_ch);
        }
    }
    if (m.has_mtp && m.ling) {                               // its MLA block's latents
        m.mtp.lat_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.kv_lora + m.n_rot, e.n_ctx);
    } else if (m.has_mtp && m.glm5) {                        // its MLA block's latents and indexer
        m.mtp.lat_cache = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.kv_lora, e.n_ctx);
        m.mtp.idx_raw = ggml_new_tensor_2d(cs, GGML_TYPE_F16, 2 * m.idx_dim, e.n_ctx);
        m.mtp.idx_pool = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.idx_dim, e.n_ctx / m.kpool + 64);
    } else if (m.has_mtp) {
        m.mtp.k_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, m.head_dim, m.n_head_kv, e.n_ctx);
        m.mtp.v_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, m.head_dim, m.n_head_kv, e.n_ctx);
    }
    for (int d = 0; d < nc; ++d) {
        if (!ggml_get_first_tensor(css[d])) continue;
        ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors(css[d], card(e, d));
        if (!bs) die("cannot allocate the state");
        ggml_backend_buffer_clear(bs, 0);
        if (nc > 1) card_buf(bs, d);
        e.bufs.push_back(bs);
    }
    for (auto & L : m.layers) L.map_host.assign(L.slot_map ? ggml_nelements(L.slot_map) : 0, 0);

    std::vector<ggml_backend_t> backends;                    // (--layer-split: every card, then the CPU)
    for (int d = 0; d < nc; ++d) backends.push_back(card(e, d));
#if defined(MOMOE_CUDA)
    if (e.helper) {                                          // --helper-tier: the second visible card
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) die("--helper-tier needs a second visible card");
        if (e.device != 0 || nc > 1) die("--helper-tier: the main card is the first visible one (no --device, no --layer-split)");
        e.helper_gpu = ggml_backend_cuda_init(1);
        if (!e.helper_gpu) die("--helper-tier: cannot start CUDA device 1");
        char desc[256];
        ggml_backend_cuda_get_device_description(1, desc, sizeof desc);
        fprintf(stderr, "momoe-cuda: helper tier: the second pool on %s\n", desc);
        backends.push_back(e.helper_gpu);
    }
#endif
    backends.push_back(e.cpu);
    e.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 32768, false, true);
    e.sched_backends = backends;
    gguf_free(g);
}

// ------------------------------------------------------------------------------------------------ making experts resident
static bool debug_on() { static int on = getenv("MOMOE_DEBUG") ? 1 : 0; return on; }

// the least used slot of the group (recency breaking ties) that is not in use from `floor` on (this layer's experts, a
// prediction) and not being read
#if defined(MOMOE_CUDA)
// does a graph of a few tokens take the mailboxes (Nemotron: always; qwen35moe: on the CPU's experts, see forward)?
static bool mb_on(const Engine & e) { return e.mailbox && (e.m.nemo || e.hyb_active); }
#endif

static int take_slot(Pool & P, Group & G, uint64_t floor, bool spare_predictions = true) {
    int best = -1;
    auto f = [&](int i) { return G.owner[i] >= 0 ? P.freq[G.owner[i]] : 0u; };
    for (int i = 0; i < G.n_slots; ++i) {
        if (G.used[i] >= floor || G.left[i].load(std::memory_order_relaxed) != 0) continue;
        // a prediction waiting for its layer (a clock tick per pause and per prediction: ~3 a layer)
        if (spare_predictions && G.by_prefetch[i] && G.used[i] + 12 > P.clock) continue;
        if (best < 0) { best = i; continue; }
        if (P.lfu ? (f(i) < f(best) || (f(i) == f(best) && G.used[i] < G.used[best])) : G.used[i] < G.used[best]) best = i;
    }
    if (best < 0) return -1;                                 // every slot in use
    if (G.owner[best] >= 0) P.slot_of[G.owner[best]] = -1;
    return best;
}

// read expert `x` of layer `il` into `slot` of its group (3 jobs: gate, up, down)
static void fill_slot(Engine & e, int il, int x, int slot, std::vector<Readers::Job> & jobs) {
    Pool & P = e.pool;
    Layer & L = e.m.layers[il];
    Group & G = P.g[L.grp];
    const int p0 = e.m.part0();
    G.left[slot].store(3 - p0, std::memory_order_relaxed);
    for (int w = p0; w < 3; ++w) {
        const size_t off = L.exp_off[w] + (size_t) x * L.exp_bytes[w], t_off = (size_t) slot * G.t[w]->nb[2];
        if (kHostMem) jobs.push_back({L.exp_fd[w], (char *) G.t[w]->data + t_off, L.exp_bytes[w], off, &G.left[slot]});
        else if (P.host)                                     // a host-mapped pool: written by the reader's CPU
            jobs.push_back({L.exp_fd[w], P.host + ((char *) G.t[w]->data - P.dev) + t_off, L.exp_bytes[w], off, &G.left[slot]});
        else jobs.push_back({L.exp_fd[w], nullptr, L.exp_bytes[w], off, &G.left[slot], L.exp_map[w] + off, G.t[w], t_off});
        P.bytes_read += L.exp_bytes[w];
    }
    const int key = il * e.m.n_expert + x;
    G.owner[slot] = key;
    P.slot_of[key] = slot;
}

// --helper-tier: expert `key` (an eviction of the main pool) read into helper slot `hs`, in the background
static void fill_hslot(Engine & e, int key, int hs, std::vector<Readers::Job> & jobs) {
    Model & m = e.m;
    Pool & H = e.hpool;
    const int il = key / m.n_expert, x = key % m.n_expert;
    Layer & L = m.layers[il];
    Group & K = H.g[L.grp];
    K.left[hs].store(3 - m.part0(), std::memory_order_relaxed);
    for (int w = m.part0(); w < 3; ++w) {
        const size_t off = L.exp_off[w] + (size_t) x * L.exp_bytes[w], t_off = (size_t) hs * K.t[w]->nb[2];
        jobs.push_back({L.exp_fd[w], nullptr, L.exp_bytes[w], off, &K.left[hs], L.exp_map[w] + off, K.t[w], t_off});
    }
    K.owner[hs] = key;
    H.slot_of[key] = hs;
    ++e.helper_down;
}

// called at a layer's pause with its chosen experts ([n_used, n_tokens] i32): every one is made resident (a hit, a
// prediction still being read: waited for, a miss: read now, ahead of the predictions), and its slot is written in
// the layer's expert -> slot table, which the expert matmuls look their slots up in (get_rows on the chosen experts:
// a dependency the Metal backend's reordering keeps)
#if defined(MOMOE_CUDA)
// CUDA's quantized batch kernels read each row on to a multiple of 512 values and multiply that tail by zeros: an
// expert's last row reads on into its pad, which must hold no NaN (Lightning's down rows, 1856 values: the bytes of
// another group's experts there, read as Q8_0 scales, made NaN). The groups share the staging buffer, so a group's pads
// are zeroed again when it follows another one there.
static void stage_pads(Engine & e, Group & G, int gi) {
    Pool & P = e.pool;
    if (P.stage_zeroed[G.dev] == gi) return;
    for (int w = e.m.part0(); w < 3; ++w)
        if (G.stage_pad[w])
            cudaMemset2DAsync((char *) G.stage[w]->data + (G.stage[w]->nb[2] - G.stage_pad[w]), G.stage[w]->nb[2], 0,
                              G.stage_pad[w], G.stage[w]->ne[2], cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
    P.stage_zeroed[G.dev] = gi;
}
#endif

static void make_resident(Engine & e, int il, const int32_t * ids, int n_tok) {
    Model & m = e.m;
    Pool & P = e.pool;
    Layer & L = m.layers[il];
    Group & G = P.g[L.grp];
    const bool stage = staged(m.n_used, G.n_slots, n_tok);
    int32_t * map = L.slots_host();
    if (!kHostMem && stage) {
        // momoe-cuda: a big batch reads its layer from the staging copy, filled from the mapped file (host to device,
        // every chosen expert): nothing goes into the pool (that copy would be a second one, for nothing)
#if defined(MOMOE_CUDA)
        stage_pads(e, G, L.grp);
#endif
        std::vector<uint8_t> seen(m.n_expert, 0);
        std::atomic<int> left{0};
        std::vector<Readers::Job> jobs;
        int n = 0;
        // an expert the pool holds (read in, nothing in flight) is copied from there, inside the GPU, instead of read
        // again (--direct-io: from the SSD; slots change only on this thread, so it stays put while copied)
        static const bool from_pool = !getenv("MOMOE_STAGE_NO_POOL");
        bool copied_in = false;
        for (size_t i = 0; i < (size_t) m.n_used * n_tok; ++i) {
            const int x = ids[i];
            if (seen[x]) continue;
            seen[x] = 1;
            ++P.freq[il * m.n_expert + x];                   // once a chunk: a long prompt's experts would otherwise
                                                             // outrank the reply's for hundreds of tokens (measured:
                                                             // 19% hits writing after 20K tokens of prompt, 55% without)
#if defined(MOMOE_CUDA)
            const int have = P.slot_of[il * m.n_expert + x];
            if (from_pool && have >= 0 && G.left[have].load(std::memory_order_acquire) == 0) {
                for (int w = m.part0(); w < 3; ++w)
                    cudaMemcpyAsync((char *) G.stage[w]->data + (size_t) x * G.stage[w]->nb[2],
                                    (const char *) G.t[w]->data + (size_t) have * G.t[w]->nb[2], L.exp_bytes[w],
                                    cudaMemcpyDeviceToDevice, cudaStreamPerThread);
                ++P.stage_from_pool;
                copied_in = true;
                continue;
            }
#endif
            for (int w = m.part0(); w < 3; ++w) {
                const size_t off = L.exp_off[w] + (size_t) x * L.exp_bytes[w], t_off = (size_t) x * G.stage[w]->nb[2];
                jobs.push_back({L.exp_fd[w], nullptr, L.exp_bytes[w], off, &left, L.exp_map[w] + off, G.stage[w], t_off});
                P.bytes_read += L.exp_bytes[w];
                ++n;
            }
        }
        left.store(n, std::memory_order_relaxed);
        const double t0 = now_s();
        e.readers.submit(jobs, true);
        while (left.load(std::memory_order_acquire)) { }
#if defined(MOMOE_CUDA)
        if (copied_in) cudaStreamSynchronize(cudaStreamPerThread);
#endif
        P.read_s += now_s() - t0;
        // the pool kept fresh for free: an expert used often lately (as --hybrid admits) and not in the pool is copied
        // there from the staging copy, inside the GPU (no PCIe)
#if defined(MOMOE_CUDA)
        static const uint32_t admit = getenv("MOMOE_ADMIT") ? (uint32_t) atoi(getenv("MOMOE_ADMIT")) : 2;
        if (mb_on(e)) {                                      // the slot tables change only between graphs: the experts
            for (int x = 0; x < m.n_expert; ++x) {           // to read in go to the next boundary
                const int key = il * m.n_expert + x;
                if (seen[x] && P.slot_of[key] < 0 && P.freq[key] > admit) e.mb.want.push_back({il, x});
            }
            return;
        }
        const uint64_t stamp = ++P.clock;
        bool copied = false;
        for (int x = 0; x < m.n_expert; ++x) {
            const int key = il * m.n_expert + x;
            // (--direct-io: every one: the next chunk copies it from the pool instead of reading it from the SSD again)
            if (!seen[x] || P.slot_of[key] >= 0 || (!g_direct && P.freq[key] <= admit)) continue;
            const int slot = take_slot(P, G, stamp);
            if (slot < 0) break;
            for (int w = m.part0(); w < 3; ++w)
                cudaMemcpyAsync((char *) G.t[w]->data + (size_t) slot * G.t[w]->nb[2],
                                (const char *) G.stage[w]->data + (size_t) x * G.stage[w]->nb[2], L.exp_bytes[w],
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread);
            G.owner[slot] = key;
            P.slot_of[key] = slot;
            G.used[slot] = stamp - 1;
            G.by_prefetch[slot] = 0;
            copied = true;
        }
        if (copied) cudaStreamSynchronize(cudaStreamPerThread);
#endif
        return;
    }
    std::vector<Readers::Job> jobs;
    std::vector<int> wait;
    std::vector<std::atomic<int>> sleft(1);                  // staging: the parts still being read or copied
    int n_stage = 0;
    const uint64_t pin = ++P.clock;                          // this layer's experts: not taken by each other
    if (++P.uses % ((uint64_t) 64 * m.n_layer) == 0)       // aging: every 64 tokens' worth of layers
        for (auto & c : P.freq) c >>= 1;
    std::vector<uint8_t> seen(stage ? m.n_expert : 0, 0);
    // --helper-tier (not a staged batch): an expert the helper card holds is computed there (the main table points at
    // the slot of zeros, the helper's at its slot); an expert the main pool evicts goes down to the helper
    const bool hp = e.helper && !stage && !L.hmap_host.empty();
    Pool & H = e.hpool;
    Group * K = hp ? &H.g[L.grp] : nullptr;
    int32_t * hmap = hp ? L.hmap_host.data() : nullptr;
    const uint64_t hpin = hp ? ++H.clock : 0;                // this layer's helper experts: not taken by the evictions
    std::vector<Readers::Job> hjobs;
    for (size_t i = 0; i < (size_t) m.n_used * n_tok; ++i) {
        const int x = ids[i];
        if (x < 0 || x >= m.n_expert)                        // (a GPU step that failed: out of its memory)
            die("%s: the GPU's work failed (out of GPU memory? a smaller --pool-gib)",
                ("layer " + std::to_string(il) + " chose expert " + std::to_string(x)).c_str());
        const int key = il * m.n_expert + x;
        ++P.freq[key];
        if (stage) {
            if (seen[x]) continue;
            seen[x] = 1;
        }
        int slot = P.slot_of[key];
        if (slot < 0 && hp && H.slot_of[key] >= 0 && K->left[H.slot_of[key]].load(std::memory_order_acquire) == 0) {
            const int hs = H.slot_of[key];
            K->used[hs] = hpin;
            ++H.freq[key];
            ++e.helper_hits;
            map[x] = G.zero;
            hmap[x] = hs;
            continue;
        }
        if (slot >= 0) {
            ++P.hits;
            if (G.by_prefetch[slot]) { ++P.prefetch_hits; G.by_prefetch[slot] = 0; }
            if (G.left[slot].load(std::memory_order_acquire)) { ++P.late; wait.push_back(slot); }
        } else {
            ++P.misses;
            slot = take_slot(P, G, pin);
            while (slot < 0 && !stage) {                     // every slot this layer's or being read: wait for a read
                bool reading = false;
                for (int i = 0; i < G.n_slots && !reading; ++i) reading = G.used[i] < pin && G.left[i].load() != 0;
                if (!reading) break;
                std::this_thread::sleep_for(std::chrono::microseconds(20));
                slot = take_slot(P, G, pin);
            }
            // every other slot a prediction for a later layer: a miss outranks a guess (Nemotron Bolt on the M4 Air, a
            // 14 GiB pool: a 282-slot format group held only predictions, and the engine stopped)
            if (slot < 0 && !stage) slot = take_slot(P, G, pin, false);
            if (hp && slot >= 0) {                           // the expert it evicts: down to the helper (read in the
                const int old = G.owner[slot];               // background; the helper's own least used goes)
                if (old >= 0 && H.slot_of[old] < 0) {
                    const int hs = take_slot(H, *K, hpin);
                    if (hs >= 0) fill_hslot(e, old, hs, hjobs);
                }
            }
            if (slot < 0) {
                if (!stage) {
                    int n_pinned = 0, n_reading = 0, n_pred = 0;
                    for (int i = 0; i < G.n_slots; ++i) {
                        if (G.used[i] >= pin) ++n_pinned;
                        else if (G.left[i].load()) ++n_reading;
                        else if (G.by_prefetch[i] && G.used[i] + 12 > P.clock) ++n_pred;
                    }
                    fprintf(stderr, "momoe-metal: layer %d, %d tokens: group %d has %d slots, none free (%d this layer's, "
                            "%d being read, %d predictions)\n", il, n_tok, L.grp, G.n_slots, n_pinned, n_reading, n_pred);
                    die("the expert pool is too small for this batch");
                }
                for (int w = m.part0(); w < 3; ++w) {       // straight into the staging copy
                    const size_t off = L.exp_off[w] + (size_t) x * L.exp_bytes[w], t_off = (size_t) x * G.stage[w]->nb[2];
                    if (kHostMem) jobs.push_back({L.exp_fd[w], (char *) G.stage[w]->data + t_off, L.exp_bytes[w], off, &sleft[0]});
                    else jobs.push_back({L.exp_fd[w], nullptr, L.exp_bytes[w], off, &sleft[0], L.exp_map[w] + off, G.stage[w], t_off});
                    P.bytes_read += L.exp_bytes[w];
                    ++n_stage;
                }
                continue;
            }
            fill_slot(e, il, x, slot, jobs);
            G.by_prefetch[slot] = 0;
            wait.push_back(slot);
        }
        G.used[slot] = pin;
        map[x] = slot;
        if (hp) hmap[x] = K->zero;
    }
    if (!hjobs.empty()) e.readers.submit(hjobs);              // (behind this layer's own reads)
    sleft[0].store(n_stage, std::memory_order_relaxed);
    const double t0 = now_s();
    (e.demand.th.empty() ? e.readers : e.demand).submit(jobs, true);
    for (int slot : wait)
        while (G.left[slot].load(std::memory_order_acquire)) { }
    P.read_s += now_s() - t0;
    if (stage) {                                             // the batch's experts in the pool: copied to the staging
        int n = 0;                                           // copy (the others are being read into it)
        std::vector<std::atomic<int>> left(1);
        for (int x = 0; x < m.n_expert; ++x) {
            if (!seen[x]) continue;
            const int slot = P.slot_of[il * m.n_expert + x];
            if (slot < 0) continue;                          // read straight into it
            for (int w = m.part0(); w < 3; ++w) {
                const size_t t_off = (size_t) x * G.stage[w]->nb[2];
                if (kHostMem) {
                    Readers::Job j{-1, (char *) G.stage[w]->data + t_off, L.exp_bytes[w], 0, &left[0],
                                   (const char *) G.t[w]->data + (size_t) slot * G.t[w]->nb[2]};
                    jobs.push_back(j);
                } else {                                     // momoe-cuda: from the mapped file (host to device)
                    const size_t off = L.exp_off[w] + (size_t) x * L.exp_bytes[w];
                    jobs.push_back({L.exp_fd[w], nullptr, L.exp_bytes[w], off, &left[0], L.exp_map[w] + off, G.stage[w], t_off});
                }
                ++n;
            }
        }
        left[0].store(n, std::memory_order_relaxed);
        e.readers.submit(jobs, true);
        while (left[0].load(std::memory_order_acquire)) { }
        while (sleft[0].load(std::memory_order_acquire)) { }
    }
    L.sync_slots();
    if (hp) tset(L.hslot_map, L.hmap_host.data(), 0, ggml_nbytes(L.hslot_map));
}

// --nostall, before a graph (the GPU idle): which experts are in the pool (read completely), as the layers'
// masks, weights and slot tables (a missing one points at the slot of zeros). False when some layer has fewer than
// 4 x n_used of its experts in the pool: the graph takes the exact path
static bool ns_publish(Engine & e, int from = 0) {
    Model & m = e.m;
    Pool & P = e.pool;
    bool ok = true;
    for (int il = from; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        Group & G = P.g[L.grp];
        float * mask = (float *) L.res_mask->data, * keep = (float *) L.res_keep->data;
        int32_t * map = (int32_t *) L.slot_map->data;
        int n = 0;
        for (int x = 0; x < m.n_expert; ++x) {
            const int slot = P.slot_of[il * m.n_expert + x];
            const bool in = slot >= 0 && G.left[slot].load(std::memory_order_acquire) == 0;
            mask[x] = in ? 0.0f : -INFINITY;
            keep[x] = in ? 1.0f : 0.0f;
            map[x] = in ? slot : G.zero;
            n += in;
        }
        if (n < 4 * m.n_used) ok = false;                     // a cold pool (the first graphs): the exact path
    }
    return ok;
}

static void prefetch(Engine & e, int il, const int32_t * ids, int n_tok);

// --nostall, after a graph: the experts each layer wanted are counted as used, the missing ones read into the
// pool (published before a later graph, once read), and the next layer's predicted ones read ahead
static bool ns_paused(const Engine & e, int il) { return !e.ns_graph || (e.ns_every > 0 && il % e.ns_every == 0); }

static void ns_layer(Engine & e, int il, int T) {
    Model & m = e.m;
    Pool & P = e.pool;
    std::vector<Readers::Job> jobs;
    {
        Layer & L = m.layers[il];
        Group & G = P.g[L.grp];
        const int32_t * top = (const int32_t *) e.topk[il]->data;
        const uint64_t pin = ++P.clock;
        if (++P.uses % ((uint64_t) 64 * m.n_layer) == 0)
            for (auto & c : P.freq) c >>= 1;
        for (int i = 0; i < m.n_used * T; ++i) {
            const int x = top[i];
            const int key = il * m.n_expert + x;
            ++P.freq[key];
            ++e.ns_wanted;
            int slot = P.slot_of[key];
            if (slot >= 0 && G.left[slot].load(std::memory_order_acquire) == 0) {
                ++P.hits;
                if (G.by_prefetch[slot]) { ++P.prefetch_hits; G.by_prefetch[slot] = 0; }
                G.used[slot] = pin;
                continue;
            }
            ++P.misses;
            ++e.ns_missing;
            if (slot >= 0) continue;                         // being read already
            slot = take_slot(P, G, pin);
            if (slot < 0) continue;
            fill_slot(e, il, x, slot, jobs);
            G.by_prefetch[slot] = 0;
            G.used[slot] = pin;
        }
        e.readers.submit(jobs, true);
        const size_t n_pred = (size_t) m.n_used * T * ((e.predict && il + 1 < m.n_layer) + (e.predict2 && il + 2 < m.n_layer));
        if (n_pred) {
            if (e.predict && il + 1 < m.n_layer) prefetch(e, il + 1, top + (size_t) m.n_used * T, T);
            if (e.predict2 && il + 2 < m.n_layer) prefetch(e, il + 2, top + (size_t) 2 * m.n_used * T, T);
        }
    }
}

static void ns_after(Engine & e, int T) {
    ++e.ns_tokens;
    for (int il = 0; il < e.m.n_layer; ++il)
        if (!ns_paused(e, il)) ns_layer(e, il, T);
}

// after a layer's pause (the GPU already released): read the next layer's predicted experts in the background
static void prefetch(Engine & e, int il, const int32_t * ids, int n_tok) {
    Model & m = e.m;
    Pool & P = e.pool;
    Group & G = P.g[m.layers[il].grp];
    // momoe-cuda: a staged batch reads its layer into the staging copy whatever the pool holds: reading predictions
    // into the pool for it costs reads (--direct-io: SSD reads, which then held every slot a group has when the
    // first token after a long prompt came: measured on the Spark)
    if (!kHostMem && staged(m.n_used, G.n_slots, n_tok)) return;
    std::vector<Readers::Job> jobs;
    const uint64_t stamp = ++P.clock;                        // newer than the layer the GPU is computing: kept
    // MOMOE_PREDICT_K: only each token's K most likely predicted experts (the top-k comes in order) are read
    static const int pk = getenv("MOMOE_PREDICT_K") ? std::max(1, atoi(getenv("MOMOE_PREDICT_K"))) : 1 << 30;
    for (size_t i = 0; i < (size_t) m.n_used * n_tok; ++i) {
        if ((int) (i % m.n_used) >= pk) continue;
        const int x = ids[i];
        if (x < 0 || x >= m.n_expert) {
            fprintf(stderr, "momoe-metal: layer %d prediction %zu is %d (tokens %d)\n", il, i, x, n_tok);
            die("a predicted expert is out of range");
        }
        if (P.slot_of[il * m.n_expert + x] >= 0) continue;
        if (e.helper && e.hpool.slot_of.size() && e.hpool.slot_of[il * m.n_expert + x] >= 0) continue;   // (computed there)
        const int slot = take_slot(P, G, stamp - 1);        // not the experts of the layer being computed
        if (slot < 0) break;                                 // a small pool: the rest is read when needed
        fill_slot(e, il, x, slot, jobs);
        G.used[slot] = stamp;
        G.by_prefetch[slot] = 1;
        ++P.prefetched;
    }
    e.readers.submit(jobs);
}

// --hybrid: does this graph split its experts between the GPU and the CPU (a few tokens, the pool path)?
static bool hybrid_on(const Engine & e, const Layer & L, int T) {
    return e.hybrid && hybrid_model(e.m) && (e.m.nemo || e.hyb_active) && L.cpu_dn && !e.preloaded && T <= HYB_MAX_T &&
           !staged(e.m.n_used, e.pool.g[L.grp].n_slots, T);   // (preloaded: nothing missing, and no stop to join it)
}

// --hybrid, at a layer's stop: the chosen experts the pool holds (read completely) are its; every other one points at
// the zero slot and goes to the CPU (launched here, joined at the layer's moedn- stop), and is read into the pool in
// the background for the next time (nothing waits for a copy)
static void hybrid_resident(Engine & e, int il, const int32_t * ids, int T) {
    Model & m = e.m;
    Pool & P = e.pool;
    Layer & L = m.layers[il];
    Group & G = P.g[L.grp];
    int32_t * map = L.slots_host();
    const uint64_t pin = ++P.clock;
    if (++P.uses % ((uint64_t) 64 * m.n_layer) == 0)
        for (auto & c : P.freq) c >>= 1;
    const int k = m.n_used;
    std::vector<int> miss_of(m.n_expert, -1);
    CpuExperts & C = e.cpu_x;
    C.miss.clear();
    // --helper-tier: a missing expert the helper card holds is computed there, the others on the CPU; an expert the pool
    // lets go goes down to the helper (as make_resident)
    const bool hp = e.helper && !L.hmap_host.empty();
    Pool & H = e.hpool;
    Group * K = hp ? &H.g[L.grp] : nullptr;
    int32_t * hmap = hp ? L.hmap_host.data() : nullptr;
    const uint64_t hpin = hp ? ++H.clock : 0;
    std::vector<Readers::Job> hjobs;
    auto evicted = [&](int slot) {                           // `slot` taken from its expert: that one down to the helper
        const int old = hp && slot >= 0 ? G.owner[slot] : -1;
        if (old >= 0 && H.slot_of[old] < 0) {
            const int hs = take_slot(H, *K, hpin);
            if (hs >= 0) fill_hslot(e, old, hs, hjobs);
        }
    };
    for (int i = 0; i < k * T; ++i) {                        // the hits first: their slots are pinned
        const int x = ids[i];
        ++P.freq[il * m.n_expert + x];
        const int slot = P.slot_of[il * m.n_expert + x];
        if (hp) hmap[x] = K->zero;
        if (slot >= 0 && G.left[slot].load(std::memory_order_acquire) == 0) {
            ++P.hits; ++e.hyb_hits;
            if (G.by_prefetch[slot]) { ++P.prefetch_hits; G.by_prefetch[slot] = 0; }
            G.used[slot] = pin;
            map[x] = slot;
        }
    }
    std::vector<Readers::Job> jobs;
    // qwen35moe's split (e.hyb_frac, Strata's way): that share of the missing experts - the most used lately first: they
    // stay for the next time - is copied into the pool now and computed by the GPU; the stop waits only for these, while
    // the CPU computes the rest
    std::vector<int> copy_wait;
    if (!m.nemo && e.hyb_frac > 0) {
        std::vector<int> want;
        for (int i = 0; i < k * T; ++i) {
            const int x = ids[i];
            const int slot = P.slot_of[il * m.n_expert + x];
            if ((slot >= 0 && G.used[slot] == pin) || std::find(want.begin(), want.end(), x) != want.end()) continue;
            want.push_back(x);
        }
        std::stable_sort(want.begin(), want.end(), [&](int a, int b) {
            return P.freq[il * m.n_expert + a] > P.freq[il * m.n_expert + b]; });
        const int n_copy = (int) (e.hyb_frac * (float) want.size() + 0.5f);
        std::vector<Readers::Job> now;
        for (int j = 0; j < n_copy; ++j) {
            const int x = want[j];
            int slot = P.slot_of[il * m.n_expert + x];       // being read already (a background fill): waited for
            if (slot < 0) {
                slot = take_slot(P, G, pin);
                if (slot < 0) break;
                fill_slot(e, il, x, slot, now);
                G.by_prefetch[slot] = 0;
            }
            G.used[slot] = pin;
            map[x] = slot;
            copy_wait.push_back(slot);
            ++P.misses; ++e.hyb_copy;
        }
        e.readers.submit(now, true);                         // ahead of the background fills
    }
    for (int i = 0; i < k * T; ++i) {
        const int x = ids[i];
        const int slot = P.slot_of[il * m.n_expert + x];
        if (slot >= 0 && G.used[slot] == pin) continue;      // a hit (or copied in now)
        map[x] = G.zero;
        const int key = il * m.n_expert + x;
        if (hp && H.slot_of[key] >= 0 && K->left[H.slot_of[key]].load(std::memory_order_acquire) == 0) {
            const int hs = H.slot_of[key];                   // on the helper card
            if (hmap[x] != hs) { K->used[hs] = hpin; ++H.freq[key]; ++e.helper_hits; }
            hmap[x] = hs;
            continue;
        }
        ++P.misses; ++e.hyb_miss;
        if (miss_of[x] < 0) {
            miss_of[x] = (int) C.miss.size();
            C.miss.push_back({x, {}});
            static const uint32_t admit = getenv("MOMOE_ADMIT") ? (uint32_t) atoi(getenv("MOMOE_ADMIT")) : 2;   // measured (0-8 within the noise)
            if (slot < 0 && P.freq[il * m.n_expert + x] > admit) {   // read for the next time (an expert used
                const int s2 = take_slot(P, G, pin);         // more than `admit` times lately: the rest stay on the CPU)
                evicted(s2);
                if (s2 >= 0) {
                    fill_slot(e, il, x, s2, jobs);
                    G.by_prefetch[s2] = 0;
                    G.used[s2] = pin - 1;                    // replaceable as any other
                }
            }
        }
        C.miss[miss_of[x]].uses.push_back({i / k, i % k});
    }
    (e.fillers.th.empty() ? e.readers : e.fillers).submit(jobs);
    if (!hjobs.empty()) e.readers.submit(hjobs);
    L.sync_slots();
    if (hp) tset(L.hslot_map, L.hmap_host.data(), 0, ggml_nbytes(L.hslot_map));
    const int n_in = m.nemo ? m.latent : m.n_embd;          // the experts' input: Nemotron's latent, else the hidden state
    C.k = k;
    C.T = T;
    C.out.assign((size_t) T * k * n_in, 0.0f);
    C.outp = C.out.data();
    static const bool skip = getenv("MOMOE_TEST_SKIP_CPU") != nullptr;   // a timing test only: the output is wrong
    if (skip) C.miss.clear();
    if (C.miss.empty()) {
        for (int slot : copy_wait)
            while (G.left[slot].load(std::memory_order_acquire)) { }
        return;
    }
    C.n_in = n_in;
    cpu_parts(m, L, C);
    e.xl_host.resize((size_t) n_in * T);
    ggml_backend_tensor_get(e.xl[il], e.xl_host.data(), 0, e.xl_host.size() * sizeof(float));
    C.launch(e.xl_host.data());
    for (int slot : copy_wait)                               // the split's copies: the GPU computes them after this stop
        while (G.left[slot].load(std::memory_order_acquire)) { }
}

#if defined(MOMOE_CUDA)
// qwen35moe, before a mailbox graph that follows graphs without them (the copies, a prompt): those set a slot table's
// entries for the experts they chose only, so every table is set again from the pool (an expert still being read: the
// zero slot until the boundary after its read), and what the mailboxes had pending is dropped (its slots may be another
// expert's now)
static void mb_rebuild(Engine & e) {
    Model & m = e.m;
    Pool & P = e.pool;
    e.mb_pending.clear();
    e.mb.want.clear();
    for (int il : e.mb.moe) {
        Layer & L = m.layers[il];
        Group & G = P.g[L.grp];
        int32_t * map = L.slots_host();
        for (int x = 0; x < m.n_expert; ++x) {
            const int slot = P.slot_of[il * m.n_expert + x];
            const bool read = slot >= 0 && G.left[slot].load(std::memory_order_acquire) == 0;
            map[x] = read ? slot : G.zero;
            if (slot >= 0 && !read) e.mb_pending.push_back({il, x, slot});
        }
        L.sync_slots();
    }
    e.mb_stale = false;
}

// the CPU's part of one MoE layer of a mailbox graph: the experts the slot table points at the zero slot, from the
// layer's posted input, into its posted outputs (zeros for the GPU's experts); the pool's bookkeeping as hybrid_resident
static void mb_layer(Engine & e, int il, int T) {
    Model & m = e.m;
    Pool & P = e.pool;
    Layer & L = m.layers[il];
    Group & G = P.g[L.grp];
    MbLayer & B = e.mb.layer[il];
    CpuExperts & C = e.cpu_x;
    const int k = m.n_used;
    const int32_t * map = L.slots_host();
    static const uint32_t admit = getenv("MOMOE_ADMIT") ? (uint32_t) atoi(getenv("MOMOE_ADMIT")) : 2;
    const uint64_t pin = ++P.clock;
    if (++P.uses % ((uint64_t) 64 * m.n_layer) == 0)
        for (auto & c : P.freq) c >>= 1;
    static thread_local std::vector<int> miss_of;
    miss_of.assign(m.n_expert, -1);
    C.miss.clear();
    for (int i = 0; i < k * T; ++i) {
        const int x = B.ids[i];
        const int key = il * m.n_expert + x;
        ++P.freq[key];
        const int slot = map[x];
        if (slot != G.zero) {
            ++P.hits; ++e.hyb_hits;
            G.used[slot] = pin;
            continue;
        }
        ++P.misses; ++e.hyb_miss;
        if (miss_of[x] < 0) {
            miss_of[x] = (int) C.miss.size();
            C.miss.push_back({x, {}});
            if (P.slot_of[key] < 0 && P.freq[key] > admit) e.mb.want.push_back({il, x});
        }
        C.miss[miss_of[x]].uses.push_back({i / k, i % k});
    }
    const int n_in = m.nemo ? m.latent : m.n_embd;
    memset(B.out, 0, sizeof(float) * (size_t) T * k * n_in);
    static const bool skip = getenv("MOMOE_TEST_SKIP_CPU") != nullptr;   // a timing test only: the output is wrong
    if (C.miss.empty() || skip) return;
    C.k = k;
    C.T = T;
    C.n_in = n_in;
    cpu_parts(m, L, C);
    C.outp = B.out;
    C.launch(B.x);
    C.wait();
}

static void rmb_layer(Engine & e, int il, int T);
static void rmb_after(Engine & e, int il, int T);

// the worker: per graph, the MoE layers in order, each when the GPU has posted it
static void mb_worker(Engine & e) {
    Mailbox & M = e.mb;
    uint64_t seen = 0;
    for (;;) {
        int T;
        uint32_t cur;
        {
            std::unique_lock<std::mutex> lk(M.mu);
            M.cv.wait(lk, [&] { return M.quit || M.job != seen; });
            if (M.quit) return;
            seen = M.job;
            T = M.T;
            cur = M.cur;
        }
        const std::vector<int> layers = M.graph;            // (copied under the lock: the next graph's list)
        for (int il : layers) {
            MbLayer & B = M.layer[il];
            const double ws = now_s();
            bool told = false;
            while (*B.sent != cur) {
                if (M.quit || *M.timeout) {
                    fprintf(stderr, "momoe-cuda: mailbox worker: layer %d never posted (its flag %u, epoch %u, graph %u)\n",
                            il, *B.sent, *M.epoch, cur);
                    return;
                }
                if (!told && now_s() - ws > 2) {
                    told = true;
                    fprintf(stderr, "momoe-cuda: mailbox worker: waiting 2 s for layer %d (its flag %u, epoch %u, graph %u)\n",
                            il, *B.sent, *M.epoch, cur);
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const double w0 = now_s();
            if (e.rmb) rmb_layer(e, il, T);
            else mb_layer(e, il, T);
            e.t_cpu_wait += now_s() - w0;
            std::atomic_thread_fence(std::memory_order_release);
            *B.done = cur;
            if (e.rmb) rmb_after(e, il, T);                  // (the GPU goes on meanwhile)
        }
        {
            std::lock_guard<std::mutex> lk(M.mu);
            M.done_job = seen;
        }
        M.cv.notify_all();
    }
}

static void mb_init(Engine & e) {
    Model & m = e.m;
    Mailbox & M = e.mb;
    const size_t n_in = m.nemo ? m.latent : m.n_embd;       // the experts' input: Nemotron's latent, else the hidden state
    const size_t x_b = sizeof(float) * n_in * HYB_MAX_T, ids_b = sizeof(int32_t) * (size_t) m.n_used * HYB_MAX_T;
    const size_t out_b = sizeof(float) * n_in * m.n_used * HYB_MAX_T;
    const size_t per = 256 + (x_b + ids_b + out_b + 255) / 256 * 256;
    auto moe = [&](const Layer & L) { return m.nemo ? L.kind == 'e' : L.cpu_dn != nullptr; };   // (qwen35moe: every layer)
    size_t total = 256;
    for (auto & L : m.layers) total += moe(L) ? per : 0;
    cudaSetDevice(e.device);
    void * h = nullptr, * d = nullptr;
    if (cudaHostAlloc(&h, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) die("cannot allocate the mailboxes (mapped host memory)");
    memset(h, 0, total);
    M.host = (char *) h;
    M.dev = (char *) d;
    auto dev = [&](const void * p) { return (void *) (M.dev + ((const char *) p - M.host)); };
    M.epoch = (volatile uint32_t *) M.host;
    M.timeout = (volatile uint32_t *) (M.host + 64);
    M.layer.resize(m.n_layer);
    size_t at = 256;
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        if (!moe(L)) continue;
        MbLayer & B = M.layer[il];
        char * base = M.host + at;
        B.sent = (volatile uint32_t *) base;
        B.done = (volatile uint32_t *) (base + 64);
        B.x = (float *) (base + 256);
        B.ids = (int32_t *) (base + 256 + x_b);
        B.out = (float *) (base + 256 + x_b + ids_b);
        at += per;
        B.send = {MOMOE_MAILBOX_MAGIC, MOMOE_MAILBOX_SEND, 0, 0, dev(B.x), dev(B.ids), (volatile uint32_t *) dev((const void *) B.sent),
                  (const volatile uint32_t *) dev((const void *) M.epoch), (volatile uint32_t *) dev((const void *) M.timeout)};
        B.wait = {MOMOE_MAILBOX_MAGIC, MOMOE_MAILBOX_WAIT, 0, 0, dev(B.out), nullptr, (volatile uint32_t *) dev((const void *) B.done),
                  (const volatile uint32_t *) dev((const void *) M.epoch), (volatile uint32_t *) dev((const void *) M.timeout)};
        M.moe.push_back(il);
        // every expert starts outside the pool: the zero slot
        Group & G = e.pool.g[L.grp];
        std::fill(L.map_host.begin(), L.map_host.end(), G.zero);
        L.sync_slots();
    }
    M.th = std::thread([&e] { mb_worker(e); });
    e.mailbox = true;
    fprintf(stderr, "momoe-cuda: mailboxes for %zu MoE layers (%.1f MiB of mapped memory)\n", M.moe.size(), total / 1048576.0);
}

// Resident mailboxes (glm5-next, --direct-io: no CPU experts): a decode token is one graph with no stops.  At each MoE
// block the GPU posts its chosen experts (and the predictions after them) to the block's mailbox and waits; the worker
// makes the chosen ones resident (the misses read in, the ones still being read waited for), writes their slot
// numbers back, lets the GPU go on, and only then reads the next block's predictions in.  The GPU takes the slots from
// the wait itself: no slot table, and only the worker touches the pool while the graph runs.
static void rmb_layer(Engine & e, int il, int T) {
    Model & m = e.m;
    Pool & P = e.pool;
    Layer & L = m.layers[il];
    Group & G = P.g[L.grp];
    MbLayer & B = e.mb.layer[il];
    const int k = m.n_used;
    const uint64_t pin = ++P.clock;
    if (++P.uses % ((uint64_t) 64 * m.n_layer) == 0)
        for (auto & c : P.freq) c >>= 1;
    int32_t * slots = (int32_t *) B.out;
    std::vector<Readers::Job> jobs;
    std::vector<int> wait;
    for (int i = 0; i < k * T; ++i) {
        const int x = B.ids[i];
        const int key = il * m.n_expert + x;
        ++P.freq[key];
        int slot = P.slot_of[key];
        if (slot >= 0) {
            ++P.hits;
            if (G.by_prefetch[slot]) { ++P.prefetch_hits; G.by_prefetch[slot] = 0; }
            if (G.left[slot].load(std::memory_order_acquire)) { ++P.late; wait.push_back(slot); }
        } else {
            ++P.misses;
            slot = take_slot(P, G, pin);
            while (slot < 0) {                               // every slot this block's or being read: wait for a read
                std::this_thread::sleep_for(std::chrono::microseconds(20));
                slot = take_slot(P, G, pin);
            }
            fill_slot(e, il, x, slot, jobs);
            G.by_prefetch[slot] = 0;
            G.owner[slot] = key;
            P.slot_of[key] = slot;
            wait.push_back(slot);
        }
        G.used[slot] = pin;
        slots[i] = slot;
    }
    const double t0 = now_s();
    const size_t n_jobs = jobs.size();
    (e.demand.th.empty() ? e.readers : e.demand).submit(jobs, true);
    for (int slot : wait) {
        bool told = false;
        while (G.left[slot].load(std::memory_order_acquire)) {
            if (!told && now_s() - t0 > 2) {
                told = true;
                fprintf(stderr, "momoe-cuda: rmb layer %d: slot %d still %d parts to read after 2 s (%zu jobs submitted)\n",
                        il, slot, G.left[slot].load(), n_jobs);
            }
        }
    }
    P.read_s += now_s() - t0;
    static const bool dbg = getenv("MOMOE_RMB_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "RMB layer %d T %d: %zu reads, %zu waited for, %.2f ms\n", il, T, n_jobs, wait.size(), 1000 * (now_s() - t0));
}

// after the GPU went on: the predictions for the next block(s), posted after this block's own experts
static void rmb_after(Engine & e, int il, int T) {
    MbLayer & B = e.mb.layer[il];
    const size_t own = (size_t) e.m.n_used * T, n = (size_t) B.send.n_i32;
    if (n > own && il + 1 < e.m.n_layer) prefetch(e, il + 1, B.ids + own, T);
    if (n > 2 * own && il + 2 < e.m.n_layer) prefetch(e, il + 2, B.ids + 2 * own, T);
}

static void rmb_init(Engine & e) {
    Model & m = e.m;
    Mailbox & M = e.mb;
    const size_t ids_b = sizeof(int32_t) * (size_t) m.n_used * HYB_MAX_T * 3;   // own + two blocks of predictions
    const size_t out_b = sizeof(int32_t) * (size_t) m.n_used * HYB_MAX_T;      // the slots
    const size_t per = 256 + (ids_b + out_b + 255) / 256 * 256;
    size_t total = 256;
    for (auto & L : m.layers) total += L.exp_fd[1] >= 0 ? per : 0;
    cudaSetDevice(e.device);
    void * h = nullptr, * d = nullptr;
    if (cudaHostAlloc(&h, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) die("cannot allocate the mailboxes (mapped host memory)");
    memset(h, 0, total);
    M.host = (char *) h;
    M.dev = (char *) d;
    auto dev = [&](const void * p) { return (void *) (M.dev + ((const char *) p - M.host)); };
    M.epoch = (volatile uint32_t *) M.host;
    M.timeout = (volatile uint32_t *) (M.host + 64);
    M.layer.resize(m.n_layer);
    size_t at = 256;
    for (int il = 0; il < m.n_layer; ++il) {
        if (m.layers[il].exp_fd[1] < 0) continue;            // (a dense block)
        MbLayer & B = M.layer[il];
        char * base = M.host + at;
        B.sent = (volatile uint32_t *) base;
        B.done = (volatile uint32_t *) (base + 64);
        B.ids = (int32_t *) (base + 256);
        B.out = (float *) (base + 256 + ids_b);
        at += per;
        B.send = {MOMOE_MAILBOX_MAGIC, MOMOE_MAILBOX_SEND, 1, 0, dev(B.out), dev(B.ids), (volatile uint32_t *) dev((const void *) B.sent),
                  (const volatile uint32_t *) dev((const void *) M.epoch), (volatile uint32_t *) dev((const void *) M.timeout)};
        B.wait = {MOMOE_MAILBOX_MAGIC, MOMOE_MAILBOX_WAIT, 0, 0, dev(B.out), nullptr, (volatile uint32_t *) dev((const void *) B.done),
                  (const volatile uint32_t *) dev((const void *) M.epoch), (volatile uint32_t *) dev((const void *) M.timeout)};
        M.moe.push_back(il);
    }
    e.rmb = true;
    M.th = std::thread([&e] { mb_worker(e); });
    fprintf(stderr, "momoe-cuda: resident mailboxes for %zu MoE layers (no stops while writing)\n", M.moe.size());
}

static void mb_stop(Engine & e) {
    if (e.rmb) {
        { std::lock_guard<std::mutex> lk(e.mb.mu); e.mb.quit = true; }
        e.mb.cv.notify_all();
        e.mb.th.join();
        return;
    }
    if (!e.mailbox) return;
    { std::lock_guard<std::mutex> lk(e.mb.mu); e.mb.quit = true; }
    e.mb.cv.notify_all();
    e.mb.th.join();
}

// between graphs: the experts read into the pool since are published in the slot tables, and new ones are read in
// (MOMOE_FILLS evictions a token at most, see below): a slot taken from another expert first leaves that expert's table
static void mb_boundary(Engine & e) {
    Model & m = e.m;
    Pool & P = e.pool;
    std::vector<uint8_t> dirty(m.n_layer, 0);
    for (size_t i = 0; i < e.mb_pending.size();) {
        const auto pd = e.mb_pending[i];
        Group & G = P.g[m.layers[pd.il].grp];
        if (G.left[pd.slot].load(std::memory_order_acquire) == 0) {
            m.layers[pd.il].slots_host()[pd.x] = pd.slot;
            dirty[pd.il] = 1;
            e.mb_pending[i] = e.mb_pending.back();
            e.mb_pending.pop_back();
        } else {
            ++i;
        }
    }
    // a free slot is filled at once; an expert put out of its slot counts against MOMOE_FILLS (each read is RAM bandwidth
    // the CPU's experts need) (measured: Nemotron no limit; qwen35moe 16: Ornith-1.5-122B on the RTX 4070 Ti + Ryzen 9 5950X, 300 tokens twice,
    // 18.2 tok/s against 16.4 without one: 15 GB read instead of 99, a third of the chosen experts in the pool instead of
    // 42%, the CPU's experts about as fast as the GPU's there; 8: 18.0, 0: 18.8 but the pool never follows the text)
    static const int max_evict = getenv("MOMOE_FILLS") ? atoi(getenv("MOMOE_FILLS")) : e.m.nemo ? 1 << 30 : 16;
    std::vector<Readers::Job> jobs;
    int n = 0, evicted = 0;
    for (const auto & w : e.mb.want) {
        const int key = w.first * m.n_expert + w.second;
        if (P.slot_of[key] >= 0) continue;
        Layer & L = m.layers[w.first];
        Group & G = P.g[L.grp];
        const int slot = take_slot(P, G, UINT64_MAX);
        if (slot < 0) continue;
        if (G.owner[slot] >= 0 && evicted >= max_evict) {    // (take_slot let it go: it is still read where it is)
            P.slot_of[G.owner[slot]] = slot;
            continue;
        }
        if (G.owner[slot] >= 0) {                            // its expert leaves its table first
            ++evicted;
            const int ok = G.owner[slot], oil = ok / m.n_expert, ox = ok % m.n_expert;
            m.layers[oil].slots_host()[ox] = G.zero;
            dirty[oil] = 1;
            G.owner[slot] = -1;
        }
        fill_slot(e, w.first, w.second, slot, jobs);
        G.by_prefetch[slot] = 0;
        G.used[slot] = P.clock;
        e.mb_pending.push_back({w.first, w.second, slot});
        ++n;
        ++e.mb_fills;
    }
    e.mb.want.clear();
    for (int il = 0; il < m.n_layer; ++il)                   // the tables first: no slot being read is referenced
        if (dirty[il]) m.layers[il].sync_slots();
    (e.fillers.th.empty() ? e.readers : e.fillers).submit(jobs);
}
#endif

static bool eval_cb(ggml_tensor * t, bool ask, void * user) {
    Engine & e = *(Engine *) user;
    int il = -1;
    if (debug_on() && t->name[0] && strncmp(t->name, "topk-", 5) && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)
            && !strstr(t->name, "(")) {                     // MOMOE_DEBUG: the sums llama-eval-callback prints
        if (ask) return true;
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        static const bool last_only = getenv("MOMOE_DEBUG_LAST") != nullptr;   // the last position's row only
        if (last_only) v.erase(v.begin(), v.end() - std::min<size_t>(v.size(), (size_t) t->ne[0]));
        double sum = 0;
        for (float f : v) sum += f;
        fprintf(stderr, "DBG %-28s sum = %f\n", t->name, sum);
        return true;
    }
    if (sscanf(t->name, "moedn-%d", &il) == 1 && !t->view_src) {   // --hybrid: the CPU's experts join here
        if (ask) return true;
        const double w0 = now_s();
        Engine & en = e;
        CpuExperts & C = en.cpu_x;
        if (!C.miss.empty()) C.wait();
        Layer & L = en.m.layers[il];
        ggml_backend_tensor_set(L.cpu_dn, C.out.data(), 0, C.out.size() * sizeof(float));
        en.t_cpu_wait += now_s() - w0;
        return true;
    }
    if (sscanf(t->name, "topk-%d", &il) != 1 || t->view_src) return ask ? false : true;   // (not "topk-N (view)")
    if (ask) return true;
    const double t0 = now_s();
    ++e.pool.calls;
    const size_t n = ggml_nelements(t);
    e.ids_host.resize(n);
    ggml_backend_tensor_get(t, e.ids_host.data(), 0, n * sizeof(int32_t));
    const int T = e.batch_tokens;
    const size_t own = (size_t) e.m.n_used * T;
    static const bool segtime = getenv("MOMOE_SEGTIME") != nullptr;   // a test: GPU time between stops, resident time
    static double t_last = 0;
    if (segtime && T >= 32) fprintf(stderr, "SEG layer %d: gpu %.1f ms", il, 1000 * (t0 - t_last));
    if (hybrid_on(e, e.m.layers[il], T)) hybrid_resident(e, il, e.ids_host.data(), T);
    else make_resident(e, il, e.ids_host.data(), T);
    if (segtime && T >= 32) fprintf(stderr, ", resident %.1f ms\n", 1000 * (now_s() - t0));
    t_last = now_s();
    // the predictions after this layer's own choice: read in the background while the GPU goes on (the CUDA readers
    // copy on their own streams)
    if (!kHostMem && n > own) {
        const int nx = e.m.nemo ? e.m.layers[il].next_moe : il + 1;
        if (nx >= 0 && nx < e.m.n_layer) prefetch(e, nx, e.ids_host.data() + own, T);
        if (!e.m.nemo && n > 2 * own && il + 2 < e.m.n_layer) prefetch(e, il + 2, e.ids_host.data() + 2 * own, T);
    }
    e.pool.cb_s += now_s() - t0;
    return true;
}

// --profile P: Strata's expert profile ("STRP", version 1: the (layer, expert) pairs ranked hottest first); the
// hottest pairs fill a fixed part of each group at the start and are never replaced (MOMOE_PIN_FRAC of the slots,
// default 0.6); the rest of the pool is replaced as usual
static void pin_profile(Engine & e, const char * path, double frac) {
    Model & m = e.m;
    Pool & P = e.pool;
    FILE * f = fopen(path, "rb");
    if (!f) die("cannot open the profile %s", path);
    char magic[4];
    uint32_t h[5];
    if (fread(magic, 1, 4, f) != 4 || fread(h, 4, 5, f) != 5 || memcmp(magic, "STRP", 4) || h[0] != 1) die("%s is not a version 1 profile", path);
    if ((int) h[1] < m.n_layer || (int) h[2] != m.n_expert) die("the profile %s is for another model", path);
    std::vector<uint16_t> raw((size_t) h[4] * 2);
    if (fread(raw.data(), 2, raw.size(), f) != raw.size()) die("the profile %s is truncated", path);
    fclose(f);
    const double t0 = now_s();
    std::vector<Readers::Job> jobs;
    std::vector<int> n_pin(P.g.size(), 0);
    int pinned = 0;
    for (int i = 0; i < (int) h[4]; ++i) {
        const int il = raw[(size_t) i * 2], x = raw[(size_t) i * 2 + 1];
        if (il >= m.n_layer || P.slot_of[il * m.n_expert + x] >= 0) continue;
        const int gi = m.layers[il].grp;
        Group & G = P.g[gi];
        if (n_pin[gi] >= (int) (G.n_slots * frac)) continue;
        const int slot = n_pin[gi]++;
        fill_slot(e, il, x, slot, jobs);
        G.used[slot] = UINT64_MAX;                           // never replaced
        ++pinned;
    }
    e.readers.submit(jobs);
    for (size_t gi = 0; gi < P.g.size(); ++gi)
        for (int i = 0; i < n_pin[gi]; ++i)
            while (P.g[gi].left[i].load(std::memory_order_acquire)) { }
    P.bytes_read = 0;
    fprintf(stderr, "momoe-metal: %d of the profile's hottest experts pinned (%.0f%% of the pool) in %.1f s\n", pinned,
            100.0 * pinned / P.n_slots, now_s() - t0);
}

// --preload, and a pool that holds every expert: every expert read into its own slot at the start (by the readers),
// the tables filled; the graphs then run without pauses or callback (nothing can be missing)
static void preload(Engine & e) {
    Model & m = e.m;
    Pool & P = e.pool;
    if (P.n_slots < m.n_exp_all) die("--preload needs a pool that holds every expert");
    const double t0 = now_s();
    std::vector<int> next(P.g.size(), 0);
    std::vector<Readers::Job> jobs;
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        if (L.exp_fd[1] < 0) continue;
        for (int x = 0; x < L.n_exp; ++x) {
            const int slot = next[L.grp]++;
            fill_slot(e, il, x, slot, jobs);
            L.slots_host()[x] = slot;
        }
        L.sync_slots();
        e.readers.submit(jobs);
    }
    for (auto & G : P.g)
        for (int i = 0; i < G.n_slots; ++i)
            while (G.left[i].load(std::memory_order_acquire)) { }
    P.bytes_read = 0;
    e.preloaded = true;
    fprintf(stderr, "momoe-metal: every expert loaded in %.1f s (no pauses)\n", now_s() - t0);
}

// ------------------------------------------------------------------------------------------------ the graph
struct Inputs {
    ggml_tensor *tokens, *pos, *mask;
    ggml_tensor *img = nullptr, *img_keep = nullptr;         // pictures: F32 [n_embd, T] rows (0 on text), [1, T] 1 on text
    ggml_tensor *mask_swa = nullptr;                         // flint: the sliding window's mask [swa_rows, T]
    ggml_tensor *kv_idx_swa = nullptr;                       // flint: I64 [T], the ring rows written (p % swa_rows)
    ggml_tensor *kv_idx = nullptr;                           // I64 [T]: the positions written (a graph reused token
                                                             // after token: the caches' rows as an input, not offsets)
    // QSA (llama.cpp's llm_graph_input_kpool, one sequence: a position's cell is its position): the blocks completed
    // by this batch (j_first.., n_new) are pooled; with more than top_k / kpool blocks the selection runs (sel)
    ggml_tensor *new_pos = nullptr;                          // I32 [4 * n_new]: rope position of each new block
    ggml_tensor *pool_idxs = nullptr;                        // F32 [kpool, n_pool]: member cells (n_kv: padding)
    ggml_tensor *pool_mask = nullptr;                        // F16 [n_pool, T]: blocks finished at or before a token
    ggml_tensor *tail_idxs = nullptr;                        // F32 [kpool - 1, T]: its unfinished block (n_kv: none)
                                                             // (cell numbers as f32: Metal casts f32 -> i32, not back)
    int j_first = 0, n_new = 0, n_pool = 0;
    bool sel = false;
    ggml_tensor *pool_src = nullptr;                         // a reused graph: I32 [kpool * n_new], the new blocks' rows
    ggml_tensor *pool_dst = nullptr;                         // in the raw cache; I64 [n_new], their rows in the pool
    ggml_tensor *sel_mask = nullptr;                         // glm5-next: F32 [n_sel, T]: 0 for a token's live slots
                                                             // (its first min(visible, n_top) pools by score), -inf
    int n_sel = 0;
    // deepseek41, per compress ratio r (1, 2): the compressed rows a token sees (F16 [ds_nc, T]: (p + 1) / r of them);
    // the groups this batch completes (ds_nb, at least one: a dummy writes the cache's last row): the rows written (I64),
    // their RoPE positions (I32, the group's first), the r rows read (I32 [r * ds_nb] into [the state's r rows | this
    // batch's]); the unfinished group's rows kept (ds_np: I64 state rows, I32 batch rows).  The candidate blocks' pin
    // (F32 [n_blk, T]: 1e30 on the block of a token's newest row; only past cand_blocks blocks); the Engram rows
    ggml_tensor *ds_cmask[3] = {}, *ds_widx[3] = {}, *ds_wpos[3] = {}, *ds_ridx[3] = {}, *ds_pdst[3] = {}, *ds_psrc[3] = {};
    int ds_nc[3] = {}, ds_nb[3] = {}, ds_np[3] = {};
    ggml_tensor *ds_pin = nullptr;
    std::vector<ggml_tensor *> ds_eg;
};

static ggml_tensor * norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// a batch's rows (w values each, contiguous) into a cache at the positions of in.kv_idx (a reused graph)
static void cache_rows(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cache, ggml_tensor * src, int64_t w, int T,
                       const Inputs & in) {
    if (ggml_nelements(cache) % w || !ggml_is_contiguous(cache)) die("a cache does not hold whole rows (cache_rows)");
    ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, cache, w, ggml_nelements(cache) / w),
                                                ggml_reshape_2d(c, ggml_cont(c, src), w, T), in.kv_idx));
}

static ggml_tensor * l2_norm(ggml_context * c, ggml_tensor * x, float eps) {   // llama.cpp's build_gdn_l2_norm
    const float n = (float) x->ne[0];
    return ggml_scale(c, ggml_rms_norm(c, x, eps / n), 1.0f / sqrtf(n));
}

// QSA's indexer (llama.cpp's build_qsa_sel): this batch's raw keys into the cache, the blocks it completes pooled
// (mean of the raw keys, normed, rotated to the block's first position); with sel, the attention mask that keeps each
// token's top_k / kpool best scoring blocks and its unfinished block
static ggml_tensor * qsa_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                               const Inputs & in, int p0, int T, int n_kv) {
    Model & m = e.m;
    const int64_t D = m.idx_dim, kp = m.kpool;
    int sections[4] = {m.sections[0], m.sections[1], m.sections[2], m.sections[3]};
    ggml_tensor * k_raw = ggml_mul_mat(c, L.idx_k, x);                                 // [D, T]
    const size_t rr = L.idx_raw->nb[1];
    if (in.kv_idx) cache_rows(c, gf, L.idx_raw, k_raw, D, T, in);
    else ggml_build_forward_expand(gf, ggml_cpy(c, k_raw, ggml_view_2d(c, L.idx_raw, D, T, rr, rr * p0)));
    if (in.n_new > 0) {
        ggml_tensor * rows = in.pool_src                     // (a reused graph: the rows by an input; f16 -> f32 either way)
            ? ggml_reshape_3d(c, ggml_get_rows(c, L.idx_raw, in.pool_src), D, kp, in.n_new)
            : ggml_view_3d(c, L.idx_raw, D, kp, in.n_new, rr, rr * kp, rr * kp * in.j_first);
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < kp; ++i) {
            ggml_tensor * sl = ggml_view_2d(c, rows, D, in.n_new, rows->nb[2], i * rows->nb[1]);
            if (sl->type != GGML_TYPE_F32) sl = ggml_cast(c, sl, GGML_TYPE_F32);
            pooled = pooled ? ggml_add(c, pooled, sl) : sl;
        }
        pooled = norm(c, ggml_scale(c, pooled, 1.0f / kp), L.idx_k_norm, m.rms_eps);
        pooled = ggml_reshape_3d(c, pooled, D, 1, in.n_new);
        pooled = ggml_rope_multi(c, pooled, in.new_pos, nullptr, m.n_rot, sections, GGML_ROPE_TYPE_IMROPE, 262144,
                                 m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        const size_t pr = L.idx_pool->nb[1];
        if (in.pool_dst)
            ggml_build_forward_expand(gf, ggml_set_rows(c, L.idx_pool, ggml_reshape_2d(c, pooled, D, in.n_new), in.pool_dst));
        else
            ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_2d(c, pooled, D, in.n_new),
                                                   ggml_view_2d(c, L.idx_pool, D, in.n_new, pr, pr * in.j_first)));
    }
    if (!in.sel) return nullptr;
    ggml_tensor * pooled = ggml_view_3d(c, L.idx_pool, D, 1, in.n_pool, L.idx_pool->nb[1], L.idx_pool->nb[1], 0);
    ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, L.idx_q, x), D, m.idx_heads, T);
    q = norm(c, q, L.idx_q_norm, m.rms_eps);
    q = ggml_rope_multi(c, q, in.pos, nullptr, m.n_rot, sections, GGML_ROPE_TYPE_IMROPE, 262144, m.rope_base, 1.0f,
                        0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * wts = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, m.idx_heads, T), 1.0f / sqrtf((float) D));
    ggml_tensor * score = ggml_lightning_indexer(c, q, pooled, wts, in.pool_mask);         // [n_pool, T]
    const int64_t n_pool = in.n_pool, n_top = std::min<int64_t>(n_pool, m.idx_top_k / kp);
    ggml_tensor * top = ggml_top_k(c, score, n_top);                                       // [n_top, T], unordered
    ggml_tensor * sel_f = ggml_get_rows(c, in.pool_idxs, ggml_reshape_1d(c, top, n_top * T));
    sel_f = ggml_concat(c, ggml_reshape_2d(c, sel_f, kp * n_top, T), in.tail_idxs, 0);     // [n_sel, T] cells
    const int64_t n_sel = sel_f->ne[0];
    // zeros for the selected cells in an all -inf row; a dead slot (an invisible block, a missing tail cell) goes to
    // its own dump row n_kv + slot, so no two slots write one cell
    ggml_tensor * mask_all = ggml_fill(c, ggml_new_tensor_4d(c, GGML_TYPE_F16, n_kv + n_sel, 1, 1, 1), -INFINITY);
    mask_all = ggml_reshape_3d(c, ggml_repeat_4d(c, mask_all, n_kv + n_sel, T, 1, 1), 1, n_kv + n_sel, T);
    ggml_tensor * zeros = ggml_fill(c, ggml_new_tensor_4d(c, GGML_TYPE_F16, n_sel, 1, 1, 1), 0.0f);
    zeros = ggml_reshape_3d(c, ggml_repeat_4d(c, zeros, n_sel, T, 1, 1), 1, n_sel, T);
    ggml_tensor * top_score = ggml_get_rows(c, ggml_reshape_3d(c, score, 1, n_pool, T), top);   // [1, n_top, T]
    ggml_tensor * live_pool = ggml_clamp(c, ggml_scale_bias(c, top_score, 1.0f, 1.0f), 0.0f, 1.0f);
    live_pool = ggml_reshape_2d(c, ggml_repeat_4d(c, live_pool, kp, n_top, T, 1), kp * n_top, T);
    ggml_tensor * live_tail = ggml_clamp(c, ggml_scale_bias(c, in.tail_idxs, -1.0f, (float) n_kv), 0.0f, 1.0f);
    ggml_tensor * live = ggml_concat(c, live_pool, live_tail, 0);                           // [n_sel, T]
    ggml_tensor * dump = ggml_scale_bias(c, ggml_cumsum(c, ggml_fill(c, live, 1.0f)), 1.0f, (float) (n_kv - 1));
    ggml_tensor * idx_f = ggml_add(c, ggml_mul(c, ggml_sub(c, sel_f, dump), live), dump);
    ggml_tensor * sel_idx = ggml_cast(c, idx_f, GGML_TYPE_I32);
    ggml_tensor * sel = ggml_set_rows(c, mask_all, zeros, ggml_reshape_3d(c, sel_idx, n_sel, T, 1));
    sel = ggml_view_2d(c, sel, n_kv, T, sel->nb[2], 0);
    return ggml_add(c, sel, in.mask);
}

static ggml_tensor * attn_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                const Inputs & in, int p0, int T, int n_kv) {
    Model & m = e.m;
    const int hd = m.head_dim;
    ggml_tensor * qsa_mask = L.idx_k ? qsa_layer(e, c, gf, L, x, in, p0, T, n_kv) : nullptr;
    ggml_tensor * qf = ggml_mul_mat(c, L.wq, x);                       // [hd*2*n_head, T]: query and gate
    ggml_tensor * kc = ggml_mul_mat(c, L.wk, x);
    ggml_tensor * vc = ggml_mul_mat(c, L.wv, x);
    const size_t es = ggml_element_size(qf);
    ggml_tensor * q = ggml_view_3d(c, qf, hd, m.n_head, T, es * hd * 2, es * hd * 2 * m.n_head, 0);
    q = norm(c, q, L.q_norm, m.rms_eps);
    ggml_tensor * k = norm(c, ggml_reshape_3d(c, kc, hd, m.n_head_kv, T), L.k_norm, m.rms_eps);
    ggml_tensor * gate = ggml_view_3d(c, qf, hd, m.n_head, T, es * hd * 2, es * hd * 2 * m.n_head, es * hd);
    gate = ggml_cont_2d(c, gate, (int64_t) hd * m.n_head, T);
    ggml_tensor * v = ggml_reshape_3d(c, vc, hd, m.n_head_kv, T);
    int sections[4] = {m.sections[0], m.sections[1], m.sections[2], m.sections[3]};
    q = ggml_rope_multi(c, q, in.pos, nullptr, m.n_rot, sections, GGML_ROPE_TYPE_IMROPE, 262144, m.rope_base, 1.0f,
                        0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_multi(c, k, in.pos, nullptr, m.n_rot, sections, GGML_ROPE_TYPE_IMROPE, 262144, m.rope_base, 1.0f,
                        0.0f, 1.0f, 32.0f, 1.0f);
    // this batch's K and V into the cache, then attention over the cache up to n_kv
    const size_t row = ggml_row_size(GGML_TYPE_F16, (int64_t) hd * m.n_head_kv);
    if (in.kv_idx) {
        cache_rows(c, gf, L.k_cache, k, (int64_t) hd * m.n_head_kv, T, in);
        cache_rows(c, gf, L.v_cache, v, (int64_t) hd * m.n_head_kv, T, in);
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, T,
                                                                   L.k_cache->nb[1], row, row * p0)));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, T,
                                                                   L.v_cache->nb[1], row, row * p0)));
    }
    ggml_tensor * kv_k = ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, n_kv, L.k_cache->nb[1], row, 0);
    ggml_tensor * kv_v = ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, n_kv, L.v_cache->nb[1], row, 0);
    ggml_tensor * qp = ggml_permute(c, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(c, kv_k, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(c, kv_v, 0, 2, 1, 3);
    ggml_tensor * o = ggml_flash_attn_ext(c, qp, kp, vp, qsa_mask ? qsa_mask : in.mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    o = ggml_reshape_2d(c, o, (int64_t) hd * m.n_head, T);
    o = ggml_mul(c, o, ggml_sigmoid(c, gate));
    return ggml_mul_mat(c, L.wo, o);
}

static ggml_tensor * delta_net_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x, int T,
                                     bool snap) {
    Model & m = e.m;
    const int hk = m.ssm_state, nk = m.ssm_groups, nv = m.ssm_dt_rank, hv = m.ssm_inner / nv;
    const int conv_ch = m.ssm_inner + 2 * nk * hk;
    ggml_tensor * qkv0 = ggml_mul_mat(c, L.wqkv, x);
    ggml_set_name(qkv0, "linear_attn_qkv_mixed");
    ggml_tensor * qkv = ggml_reshape_3d(c, qkv0, conv_ch, T, 1);
    ggml_tensor * z = ggml_mul_mat(c, L.wgate, x);
    ggml_set_name(z, "z");
    ggml_tensor * beta = ggml_sigmoid(c, ggml_reshape_4d(c, ggml_mul_mat(c, L.ssm_beta, x), 1, nv, T, 1));
    ggml_tensor * alpha = ggml_reshape_3d(c, ggml_mul_mat(c, L.ssm_alpha, x), nv, T, 1);
    ggml_tensor * g = ggml_mul(c, ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt)), L.ssm_a);
    g = ggml_reshape_4d(c, g, 1, nv, T, 1);
    // the short convolution over the previous (conv-1) inputs and this batch
    ggml_tensor * cst = ggml_reshape_3d(c, L.conv_state, m.ssm_conv - 1, conv_ch, 1);
    ggml_tensor * cin = ggml_concat(c, cst, ggml_transpose(c, qkv), 0);     // [conv-1+T, conv_ch, 1]
    ggml_tensor * last = ggml_view_3d(c, cin, m.ssm_conv - 1, conv_ch, 1, cin->nb[1], cin->nb[2],
                                      ggml_row_size(cin->type, T));
    ggml_build_forward_expand(gf, ggml_cpy(c, last, cst));
    if (snap)                                                // the window's inputs: a rollback cuts the state from them
        ggml_build_forward_expand(gf, ggml_cpy(c, cin, ggml_view_3d(c, L.conv_snap, m.ssm_conv - 1 + T, conv_ch, 1,
                ggml_row_size(GGML_TYPE_F32, m.ssm_conv - 1 + T), ggml_row_size(GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1 + T) * conv_ch), 0)));
    ggml_tensor * co = ggml_silu(c, ggml_ssm_conv(c, cin, L.conv1d));      // [conv_ch, T, 1]
    ggml_set_name(co, "conv_output_silu");
    const size_t nb1 = ggml_row_size(co->type, conv_ch);
    ggml_tensor * q = ggml_view_4d(c, co, hk, nk, T, 1, ggml_row_size(co->type, hk), nb1, nb1 * T, 0);
    ggml_tensor * k = ggml_view_4d(c, co, hk, nk, T, 1, ggml_row_size(co->type, hk), nb1, nb1 * T,
                                   ggml_row_size(co->type, (int64_t) hk * nk));
    ggml_tensor * v = ggml_view_4d(c, co, hv, nv, T, 1, ggml_row_size(co->type, hv), nb1, nb1 * T,
                                   ggml_row_size(co->type, (int64_t) 2 * hk * nk));
    q = l2_norm(c, q, m.rms_eps);
    ggml_set_name(q, "q_conv_predelta");
    k = l2_norm(c, k, m.rms_eps);
    ggml_tensor * s = ggml_reshape_4d(c, L.ssm_state, hv, hv, nv, 1);
    ggml_tensor * r = ggml_gated_delta_net(c, q, k, v, g, beta, s, snap ? T : 1);   // snap: a state per token
    ggml_tensor * out = ggml_view_4d(c, r, hv, nv, T, 1, ggml_row_size(r->type, hv), ggml_row_size(r->type, (int64_t) hv * nv),
                                     ggml_row_size(r->type, (int64_t) hv * nv * T), 0);
    ggml_tensor * ns = ggml_view_4d(c, r, hv, hv, nv, 1, ggml_row_size(r->type, hv), ggml_row_size(r->type, (int64_t) hv * hv),
                                    ggml_row_size(r->type, (int64_t) hv * hv * nv), ggml_row_size(r->type, (int64_t) hv * nv * T));
    // into the state as one row [hv*hv*nv]: the shape ggml-metal's GATED_DELTA_NET + CPY fusion takes (the kernel then
    // writes the state in place, no copy after it)
    ggml_build_forward_expand(gf, ggml_cpy(c, ns, ggml_reshape_4d(c, L.ssm_state, (int64_t) hv * hv * nv, 1, 1, 1)));
    if (snap) {                                              // every token's state, most recent first
        const int64_t D = (int64_t) hv * hv * nv;
        ggml_tensor * all = ggml_view_2d(c, r, D, T, ggml_row_size(r->type, D), ggml_row_size(r->type, (int64_t) hv * nv * T));
        ggml_build_forward_expand(gf, ggml_cpy(c, all, ggml_view_2d(c, L.ssm_snap, D, T, L.ssm_snap->nb[1], 0)));
    }
    ggml_tensor * z4 = ggml_reshape_4d(c, z, hv, nv, T, 1);
    ggml_tensor * o = ggml_mul(c, norm(c, out, L.ssm_norm, m.rms_eps), m.qwen4 ? ggml_sigmoid(c, z4) : ggml_silu(c, z4));
    o = ggml_reshape_2d(c, o, (int64_t) hv * nv, T);
    return ggml_mul_mat(c, L.ssm_out, o);
}

// qwen4exp's hyper-connections: the residual is hc streams [n_embd, hc, T]. The mix (llama.cpp's build_hc_mix): a
// grouped RMS norm (one stream at a time, the [n_embd, hc] gamma), a low-rank gate (down, silu of 1/hc of it, up), the
// streams gated by its sigmoid and averaged into the block's input [n_embd, T]; `inject` gets the [hc, T] weights the
// block's output is added back to each stream with (build_hc_combine: 2 sigmoid(inject / hc))
static ggml_tensor * hc_mix(Engine & e, ggml_context * c, ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down,
                            ggml_tensor * w_up, ggml_tensor * w_inject, ggml_tensor ** inject) {
    Model & m = e.m;
    const int64_t T = x->ne[2];
    ggml_tensor * xn = ggml_mul(c, ggml_rms_norm(c, x, m.rms_eps), ggml_reshape_2d(c, w_norm, m.n_embd, m.hc));
    xn = ggml_reshape_2d(c, xn, (int64_t) m.n_embd * m.hc, T);
    ggml_tensor * lo = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, w_down, xn), 1.0f / m.hc));
    ggml_tensor * gate = ggml_mul_mat(c, w_up, lo);
    ggml_tensor * mixed = ggml_dsv4_hc_pre_gated(c, ggml_reshape_3d(c, xn, m.n_embd, m.hc, T),
                                                 ggml_reshape_3d(c, gate, m.n_embd, m.hc, T), 1.0f / m.hc);
    if (inject) *inject = ggml_mul_mat(c, w_inject, xn);
    return mixed;
}

static ggml_tensor * hc_combine(Engine & e, ggml_context * c, ggml_tensor * res, ggml_tensor * out, ggml_tensor * inject) {
    ggml_tensor * w = ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, inject, 1.0f / e.m.hc)), 2.0f);
    return ggml_dsv4_hc_post(c, out, res, w, nullptr);
}

// qwen4exp's n-gram embedding (llama.cpp's build_ple): `emb` [ple_dim * ple_heads, T] (the table's rows, read on the
// CPU) gated against each stream of the residual, added to it with a dilated causal convolution of the gated value
// (its history: the layer's ple_state)
static ggml_tensor * ple_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * emb,
                               ggml_tensor * hidden, int T) {
    Model & m = e.m;
    const int64_t hc_dim = (int64_t) m.hc * m.n_embd;
    ggml_tensor * key = ggml_mul_mat(c, L.ple_key, emb);
    ggml_tensor * value = ggml_mul_mat(c, L.ple_value, emb);
    auto gnorm = [&](ggml_tensor * x, ggml_tensor * w) {
        return ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, x, m.n_embd, m.hc, T), m.rms_eps),
                        ggml_reshape_2d(c, w, m.n_embd, m.hc));
    };
    key = gnorm(key, L.ple_norm_key);
    ggml_tensor * query = gnorm(hidden, L.ple_norm_query);
    ggml_tensor * sc = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, key, query)), 1.0f / sqrtf((float) m.n_embd));
    ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, sc), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, sc), mag));               // [1, hc, T]
    ggml_tensor * v3 = ggml_repeat_4d(c, ggml_reshape_3d(c, value, m.n_embd, 1, T), m.n_embd, m.hc, T, 1);
    ggml_tensor * gated = ggml_mul(c, v3, gate);
    ggml_tensor * nrm = ggml_reshape_2d(c, gnorm(ggml_reshape_2d(c, gated, hc_dim, T), L.ple_norm_conv), hc_dim, T);
    // out[ch, t] = sum_k w[k, ch] * x[ch, t - (kernel-1-k) * ngram], over the history and this batch
    const int64_t kern = m.ple_kernel, dil = m.ple_ngram, hist = (kern - 1) * dil;
    ggml_tensor * st = ggml_reshape_2d(c, L.ple_state, hist, hc_dim);
    ggml_tensor * padded = ggml_concat(c, st, ggml_transpose(c, nrm), 0);               // [hist + T, hc_dim]
    ggml_tensor * tail = ggml_view_2d(c, padded, hist, hc_dim, padded->nb[1], ggml_row_size(padded->type, T));
    ggml_build_forward_expand(gf, ggml_cpy(c, tail, st));
    ggml_tensor * co = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * sh = ggml_cont(c, ggml_transpose(c, ggml_view_2d(c, padded, T, hc_dim, padded->nb[1],
                                                                         ggml_row_size(padded->type, start))));
        ggml_tensor * wk = ggml_cont(c, ggml_view_2d(c, L.ple_conv1d, 1, hc_dim, L.ple_conv1d->nb[1], k * L.ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(c, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(c, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(c, sh, wk);
        co = co ? ggml_add(c, co, term) : term;
    }
    co = ggml_reshape_3d(c, ggml_cont(c, ggml_silu(c, co)), m.n_embd, m.hc, T);
    ggml_format_name(co, "ple_conv_out-%d", m.ple_layer);
    return ggml_add(c, hidden, ggml_add(c, gated, co));
}

// the PLE rows of T tokens at p0.. (e.tok_at holds the earlier ones), read from the table and converted to f32:
// [T, ple_heads * ple_dim]. Each token hashes the n-grams ending at it (2..ngram tokens); an EOS before it cuts its
// context, a position before the start reads as EOS (llama.cpp's llm_graph_input_qwen4exp_ple)
static void ple_rows(Engine & e, const int32_t * toks, int T, int p0, std::vector<float> & out) {
    Model & m = e.m;
    const int n = m.ple_ngram, H = m.ple_heads;
    out.resize((size_t) T * H * m.ple_dim);
    std::vector<char> raw(m.ple_row_bytes);
    const auto * tr = ggml_get_type_traits(m.ple_type);
    std::vector<int64_t> ctx(n);
    for (int i = 0; i < T; ++i) {
        ctx[0] = toks[i];
        bool cut = false;
        for (int s = 1; s < n; ++s) {
            const int pos = p0 + i - s;
            const int64_t t = cut || pos < 0 ? -1 : pos >= p0 ? toks[pos - p0] : e.tok_at[pos];
            cut = cut || t < 0 || t == m.ple_eos;
            ctx[s] = cut ? m.ple_eos : t;
        }
        for (int g = 2; g <= n; ++g) {
            uint64_t mixed = (uint64_t) ctx[0] * m.ple_mult[0];
            for (int j = 1; j < g; ++j) mixed ^= (uint64_t) ctx[j] * m.ple_mult[j];
            for (int q = 0; q < m.ple_per_gram; ++q) {
                const int h = (g - 2) * m.ple_per_gram + q;
                const uint64_t row = mixed % m.ple_vocab[h] + m.ple_off[h];
                if ((int64_t) row >= m.ple_rows) die("a PLE row is outside the table");
                read_at(m.ple_fd, raw.data(), m.ple_row_bytes, m.ple_table_off + row * m.ple_row_bytes);
                tr->to_float(raw.data(), out.data() + ((size_t) i * H + h) * m.ple_dim, m.ple_dim);
            }
        }
    }
}

// a batch that may send several of a token's experts to the zero slot (--hybrid, --helper-tier; not the staging copy:
// every expert is there): CUDA's batch kernels are told (MOMOE_MMID_REPEATS), or they computed only a token's worth of
// columns for it and left the rest as they were (measured: Lightning's 9-16-token batches, wrong tokens or a crash)
static ggml_tensor * may_repeat(const Engine & e, bool big, ggml_tensor * t) {
#if defined(MOMOE_CUDA)
    if (!big && (e.hybrid || e.helper)) ((int32_t *) t->op_params)[0] = (int32_t) MOMOE_MMID_REPEATS;
#else
    (void) e; (void) big;
#endif
    return t;
}

// --helper-tier (not a staged batch): the chosen experts the helper card holds, computed there through its slot table
// (its slot of zeros for the others; the main table points at the main pool's for these): their down projections, to
// add to the main card's.  act(gate or nullptr, up): the model's activation
template <class Act>
static ggml_tensor * helper_down(Engine & e, ggml_context * c, Layer & L, ggml_tensor * x3, ggml_tensor * ids, int T,
                                 Act act) {
    const int k = e.m.n_used;
    Group & K = e.hpool.g[L.grp];
    ggml_tensor * hs = ggml_reshape_2d(c, ggml_get_rows(c, L.hslot_map, ggml_reshape_1d(c, ids, (int64_t) k * T)), k, T);
    ggml_tensor * up = may_repeat(e, false, ggml_mul_mat_id(c, K.t[1], x3, hs));
    ggml_tensor * gt = K.t[0] ? may_repeat(e, false, ggml_mul_mat_id(c, K.t[0], x3, hs)) : nullptr;
    return may_repeat(e, false, ggml_mul_mat_id(c, K.t[2], act(gt, up), hs));
}

static bool helper_on(const Engine & e, const Layer & L, bool big) { return e.helper && !big && L.hslot_map; }


static ggml_tensor * moe_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x, int T,
                               ggml_tensor * pred_in, ggml_tensor * pred_in2) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    if (hybrid_on(e, L, T)) {                                // --hybrid: the experts' input, read at the layer's stop
        if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);     // (the CPU computes the missing experts from it)
        ggml_build_forward_expand(gf, x);                    // or posted to its mailbox
        e.xl[il] = x;
    }
#if defined(MOMOE_CUDA)
    const bool mbx = mb_on(e) && hybrid_on(e, L, T) && ns_paused(e, il);   // the mailboxes: no stops in this layer
    ggml_tensor * send = nullptr;
#else
    const bool mbx = false;
#endif
    ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, x);
    ggml_format_name(lg, "ffn_moe_logits-%d", il);
    ggml_tensor * probs = ggml_soft_max(c, lg);                                         // [n_expert, T]
    const bool masked = !ns_paused(e, il);
    ggml_tensor * top = nullptr, * ids = nullptr, * w = nullptr;
    bool direct = false;                                     // the choice straight to the experts (below)
    if (!masked) {
        // llama.cpp's order, so that the backends fuse the routing as there (CUDA's and Metal's topk-moe kernels): the
        // choice and its normalized weights first, then the next layers' predicted experts (see below) beside it
        ggml_tensor * chosen = ggml_argsort_top_k(c, probs, m.n_used);                 // [n_used, T] (a view)
        w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), chosen);      // [1, n_used, T]
        w = ggml_reshape_2d(c, w, m.n_used, T);
        w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        w = ggml_reshape_3d(c, w, 1, m.n_used, T);
        ggml_build_forward_expand(gf, w);
        // every expert in its slot and nothing reading the choice on the CPU (no pause, no callback): the view itself,
        // as llama.cpp passes it, and the layer's experts as a view of the pool (no copy of the choice, no slot lookup:
        // two small kernels a layer, 80 a token for Qwen3.6). MOMOE_NO_DIRECT=1: as before. (With more than 8 experts a
        // token, ggml-metal sums them in a different order then: the same tokens, probabilities a little apart)
        // (momoe-metal only: on momoe-cuda, with its CUDA graphs, it measured no faster, and its tokens parted)
        static const bool force_cb = getenv("MOMOE_FORCE_CB") != nullptr || getenv("MOMOE_NO_DIRECT") != nullptr;
        direct = !kHostMem ? false : e.preloaded && !e.pause && !e.ns_graph && !e.hybrid && !e.helper && !debug_on() && !force_cb && !pred_in
                 && !pred_in2;
        if (direct) {
            ids = chosen;
            e.topk[il] = nullptr;
        } else {
            top = ggml_cont(c, chosen);
            for (int j = 1; j <= 2; ++j) {
                ggml_tensor * in = j == 1 ? pred_in : pred_in2;
                if (!in) continue;
                ggml_tensor * pl = ggml_soft_max(c, ggml_mul_mat(c, m.layers[il + j].gate_inp, in));
                top = ggml_concat(c, top, ggml_cont(c, ggml_argsort_top_k(c, pl, m.n_used)), 1);
            }
            ggml_format_name(top, mbx ? "mbtopk-%d" : "topk-%d", il);   // made resident here: a pause point (or the
            if (e.ns_graph) ggml_set_output(top);                                     // callback); mbtopk-: not
            e.topk[il] = top;
            ggml_build_forward_expand(gf, top);
            ids = pred_in ? ggml_view_2d(c, top, m.n_used, T, top->nb[1], 0) : top;       // [n_used, T]
#if defined(MOMOE_CUDA)
            if (mbx) {                                       // the layer's input and choice to the CPU, at once
                MbLayer & B = e.mb.layer[il];
                B.send.n_f32 = m.n_embd * T;
                B.send.n_i32 = m.n_used * T;
                ggml_tensor * args[2] = {x, pred_in ? ggml_cont(c, ids) : ids};
                send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
                ggml_format_name(send, "mbsend-%d", il);
                ggml_build_forward_expand(gf, send);
                e.force_gpu.push_back(send);
                e.mb_graph = true;
                e.mb.graph.push_back(il);
            }
#endif
        }
    } else {
        // --nostall: the next layer's router on this layer's hidden state predicts its experts: computed in the same
        // top-k, so the pause has both (the prediction is read while the GPU goes on) (and the one after it, on the
        // same state: less accurate, but twice the time to read)
        ggml_tensor * both = probs;
        if (pred_in) {
            ggml_tensor * p2 = ggml_soft_max(c, ggml_mul_mat(c, m.layers[il + 1].gate_inp, pred_in));
            both = ggml_concat(c, probs, p2, 1);                                            // [n_expert, 2T]
        }
        if (pred_in2) {
            ggml_tensor * p3 = ggml_soft_max(c, ggml_mul_mat(c, m.layers[il + 2].gate_inp, pred_in2));
            both = ggml_concat(c, both, p3, 1);                                             // [n_expert, 3T]
        }
        top = ggml_cont(c, ggml_argsort_top_k(c, both, m.n_used));          // [n_used, T or 2T]
        // made resident here: a pause point (or the callback); no-stall: what the layer wanted, read after the graph
        ggml_format_name(top, masked ? "want-%d" : "topk-%d", il);
        if (e.ns_graph) ggml_set_output(top);
        e.topk[il] = top;
        ggml_build_forward_expand(gf, top);
        ids = pred_in ? ggml_view_2d(c, top, m.n_used, T, top->nb[1], 0) : top;   // [n_used, T]
        if (masked && e.nostall == 1)                            // sub: the best of the experts in the pool
            ids = ggml_cont(c, ggml_argsort_top_k(c, ggml_add(c, probs, L.res_mask), m.n_used));
        w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), ids);   // [1, n_used, T]
        if (masked && e.nostall == 2)                            // drop: the missing ones weigh nothing
            w = ggml_mul(c, w, ggml_reshape_3d(c, ggml_get_rows(c, L.res_keep,
                                                                ggml_reshape_1d(c, ids, (int64_t) m.n_used * T)),
                                               1, m.n_used, T));
        w = ggml_reshape_2d(c, w, m.n_used, T);
        w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        w = ggml_reshape_3d(c, w, 1, m.n_used, T);
    }
    ggml_tensor * x3 = ggml_reshape_3d(c, x, m.n_embd, 1, T);
    Group & G = e.pool.g[L.grp];
    const bool big = staged(m.n_used, G.n_slots, T) || direct;   // the staging copy, by expert number (see
    ggml_tensor * slots = nullptr;                           // STAGE_MIN_TOKENS), or the layer's experts in a row
    if (!big) {
        slots = ggml_get_rows(c, L.slot_map, ggml_reshape_1d(c, ids, (int64_t) m.n_used * T));
        slots = ggml_reshape_2d(c, slots, m.n_used, T);                                 // [n_used, T]
    }
    ggml_tensor ** as = big ? G.stage : G.t;
    ggml_tensor * layer_t[3];
    if (big && e.preloaded) {                                // no pauses fill the staging copy: the layer's experts are
        const int64_t s0 = L.slots_host()[0];                // in a row in the pool (preload), a view of them
        for (int k = m.part0(); k < 3; ++k)
            layer_t[k] = ggml_view_3d(c, G.t[k], G.t[k]->ne[0], G.t[k]->ne[1], L.n_exp, G.t[k]->nb[1], G.t[k]->nb[2],
                                      (size_t) s0 * G.t[k]->nb[2]);
        as = layer_t;
    }
    if (big) slots = ids;
    ggml_tensor * up = may_repeat(e, big, ggml_mul_mat_id(c, as[1], x3, slots));                            // [n_ff, n_used, T]
    ggml_tensor * gt = may_repeat(e, big, ggml_mul_mat_id(c, as[0], x3, slots));
    ggml_tensor * h = ggml_swiglu_split(c, gt, up);
    ggml_tensor * dn = may_repeat(e, big, ggml_mul_mat_id(c, as[2], h, slots));                             // [n_embd, n_used, T]
    if (helper_on(e, L, big))
        dn = ggml_add(c, dn, helper_down(e, c, L, x3, ids, T, [&](ggml_tensor * g, ggml_tensor * u) { return ggml_swiglu_split(c, g, u); }));
#if defined(MOMOE_CUDA)
    if (send) {                                              // the CPU's experts, waited for after the GPU's own
        MbLayer & B = e.mb.layer[il];
        B.wait.n_f32 = m.n_embd * m.n_used * T;
        ggml_tensor * args[1] = {send};
        ggml_tensor * cpu = ggml_custom_4d(c, GGML_TYPE_F32, m.n_embd, m.n_used, T, 1, args, 1, momoe_mailbox_cpu_stub, 1,
                                           &B.wait);
        ggml_format_name(cpu, "mbwait-%d", il);
        e.force_gpu.push_back(cpu);
        dn = ggml_add(c, dn, cpu);
    } else
#endif
    if (hybrid_on(e, L, T)) {                                // the missing experts' outputs (zero slots here), from the CPU
        ggml_format_name(dn, "moedn-%d", il);                // (joined at this stop: eval_cb)
        dn = ggml_add(c, dn, ggml_view_3d(c, L.cpu_dn, m.n_embd, m.n_used, T, L.cpu_dn->nb[1], L.cpu_dn->nb[2], 0));
    }
    ggml_format_name(up, "moe_up-%d", il);
    if (!hybrid_on(e, L, T)) ggml_format_name(dn, "moe_down-%d", il);   // (mailboxes: no stop either way)
    ggml_format_name(w, "moe_w-%d", il);
    dn = ggml_mul(c, dn, w);
    ggml_build_forward_expand(gf, dn);
    ggml_tensor * v[64];                                     // the views before the adds (llama.cpp's order)
    GGML_ASSERT(m.n_used <= 64);
    for (int i = 0; i < m.n_used; ++i) {
        v[i] = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
        ggml_build_forward_expand(gf, v[i]);
    }
    ggml_tensor * moe = v[0];
    for (int i = 1; i < m.n_used; ++i) {
        moe = ggml_add(c, moe, v[i]);
        ggml_build_forward_expand(gf, moe);
    }
    // the shared expert, gated by a sigmoid
    ggml_tensor * sh = ggml_swiglu_split(c, ggml_mul_mat(c, L.gate_sh, x), ggml_mul_mat(c, L.up_sh, x));
    sh = ggml_mul_mat(c, L.down_sh, sh);
    sh = ggml_mul(c, sh, ggml_sigmoid(c, ggml_mul_mat(c, L.gate_inp_sh, x)));
    ggml_format_name(sh, "ffn_shexp_gated-%d", il);
    ggml_format_name(moe, "ffn_moe_out-%d", il);
    return ggml_add(c, moe, sh);
}

// ------------------------------------------------------------------------------------------------ nemotron_h_moe
// Nemotron 3.5 Bolt / Super: llama.cpp's nemotron-h.cpp and llm_build_mamba_base::build_mamba2_layer, op for op.

// Mamba-2: in_proj -> z | xBC | dt; a short convolution over xBC (its history the conv state); the selective scan
// (the state [d_state, head_dim, n_head]); y + x * D, gated by silu(z), a grouped RMS norm, out_proj
static ggml_tensor * mamba2_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                  ggml_tensor * seq0, int T) {
    Model & m = e.m;
    const int64_t d_inner = m.ssm_inner, ds = m.ssm_state, ng = m.ssm_groups, nh = m.ssm_dt_rank, hd = d_inner / nh;
    const int64_t conv_ch = d_inner + 2 * ng * ds;
    ggml_tensor * zxBCdt = ggml_mul_mat(c, L.ssm_in, x);                                 // [d_in_proj, T]
    const size_t es = ggml_element_size(zxBCdt);
    ggml_tensor * z = ggml_view_4d(c, zxBCdt, hd, nh, T, 1, hd * es, zxBCdt->nb[1], zxBCdt->nb[1] * T, 0);
    ggml_tensor * xBC = ggml_view_3d(c, zxBCdt, conv_ch, T, 1, zxBCdt->nb[1], zxBCdt->nb[1] * T, d_inner * es);
    ggml_tensor * dt = ggml_view_3d(c, zxBCdt, nh, T, 1, zxBCdt->nb[1], zxBCdt->nb[1] * T, (2 * d_inner + 2 * ng * ds) * es);
    ggml_tensor * cst = ggml_reshape_3d(c, L.conv_state, m.ssm_conv - 1, conv_ch, 1);
    ggml_tensor * cin = ggml_concat(c, cst, ggml_transpose(c, xBC), 0);                   // [conv-1+T, conv_ch, 1]
    ggml_tensor * last = ggml_view_3d(c, cin, m.ssm_conv - 1, conv_ch, 1, cin->nb[1], cin->nb[2], ggml_row_size(cin->type, T));
    ggml_build_forward_expand(gf, ggml_cpy(c, last, cst));
    ggml_tensor * co = ggml_ssm_conv(c, cin, L.conv1d);                                   // [conv_ch, T, 1]
    co = ggml_silu(c, ggml_add(c, co, L.conv1d_b));
    const size_t cs = ggml_element_size(co);
    ggml_tensor * xs = ggml_view_4d(c, co, hd, nh, T, 1, hd * cs, co->nb[1], co->nb[2], 0);
    ggml_tensor * B = ggml_view_4d(c, co, ds, ng, T, 1, ds * cs, co->nb[1], co->nb[2], d_inner * cs);
    ggml_tensor * C = ggml_view_4d(c, co, ds, ng, T, 1, ds * cs, co->nb[1], co->nb[2], (d_inner + ng * ds) * cs);
    dt = ggml_add(c, ggml_cont(c, dt), L.ssm_dt);
    ggml_tensor * st = ggml_reshape_4d(c, L.ssm_state, ds, hd, nh, 1);
    ggml_tensor * ys = ggml_ssm_scan(c, st, xs, dt, L.ssm_a, B, C, seq0, 1);            // y, then the new state
    const int64_t D = ds * hd * nh;
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_view_1d(c, ys, D, ggml_nelements(xs) * ggml_element_size(xs)),
                                           L.ssm_state));
    ggml_tensor * y = ggml_view_4d(c, ys, hd, nh, T, 1, xs->nb[1], nh * xs->nb[1], T * nh * xs->nb[1], 0);
    y = ggml_add(c, y, ggml_mul(c, xs, L.ssm_d));
    y = ggml_swiglu_split(c, ggml_cont(c, z), y);                                         // silu(z) * y
    y = norm(c, ggml_reshape_4d(c, y, d_inner / ng, ng, T, 1), L.ssm_norm, m.rms_eps);    // grouped RMS norm
    y = ggml_reshape_2d(c, y, d_inner, T);
    return ggml_mul_mat(c, L.ssm_out, y);
}

// attention without positions (NoPE), n_head queries over n_head_kv keys
static ggml_tensor * nemo_attn_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                     const Inputs & in, int p0, int T, int n_kv, ggml_tensor * kv_idx = nullptr) {
    Model & m = e.m;
    if (!kv_idx) kv_idx = in.kv_idx;
    const int hd = m.head_dim;
    ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, x), hd, m.n_head, T);
    ggml_tensor * k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, x), hd, m.n_head_kv, T);
    ggml_tensor * v = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, x), hd, m.n_head_kv, T);
    const size_t row = ggml_row_size(GGML_TYPE_F16, (int64_t) hd * m.n_head_kv);
    if (kv_idx) {                                            // the positions as an input (a graph reused token after token)
        const int64_t w = (int64_t) hd * m.n_head_kv;
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.k_cache, w, e.n_ctx), ggml_reshape_2d(c, k, w, T), kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.v_cache, w, e.n_ctx), ggml_reshape_2d(c, v, w, T), kv_idx));
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, T, L.k_cache->nb[1], row, row * p0)));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, T, L.v_cache->nb[1], row, row * p0)));
    }
    ggml_tensor * kv_k = ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, n_kv, L.k_cache->nb[1], row, 0);
    ggml_tensor * kv_v = ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, n_kv, L.v_cache->nb[1], row, 0);
    ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, kv_k, 0, 2, 1, 3),
                                          ggml_permute(c, kv_v, 0, 2, 1, 3), in.mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    o = ggml_reshape_2d(c, o, (int64_t) hd * m.n_head, T);
    return ggml_mul_mat(c, L.wo, o);
}

// the latent MoE: a sigmoid router (its bias only chooses), the chosen weights normalized and scaled; the experts
// (up, relu^2, down: no gate) work in the latent width, between latent_down and latent_up; plus the shared expert
// pred_in: this block's residual input normed as the next MoE block normalizes its own: that block's router on it
// predicts its experts, chosen in the same step (read in the background while the GPU goes on)
static ggml_tensor * nemo_moe_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x, int T,
                                    ggml_tensor * pred_in) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    const int n_exp = L.n_exp, k = m.n_used;
    // the latent input before the router: computed by the layer's stop (--hybrid: the CPU's experts read it there)
    ggml_tensor * xl2 = L.latent_down ? ggml_mul_mat(c, L.latent_down, x)                // [latent, T]
                                      : ggml_is_contiguous(x) ? x : ggml_cont(c, x);    // (Lightning: the hidden state)
    ggml_build_forward_expand(gf, xl2);
    e.xl[il] = xl2;
    ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, x);                                    // [n_exp, T]
    ggml_format_name(lg, "ffn_moe_logits-%d", il);
    ggml_tensor * probs = ggml_sigmoid(c, lg);
    // llama.cpp's order (the routing's nodes in a row: CUDA's topk-moe kernel): the choice and its weights first
    ggml_tensor * chosen = ggml_argsort_top_k(c, ggml_add(c, probs, L.exp_probs_b), k);   // [k, T] (a view)
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, n_exp, T), chosen);   // [1, k, T]
    w = ggml_reshape_2d(c, w, k, T);
    w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
    w = ggml_reshape_3d(c, w, 1, k, T);
    w = ggml_scale(c, w, m.exp_scale);
    ggml_build_forward_expand(gf, w);
    ggml_tensor * top = ggml_cont(c, chosen);                                             // [k, T]
    ggml_tensor * both = top;
    if (pred_in) {                                           // [k, 2T]: this block's, then the next one's predicted
        const Layer & N = m.layers[L.next_moe];
        ggml_tensor * pl = ggml_add(c, ggml_sigmoid(c, ggml_mul_mat(c, N.gate_inp, pred_in)), N.exp_probs_b);
        both = ggml_concat(c, top, ggml_cont(c, ggml_argsort_top_k(c, pl, k)), 1);
    }
#if defined(MOMOE_CUDA)
    const bool mbx = e.mailbox && hybrid_on(e, L, T);        // the mailboxes: no stop here
    static const bool mbdbg = getenv("MOMOE_MB_DEBUG") != nullptr;
    if (mbdbg && T == 1)
        fprintf(stderr, "MB layer %d: mailbox %d (hybrid %d, cpu_dn %d, slots %d, staged %d)\n", il, mbx, e.hybrid,
                L.cpu_dn != nullptr, e.pool.g[L.grp].n_slots, staged(m.n_used, e.pool.g[L.grp].n_slots, T));
    ggml_tensor * send = nullptr;
#else
    const bool mbx = false;
#endif
    ggml_format_name(both, mbx ? "mbtopk-%d" : "topk-%d", il);   // topk-: made resident here (the callback / a pause point)
    e.topk[il] = both;
    ggml_build_forward_expand(gf, both);
    if (pred_in) top = ggml_view_2d(c, both, k, T, both->nb[1], 0);
#if defined(MOMOE_CUDA)
    if (mbx) {                                               // the layer's input and choice to the CPU, at once
        MbLayer & B = e.mb.layer[il];
        B.send.n_f32 = m.latent * T;
        B.send.n_i32 = k * T;
        ggml_tensor * args[2] = {xl2, top};
        send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
        ggml_format_name(send, "mbsend-%d", il);
        ggml_build_forward_expand(gf, send);
        e.force_gpu.push_back(send);
        e.mb_graph = true;
        e.mb.graph.push_back(il);
    }
#endif
    ggml_tensor * slots = ggml_reshape_2d(c, ggml_get_rows(c, L.slot_map, ggml_reshape_1d(c, top, (int64_t) k * T)), k, T);
    Group & G = e.pool.g[L.grp];
    const bool big = staged(k, G.n_slots, T);
    ggml_tensor * as[3] = {nullptr, big ? G.stage[1] : G.t[1], big ? G.stage[2] : G.t[2]};
    if (big && e.preloaded) {                                // the layer's experts in a row in the pool: a view
        const int64_t s0 = L.slots_host()[0];
        for (int j = 1; j < 3; ++j)
            as[j] = ggml_view_3d(c, G.t[j], G.t[j]->ne[0], G.t[j]->ne[1], n_exp, G.t[j]->nb[1], G.t[j]->nb[2],
                                 (size_t) s0 * G.t[j]->nb[2]);
    }
    if (big) slots = top;
    ggml_tensor * xl = ggml_reshape_3d(c, xl2, m.latent, 1, T);
    ggml_tensor * up = may_repeat(e, big, ggml_mul_mat_id(c, as[1], xl, slots));                              // [n_ff, k, T]
    ggml_tensor * h = ggml_sqr(c, ggml_relu(c, up));
    ggml_tensor * dn = may_repeat(e, big, ggml_mul_mat_id(c, as[2], h, slots));                               // [latent, k, T]
    if (helper_on(e, L, big))
        dn = ggml_add(c, dn, helper_down(e, c, L, xl, top, T, [&](ggml_tensor *, ggml_tensor * u) { return ggml_sqr(c, ggml_relu(c, u)); }));
#if defined(MOMOE_CUDA)
    if (mbx) {                                               // the CPU's experts, waited for after the GPU's own
        MbLayer & B = e.mb.layer[il];
        B.wait.n_f32 = m.latent * k * T;
        ggml_tensor * args[1] = {send};
        ggml_tensor * cpu = ggml_custom_4d(c, GGML_TYPE_F32, m.latent, k, T, 1, args, 1, momoe_mailbox_cpu_stub, 1, &B.wait);
        ggml_format_name(cpu, "mbwait-%d", il);
        e.force_gpu.push_back(cpu);
        dn = ggml_add(c, dn, cpu);
    } else
#endif
    if (hybrid_on(e, L, T)) {                                // the missing experts' outputs (zero slots here), from the CPU
        ggml_format_name(dn, "moedn-%d", il);
        dn = ggml_add(c, dn, ggml_view_3d(c, L.cpu_dn, m.latent, k, T, L.cpu_dn->nb[1], L.cpu_dn->nb[2], 0));
    }
    dn = ggml_mul(c, dn, w);
    ggml_build_forward_expand(gf, dn);
    ggml_tensor * v[64];                                     // the views before the adds (llama.cpp's order)
    GGML_ASSERT(k <= 64);
    for (int i = 0; i < k; ++i) {
        v[i] = ggml_view_2d(c, dn, m.latent, T, dn->nb[2], i * dn->nb[1]);
        ggml_build_forward_expand(gf, v[i]);
    }
    ggml_tensor * moe = v[0];
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(c, moe, v[i]);
        ggml_build_forward_expand(gf, moe);
    }
    if (L.latent_up) moe = ggml_mul_mat(c, L.latent_up, moe);
    ggml_tensor * sh = ggml_mul_mat(c, L.down_sh, ggml_sqr(c, ggml_relu(c, ggml_mul_mat(c, L.up_sh, x))));
    ggml_format_name(moe, "ffn_moe_out-%d", il);
    return ggml_add(c, moe, sh);
}

// ------------------------------------------------------------------------------------------------ lfm2moe
// LFM2.5-24B-A2B: llama.cpp's lfm2.cpp (llama_model_lfm2::graph, lfm2moe's), op for op, for one sequence.

// the short convolution: in_proj -> b | c | x; b * x through a depthwise convolution of width l_cache over the last
// l_cache - 1 positions (the conv state) and this batch; c * that; out_proj
static ggml_tensor * lfm_conv_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x, int T) {
    Model & m = e.m;
    const int64_t n = m.n_embd, dc = m.ssm_conv - 1;
    ggml_tensor * bcx = ggml_reshape_3d(c, ggml_mul_mat(c, L.ssm_in, x), 3 * n, T, 1);  // [3 n_embd, T, 1]
    const size_t es = ggml_element_size(bcx);
    ggml_tensor * b = ggml_view_3d(c, bcx, n, T, 1, bcx->nb[1], bcx->nb[2], 0);
    ggml_tensor * cg = ggml_view_3d(c, bcx, n, T, 1, bcx->nb[1], bcx->nb[2], n * es);
    ggml_tensor * xx = ggml_view_3d(c, bcx, n, T, 1, bcx->nb[1], bcx->nb[2], 2 * n * es);
    ggml_tensor * bx = ggml_transpose(c, ggml_mul(c, b, xx));                             // [T, n_embd, 1]
    ggml_tensor * cst = ggml_reshape_3d(c, L.conv_state, dc, n, 1);
    ggml_tensor * cin = ggml_concat(c, cst, bx, 0);                                       // [dc + T, n_embd, 1]
    ggml_tensor * last = ggml_view_3d(c, cin, dc, n, 1, cin->nb[1], cin->nb[2], ggml_row_size(cin->type, T));
    ggml_build_forward_expand(gf, ggml_cpy(c, last, cst));
    ggml_tensor * co = ggml_ssm_conv(c, cin, L.conv1d);                                   // [n_embd, T, 1]
    ggml_tensor * y = ggml_mul(c, cg, co);
    return ggml_mul_mat(c, L.ssm_out, ggml_reshape_2d(c, y, n, T));
}

// GQA attention: q and k normed per head, NEOX RoPE, the cache [head_dim, n_head_kv, n_ctx] f16
static ggml_tensor * lfm_attn_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                    const Inputs & in, int p0, int T, int n_kv, ggml_tensor * kv_idx = nullptr) {
    Model & m = e.m;
    if (!kv_idx) kv_idx = in.kv_idx;
    const int hd = m.head_dim;
    ggml_tensor * q = norm(c, ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, x), hd, m.n_head, T), L.q_norm, m.rms_eps);
    ggml_tensor * k = norm(c, ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, x), hd, m.n_head_kv, T), L.k_norm, m.rms_eps);
    ggml_tensor * v = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, x), hd, m.n_head_kv, T);
    if (L.rope) {                                            // (flint: no RoPE on its full-attention blocks)
        ggml_tensor * pos = ggml_view_1d(c, in.pos, T, 0);                                // (section 0: p0 + i)
        q = ggml_rope_ext(c, q, pos, nullptr, m.n_rot, GGML_ROPE_TYPE_NEOX, 0, m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, m.n_rot, GGML_ROPE_TYPE_NEOX, 0, m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    }
    const size_t row = ggml_row_size(GGML_TYPE_F16, (int64_t) hd * m.n_head_kv);
    if (L.swa) {                                             // flint's ring: written at p % swa_rows, read whole
        const int64_t w = (int64_t) hd * m.n_head_kv;
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.k_cache, w, m.swa_rows), ggml_reshape_2d(c, k, w, T), in.kv_idx_swa));
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.v_cache, w, m.swa_rows), ggml_reshape_2d(c, v, w, T), in.kv_idx_swa));
        n_kv = std::min(n_kv, m.swa_rows);                   // (before it wraps: only the rows written so far)
    } else if (kv_idx) {                                            // the positions as an input (a graph reused token after token)
        const int64_t w = (int64_t) hd * m.n_head_kv;
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.k_cache, w, e.n_ctx), ggml_reshape_2d(c, k, w, T), kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.v_cache, w, e.n_ctx), ggml_reshape_2d(c, v, w, T), kv_idx));
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, T, L.k_cache->nb[1], row, row * p0)));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, T, L.v_cache->nb[1], row, row * p0)));
    }
    ggml_tensor * kv_k = ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, n_kv, L.k_cache->nb[1], row, 0);
    ggml_tensor * kv_v = ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, n_kv, L.v_cache->nb[1], row, 0);
    ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, kv_k, 0, 2, 1, 3),
                                          ggml_permute(c, kv_v, 0, 2, 1, 3), L.swa ? in.mask_swa : in.mask,
                                          1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    if (L.sinks) ggml_flash_attn_ext_add_sinks(o, L.sinks);  // (flint: one learned logit per head beside the keys)
    ggml_prec_set_acc(o, GGML_PREC_F32);
    o = ggml_reshape_2d(c, o, (int64_t) hd * m.n_head, T);
    return ggml_mul_mat(c, L.wo, o);
}

// flint: the down projection's input divided per expert and token by a power of two that brings its largest value
// under 1024 (llama.cpp's build_amax_src1_f16: 32 values of an 8-bit activation block summed in f16 stay finite),
// the same factor multiplied back on the output (exact).  act: [n_ff, k, T]; *scale: [1, k, T]
static ggml_tensor * flint_down_scale(ggml_context * c, ggml_tensor * act, ggml_tensor ** scale) {
    ggml_tensor * amax = ggml_pool_1d(c, ggml_abs(c, act), GGML_OP_POOL_MAX, (int) act->ne[0], (int) act->ne[0], 0);
    ggml_tensor * f = ggml_scale(c, amax, 1.0f / 1024.0f);
    f = ggml_scale(c, ggml_log(c, f), 1.4426950408889634f);                              // log2
    f = ggml_clamp(c, ggml_ceil(c, f), 0.0f, 127.0f);
    f = ggml_exp(c, ggml_scale(c, f, 0.6931471805599453f));                              // 2^n
    *scale = f;
    return ggml_div(c, act, f);
}

// the MoE (llama.cpp's build_moe_ffn with the sigmoid gate, norm_w): the bias only chooses, the chosen experts'
// sigmoid weights normalized; SwiGLU experts, no shared expert.  Built in llama.cpp's order, so that the backends fuse
// it as there: the choice and its weights first (sigmoid, bias, top-k, weights, norm: CUDA's topk-moe kernel), and the
// weighted experts' views before their sum (Metal's MOE_REDUCE).  pred_in / pred_in2: the next blocks' routers on this
// block's residual, chosen after it (read ahead while the GPU goes on), as moe_layer.  (--nostall: not for lfm2moe.)
static ggml_tensor * lfm_moe_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x, int T,
                                   ggml_tensor * pred_in, ggml_tensor * pred_in2) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    const int k = m.n_used;
    const bool hyb = hybrid_on(e, L, T);                     // flint --hybrid: the CPU computes the experts the pool
    if (hyb) {                                               // lacks, from the experts' input (as moe_layer)
        if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
        ggml_build_forward_expand(gf, x);
        e.xl[il] = x;
    }
#if defined(MOMOE_CUDA)
    const bool mbx = mb_on(e) && hyb && ns_paused(e, il);   // the mailboxes: no stops in this layer
    ggml_tensor * send = nullptr;
#else
    const bool mbx = false;
#endif
    ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, x);
    ggml_format_name(lg, "ffn_moe_logits-%d", il);
    ggml_tensor * probs = ggml_sigmoid(c, lg);                                            // [n_expert, T]
    ggml_tensor * chosen = ggml_argsort_top_k(c, L.exp_probs_b ? ggml_add(c, probs, L.exp_probs_b) : probs, k);   // [k, T]
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), chosen);   // [1, k, T]
    w = ggml_reshape_2d(c, w, k, T);
    w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
    w = ggml_reshape_3d(c, w, 1, k, T);
    if (m.exp_scale != 1.0f) w = ggml_scale(c, w, m.exp_scale);
    ggml_build_forward_expand(gf, w);                        // (the routing's nodes in a row: fused)
    // every expert in its slot and nothing reading the choice (momoe-metal, as moe_layer's `direct`): the choice as it
    // is and the layer's experts as a view of the pool (flint on the M4 Air, UD-IQ3_XXS)
    static const bool no_direct = getenv("MOMOE_FORCE_CB") != nullptr || getenv("MOMOE_NO_DIRECT") != nullptr;
    const bool direct = kHostMem && e.preloaded && !e.pause && !e.ns_graph && !e.hybrid && !e.helper && !debug_on() &&
                        !no_direct && !pred_in && !pred_in2;
    // this block's choice, then the predicted blocks', in one tensor: what the stop or the mailbox reads
    ggml_tensor * top = direct ? chosen : ggml_cont(c, chosen);                           // [k, T (2T, 3T)]
    for (int j = 1; j <= 2 && !direct; ++j) {
        ggml_tensor * in = j == 1 ? pred_in : pred_in2;
        if (!in) continue;
        const Layer & N = m.layers[il + j];
        ggml_tensor * pl = ggml_sigmoid(c, ggml_mul_mat(c, N.gate_inp, in));
        if (N.exp_probs_b) pl = ggml_add(c, pl, N.exp_probs_b);
        top = ggml_concat(c, top, ggml_cont(c, ggml_argsort_top_k(c, pl, k)), 1);
    }
    Group & G = e.pool.g[L.grp];
    const bool big = staged(k, G.n_slots, T) || direct;      // the staging copy, by expert number (direct: the pool)
#if defined(MOMOE_CUDA)
    const bool rmb = e.rmb && !big && T <= HYB_MAX_T && !mbx;   // the resident mailbox (rmb_layer): no stop here
#else
    const bool rmb = false;
#endif
    if (!direct) {
        ggml_format_name(top, rmb ? "rmbtopk-%d" : mbx ? "mbtopk-%d" : "topk-%d", il);
        e.topk[il] = top;
        ggml_build_forward_expand(gf, top);
    } else e.topk[il] = nullptr;
    ggml_tensor * ids = pred_in ? ggml_view_2d(c, top, k, T, top->nb[1], 0) : top;     // [k, T], contiguous
#if defined(MOMOE_CUDA)
    if (mbx) {                                               // the layer's input and choice to the CPU, at once
        MbLayer & B = e.mb.layer[il];
        B.send.n_f32 = m.n_embd * T;
        B.send.n_i32 = k * T;
        ggml_tensor * args[2] = {x, pred_in ? ggml_cont(c, ids) : ids};
        send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
        ggml_format_name(send, "mbsend-%d", il);
        ggml_build_forward_expand(gf, send);
        e.force_gpu.push_back(send);
        e.mb_graph = true;
        e.mb.graph.push_back(il);
    }
#endif
    ggml_tensor * slots = nullptr;
#if defined(MOMOE_CUDA)
    if (rmb) {                                               // the choice to the worker; the slots back from it
        MbLayer & B = e.mb.layer[il];
        B.send.n_i32 = (int32_t) ggml_nelements(top);
        B.wait.n_f32 = k * T;
        ggml_tensor * args[2] = {ggml_view_1d(c, lg, 1, 0), top};
        ggml_tensor * send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
        ggml_format_name(send, "rmbsend-%d", il);
        ggml_build_forward_expand(gf, send);
        ggml_tensor * wargs[1] = {send};
        slots = ggml_custom_4d(c, GGML_TYPE_I32, k, T, 1, 1, wargs, 1, momoe_mailbox_cpu_stub, 1, &B.wait);
        ggml_format_name(slots, "rmbwait-%d", il);
        e.force_gpu.push_back(send);
        e.force_gpu.push_back(slots);
        e.mb_graph = true;
        e.mb.graph.push_back(il);
    } else
#endif
    if (!big) slots = ggml_reshape_2d(c, ggml_get_rows(c, L.slot_map, ggml_reshape_1d(c, ids, (int64_t) k * T)), k, T);
    ggml_tensor * x3 = ggml_reshape_3d(c, x, m.n_embd, 1, T);
    ggml_tensor ** as = big ? G.stage : G.t;
    ggml_tensor * layer_t[3];
    if (big && e.preloaded) {                                // the layer's experts in a row in the pool (preload)
        const int64_t s0 = L.slots_host()[0];
        for (int j = 0; j < 3; ++j)
            layer_t[j] = ggml_view_3d(c, G.t[j], G.t[j]->ne[0], G.t[j]->ne[1], L.n_exp, G.t[j]->nb[1], G.t[j]->nb[2],
                                      (size_t) s0 * G.t[j]->nb[2]);
        as = layer_t;
    }
    if (big) slots = ids;
    ggml_tensor * up = may_repeat(e, big, ggml_mul_mat_id(c, as[1], x3, slots));                            // [n_ff, k, T]
    ggml_tensor * gt = may_repeat(e, big, ggml_mul_mat_id(c, as[0], x3, slots));
    ggml_tensor * act = ggml_swiglu_split(c, gt, up), * dscale = nullptr;
    static const bool no_scale = getenv("MOMOE_FLINT_NO_SCALE") != nullptr;   // a test: the down projection unscaled
    // (momoe-metal, a few tokens: its matrix-vector kernels read the activations as f32, nothing to keep finite; the
    // power-of-two factor cancels exactly.  Its batched kernels take them as f16: scaled, as on CUDA, whose
    // matrix-vector kernels sum 8-bit blocks in f16.  M4 Air, UD-IQ3_XXS: 38.4 -> 42.9 tok/s, the same tokens)
    if (m.flint && !no_scale && (!kHostMem || T > 8)) act = flint_down_scale(c, act, &dscale);
    ggml_tensor * dn = may_repeat(e, big, ggml_mul_mat_id(c, as[2], act, slots));                             // [n_embd, k, T]
    if (dscale) dn = ggml_mul(c, dn, dscale);                // (before the CPU's part: that one was not scaled)
    if (helper_on(e, L, big))
        dn = ggml_add(c, dn, helper_down(e, c, L, x3, ids, T, [&](ggml_tensor * g, ggml_tensor * u) { return ggml_swiglu_split(c, g, u); }));
    ggml_tensor * sh = nullptr;                              // flint's shared expert, before the CPU's experts are
    if (L.gate_sh) {                                         // waited for: the GPU computes it while the CPU works
        sh = ggml_mul_mat(c, L.down_sh, ggml_swiglu_split(c, ggml_mul_mat(c, L.gate_sh, x), ggml_mul_mat(c, L.up_sh, x)));
        ggml_format_name(sh, "ffn_shexp-%d", il);
        ggml_build_forward_expand(gf, sh);
    }
#if defined(MOMOE_CUDA)
    if (send) {                                              // the CPU's experts, waited for after the GPU's own
        MbLayer & B = e.mb.layer[il];
        B.wait.n_f32 = m.n_embd * k * T;
        ggml_tensor * args[1] = {send};
        ggml_tensor * cpu = ggml_custom_4d(c, GGML_TYPE_F32, m.n_embd, k, T, 1, args, 1, momoe_mailbox_cpu_stub, 1, &B.wait);
        ggml_format_name(cpu, "mbwait-%d", il);
        e.force_gpu.push_back(cpu);
        dn = ggml_add(c, dn, cpu);
    } else
#endif
    if (hyb) {                                               // the missing experts' outputs (zero slots here), from the CPU
        ggml_format_name(dn, "moedn-%d", il);                // (joined at this stop: eval_cb)
        dn = ggml_add(c, dn, ggml_view_3d(c, L.cpu_dn, m.n_embd, k, T, L.cpu_dn->nb[1], L.cpu_dn->nb[2], 0));
    }
    if (!hyb) ggml_format_name(dn, "moe_down-%d", il);
    dn = ggml_mul(c, dn, w);
    ggml_build_forward_expand(gf, dn);
    ggml_tensor * v[64];                                     // the views before the adds (llama.cpp's order)
    GGML_ASSERT(k <= 64);
    for (int i = 0; i < k; ++i) {
        v[i] = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
        ggml_build_forward_expand(gf, v[i]);
    }
    ggml_tensor * moe = v[0];
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(c, moe, v[i]);
        ggml_build_forward_expand(gf, moe);
    }
    ggml_format_name(moe, "ffn_moe_out-%d", il);
    return sh ? ggml_add(c, moe, sh) : moe;                  // (as llama.cpp: the routed experts' sum, then the shared)
}

// ------------------------------------------------------------------------------------------------ gemma4
// Gemma 4 26B-A4B: llama.cpp's gemma4.cpp (llama_model_gemma4::graph), op for op, for one sequence.

// attention: q / k RMS-normed per head with their weights, v without one (a full block has no v projection: v is k's
// projection before its norm); NEOX RoPE over the whole head (a full block: its own base and rope_freqs); scale 1; a
// sliding-window block keeps its ring (as flint's)
static ggml_tensor * gemma_attn_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                      const Inputs & in, int p0, int T, int n_kv, ggml_tensor * kv_idx = nullptr) {
    Model & m = e.m;
    if (!kv_idx) kv_idx = in.kv_idx;
    const int hd = L.hd, nkv = L.n_kv;
    ggml_tensor * pos = ggml_view_1d(c, in.pos, T, 0);
    ggml_tensor * ff = L.swa ? nullptr : m.rope_freqs;
    const float base = L.swa ? m.rope_base_swa : m.rope_base;
    ggml_tensor * q = norm(c, ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, x), hd, m.n_head, T), L.q_norm, m.rms_eps);
    q = ggml_rope_ext(c, q, pos, ff, hd, GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_tensor * kr = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, x), hd, nkv, T);
    ggml_tensor * v = L.wv ? ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, x), hd, nkv, T) : kr;
    ggml_tensor * k = norm(c, kr, L.k_norm, m.rms_eps);
    v = ggml_rms_norm(c, v, m.rms_eps);
    k = ggml_rope_ext(c, k, pos, ff, hd, GGML_ROPE_TYPE_NEOX, 0, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    const int64_t w = (int64_t) hd * nkv;
    const size_t row = ggml_row_size(GGML_TYPE_F16, w);
    if (L.swa) {                                             // the ring: written at p % swa_rows, read whole
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.k_cache, w, m.swa_rows), ggml_reshape_2d(c, k, w, T), in.kv_idx_swa));
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.v_cache, w, m.swa_rows), ggml_reshape_2d(c, v, w, T), in.kv_idx_swa));
        n_kv = std::min(n_kv, m.swa_rows);
    } else if (kv_idx) {                                     // the positions as an input (a reused graph)
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.k_cache, w, e.n_ctx), ggml_reshape_2d(c, k, w, T), kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(c, ggml_reshape_2d(c, L.v_cache, w, e.n_ctx), ggml_reshape_2d(c, v, w, T), kv_idx));
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, nkv, T, L.k_cache->nb[1], row, row * p0)));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, nkv, T, L.v_cache->nb[1], row, row * p0)));
    }
    ggml_tensor * kv_k = ggml_view_3d(c, L.k_cache, hd, nkv, n_kv, L.k_cache->nb[1], row, 0);
    ggml_tensor * kv_v = ggml_view_3d(c, L.v_cache, hd, nkv, n_kv, L.v_cache->nb[1], row, 0);
    ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, kv_k, 0, 2, 1, 3),
                                          ggml_permute(c, kv_v, 0, 2, 1, 3), L.swa ? in.mask_swa : in.mask, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    o = ggml_reshape_2d(c, o, (int64_t) hd * m.n_head, T);
    return ggml_mul_mat(c, L.wo, o);
}

// the router's input (llama.cpp's "custom MoE logits"): the residual RMS-normed, over sqrt(n_embd), times its scale
static ggml_tensor * gemma_router_in(const Model & m, ggml_context * c, const Layer & L, ggml_tensor * res) {
    return ggml_mul(c, ggml_scale(c, ggml_rms_norm(c, res, m.rms_eps), 1.0f / sqrtf((float) m.n_embd)), L.gate_inp_s);
}

// the MoE (llama.cpp's build_moe_ffn: softmax router, norm_w, GELU experts whose gate and up are one tensor, each
// expert's output scaled by down_exps.scale): the router reads its own input (rin), the experts theirs (x, the
// pre_ffw_norm_2 one).  Built in llama.cpp's order (the routing's nodes in a row, the views before the adds), as
// lfm_moe_layer, which this follows for the pool and the stops.  pred_in: the next block's router input from this
// block's residual, chosen after it (read ahead while the GPU goes on)
static ggml_tensor * gemma_moe_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x,
                                     ggml_tensor * rin, int T, ggml_tensor * pred_in) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    const int k = m.n_used;
    const int64_t n_exp = L.gate_inp->ne[1];
    const bool hyb = hybrid_on(e, L, T);                     // --hybrid: the CPU computes the experts the pool lacks,
    if (hyb) {                                               // from the experts' input (as lfm_moe_layer)
        if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
        ggml_build_forward_expand(gf, x);
        e.xl[il] = x;
    }
#if defined(MOMOE_CUDA)
    const bool mbx = mb_on(e) && hyb && ns_paused(e, il);   // the mailboxes: no stops in this layer
    ggml_tensor * send = nullptr;
#else
    const bool mbx = false;
#endif
    ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, rin);
    ggml_format_name(lg, "ffn_moe_logits-%d", il);
    ggml_tensor * probs = ggml_soft_max(c, lg);                                           // [n_expert, T]
    ggml_tensor * chosen = ggml_argsort_top_k(c, probs, k);                               // [k, T]
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, n_exp, T), chosen);   // [1, k, T]
    w = ggml_reshape_2d(c, w, k, T);
    w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
    w = ggml_reshape_3d(c, w, 1, k, T);
    ggml_build_forward_expand(gf, w);
    static const bool no_direct = getenv("MOMOE_FORCE_CB") != nullptr || getenv("MOMOE_NO_DIRECT") != nullptr;
    const bool direct = kHostMem && e.preloaded && !e.pause && !e.ns_graph && !e.hybrid && !e.helper && !debug_on() &&
                        !no_direct && !pred_in;
    ggml_tensor * top = direct ? chosen : ggml_cont(c, chosen);                           // [k, T (2T)]
    if (pred_in && !direct) {
        const Layer & N = m.layers[il + 1];
        top = ggml_concat(c, top, ggml_cont(c, ggml_argsort_top_k(c, ggml_soft_max(c, ggml_mul_mat(c, N.gate_inp, pred_in)), k)), 1);
    }
    Group & G = e.pool.g[L.grp];
    const bool big = staged(k, G.n_slots, T) || direct;
    if (!direct) {
        ggml_format_name(top, mbx ? "mbtopk-%d" : "topk-%d", il);
        e.topk[il] = top;
        ggml_build_forward_expand(gf, top);
    } else e.topk[il] = nullptr;
    ggml_tensor * ids = pred_in && !direct ? ggml_view_2d(c, top, k, T, top->nb[1], 0) : top;   // [k, T]
#if defined(MOMOE_CUDA)
    if (mbx) {                                               // the layer's input and choice to the CPU, at once
        MbLayer & B = e.mb.layer[il];
        B.send.n_f32 = m.n_embd * T;
        B.send.n_i32 = k * T;
        ggml_tensor * args[2] = {x, pred_in ? ggml_cont(c, ids) : ids};
        send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
        ggml_format_name(send, "mbsend-%d", il);
        ggml_build_forward_expand(gf, send);
        e.force_gpu.push_back(send);
        e.mb_graph = true;
        e.mb.graph.push_back(il);
    }
#endif
    ggml_tensor * slots = big ? ids : ggml_reshape_2d(c, ggml_get_rows(c, L.slot_map, ggml_reshape_1d(c, ids, (int64_t) k * T)), k, T);
    ggml_tensor * x3 = ggml_reshape_3d(c, x, m.n_embd, 1, T);
    ggml_tensor ** as = big ? G.stage : G.t;
    ggml_tensor * layer_t[3];
    if (big && e.preloaded) {                                // the layer's experts in a row in the pool (preload)
        const int64_t s0 = L.slots_host()[0];
        for (int j = m.part0(); j < 3; ++j)
            layer_t[j] = ggml_view_3d(c, G.t[j], G.t[j]->ne[0], G.t[j]->ne[1], L.n_exp, G.t[j]->nb[1], G.t[j]->nb[2],
                                      (size_t) s0 * G.t[j]->nb[2]);
        as = layer_t;
    }
    ggml_tensor * gu = may_repeat(e, big, ggml_mul_mat_id(c, as[1], x3, slots));        // [2 n_ff, k, T]
    const int64_t n_ff = gu->ne[0] / 2;
    ggml_tensor * gt = ggml_view_3d(c, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
    ggml_tensor * up = ggml_view_3d(c, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], n_ff * gu->nb[0]);
    ggml_tensor * dn = may_repeat(e, big, ggml_mul_mat_id(c, as[2], ggml_geglu_split(c, gt, up), slots));   // [n_embd, k, T]
#if defined(MOMOE_CUDA)
    if (send) {                                              // the CPU's experts, waited for after the GPU's own
        MbLayer & B = e.mb.layer[il];
        B.wait.n_f32 = m.n_embd * k * T;
        ggml_tensor * args[1] = {send};
        ggml_tensor * cpu = ggml_custom_4d(c, GGML_TYPE_F32, m.n_embd, k, T, 1, args, 1, momoe_mailbox_cpu_stub, 1, &B.wait);
        ggml_format_name(cpu, "mbwait-%d", il);
        e.force_gpu.push_back(cpu);
        dn = ggml_add(c, dn, cpu);
    } else
#endif
    if (hyb) {                                               // the missing experts' outputs (zero slots here), from the CPU
        ggml_format_name(dn, "moedn-%d", il);                // (joined at this stop: eval_cb)
        dn = ggml_add(c, dn, ggml_view_3d(c, L.cpu_dn, m.n_embd, k, T, L.cpu_dn->nb[1], L.cpu_dn->nb[2], 0));
    }
    if (L.down_s) {                                          // each chosen expert's own scale (the CPU's part too)
        ggml_tensor * sc = ggml_repeat_4d(c, ggml_reshape_3d(c, L.down_s, 1, n_exp, 1), 1, n_exp, T, 1);
        dn = ggml_mul(c, dn, ggml_get_rows(c, sc, ids));
    }
    if (!hyb) ggml_format_name(dn, "moe_down-%d", il);
    dn = ggml_mul(c, dn, w);
    ggml_build_forward_expand(gf, dn);
    ggml_tensor * v[64];                                     // the views before the adds (llama.cpp's order)
    GGML_ASSERT(k <= 64);
    for (int i = 0; i < k; ++i) {
        v[i] = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
        ggml_build_forward_expand(gf, v[i]);
    }
    ggml_tensor * moe = v[0];
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(c, moe, v[i]);
        ggml_build_forward_expand(gf, moe);
    }
    ggml_format_name(moe, "ffn_moe_out-%d", il);
    return moe;
}

// one block: attention (post-normed, residual), then the dense GELU FFN and the MoE side by side on that residual
// (each post-normed, summed, post-normed again), the residual, the block's output scale
static ggml_tensor * gemma_block(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x,
                                 const Inputs & in, int p0, int T, int n_kv, ggml_tensor * kv_idx = nullptr) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
    ggml_format_name(h, "attn_norm-%d", il);
    h = norm(c, gemma_attn_layer(e, c, gf, L, h, in, p0, T, n_kv, kv_idx), L.post_norm, m.rms_eps);
    ggml_tensor * ao = ggml_add(c, h, x);
    ggml_format_name(ao, "attn_out-%d", il);
    ggml_tensor * mlp = norm(c, ao, L.ffn_norm, m.rms_eps);
    mlp = ggml_mul_mat(c, L.ffn_down, ggml_geglu_split(c, ggml_mul_mat(c, L.ffn_gate, mlp), ggml_mul_mat(c, L.ffn_up, mlp)));
    mlp = norm(c, mlp, L.post_ffw1, m.rms_eps);
    ggml_tensor * f = mlp;
    if (L.gate_inp) {
        ggml_tensor * mi = norm(c, ao, L.pre_ffw2, m.rms_eps);
        ggml_tensor * pred = e.predict && il + 1 < m.n_layer && m.layers[il + 1].gate_inp
                           ? gemma_router_in(m, c, m.layers[il + 1], ao) : nullptr;
        ggml_tensor * moe = gemma_moe_layer(e, c, gf, il, mi, gemma_router_in(m, c, L, ao), T, pred);
        f = ggml_add(c, mlp, norm(c, moe, L.post_ffw2, m.rms_eps));
    }
    f = norm(c, f, L.post_ffw, m.rms_eps);
    x = ggml_add(c, f, ao);
    if (L.out_scale) x = ggml_mul(c, x, L.out_scale);
    ggml_format_name(x, "l_out-%d", il);
    return x;
}

// ------------------------------------------------------------------------------------------------ glm5-next
// GLM-5.3-Flash: llama.cpp's glm5-next.cpp, op for op (its fused mHC and lightning-indexer ops, as llama.cpp picks on
// CUDA), for one sequence: a position's cell is its position, the k-pool j is cells kpool*j .. kpool*j + kpool-1.

// mHC before a block (build_hc_pre): the streams [n_embd, hc, T] normed as one row; their mixes give the block's input
// weights (pre), its output's weights per stream (post) and the streams' Sinkhorn-normalized mixing (comb)
static ggml_tensor * glm_hc_pre(Engine & e, ggml_context * c, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale,
                                ggml_tensor * base, ggml_tensor ** post, ggml_tensor ** comb) {
    Model & m = e.m;
    const int64_t hc = m.hc, T = x->ne[2];
    ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, (int64_t) hc * m.n_embd, T), m.rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                      // [(2 + hc) hc, T]
    auto v1 = [&](ggml_tensor * t, int64_t n, int64_t i0) { return ggml_view_1d(c, t, n, ggml_row_size(t->type, i0)); };
    auto v2 = [&](ggml_tensor * t, int64_t n, int64_t i0) {
        return ggml_view_2d(c, t, n, T, t->nb[1], ggml_row_size(t->type, i0));
    };
    ggml_tensor * pre = ggml_add(c, ggml_mul(c, v2(mixes, hc, 0), v1(scale, 1, 0)), v1(base, hc, 0));
    pre = ggml_scale_bias(c, ggml_sigmoid(c, pre), 1.0f, m.hc_eps);
    ggml_tensor * po = ggml_add(c, ggml_mul(c, v2(mixes, hc, hc), v1(scale, 1, 1)), v1(base, hc, hc));
    *post = ggml_scale(c, ggml_sigmoid(c, po), 2.0f);
    *comb = ggml_dsv4_hc_comb(c, mixes, scale, base, m.hc_eps, m.hc_iters);
    return ggml_dsv4_hc_pre(c, x, pre);
}

// KDA (build_kda_layer): q, k, v each through a short causal convolution (its own history), silu; a per-channel decay
// (f_a -> f_b, + dt bias, times A, into [lower bound, 0] by a sigmoid) and beta; the gated delta rule over the state;
// the output RMS-normed per head and gated by sigmoid(g_a -> g_b)
static ggml_tensor * glm_kda_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x, int T) {
    Model & m = e.m;
    const int64_t hd = m.kda_hd, nh = m.n_head, di = m.ssm_inner, w1 = m.ssm_conv - 1;
    auto conv = [&](int which, ggml_tensor * w, ggml_tensor * cw) {
        ggml_tensor * xp = ggml_reshape_3d(c, ggml_mul_mat(c, w, x), di, T, 1);
        ggml_tensor * st = ggml_view_3d(c, L.conv_state, w1, di, 1, ggml_row_size(GGML_TYPE_F32, w1),
                                        ggml_row_size(GGML_TYPE_F32, w1 * di), ggml_row_size(GGML_TYPE_F32, which * w1 * di));
        ggml_tensor * cin = ggml_concat(c, st, ggml_transpose(c, xp), 0);                 // [w1 + T, di, 1]
        ggml_tensor * last = ggml_view_3d(c, cin, w1, di, 1, cin->nb[1], cin->nb[2], ggml_row_size(cin->type, T));
        ggml_build_forward_expand(gf, ggml_cpy(c, last, st));
        ggml_tensor * co = ggml_silu(c, ggml_ssm_conv(c, cin, ggml_reshape_2d(c, cw, m.ssm_conv, di)));
        return ggml_reshape_4d(c, co, hd, nh, T, 1);
    };
    ggml_tensor * q = conv(0, L.wq, L.conv_q), * k = conv(1, L.wk, L.conv_k), * v = conv(2, L.wv, L.conv_v);
    ggml_tensor * g = L.ssm_f_b ? ggml_mul_mat(c, L.ssm_f_b, ggml_mul_mat(c, L.ssm_f_a, x)) : ggml_mul_mat(c, L.ssm_f_a, x);
    g = ggml_add(c, g, L.ssm_dt);                                                         // (ling: no LoRA)
    ggml_tensor * A = ggml_reshape_3d(c, L.ssm_a, 1, nh, 1);                              // -exp(A_log); ling: exp(A_log)
    if (m.ling) {                                            // (bailingmoe3.cpp: the same gate, A's sign in the file)
        g = ggml_mul(c, ggml_reshape_3d(c, g, hd, nh, T), A);
        g = ggml_scale(c, ggml_sigmoid(c, g), m.kda_lb);
    } else if (m.kda_lb > -INFINITY) {
        g = ggml_mul(c, ggml_reshape_3d(c, g, hd, nh, T), A);
        g = ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, g, -1.0f)), m.kda_lb);
    } else {
        g = ggml_mul(c, ggml_reshape_3d(c, ggml_softplus(c, g), hd, nh, T), A);
    }
    g = ggml_reshape_4d(c, g, hd, nh, T, 1);
    ggml_tensor * beta = ggml_sigmoid(c, ggml_reshape_4d(c, ggml_mul_mat(c, L.ssm_beta, x), 1, nh, T, 1));
    q = l2_norm(c, q, 1e-6f);                                                             // (FLA's l2 norm)
    k = l2_norm(c, k, 1e-6f);
    ggml_tensor * s = ggml_reshape_4d(c, L.ssm_state, hd, hd, nh, 1);
    ggml_tensor * r = ggml_gated_delta_net(c, q, k, v, g, beta, s, 1);
    ggml_tensor * out = ggml_view_4d(c, r, hd, nh, T, 1, ggml_row_size(r->type, hd), ggml_row_size(r->type, hd * nh),
                                     ggml_row_size(r->type, hd * nh * T), 0);
    ggml_tensor * ns = ggml_view_4d(c, r, hd, hd, nh, 1, ggml_row_size(r->type, hd), ggml_row_size(r->type, hd * hd),
                                    ggml_row_size(r->type, hd * hd * nh), ggml_row_size(r->type, hd * nh * T));
    ggml_build_forward_expand(gf, ggml_cpy(c, ns, ggml_reshape_4d(c, L.ssm_state, (int64_t) hd * hd * nh, 1, 1, 1)));   // (as above)
    ggml_tensor * g2 = ggml_reshape_3d(c, L.ssm_g_b ? ggml_mul_mat(c, L.ssm_g_b, ggml_mul_mat(c, L.ssm_g_a, x))
                                                    : ggml_mul_mat(c, L.ssm_g_a, x), hd, nh, T);
    ggml_tensor * o = norm(c, ggml_reshape_3d(c, ggml_cont(c, out), hd, nh, T), L.ssm_norm, m.rms_eps);
    o = ggml_cont_2d(c, ggml_mul(c, o, ggml_sigmoid(c, g2)), di, T);
    return ggml_mul_mat(c, L.wo, o);
}

// the k-pool indexer (build_kpool_select): this batch's indexer key (LayerNorm) and pool gate into the cache; the pools
// it completes pooled (a softmax over a pool's members of gate + ape, per channel, weighting their keys); every pool
// scored against the indexer's queries (lightning indexer: relu(q.k), weighted per head by proj(x)); each token's
// n_top best pools, ordered by score, become its attention mask (with the causal mask added)
static ggml_tensor * glm_kpool_select(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                      ggml_tensor * qr, const Inputs & in, int p0, int T, int n_kv) {
    Model & m = e.m;
    const int64_t D = m.idx_dim, kp = m.kpool, nh = m.idx_heads;
    ggml_tensor * iq = ggml_reshape_3d(c, ggml_mul_mat(c, L.idx_q, qr), D, nh, T);
    ggml_tensor * ik = ggml_mul_mat(c, L.idx_k, x);
    ik = ggml_add(c, ggml_mul(c, ggml_norm(c, ik, m.norm_eps), L.idx_k_norm), L.idx_k_norm_b);
    ggml_tensor * ig = ggml_mul_mat(c, L.idx_gate, x);
    const size_t rr = L.idx_raw->nb[1];
    if (in.kv_idx) cache_rows(c, gf, L.idx_raw, ggml_concat(c, ik, ig, 0), 2 * D, T, in);
    else ggml_build_forward_expand(gf, ggml_cpy(c, ggml_concat(c, ik, ig, 0), ggml_view_2d(c, L.idx_raw, 2 * D, T, rr, rr * p0)));
    if (in.n_new > 0) {
        ggml_tensor * pk = ggml_view_3d(c, L.idx_raw, D, kp, in.n_new, rr, rr * kp, rr * kp * in.j_first);
        ggml_tensor * pg = ggml_view_3d(c, L.idx_raw, D, kp, in.n_new, rr, rr * kp, rr * kp * in.j_first + ggml_row_size(L.idx_raw->type, D));
        pk = ggml_cast(c, pk, GGML_TYPE_F32);
        pg = ggml_cast(c, pg, GGML_TYPE_F32);
        ggml_tensor * lg = ggml_cont(c, ggml_permute(c, ggml_add(c, pg, L.idx_ape), 1, 0, 2, 3));   // [kp, D, n_new]
        ggml_tensor * pr = ggml_reshape_3d(c, ggml_soft_max(c, ggml_reshape_2d(c, lg, kp, D * in.n_new)), kp, D, in.n_new);
        pk = ggml_cont(c, ggml_permute(c, pk, 1, 0, 2, 3));
        ggml_tensor * pooled = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_mul(c, pr, pk)), D, in.n_new);
        const size_t pr2 = L.idx_pool->nb[1];
        ggml_build_forward_expand(gf, ggml_cpy(c, pooled, ggml_view_2d(c, L.idx_pool, D, in.n_new, pr2, pr2 * in.j_first)));
    }
    ggml_tensor * pooled = ggml_view_3d(c, L.idx_pool, D, 1, in.n_pool, L.idx_pool->nb[1], L.idx_pool->nb[1], 0);
    ggml_tensor * wts = ggml_scale(c, ggml_mul_mat(c, L.idx_proj, x), 1.0f / sqrtf((float) (D * nh)));
    ggml_tensor * score = ggml_lightning_indexer(c, iq, pooled, wts, in.pool_mask);       // [n_pool, T]
    const int64_t n_pool = in.n_pool, n_top = std::min<int64_t>(n_pool, m.idx_top_k / kp);
    ggml_tensor * top = ggml_top_k(c, score, n_top);                                      // [n_top, T], unordered
    ggml_tensor * ss = ggml_get_rows(c, ggml_reshape_3d(c, score, 1, n_pool, T), top);    // ordered by score
    ggml_tensor * order = ggml_argsort(c, ggml_reshape_2d(c, ss, n_top, T), GGML_SORT_ORDER_DESC);
    top = ggml_get_rows(c, ggml_reshape_3d(c, ggml_cast(c, top, GGML_TYPE_F32), 1, n_top, T), order);
    top = ggml_cast(c, ggml_cont(c, ggml_reshape_2d(c, top, n_top, T)), GGML_TYPE_I32);
    ggml_tensor * sel_f = ggml_get_rows(c, in.pool_idxs, ggml_reshape_1d(c, top, n_top * T));
    sel_f = ggml_reshape_2d(c, sel_f, kp * n_top, T);                                     // cells (f32)
    if (m.kpool_tail) sel_f = ggml_concat(c, sel_f, in.tail_idxs, 0);
    const int64_t n_sel = sel_f->ne[0];
    if (n_sel != in.n_sel) die("glm5-next: the selection's width");
    ggml_tensor * mask_all = ggml_fill(c, ggml_new_tensor_4d(c, GGML_TYPE_F16, n_kv + n_sel, 1, 1, 1), -INFINITY);
    mask_all = ggml_reshape_3d(c, ggml_repeat_4d(c, mask_all, n_kv + n_sel, T, 1, 1), 1, n_kv + n_sel, T);
    ggml_tensor * zeros = ggml_fill(c, ggml_new_tensor_4d(c, GGML_TYPE_F16, n_sel, 1, 1, 1), 0.0f);
    zeros = ggml_reshape_3d(c, ggml_repeat_4d(c, zeros, n_sel, T, 1, 1), 1, n_sel, T);
    // a live slot writes its cell, a dead one its own dump row n_kv + slot: idx = dump + live (idx - dump)
    ggml_tensor * live = ggml_exp(c, in.sel_mask);
    ggml_tensor * dump = ggml_arange(c, (float) n_kv, (float) (n_kv + n_sel), 1.0f);
    ggml_tensor * idx_f = ggml_add(c, ggml_mul(c, ggml_sub(c, sel_f, dump), live), dump);
    ggml_tensor * sel = ggml_set_rows(c, mask_all, zeros, ggml_reshape_3d(c, ggml_cast(c, idx_f, GGML_TYPE_I32), n_sel, T, 1));
    sel = ggml_view_2d(c, sel, n_kv, T, sel->nb[2], 0);
    return ggml_add(c, sel, in.mask);
}

// MLA without positions (build_dsa_layer): the query through q_a (normed) and q_b, the latent through kv_a (normed)
// into the cache; wk_b absorbed into the query, attention over the latents (key and value), wv_b after it
static ggml_tensor * glm_mla_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                   const Inputs & in, int p0, int T, int n_kv) {
    Model & m = e.m;
    ggml_tensor * qr = norm(c, ggml_mul_mat(c, L.wq_a, x), L.q_a_norm, m.rms_eps);        // [q_lora, T]
    ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq_b, qr), m.mla_k, m.n_head, T);
    ggml_tensor * kv = norm(c, ggml_mul_mat(c, L.wkv_a, x), L.kv_a_norm, m.rms_eps);      // [kv_lora, T]
    ggml_tensor * qa = ggml_permute(c, ggml_mul_mat(c, L.wk_b, ggml_permute(c, q, 0, 2, 1, 3)), 0, 2, 1, 3);   // [kv_lora, n_head, T]
    ggml_tensor * mask = glm_kpool_select(e, c, gf, L, x, qr, in, p0, T, n_kv);
    const size_t row = L.lat_cache->nb[1];
    if (in.kv_idx) cache_rows(c, gf, L.lat_cache, kv, m.kv_lora, T, in);
    else ggml_build_forward_expand(gf, ggml_cpy(c, kv, ggml_view_2d(c, L.lat_cache, m.kv_lora, T, row, row * p0)));
    ggml_tensor * lat = ggml_view_3d(c, L.lat_cache, m.kv_lora, 1, n_kv, row, row, 0);
    ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, qa, 0, 2, 1, 3), ggml_permute(c, lat, 0, 2, 1, 3),
                                          ggml_permute(c, lat, 0, 2, 1, 3), mask, 1.0f / sqrtf((float) m.mla_k), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(o, in.n_sel);
    ggml_prec_set_acc(o, GGML_PREC_F32);                                                  // [kv_lora, n_head, T]
    ggml_format_name(o, "fattn-%d", e.cur_layer);
    ggml_format_name(mask, "kq_mask_dsa-%d", e.cur_layer);
    ggml_format_name(qa, "q_absorbed-%d", e.cur_layer);
    ggml_format_name(kv, "kv_cmpr-%d", e.cur_layer);
    o = ggml_permute(c, ggml_mul_mat(c, L.wv_b, ggml_permute(c, o, 0, 2, 1, 3)), 0, 2, 1, 3);   // [mla_v, n_head, T]
    o = ggml_reshape_2d(c, ggml_cont(c, o), (int64_t) m.mla_v * m.n_head, T);
    return ggml_mul_mat(c, L.wo, o);
}

// bailingmoe3's MLA (its graph): the query straight from x, split into its NoPE part (wk_b absorbed into it) and its
// RoPE part; the latent (normed) and the key's RoPE part from kv_a into the cache, one row; attention over the rows
// (key: the whole row, value: the latent), wv_b after it; each head gated by sigmoid(attn_gate x)
static ggml_tensor * ling_mla_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x,
                                    const Inputs & in, int p0, int T, int n_kv) {
    Model & m = e.m;
    const int64_t R = m.n_rot, nope = m.mla_k - R, KL = m.kv_lora, nh = m.n_head;
    ggml_tensor * qf = ggml_mul_mat(c, L.wq, x);                                          // [nh * mla_k, T]
    const size_t eq = ggml_element_size(qf), ek = sizeof(float);
    ggml_tensor * q_nope = ggml_view_3d(c, qf, nope, nh, T, eq * m.mla_k, eq * m.mla_k * nh, 0);
    ggml_tensor * q_pe = ggml_view_3d(c, qf, R, nh, T, eq * m.mla_k, eq * m.mla_k * nh, eq * nope);
    ggml_tensor * kva = ggml_mul_mat(c, L.wkv_a, x);                                      // [KL + R, T]
    ggml_tensor * kv = ggml_view_2d(c, kva, KL, T, ek * (KL + R), 0);
    ggml_tensor * k_pe = ggml_view_3d(c, kva, R, 1, T, ek * (KL + R), ek * (KL + R), ek * KL);
    ggml_tensor * pos = ggml_view_1d(c, in.pos, T, 0);                                    // (the first section: t)
    q_pe = ggml_rope_ext(c, q_pe, pos, nullptr, (int) R, GGML_ROPE_TYPE_NORMAL, 262144, m.rope_base, 1.0f, 0.0f, 1.0f,
                         32.0f, 1.0f);
    k_pe = ggml_rope_ext(c, k_pe, pos, nullptr, (int) R, GGML_ROPE_TYPE_NORMAL, 262144, m.rope_base, 1.0f, 0.0f, 1.0f,
                         32.0f, 1.0f);
    kv = norm(c, kv, L.kv_a_norm, m.rms_eps);
    ggml_tensor * qa = ggml_permute(c, ggml_mul_mat(c, L.wk_b, ggml_permute(c, q_nope, 0, 2, 1, 3)), 0, 2, 1, 3);   // [KL, nh, T]
    ggml_tensor * q = ggml_concat(c, qa, q_pe, 0);                                        // [KL + R, nh, T]
    ggml_tensor * kn = ggml_concat(c, ggml_reshape_2d(c, kv, KL, T), ggml_reshape_2d(c, k_pe, R, T), 0);   // [KL + R, T]
    const size_t row = L.lat_cache->nb[1];
    if (in.kv_idx) cache_rows(c, gf, L.lat_cache, kn, KL + R, T, in);
    else ggml_build_forward_expand(gf, ggml_cpy(c, kn, ggml_view_2d(c, L.lat_cache, KL + R, T, row, row * p0)));
    ggml_tensor * lk = ggml_view_3d(c, L.lat_cache, KL + R, 1, n_kv, row, row, 0);
    ggml_tensor * lv = ggml_view_3d(c, L.lat_cache, KL, 1, n_kv, row, row, 0);
    ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_permute(c, lk, 0, 2, 1, 3),
                                          ggml_permute(c, lv, 0, 2, 1, 3), in.mask, 1.0f / sqrtf((float) m.mla_k), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);                                                  // [KL, nh, T]
    o = ggml_permute(c, ggml_mul_mat(c, L.wv_b, ggml_permute(c, o, 0, 2, 1, 3)), 0, 2, 1, 3);   // [mla_v, nh, T]
    ggml_tensor * gate = ggml_sigmoid(c, ggml_reshape_3d(c, ggml_mul_mat(c, L.wgate, x), 1, nh, T));
    o = ggml_mul(c, ggml_cont(c, o), gate);
    return ggml_mul_mat(c, L.wo, ggml_reshape_2d(c, o, (int64_t) m.mla_v * nh, T));
}

// DeepSeek's group selection (llama.cpp's build_moe_ffn, n_expert_groups > 1): the biased scores [n_expert, T] with
// every expert outside the n_group_used groups of the best two-expert sums at -inf
static ggml_tensor * group_select(const Model & m, ggml_context * c, ggml_tensor * sel) {
    if (m.n_groups <= 1) return sel;
    const int64_t T = sel->ne[1], per = m.n_expert / m.n_groups;
    ggml_tensor * grp = ggml_reshape_3d(c, sel, per, m.n_groups, T);
    ggml_tensor * best2 = ggml_argsort_top_k(c, grp, 2);                                  // [2, n_groups, T]
    best2 = ggml_get_rows(c, ggml_reshape_4d(c, grp, 1, per, m.n_groups, T), best2);      // [1, 2, n_groups, T]
    ggml_tensor * score = ggml_sum_rows(c, ggml_reshape_3d(c, best2, 2, m.n_groups, T));  // [1, n_groups, T]
    ggml_tensor * keep = ggml_argsort_top_k(c, ggml_reshape_2d(c, score, m.n_groups, T), m.n_group_used);
    ggml_tensor * kept = ggml_get_rows(c, grp, keep);                                     // [per, n_group_used, T]
    kept = ggml_set_rows(c, ggml_fill(c, grp, -INFINITY), kept, keep);
    return ggml_reshape_2d(c, kept, m.n_expert, T);
}

// SwiGLU, clamped as llama.cpp's build_ffn does for glm5-next (limit: swiglu_clamp_shexp of the block)
static ggml_tensor * glm_swiglu(ggml_context * c, ggml_tensor * gate, ggml_tensor * up, float limit) {
    return limit > 1e-6f ? ggml_swiglu_clamp(c, gate, up, limit) : ggml_swiglu_split(c, gate, up);
}

// the MoE (build_moe_ffn, sigmoid gating): the bias only chooses; the chosen weights normalized and scaled; the experts
// SwiGLU (clamped) from the pool; plus the shared expert
// pred_in: the next block's MoE input as this block's residual gives it (its FFN mix and norm, without its attention):
// that block's router on it predicts its experts, chosen in the same step (read while the GPU goes on)
static ggml_tensor * glm_moe_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x, int T,
                                   ggml_tensor * pred_in, ggml_tensor * pred_in2 = nullptr) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    const int k = m.n_used;
    const bool hyb = hybrid_on(e, L, T);                     // ling --hybrid: the CPU computes the experts the pool
    if (hyb) {                                               // lacks, from the experts' input (as lfm_moe_layer)
        if (!ggml_is_contiguous(x)) x = ggml_cont(c, x);
        ggml_build_forward_expand(gf, x);
        e.xl[il] = x;
    }
#if defined(MOMOE_CUDA)
    const bool mbx = mb_on(e) && hyb && ns_paused(e, il);   // the mailboxes: no stops in this layer
    ggml_tensor * send = nullptr;
#else
    const bool mbx = false;
#endif
    ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, x);
    ggml_format_name(lg, "ffn_moe_logits-%d", il);
    // (deepseek41: sqrt(softplus) scores, accumulated in F32 as llama.cpp's: the choice is sensitive to near-ties)
    auto score = [&](ggml_tensor * t) {
        if (!m.ds41) return ggml_sigmoid(c, t);
        ggml_prec_set_acc(t, GGML_PREC_F32);
        return ggml_sqrt(c, ggml_softplus(c, t));
    };
    ggml_tensor * probs = score(lg);                                                      // [n_expert, T]
    // llama.cpp's order (the routing's nodes in a row: CUDA's topk-moe kernel): the choice and its weights first
    ggml_tensor * chosen = ggml_argsort_top_k(c, group_select(m, c, ggml_add(c, probs, L.exp_probs_b)), k);   // [k, T]
    ggml_tensor * w = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), chosen), k, T);
    w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
    w = ggml_scale(c, ggml_reshape_3d(c, w, 1, k, T), m.exp_scale);
    ggml_build_forward_expand(gf, w);
    ggml_tensor * top = ggml_cont(c, chosen);
    ggml_tensor * both = top;
    if (pred_in) {                                           // [k, 2T]: this block's, then the next one's predicted
        const Layer & N = m.layers[il + 1];
        ggml_tensor * pl = group_select(m, c, ggml_add(c, score(ggml_mul_mat(c, N.gate_inp, pred_in)), N.exp_probs_b));
        both = ggml_concat(c, top, ggml_cont(c, ggml_argsort_top_k(c, pl, k)), 1);
    }
    if (pred_in && pred_in2) {                               // [k, 3T]: and the one after it (MOMOE_PREDICT2)
        const Layer & N2 = m.layers[il + 2];
        ggml_tensor * pl = group_select(m, c, ggml_add(c, score(ggml_mul_mat(c, N2.gate_inp, pred_in2)), N2.exp_probs_b));
        both = ggml_concat(c, both, ggml_cont(c, ggml_argsort_top_k(c, pl, k)), 1);
    }
    Group & G = e.pool.g[L.grp];
    const bool big = staged(k, G.n_slots, T);
#if defined(MOMOE_CUDA)
    const bool rmb = e.rmb && !big && T <= HYB_MAX_T && !mbx;   // the resident mailbox (rmb_layer): no stop here
#else
    const bool rmb = false;
#endif
    ggml_format_name(both, rmb ? "rmbtopk-%d" : mbx ? "mbtopk-%d" : "topk-%d", il);   // topk-: made resident here (the stop)
    e.topk[il] = both;
    ggml_build_forward_expand(gf, both);
    if (pred_in) top = ggml_view_2d(c, both, k, T, both->nb[1], 0);
#if defined(MOMOE_CUDA)
    if (mbx) {                                               // the layer's input and choice to the CPU, at once
        MbLayer & B = e.mb.layer[il];
        B.send.n_f32 = m.n_embd * T;
        B.send.n_i32 = k * T;
        ggml_tensor * args[2] = {x, pred_in ? ggml_cont(c, top) : top};
        send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
        ggml_format_name(send, "mbsend-%d", il);
        ggml_build_forward_expand(gf, send);
        e.force_gpu.push_back(send);
        e.mb_graph = true;
        e.mb.graph.push_back(il);
    }
#endif
    ggml_tensor * slots = nullptr;
#if defined(MOMOE_CUDA)
    if (rmb) {                                               // the choice to the worker; the slots back from it
        MbLayer & B = e.mb.layer[il];
        B.send.n_i32 = (int32_t) ggml_nelements(both);
        B.wait.n_f32 = k * T;
        ggml_tensor * args[2] = {ggml_view_1d(c, lg, 1, 0), both};
        ggml_tensor * send = ggml_custom_4d(c, GGML_TYPE_I32, 1, 1, 1, 1, args, 2, momoe_mailbox_cpu_stub, 1, &B.send);
        ggml_format_name(send, "rmbsend-%d", il);
        ggml_build_forward_expand(gf, send);
        ggml_tensor * wargs[1] = {send};
        slots = ggml_custom_4d(c, GGML_TYPE_I32, k, T, 1, 1, wargs, 1, momoe_mailbox_cpu_stub, 1, &B.wait);
        ggml_format_name(slots, "rmbwait-%d", il);
        e.force_gpu.push_back(send);
        e.force_gpu.push_back(slots);
        e.mb_graph = true;
        e.mb.graph.push_back(il);
    } else
#endif
    slots = ggml_reshape_2d(c, ggml_get_rows(c, L.slot_map, ggml_reshape_1d(c, top, (int64_t) k * T)), k, T);
    ggml_tensor * as[3] = {big ? G.stage[0] : G.t[0], big ? G.stage[1] : G.t[1], big ? G.stage[2] : G.t[2]};
    if (big && e.preloaded) {                                // the layer's experts in a row in the pool: a view
        const int64_t s0 = L.slots_host()[0];
        for (int j = 0; j < 3; ++j)
            as[j] = ggml_view_3d(c, G.t[j], G.t[j]->ne[0], G.t[j]->ne[1], L.n_exp, G.t[j]->nb[1], G.t[j]->nb[2],
                                 (size_t) s0 * G.t[j]->nb[2]);
    }
    if (big) slots = top;
    ggml_tensor * x3 = ggml_reshape_3d(c, x, m.n_embd, 1, T);
    ggml_tensor * up = may_repeat(e, big, ggml_mul_mat_id(c, as[1], x3, slots));                              // [n_ff, k, T]
    ggml_tensor * gt = may_repeat(e, big, ggml_mul_mat_id(c, as[0], x3, slots));
    ggml_tensor * dn = may_repeat(e, big, ggml_mul_mat_id(c, as[2], glm_swiglu(c, gt, up, m.clamp_exp[il]), slots));   // [n_embd, k, T]
    if (helper_on(e, L, big))
        dn = ggml_add(c, dn, helper_down(e, c, L, x3, top, T,
                                         [&](ggml_tensor * g, ggml_tensor * u) { return glm_swiglu(c, g, u, m.clamp_exp[il]); }));
    // the shared expert before the CPU's experts are waited for: the GPU computes it while the CPU works
    ggml_tensor * sh = glm_swiglu(c, ggml_mul_mat(c, L.gate_sh, x), ggml_mul_mat(c, L.up_sh, x), m.clamp_sh[il]);
    sh = ggml_mul_mat(c, L.down_sh, sh);
    if (hyb) ggml_build_forward_expand(gf, sh);
#if defined(MOMOE_CUDA)
    if (send) {                                              // the CPU's experts, waited for after the GPU's own
        MbLayer & B = e.mb.layer[il];
        B.wait.n_f32 = m.n_embd * k * T;
        ggml_tensor * args[1] = {send};
        ggml_tensor * cpu = ggml_custom_4d(c, GGML_TYPE_F32, m.n_embd, k, T, 1, args, 1, momoe_mailbox_cpu_stub, 1, &B.wait);
        ggml_format_name(cpu, "mbwait-%d", il);
        e.force_gpu.push_back(cpu);
        dn = ggml_add(c, dn, cpu);
    } else
#endif
    if (hyb) {                                               // the missing experts' outputs (zero slots here), from the CPU
        ggml_format_name(dn, "moedn-%d", il);                // (joined at this stop: eval_cb)
        dn = ggml_add(c, dn, ggml_view_3d(c, L.cpu_dn, m.n_embd, k, T, L.cpu_dn->nb[1], L.cpu_dn->nb[2], 0));
    }
    dn = ggml_mul(c, dn, w);
    ggml_build_forward_expand(gf, dn);
    ggml_tensor * v[64];                                     // the views before the adds (llama.cpp's order)
    GGML_ASSERT(k <= 64);
    for (int i = 0; i < k; ++i) {
        v[i] = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
        ggml_build_forward_expand(gf, v[i]);
    }
    ggml_tensor * moe = v[0];
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(c, moe, v[i]);
        ggml_build_forward_expand(gf, moe);
    }
    ggml_format_name(moe, "ffn_moe_out-%d", il);
    return ggml_add(c, moe, sh);
}

// ------------------------------------------------------------------------------------------------ deepseek41
// DeepSeek-V4.1-Flash (llama.cpp's deepseek41.cpp, one sequence).  RoPE on the last n_rot of a row (ggml_rope_set_offset):
// a compressed block's (and its compressed rows', indexer's) at ds_comp_base with YaRN, a window-only block's plain
static ggml_tensor * ds_rope(const Model & m, ggml_context * c, ggml_tensor * t, ggml_tensor * pos, bool comp, bool back = false) {
    const int64_t nope = t->ne[0] - m.n_rot;
    t = (back ? ggml_rope_ext_back : ggml_rope_ext)(c, t, pos, nullptr, m.n_rot, 0, comp ? m.ds_ctx_orig : 0,
            comp ? m.ds_comp_base : m.rope_base, comp ? m.ds_freq_scale : 1.0f, comp ? m.ds_ext : 0.0f,
            comp ? m.ds_attn_factor : 1.0f, comp ? m.ds_beta_fast : 0.0f, comp ? m.ds_beta_slow : 0.0f);
    return ggml_rope_set_offset(t, (int) nope);
}

// mHC's three mixes of the streams x [n_embd, hc, T] (glm_hc_pre's, not applied): pre [hc, T], post [hc, T], comb
static void ds_hc_mixes(Engine & e, ggml_context * c, ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale,
                        ggml_tensor * base, ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) {
    Model & m = e.m;
    const int64_t hc = m.hc, T = x->ne[2];
    ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, (int64_t) hc * m.n_embd, T), m.rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);                                      // [(2 + hc) hc, T]
    auto v1 = [&](ggml_tensor * t, int64_t n, int64_t i0) { return ggml_view_1d(c, t, n, ggml_row_size(t->type, i0)); };
    auto v2 = [&](ggml_tensor * t, int64_t n, int64_t i0) {
        return ggml_view_2d(c, t, n, T, t->nb[1], ggml_row_size(t->type, i0));
    };
    ggml_tensor * p = ggml_add(c, ggml_mul(c, v2(mixes, hc, 0), v1(scale, 1, 0)), v1(base, hc, 0));
    *pre = ggml_scale_bias(c, ggml_sigmoid(c, p), 1.0f, m.hc_eps);
    ggml_tensor * po = ggml_add(c, ggml_mul(c, v2(mixes, hc, hc), v1(scale, 1, 1)), v1(base, hc, hc));
    *post = ggml_scale(c, ggml_sigmoid(c, po), 2.0f);
    *comb = ggml_dsv4_hc_comb(c, mixes, scale, base, m.hc_eps, m.hc_iters);
}

// Engram (build_engram): the hashed rows [eg_heads * eg_dim, T] -> a key per stream and one value; each stream gains
// the value, gated by sigmoid(signed sqrt(rms(x) . rms(key) . (q k) / sqrt(n_embd)))
static ggml_tensor * ds_engram(Engine & e, ggml_context * c, Layer & L, ggml_tensor * x, ggml_tensor * rows, int T) {
    Model & m = e.m;
    const int64_t hc = m.hc, d = m.n_embd;
    ggml_tensor * kv = ggml_mul_mat(c, L.eg_wkv, rows);                                  // [d (hc + 1), T]
    ggml_tensor * key = ggml_cont(c, ggml_view_3d(c, kv, d, hc, T, ggml_row_size(kv->type, d), kv->nb[1], 0));
    ggml_tensor * val = ggml_cont(c, ggml_view_3d(c, kv, d, 1, T, ggml_row_size(kv->type, d), kv->nb[1],
                                                  ggml_row_size(kv->type, hc * d)));
    ggml_tensor * dot = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, x, m.rms_eps), ggml_rms_norm(c, key, m.rms_eps)),
                                 ggml_mul(c, L.eg_q, L.eg_k));
    dot = ggml_scale(c, ggml_sum_rows(c, dot), 1.0f / sqrtf((float) d));                  // [1, hc, T]
    ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(c, ggml_step(c, ggml_neg(c, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(c, ggml_mul(c, mag, sign));
    return ggml_add(c, x, ggml_mul(c, ggml_repeat_4d(c, val, d, hc, T, 1), gate));
}

// a KV source's compressor: the groups this batch completes (in.ds_nb[r], a dummy one when none) from the state's and
// this batch's rows, normed, their index key, both rotated to the group's first position, into comp_cache; the
// unfinished group's rows kept
static void ds_compress(Engine & e, ggml_context * c, ggml_cgraph * gf, Layer & L, ggml_tensor * x, const Inputs & in, int T) {
    Model & m = e.m;
    const int r = L.ratio;
    const int64_t hd = m.ds_hd, nb = in.ds_nb[r];
    ggml_tensor * skv = ggml_mul_mat(c, L.comp_kv, x);                                    // [hd, T]
    ggml_tensor * comp;
    if (r > 1) {
        ggml_tensor * ssc = ggml_mul_mat(c, L.comp_gate, x);
        ggml_tensor * v = ggml_get_rows(c, ggml_concat(c, L.cst_kv, skv, 1), in.ds_ridx[r]);   // [hd, r nb]
        ggml_tensor * sc = ggml_get_rows(c, ggml_concat(c, L.cst_sc, ssc, 1), in.ds_ridx[r]);
        v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, hd, r, nb), 1, 0, 2, 3));   // [r, hd, nb]
        sc = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, sc, hd, r, nb), 1, 0, 2, 3));
        comp = ggml_sum_rows(c, ggml_mul(c, v, ggml_soft_max(c, sc)));                     // [1, hd, nb]
        comp = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, comp, 1, 0, 2, 3)), hd, nb);
        ggml_build_forward_expand(gf, comp);                 // (read before the state is written)
        ggml_build_forward_expand(gf, ggml_set_rows(c, L.cst_kv, ggml_get_rows(c, skv, in.ds_psrc[r]), in.ds_pdst[r]));
        ggml_build_forward_expand(gf, ggml_set_rows(c, L.cst_sc, ggml_get_rows(c, ssc, in.ds_psrc[r]), in.ds_pdst[r]));
    } else {
        comp = ggml_get_rows(c, skv, in.ds_ridx[r]);         // ratio 1: a row per position (a softmax over one)
    }
    comp = norm(c, comp, L.comp_norm, m.rms_eps);
    ggml_tensor * ik = norm(c, ggml_mul_mat(c, L.idx_k, comp), L.idx_k_norm, m.rms_eps);  // [idx_dim, nb]
    ik = ds_rope(m, c, ggml_reshape_3d(c, ik, m.idx_dim, 1, nb), in.ds_wpos[r], true);
    comp = ds_rope(m, c, ggml_reshape_3d(c, comp, hd, 1, nb), in.ds_wpos[r], true);
    ggml_tensor * row = ggml_concat(c, ggml_reshape_2d(c, comp, hd, nb), ggml_reshape_2d(c, ik, m.idx_dim, nb), 0);
    ggml_format_name(row, "comp_row-%d", e.cur_layer);
    ggml_build_forward_expand(gf, ggml_set_rows(c, L.comp_cache, row, in.ds_widx[r]));
}

// the products of earlier blocks a graph's later ones read: each index source's top-k, the candidate blocks
struct DsGraph {
    std::vector<ggml_tensor *> topk;
    ggml_tensor * cand = nullptr;
};

// an index source's selection: every visible compressed row scored (the lightning indexer), the top idx_top_k kept;
// past cand_blocks blocks, cand_src picks its best blocks (the newest pinned) and the sources after it score inside them
static ggml_tensor * ds_indexer(Engine & e, ggml_context * c, Layer & L, const Layer & S, ggml_tensor * x, ggml_tensor * qr,
                                ggml_tensor * pos, const Inputs & in, DsGraph & dg, int il, int T) {
    Model & m = e.m;
    const int r = L.ratio;
    const int64_t id = m.idx_dim, ih = m.idx_heads, nc = in.ds_nc[r];
    ggml_tensor * ik = ggml_view_4d(c, S.comp_cache, id, 1, nc, 1, S.comp_cache->nb[1], S.comp_cache->nb[1],
                                    S.comp_cache->nb[1] * nc, ggml_row_size(S.comp_cache->type, m.ds_hd));
    ggml_tensor * iq = ds_rope(m, c, ggml_reshape_3d(c, ggml_mul_mat(c, L.idx_q, qr), id, ih, T), pos, true);
    iq = ggml_reshape_4d(c, iq, id, ih, T, 1);
    ggml_tensor * iw = ggml_scale(c, ggml_mul_mat(c, L.idx_proj, x), 1.0f / sqrtf((float) (id * ih)));
    iw = ggml_reshape_4d(c, iw, ih, T, 1, 1);
    ggml_tensor * score = ggml_lightning_indexer(c, iq, ik, iw, ggml_reshape_4d(c, in.ds_cmask[r], nc, T, 1, 1));
    score = ggml_reshape_2d(c, score, nc, T);
    const int64_t bs = m.cand_bs, n_blk = bs ? (nc + bs - 1) / bs : 0;
    if (m.cand_src == il && in.ds_pin) {                     // the best row of each block, the newest block pinned
        ggml_tensor * s2 = score;
        if (n_blk * bs != nc)
            s2 = ggml_concat(c, s2, ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, n_blk * bs - nc, T), -INFINITY), 0);
        ggml_tensor * bmax = ggml_add(c, ggml_pool_1d(c, s2, GGML_OP_POOL_MAX, (int) bs, (int) bs, 0), in.ds_pin);
        dg.cand = ggml_top_k(c, bmax, (int) std::min<int64_t>(m.cand_blocks, n_blk));    // [cand_blocks, T]
    } else if (m.cand_src >= 0 && m.cand_src < il && dg.cand) {   // 0 inside the source's blocks, -inf elsewhere
        const int64_t kb = dg.cand->ne[0];
        ggml_tensor * bm = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n_blk, T), -INFINITY);
        ggml_tensor * zero = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, kb, T), 0.0f);
        bm = ggml_set_rows(c, bm, zero, ggml_reshape_2d(c, dg.cand, kb, T));
        bm = ggml_reshape_2d(c, ggml_repeat_4d(c, bm, bs, n_blk, T, 1), bs * n_blk, T);
        score = ggml_add(c, score, ggml_view_2d(c, bm, nc, T, bm->nb[1], 0));
    }
    return ggml_top_k(c, score, (int) std::min<int64_t>(nc, m.idx_top_k));               // [k, T]
}

// the attention of a block (build_attention41): the query (wq_a -> norm -> wq_b) and the KV row, rotated; the row into
// the window's ring; a compressed block also attends to its source's compressed rows (the top-k of its index source,
// as a mask), the source compressing first; per-head sinks; the output rotated back, through the grouped LoRAs
static ggml_tensor * ds_attn_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x,
                                   const Inputs & in, DsGraph & dg, int T) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    const int r = L.ratio;
    const int64_t hd = m.ds_hd;
    const bool comp = r != 0;
    ggml_tensor * pos = ggml_view_1d(c, in.pos, T, 0);                                    // (section 0: p0 + i)
    ggml_tensor * qr = norm(c, ggml_mul_mat(c, L.wq_a, x), L.q_a_norm, m.rms_eps);        // [q_lora, T]
    ggml_tensor * q = ds_rope(m, c, ggml_reshape_3d(c, ggml_mul_mat(c, L.wq_b, qr), hd, m.n_head, T), pos, comp);
    ggml_tensor * kv = norm(c, ggml_mul_mat(c, L.wkv_a, x), L.kv_a_norm, m.rms_eps);      // [hd, T]
    kv = ds_rope(m, c, ggml_reshape_3d(c, kv, hd, 1, T), pos, comp);
    ggml_build_forward_expand(gf, ggml_set_rows(c, L.k_cache, ggml_reshape_2d(c, kv, hd, T), in.kv_idx_swa));
    const int64_t R = in.mask_swa->ne[0];
    ggml_tensor * k_all = ggml_view_3d(c, L.k_cache, hd, 1, R, L.k_cache->nb[1], L.k_cache->nb[1], 0);
    ggml_tensor * mask = in.mask_swa;
    if (comp) {
        const Layer & S = m.layers[L.kv_src];
        if (L.kv_src == il) ds_compress(e, c, gf, L, x, in, T);
        if (L.topk_src == il) dg.topk[il] = ds_indexer(e, c, L, S, x, qr, pos, in, dg, il, T);
        ggml_tensor * top = dg.topk[L.topk_src];
        if (!top) die("deepseek41: block %s's index source made no selection", std::to_string(il).c_str());
        const int64_t nc = in.ds_nc[r], k = top->ne[0];
        ggml_tensor * ck = ggml_cont(c, ggml_view_3d(c, S.comp_cache, hd, 1, nc, S.comp_cache->nb[1], S.comp_cache->nb[1], 0));
        k_all = ggml_concat(c, ggml_cont(c, k_all), ck, 2);                               // [hd, 1, R + nc]
        // the selection as a mask: -inf, 0 at the top-k rows, plus the rows' visibility (build_top_k_mask)
        ggml_tensor * sel = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F16, 1, nc, T), -INFINITY);
        ggml_tensor * zero = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F16, 1, k, T), 0.0f);
        sel = ggml_reshape_2d(c, ggml_set_rows(c, sel, zero, top), nc, T);
        sel = ggml_add(c, sel, in.ds_cmask[r]);
        mask = ggml_concat(c, in.mask_swa, sel, 0);                                       // [R + nc, T]
    }
    ggml_tensor * kp = ggml_permute(c, k_all, 0, 2, 1, 3);
    ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), kp, kp, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(o, L.sinks);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    ggml_format_name(o, "fattn-%d", il);
    o = ds_rope(m, c, ggml_reshape_3d(c, o, hd, m.n_head, T), pos, comp, true);
    o = ggml_permute(c, ggml_reshape_3d(c, o, m.n_head / m.o_groups * hd, m.o_groups, T), 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(c, L.wo_a, o);                                        // [o_lora, T, o_groups]
    oa = ggml_cont_2d(c, ggml_permute(c, oa, 0, 2, 1, 3), (int64_t) m.o_lora * m.o_groups, T);
    return ggml_mul_mat(c, L.wo, oa);
}

// the Engram rows of a batch for module j: each token's n-gram hash (engram_hash: its last eg_ngram - 1 tokens mapped
// through eg_map, a dead one or the sequence's start padding every longer n-gram) -> eg_heads rows of its table
static void ds_engram_rows(Engine & e, int j, const int32_t * toks, int T, int p0, std::vector<float> & out) {
    Model & m = e.m;
    const int n = m.eg_ngram, H = m.eg_heads, per = m.eg_per;
    out.resize((size_t) T * H * m.eg_dim);
    std::vector<char> raw(m.eg_row[j]);
    const auto * tr = ggml_get_type_traits(m.eg_type);
    const uint64_t * mult = m.eg_mult.data() + (size_t) j * n;
    const uint64_t * primes = m.eg_primes.data() + (size_t) j * H, * offs = m.eg_offs.data() + (size_t) j * H;
    const int64_t pad = m.eg_map[m.eg_pad];
    auto cid = [&](int64_t t) -> int64_t { return t < 0 || t >= (int64_t) m.eg_map.size() ? -1 : m.eg_map[t]; };
    std::vector<uint64_t> g(n);
    for (int i = 0; i < T; ++i) {
        int64_t c0 = cid(toks[i]);
        bool blocked = c0 < 0;
        g[0] = (uint64_t) (blocked ? pad : c0);
        for (int s = 1; s < n; ++s) {
            const int p = p0 + i - s;
            const int64_t cs = p < 0 ? -1 : cid(p >= p0 ? toks[p - p0] : e.tok_at[p]);
            blocked = blocked || cs < 0;
            g[s] = (uint64_t) (blocked ? pad : cs);
        }
        uint64_t rolling = g[0] * mult[0];
        for (int o = 1; o < n; ++o) {
            rolling ^= g[o] * mult[o];
            for (int h = 0; h < per; ++h) {
                const int col = (o - 1) * per + h;
                const uint64_t row = rolling % primes[col] + offs[col];
                read_at(m.eg_fd[j], raw.data(), m.eg_row[j], m.eg_off[j] + row * m.eg_row[j]);
                tr->to_float(raw.data(), out.data() + ((size_t) i * H + col) * m.eg_dim, m.eg_dim);
            }
        }
    }
}

// a batch's compressed-row inputs for ratio r: how many groups it completes (at least one), how many state rows it keeps
static void ds_shape(const Engine & e, Inputs & in, int p0, int T) {
    const Model & m = e.m;
    for (int r = 1; r <= 2; ++r) {
        int done = 0;
        for (int i = 0; i < T; ++i) done += (p0 + i + 1) % r == 0;
        in.ds_nb[r] = std::max(1, done);
        in.ds_np[r] = std::min(T, r);
        const int rows = e.n_ctx / r + 1;                    // (the last: the dummy group's)
        in.ds_nc[r] = std::min(rows, std::max(256, ((p0 + T) / r + 255) / 256 * 256));
    }
    (void) m;
}

static void ds_new(Engine & e, ggml_context * c, Inputs & in, int p0, int T) {
    Model & m = e.m;
    ds_shape(e, in, p0, T);
    bool has[3] = {false, false, false};
    for (const Layer & L : m.layers) has[L.ratio] = has[L.ratio] || L.ratio > 0;
    for (int r = 1; r <= 2; ++r) {
        if (!has[r]) continue;
        auto mk = [&](ggml_type t, int64_t n) { ggml_tensor * x = ggml_new_tensor_1d(c, t, n); ggml_set_input(x); return x; };
        in.ds_cmask[r] = ggml_new_tensor_2d(c, GGML_TYPE_F16, in.ds_nc[r], T);
        ggml_set_input(in.ds_cmask[r]);
        in.ds_widx[r] = mk(GGML_TYPE_I64, in.ds_nb[r]);
        in.ds_wpos[r] = mk(GGML_TYPE_I32, in.ds_nb[r]);
        in.ds_ridx[r] = mk(GGML_TYPE_I32, (int64_t) r * in.ds_nb[r]);
        if (r > 1) {
            in.ds_pdst[r] = mk(GGML_TYPE_I64, in.ds_np[r]);
            in.ds_psrc[r] = mk(GGML_TYPE_I32, in.ds_np[r]);
        }
    }
    if (m.cand_src >= 0) {
        const int r = m.layers[m.cand_src].ratio, n_blk = (in.ds_nc[r] + m.cand_bs - 1) / m.cand_bs;
        if (n_blk > m.cand_blocks) {                         // (fewer blocks: all of them are kept anyway)
            in.ds_pin = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_blk, T);
            ggml_set_input(in.ds_pin);
        }
    }
    in.ds_eg.clear();
    for (size_t j = 0; j < m.eg_fd.size(); ++j) {
        ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t) m.eg_heads * m.eg_dim, T);
        ggml_set_input(t);
        in.ds_eg.push_back(t);
    }
}

static void ds_set(Engine & e, const Inputs & in, const int32_t * toks, int p0, int T) {
    Model & m = e.m;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int r = 1; r <= 2; ++r) {
        if (!in.ds_cmask[r]) continue;
        const int nc = in.ds_nc[r], nb = in.ds_nb[r];
        std::vector<ggml_fp16_t> mask((size_t) nc * T);
        for (int i = 0; i < T; ++i) {
            const int vis = (p0 + i + 1) / r;
            for (int j = 0; j < nc; ++j) mask[(size_t) i * nc + j] = j < vis ? zero : ninf;
        }
        ggml_backend_tensor_set(in.ds_cmask[r], mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        std::vector<int64_t> widx;
        std::vector<int32_t> wpos, ridx;
        for (int i = 0; i < T; ++i) {
            const int p = p0 + i;
            if ((p + 1) % r) continue;
            widx.push_back(p / r);
            wpos.push_back(p + 1 - r);
            for (int s = p + 1 - r; s <= p; ++s) ridx.push_back(r > 1 ? (s >= p0 ? r + (s - p0) : s % r) : s - p0);
        }
        while ((int) widx.size() < nb) {                     // the dummy group: the cache's last row, never visible
            widx.push_back(e.n_ctx / r);
            wpos.push_back(0);
            for (int s = 0; s < r; ++s) ridx.push_back(r > 1 ? r : 0);
        }
        ggml_backend_tensor_set(in.ds_widx[r], widx.data(), 0, widx.size() * sizeof(int64_t));
        ggml_backend_tensor_set(in.ds_wpos[r], wpos.data(), 0, wpos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in.ds_ridx[r], ridx.data(), 0, ridx.size() * sizeof(int32_t));
        if (in.ds_pdst[r]) {                                 // the last tokens' rows, at p % r
            std::vector<int64_t> dst;
            std::vector<int32_t> src;
            for (int i = T - in.ds_np[r]; i < T; ++i) {
                dst.push_back((p0 + i) % r);
                src.push_back(i);
            }
            ggml_backend_tensor_set(in.ds_pdst[r], dst.data(), 0, dst.size() * sizeof(int64_t));
            ggml_backend_tensor_set(in.ds_psrc[r], src.data(), 0, src.size() * sizeof(int32_t));
        }
    }
    if (in.ds_pin) {
        const int r = m.layers[m.cand_src].ratio;
        const int64_t n_blk = in.ds_pin->ne[0];
        std::vector<float> pin((size_t) n_blk * T, 0.0f);
        for (int i = 0; i < T; ++i) {
            const int nv = (p0 + i + 1) / r;
            if (nv > 0) pin[(size_t) i * n_blk + (nv - 1) / m.cand_bs] = 1e30f;
        }
        ggml_backend_tensor_set(in.ds_pin, pin.data(), 0, pin.size() * sizeof(float));
    }
    for (size_t j = 0; j < in.ds_eg.size(); ++j) {
        std::vector<float> rows;
        ds_engram_rows(e, (int) j, toks, T, p0, rows);
        ggml_backend_tensor_set(in.ds_eg[j], rows.data(), 0, rows.size() * sizeof(float));
    }
}

// run T tokens at positions p0..p0+T-1; returns the greedy next token after the last one
struct Out {
    bool all = false;                                        // the next token after every position (else the last)
    bool snap = false;                                       // keep the delta net's per-token states (a rollback)
    std::vector<int32_t> next;                               // the greedy next token(s)
    std::vector<float> hidden;                               // [T, n_embd]: the normalized hidden state of each position
    bool logits = false;                                     // the last position's logits (sampling)
    std::vector<float> last_logits;
    std::vector<float> all_logits;                           // logits and all: every position's [T, n_vocab]
};

static uint64_t g_sched_gen = 1;                             // a reused decode graph's allocation: bumped by every other
                                                             // graph the scheduler allocates
// flint: the sliding window's mask over the ring and the rows a batch of T tokens at p0 writes.  Row r holds the
// newest position <= p0 + T - 1 that is r mod swa_rows (one the batch writes, or one before it); token i sees it when
// it is no later than the token and fewer than n_swa positions back.  (A batch of at most swa_rows - n_swa + 1 tokens
// overwrites no row an earlier token of it still sees: the ring's size, see load)
// a picture whose cells see each other (llama.cpp's mtmd_decode_use_non_causal: Gemma 4 26B-A4B and 31B): the cells
// [lo, hi) the cell q also attends; none (lo = hi = 0) for text and for every other model
static void img_span(const Engine & e, int q, int & lo, int & hi) {
    lo = hi = 0;
    if (!e.m.gemma4 || e.m.n_embd == 1536 || e.m.n_embd == 2560) return;
    for (const auto & I : e.imgs)
        if (q >= I.s && q < I.s + I.n) { lo = I.s; hi = I.s + I.n; return; }
}

static void flint_swa_inputs(Engine & e, ggml_tensor * mask_t, ggml_tensor * idx_t, int p0, int T) {
    const Model & m = e.m;
    const int W = m.swa_rows, R = (int) mask_t->ne[0];      // the ring; the rows the graph reads (all once it wraps)
    if (T > W - m.n_swa + 1) die("flint: a batch larger than the sliding window's ring allows");
    std::vector<ggml_fp16_t> mask((size_t) R * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    const int last = p0 + T - 1;
    for (int r = 0; r < R; ++r) {
        const int p = last - (((last - r) % W) + W) % W;     // the position row r holds (< 0: never written)
        for (int i = 0; i < T; ++i) {
            const int q = p0 + i;
            int lo, hi;
            img_span(e, q, lo, hi);
            mask[(size_t) i * R + r] = p >= 0 && ((p <= q && q - p < m.n_swa) || (p >= lo && p < hi)) ? zero : ninf;
        }
    }
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    std::vector<int64_t> rows(T);
    for (int i = 0; i < T; ++i) rows[i] = (p0 + i) % W;
    ggml_backend_tensor_set(idx_t, rows.data(), 0, rows.size() * sizeof(int64_t));
}

#if defined(MOMOE_CUDA)
// momoe-cuda, Bolt's one-token decode through mailboxes: the graph is built once per attention window (256 positions)
// and kept, its per-token values (the token, the mask, the KV cache's row) set as inputs, so every token computes the
// same graph and ggml-cuda can replay it as a CUDA graph (a fresh graph a token launched ~1,500 kernels one by one)
struct DecodeCache {
    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tokens = nullptr, *mask = nullptr, *kv_idx = nullptr, *seq0 = nullptr, *logits = nullptr, *best = nullptr;
    ggml_tensor *on_cpu = nullptr, *on_gpu = nullptr;
    ggml_tensor *pos = nullptr, *mask_swa = nullptr, *kv_idx_swa = nullptr;   // flint: its RoPE positions, its window's mask, ring rows
    std::vector<ggml_tensor *> force_gpu;
    std::vector<int> mb_layers;
    int n_kv = -1;
    uint64_t sched_gen = 0;                                  // the scheduler's allocation it was made with
};
static DecodeCache g_dec;

static bool decode_cache_ok(const Engine & e, int T, const Out * out) {
    static const bool off = getenv("MOMOE_NO_GRAPH_REUSE") != nullptr;
    // Bolt; flint while its CPU path runs (the copies stop at every layer: no graph to keep)
    if (off || !e.mailbox || !(e.m.nemo || ((e.m.flint || e.m.gemma4 || e.m.ling) && e.hyb_active)) || T != 1 ||
        (out && (out->all || out->snap)))
        return false;
    for (const auto & G : e.pool.g)
        if (staged(e.m.n_used, G.n_slots, 1)) return false;  // every MoE layer through its mailbox
    return true;
}

static int forward_decode(Engine & e, const int32_t * toks, int p0, Out * out) {
    Model & m = e.m;
    const double tb0 = now_s();
    const int T = 1;
    const int n_kv = std::min(e.n_ctx, ((p0 + T + 255) / 256) * 256);
    DecodeCache & D = g_dec;
    // reused as it is: allocated once (an allocation rewires the graph's inputs to the scheduler's copies, so a second
    // one of the same graph breaks it); built and allocated again for a new attention window, or after another graph
    const bool reuse = D.n_kv == n_kv && D.sched_gen == g_sched_gen;
    if (!reuse) {
        if (D.c) ggml_free(D.c);
        ggml_init_params ip = {ggml_tensor_overhead() * 40000 + ggml_graph_overhead_custom(32768, false), nullptr, true};
        D.c = ggml_init(ip);
        ggml_context * c = D.c;
        D.gf = ggml_new_graph_custom(c, 32768, false);
        e.topk.assign(m.n_layer, nullptr);
        e.xl.assign(m.n_layer, nullptr);
        e.mb_graph = false;
        e.force_gpu.clear();
        e.mb.graph.clear();
        Inputs in;
        in.tokens = D.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
        ggml_set_input(in.tokens);
        in.mask = D.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, T);
        ggml_set_input(in.mask);
        D.kv_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);
        ggml_set_input(D.kv_idx);
        D.seq0 = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_input(D.seq0);
        if (m.flint || m.gemma4) {
            in.pos = D.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) T * 4);
            ggml_set_input(in.pos);
            in.mask_swa = D.mask_swa = ggml_new_tensor_2d(c, GGML_TYPE_F16, std::min(n_kv, m.swa_rows), T);
            ggml_set_input(in.mask_swa);
            in.kv_idx_swa = D.kv_idx_swa = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);
            ggml_set_input(in.kv_idx_swa);
        } else if (m.ling) {                                 // its MLA blocks: RoPE, the cache row as an input
            in.pos = D.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) T * 4);
            ggml_set_input(in.pos);
            in.kv_idx = D.kv_idx;
        }
        ggml_tensor * x = ggml_get_rows(c, m.tok_embd, in.tokens);
        D.on_cpu = D.on_gpu = nullptr;
        if (m.tok_embd->buffer && ggml_backend_buffer_is_host(m.tok_embd->buffer)) {
            D.on_cpu = x;
            x = D.on_gpu = ggml_scale(c, x, 1.0f);
        }
        if (m.gemma4) x = ggml_scale(c, x, sqrtf((float) m.n_embd));
        for (int il = 0; il < m.n_layer && m.gemma4; ++il) x = gemma_block(e, c, D.gf, il, x, in, p0, T, n_kv, D.kv_idx);
        for (int il = 0; il < m.n_layer && m.ling; ++il) {   // ling: as forward's blocks, without the predictions
            Layer & L = m.layers[il];
            ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
            e.cur_layer = il;
            h = L.kind == 'k' ? glm_kda_layer(e, c, D.gf, L, h, T) : ling_mla_layer(e, c, D.gf, L, h, in, p0, T, n_kv);
            x = ggml_add(c, h, x);
            ggml_tensor * f = norm(c, x, L.ffn_norm, m.rms_eps);
            if (il < m.n_dense)
                f = ggml_mul_mat(c, L.ffn_down, glm_swiglu(c, ggml_mul_mat(c, L.ffn_gate, f), ggml_mul_mat(c, L.ffn_up, f),
                                                           m.clamp_sh[il]));
            else
                f = glm_moe_layer(e, c, D.gf, il, f, T, nullptr);
            x = ggml_add(c, f, x);
        }
        for (int il = 0; il < m.n_layer && !m.flint && !m.gemma4 && !m.ling; ++il) {
            Layer & L = m.layers[il];
            ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
            h = L.kind == 'm' ? mamba2_layer(e, c, D.gf, L, h, D.seq0, T)
              : L.kind == 'a' ? nemo_attn_layer(e, c, D.gf, L, h, in, p0, T, n_kv, D.kv_idx)
                              : nemo_moe_layer(e, c, D.gf, il, h, T, nullptr);
            x = ggml_add(c, h, x);
        }
        for (int il = 0; il < m.n_layer && m.flint; ++il) {  // flint: as forward's lfm2moe blocks, every one attention
            Layer & L = m.layers[il];
            x = ggml_add(c, x, lfm_attn_layer(e, c, D.gf, L, norm(c, x, L.attn_norm, m.rms_eps), in, p0, T, n_kv, D.kv_idx));
            ggml_tensor * f = norm(c, x, L.ffn_norm, m.rms_eps);
            if (il < m.n_dense) {
                f = ggml_mul_mat(c, L.ffn_down, ggml_swiglu_split(c, ggml_mul_mat(c, L.ffn_gate, f), ggml_mul_mat(c, L.ffn_up, f)));
            } else {
                f = lfm_moe_layer(e, c, D.gf, il, f, T, nullptr, nullptr);
            }
            x = ggml_add(c, x, f);
        }
        x = norm(c, x, m.out_norm, m.rms_eps);
        D.logits = ggml_mul_mat(c, m.output, x);
        if (m.softcap > 0) D.logits = ggml_scale(c, ggml_tanh(c, ggml_scale(c, D.logits, 1.0f / m.softcap)), m.softcap);
        ggml_set_output(D.logits);
        D.best = ggml_argmax(c, D.logits);
        ggml_set_output(D.best);
        ggml_build_forward_expand(D.gf, D.best);
        D.force_gpu = e.force_gpu;
        D.mb_layers = e.mb.graph;
        D.n_kv = n_kv;
        if (D.mb_layers.size() != e.mb.moe.size()) die("the decode graph has a MoE layer without its mailbox");
        ggml_backend_sched_reset(e.sched);
        if (D.on_gpu) {
            ggml_backend_sched_set_tensor_backend(e.sched, D.on_cpu, e.cpu);
            ggml_backend_sched_set_tensor_backend(e.sched, D.on_gpu, e.metal);
        }
        for (ggml_tensor * t : D.force_gpu) ggml_backend_sched_set_tensor_backend(e.sched, t, e.metal);
        if (!ggml_backend_sched_alloc_graph(e.sched, D.gf)) die("cannot allocate the decode graph");
        D.sched_gen = ++g_sched_gen;
    }
    ggml_backend_tensor_set(D.tokens, toks, 0, sizeof(int32_t));
    std::vector<ggml_fp16_t> mask(n_kv);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < n_kv; ++j) mask[j] = j <= p0 ? zero : ninf;
    ggml_backend_tensor_set(D.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (D.mask_swa) {                                        // flint: the window over its ring, the RoPE position
        flint_swa_inputs(e, D.mask_swa, D.kv_idx_swa, p0, 1);
        const int32_t pp[4] = {p0, p0, p0, 0};
        ggml_backend_tensor_set(D.pos, pp, 0, sizeof pp);
    } else if (m.ling) {                                     // (ling: its MLA blocks' RoPE position)
        const int32_t pp[4] = {p0, p0, p0, 0};
        ggml_backend_tensor_set(D.pos, pp, 0, sizeof pp);
    }
    const int64_t pos = p0;
    ggml_backend_tensor_set(D.kv_idx, &pos, 0, sizeof pos);
    const int32_t zero_id = 0;
    if (D.seq0->buffer) ggml_backend_tensor_set(D.seq0, &zero_id, 0, sizeof zero_id);   // (flint: no Mamba state)
    e.batch_tokens = T;
    const double tc0 = now_s();
    e.t_build += tc0 - tb0;
    if (e.mb_stale) mb_rebuild(e);                           // (experts moved by a graph without the mailboxes)
    ++e.mb_graphs;
    mb_boundary(e);
    uint64_t mb_job;
    {
        std::lock_guard<std::mutex> lk(e.mb.mu);
        e.mb.graph = D.mb_layers;
        ++e.mb.cur;
        *e.mb.epoch = e.mb.cur;
        e.mb.T = T;
        mb_job = ++e.mb.job;
    }
    e.mb.cv.notify_all();
    ggml_backend_sched_set_eval_callback(e.sched, nullptr, nullptr);   // no stops: one graph (a CUDA graph once warm)
    const ggml_status st = ggml_backend_sched_graph_compute(e.sched, D.gf);
    ggml_backend_sched_set_eval_callback(e.sched, eval_cb, &e);
    if (st != GGML_STATUS_SUCCESS) die("the decode graph failed");
    if (*e.mb.timeout) die("the GPU waited for the CPU's experts and gave up (the mailbox worker stopped)");
    {
        std::unique_lock<std::mutex> lk(e.mb.mu);
        e.mb.cv.wait(lk, [&] { return e.mb.done_job == mb_job; });
    }
    e.t_compute += now_s() - tc0;
    int32_t best = 0;
    ggml_backend_tensor_get(D.best, &best, 0, sizeof best);
    if (out) {
        out->next.assign(1, best);
        if (out->logits) {
            out->last_logits.resize(D.logits->ne[0]);
            ggml_backend_tensor_get(D.logits, out->last_logits.data(), 0, out->last_logits.size() * sizeof(float));
        }
    }
    return best;
}
#endif


// the k-pool inputs (QSA's, glm5-next's) of a batch at positions p0..p0+T-1: made, then set after the allocation
static void kpool_shape(const Engine & e, Inputs & in, int p0, int T) {
    const Model & m = e.m;
    if (!m.kpool) return;
    const int kp = m.kpool, total = p0 + T;
    in.j_first = p0 / kp;
    in.n_new = std::max(0, total / kp - in.j_first);
    in.n_pool = std::max(64, (total / kp + 1 + 63) / 64 * 64);
    static const bool no_qsa = getenv("MOMOE_NO_QSA") != nullptr;   // a test: attention to every position
    in.sel = total / kp > m.idx_top_k / kp && !no_qsa;       // else every finished block is kept: dense attention
    if (m.glm5) in.sel = true;                               // (glm5-next: the selection is the mask, always: a token
                                                             // never sees its own unfinished pool)
    if (in.sel && m.glm5) in.n_sel = kp * std::min(in.n_pool, m.idx_top_k / kp) + (m.kpool_tail ? kp - 1 : 0);
}

static void kpool_new(Engine & e, ggml_context * c, Inputs & in, int p0, int T) {
    Model & m = e.m;
    if (!m.kpool) return;
    {
        const int kp = m.kpool;
        kpool_shape(e, in, p0, T);
        if (in.n_new > 0 && !m.glm5) {                       // (glm5-next: no positions)
            in.new_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * (int64_t) in.n_new);
            ggml_set_input(in.new_pos);
            if (in.kv_idx) {                                 // a reused graph: the new blocks' rows as inputs
                in.pool_src = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) kp * in.n_new);
                in.pool_dst = ggml_new_tensor_1d(c, GGML_TYPE_I64, in.n_new);
                ggml_set_input(in.pool_src);
                ggml_set_input(in.pool_dst);
            }
        }
        if (in.sel) {
            in.pool_idxs = ggml_new_tensor_2d(c, GGML_TYPE_F32, kp, in.n_pool);
            in.pool_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, in.n_pool, T);
            in.tail_idxs = ggml_new_tensor_2d(c, GGML_TYPE_F32, kp - 1, T);
            ggml_set_input(in.pool_idxs);
            ggml_set_input(in.pool_mask);
            ggml_set_input(in.tail_idxs);
            if (m.glm5) {
                in.sel_mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, in.n_sel, T);
                ggml_set_input(in.sel_mask);
            }
        }
    }
}

static void kpool_set(Engine & e, const Inputs & in, int p0, int T, int n_kv) {
    Model & m = e.m;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    const int j_first = m.kpool ? p0 / m.kpool : 0;         // (this call's: a reused graph was built at another)
    if (in.new_pos) {                                        // a block's keys rotate to its first position
        std::vector<int32_t> np(4 * (size_t) in.n_new);
        for (int j = 0; j < in.n_new; ++j)
            for (int s4 = 0; s4 < 4; ++s4) np[(size_t) s4 * in.n_new + j] = (j_first + j) * m.kpool;
        ggml_backend_tensor_set(in.new_pos, np.data(), 0, np.size() * sizeof(int32_t));
    }
    if (in.pool_src) {
        std::vector<int32_t> src((size_t) m.kpool * in.n_new);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (int32_t) ((size_t) j_first * m.kpool + i);
        ggml_backend_tensor_set(in.pool_src, src.data(), 0, src.size() * sizeof(int32_t));
        std::vector<int64_t> dst(in.n_new);
        for (int j = 0; j < in.n_new; ++j) dst[j] = j_first + j;
        ggml_backend_tensor_set(in.pool_dst, dst.data(), 0, dst.size() * sizeof(int64_t));
    }
    if (in.sel) {
        const int kp = m.kpool, n_real = (p0 + T) / kp;
        std::vector<float> pi((size_t) kp * in.n_pool);
        for (int j = 0; j < in.n_pool; ++j)
            for (int k = 0; k < kp; ++k) pi[(size_t) j * kp + k] = (float) (j < n_real ? j * kp + k : n_kv);
        ggml_backend_tensor_set(in.pool_idxs, pi.data(), 0, pi.size() * sizeof(float));
        std::vector<ggml_fp16_t> pm((size_t) in.n_pool * T);
        std::vector<float> ti((size_t) (kp - 1) * T);
        for (int i = 0; i < T; ++i) {
            const int p = p0 + i, nv = (p + 1) / kp, n_tail = (p + 1) % kp;
            for (int j = 0; j < in.n_pool; ++j) pm[(size_t) i * in.n_pool + j] = j < nv ? zero : ninf;
            for (int k = 0; k < kp - 1; ++k) ti[(size_t) i * (kp - 1) + k] = (float) (k < n_tail ? p - k : n_kv);
        }
        ggml_backend_tensor_set(in.pool_mask, pm.data(), 0, pm.size() * sizeof(ggml_fp16_t));
        if (in.tail_idxs->buffer) ggml_backend_tensor_set(in.tail_idxs, ti.data(), 0, ti.size() * sizeof(float));
        if (in.sel_mask) {                                   // glm5-next: a token's first min(visible, n_top) pools
            const int n_top = std::min(in.n_pool, m.idx_top_k / kp);   // (ordered by score) are live, then its tail
            std::vector<float> sm((size_t) in.n_sel * T, -INFINITY);
            for (int i = 0; i < T; ++i) {
                const int p = p0 + i, nvc = std::min((p + 1) / kp, n_top), n_tail = (p + 1) % kp;
                float * r = sm.data() + (size_t) i * in.n_sel;
                for (int j = 0; j < nvc * kp; ++j) r[j] = 0.0f;
                if (m.kpool_tail)
                    for (int k2 = 0; k2 < kp - 1; ++k2) r[n_top * kp + k2] = k2 < n_tail ? 0.0f : -INFINITY;
            }
            ggml_backend_tensor_set(in.sel_mask, sm.data(), 0, sm.size() * sizeof(float));
        }
    }
}

// one forward graph: its tensors, and what forward() sets and reads back
struct Built {
    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    Inputs in;
    ggml_tensor *seq0 = nullptr, *ple_in = nullptr, *on_cpu = nullptr, *on_gpu = nullptr;
    ggml_tensor *hid = nullptr, *logits = nullptr, *best = nullptr;
};

// the graph of T tokens at p0.. (by_input: the caches' rows as an input, so the graph can be computed again at other
// positions with the same shapes)
static void build_forward(Engine & e, int T, int p0, int n_kv, Out * out, Built & b, bool by_input) {
    Model & m = e.m;
    ggml_init_params ip = {ggml_tensor_overhead() * 40000 + ggml_graph_overhead_custom(32768, false), nullptr, true};
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 32768, false);
    Inputs in;
    e.topk.assign(m.n_layer, nullptr);
    e.xl.assign(m.n_layer, nullptr);
#if defined(MOMOE_CUDA)
    e.mb_graph = false;
    e.force_gpu.clear();
    e.mb.graph.clear();
#endif
    e.ns_graph = false;
    if (e.nostall && T <= 8) {                               // a few tokens: no layer's experts staged
        bool small = true;
        for (auto & G : e.pool.g) small = small && !staged(m.n_used, G.n_slots, T);
        e.ns_graph = small && ns_publish(e);
    }
    in.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(in.tokens);
    in.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) T * 4);
    ggml_set_input(in.pos);
    in.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, T);
    ggml_set_input(in.mask);
    if (m.flint || m.gemma4 || m.ds41) {
        in.mask_swa = ggml_new_tensor_2d(c, GGML_TYPE_F16, std::min(n_kv, m.swa_rows), T);
        ggml_set_input(in.mask_swa);
        in.kv_idx_swa = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);
        ggml_set_input(in.kv_idx_swa);
    }
    if (by_input) {                                          // the rows written: an input (a reused graph)
        in.kv_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);
        ggml_set_input(in.kv_idx);
    }
    ggml_tensor * x = ggml_get_rows(c, m.tok_embd, in.tokens);
    ggml_set_name(x, "model.input_embed");
    ggml_tensor * on_cpu = nullptr, * on_gpu = nullptr;      // momoe-cuda: the embeddings are in RAM (see load): their
    if (!kHostMem && m.tok_embd->buffer && ggml_backend_buffer_is_host(m.tok_embd->buffer)) {   // rows are looked up on
        on_cpu = x;                                          // the CPU (else the scheduler copies the whole table to the
        x = on_gpu = ggml_scale(c, x, 1.0f);                 // GPU every graph), and everything after runs on the GPU
    }
    if (m.gemma4) x = ggml_scale(c, x, sqrtf((float) m.n_embd));   // (gemma4: the embeddings scaled)
    if (e.img_now) {                                         // pictures: their cells take the encoder's rows (not scaled)
        in.img = ggml_new_tensor_2d(c, GGML_TYPE_F32, m.n_embd, T);
        ggml_set_input(in.img);
        in.img_keep = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T);
        ggml_set_input(in.img_keep);
        x = ggml_add(c, ggml_mul(c, x, in.img_keep), in.img);
    }
    kpool_new(e, c, in, p0, T);
    if (m.ds41) ds_new(e, c, in, p0, T);
    ggml_tensor * ple_in = nullptr;
    if (m.ple_layer >= 0) {
        if (out && out->snap) die("verify-window snapshots are not supported with qwen4exp");
        ple_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t) m.ple_dim * m.ple_heads, T);
        ggml_set_input(ple_in);
    }
    ggml_tensor * seq0 = nullptr;                            // nemo: the scan's state slot (one sequence: 0)
    if (m.nemo) {
        seq0 = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_input(seq0);
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
            ggml_format_name(h, "attn_norm-%d", il);
            ggml_tensor * pred = L.kind == 'e' && e.predict && L.next_moe >= 0
                ? norm(c, x, m.layers[L.next_moe].attn_norm, m.rms_eps) : nullptr;
            h = L.kind == 'm' ? mamba2_layer(e, c, gf, L, h, seq0, T)
              : L.kind == 'a' ? nemo_attn_layer(e, c, gf, L, h, in, p0, T, n_kv)
                              : nemo_moe_layer(e, c, gf, il, h, T, pred);
            x = ggml_add(c, h, x);
            ggml_format_name(x, "l_out-%d", il);
        }
    }
    if (m.lfm) {                                             // operator (conv or attention), then the FFN, each with
        for (int il = 0; il < m.n_layer; ++il) {             // its pre-norm and residual
            Layer & L = m.layers[il];
            ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
            ggml_format_name(h, "operator_norm-%d", il);
            h = L.kind == 'c' ? lfm_conv_layer(e, c, gf, L, h, T) : lfm_attn_layer(e, c, gf, L, h, in, p0, T, n_kv);
            ggml_format_name(h, L.kind == 'c' ? "conv_out-%d" : "attn_out-%d", il);
            x = ggml_add(c, x, h);
            ggml_tensor * f = norm(c, x, L.ffn_norm, m.rms_eps);
            ggml_format_name(f, "ffn_norm-%d", il);
            if (il < m.n_dense) {
                f = ggml_mul_mat(c, L.ffn_down, ggml_swiglu_split(c, ggml_mul_mat(c, L.ffn_gate, f), ggml_mul_mat(c, L.ffn_up, f)));
            } else {                                         // the next blocks' routers on their FFN norm of this residual
                ggml_tensor * pred = e.predict && il + 1 < m.n_layer ? norm(c, x, m.layers[il + 1].ffn_norm, m.rms_eps) : nullptr;
                ggml_tensor * pred2 = pred && e.predict2 && il + 2 < m.n_layer
                                    ? norm(c, x, m.layers[il + 2].ffn_norm, m.rms_eps) : nullptr;
                f = lfm_moe_layer(e, c, gf, il, f, T, pred, pred2);   // (flint: with its shared expert)
            }
            ggml_format_name(f, "ffn_out-%d", il);
            x = ggml_add(c, x, f);
            ggml_format_name(x, "l_out-%d", il);
        }
    }
    if (m.glm5) {                                            // mHC: the streams start as hc copies of the embedding
        ggml_tensor * res = ggml_repeat_4d(c, ggml_reshape_3d(c, x, m.n_embd, 1, T), m.n_embd, m.hc, T, 1);
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            ggml_tensor * post = nullptr, * comb = nullptr;
            ggml_tensor * h = glm_hc_pre(e, c, res, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &post, &comb);
            h = norm(c, h, L.attn_norm, m.rms_eps);
            ggml_format_name(h, "attn_norm-%d", il);
            ggml_build_forward_expand(gf, h);
            e.cur_layer = il;
            h = L.kind == 'k' ? glm_kda_layer(e, c, gf, L, h, T) : glm_mla_layer(e, c, gf, L, h, in, p0, T, n_kv);
            ggml_format_name(h, L.kind == 'k' ? "kda_out-%d" : "attn_out-%d", il);
            res = ggml_dsv4_hc_post(c, h, res, post, comb);
            ggml_tensor * f = glm_hc_pre(e, c, res, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &post, &comb);
            f = norm(c, f, L.ffn_norm, m.rms_eps);
            ggml_format_name(f, "ffn_norm-%d", il);
            if (il < m.n_dense)
                f = ggml_mul_mat(c, L.ffn_down, glm_swiglu(c, ggml_mul_mat(c, L.ffn_gate, f), ggml_mul_mat(c, L.ffn_up, f), m.clamp_sh[il]));
            else {
                ggml_tensor * pred = nullptr, * pred2 = nullptr;   // the next blocks' predicted MoE inputs
                auto pred_of = [&](int j) {                  // (see glm_moe_layer)
                    const Layer & N = m.layers[j];
                    ggml_tensor * po = nullptr, * co = nullptr;
                    return norm(c, glm_hc_pre(e, c, res, N.hc_ffn_fn, N.hc_ffn_scale, N.hc_ffn_base, &po, &co),
                                N.ffn_norm, m.rms_eps);
                };
                if (e.predict && il + 1 < m.n_layer) pred = pred_of(il + 1);
                if (pred && e.predict2 && il + 2 < m.n_layer) pred2 = pred_of(il + 2);
                f = glm_moe_layer(e, c, gf, il, f, T, pred, pred2);
            }
            ggml_format_name(f, "ffn_out-%d", il);
            res = ggml_dsv4_hc_post(c, f, res, post, comb);
            ggml_format_name(res, "l_out-%d", il);
        }
        x = res;                                             // [n_embd, hc, T]
    }
    if (m.ds41) {                                            // mHC as glm5-next's, its pre-mix one sublayer late (llama.cpp's
        DsGraph dg;                                          // "single-pass mHC"): the attention collapses the streams with
        dg.topk.assign(m.n_layer, nullptr);                  // the previous block's FFN pre-mix (block 0: stream 0), the FFN
        ggml_tensor * res = ggml_repeat_4d(c, ggml_reshape_3d(c, x, m.n_embd, 1, T), m.n_embd, m.hc, T, 1);   // with this
        ggml_tensor * pre_prev = nullptr;                    // block's attention pre-mix; the head with the last FFN's
        std::vector<ggml_tensor *> feats;                    // the DSpark drafter's input: its target blocks' inputs
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            if (L.eg >= 0) res = ds_engram(e, c, L, res, in.ds_eg[L.eg], T);
            if (e.ds.on && std::find(e.ds.targets.begin(), e.ds.targets.end(), il) != e.ds.targets.end()) {
                ggml_tensor * acc = nullptr;                 // (build_hc_mean: the streams' mean)
                for (int64_t s4 = 0; s4 < m.hc; ++s4) {
                    ggml_tensor * v = ggml_view_2d(c, res, m.n_embd, T, res->nb[2], s4 * res->nb[1]);
                    acc = acc ? ggml_add(c, acc, v) : v;
                }
                feats.push_back(ggml_scale(c, acc, 1.0f / m.hc));
            }
            ggml_tensor * apre, * apost, * acomb, * fpre, * fpost, * fcomb;
            ds_hc_mixes(e, c, res, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &apre, &apost, &acomb);
            ggml_tensor * h = pre_prev ? ggml_dsv4_hc_pre(c, res, pre_prev)
                                       : ggml_cont(c, ggml_view_2d(c, res, m.n_embd, T, res->nb[2], 0));
            h = norm(c, h, L.attn_norm, m.rms_eps);
            ggml_format_name(h, "attn_norm-%d", il);
            e.cur_layer = il;
            h = ds_attn_layer(e, c, gf, il, h, in, dg, T);
            ggml_format_name(h, "attn_out-%d", il);
            res = ggml_dsv4_hc_post(c, h, res, apost, acomb);
            ds_hc_mixes(e, c, res, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &fpre, &fpost, &fcomb);
            ggml_tensor * f = norm(c, ggml_dsv4_hc_pre(c, res, apre), L.ffn_norm, m.rms_eps);
            ggml_format_name(f, "ffn_norm-%d", il);
            ggml_tensor * pred = nullptr, * pred2 = nullptr;     // the next blocks' MoE inputs as guessed from here: their
            auto pred_of = [&](int j) {                      // attention pre-mix of these streams (see glm_moe_layer)
                const Layer & N = m.layers[j];
                ggml_tensor * pp, * po, * co;
                ds_hc_mixes(e, c, res, N.hc_attn_fn, N.hc_attn_scale, N.hc_attn_base, &pp, &po, &co);
                return norm(c, ggml_dsv4_hc_pre(c, res, pp), N.ffn_norm, m.rms_eps);
            };
            if (e.predict && il + 1 < m.n_layer) pred = pred_of(il + 1);
            if (pred && e.predict2 && il + 2 < m.n_layer) pred2 = pred_of(il + 2);
            f = glm_moe_layer(e, c, gf, il, f, T, pred, pred2);
            ggml_format_name(f, "ffn_out-%d", il);
            res = ggml_dsv4_hc_post(c, f, res, fpost, fcomb);
            ggml_format_name(res, "l_out-%d", il);
            pre_prev = fpre;
        }
        if (e.ds.on) {                                       // the drafter's rings at these positions (graph_dsv41's
            ggml_tensor * f = feats[0];                      // embd batch): fc, enc_norm, each block's KV row, rotated
            for (size_t j = 1; j < feats.size(); ++j) f = ggml_concat(c, f, feats[j], 0);
            ggml_tensor * xe = norm(c, ggml_mul_mat(c, e.ds.fc, f), e.ds.enc_norm, m.rms_eps);
            ggml_tensor * pos = ggml_view_1d(c, in.pos, T, 0);
            for (Layer & D : e.ds.layers) {
                ggml_tensor * kv = norm(c, ggml_mul_mat(c, D.wkv_a, xe), D.kv_a_norm, m.rms_eps);
                kv = ds_rope(m, c, ggml_reshape_3d(c, kv, m.ds_hd, 1, T), pos, false);
                ggml_build_forward_expand(gf, ggml_set_rows(c, D.k_cache, ggml_reshape_2d(c, kv, m.ds_hd, T), in.kv_idx_swa));
            }
        }
        const int64_t To = out && out->all ? T : 1;          // (only the positions whose logits are wanted)
        res = ggml_view_3d(c, res, m.n_embd, m.hc, To, res->nb[1], res->nb[2], (size_t) (T - To) * res->nb[2]);
        pre_prev = ggml_view_2d(c, pre_prev, m.hc, To, pre_prev->nb[1], (size_t) (T - To) * pre_prev->nb[1]);
        x = ggml_dsv4_hc_pre(c, res, pre_prev);              // [n_embd, To]
    }
    if (m.ling) {                                            // pre-norm blocks: KDA or MLA, then the FFN, each with
        for (int il = 0; il < m.n_layer; ++il) {             // its residual (bailingmoe3.cpp)
            Layer & L = m.layers[il];
            ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
            ggml_format_name(h, "attn_norm-%d", il);
            e.cur_layer = il;
            h = L.kind == 'k' ? glm_kda_layer(e, c, gf, L, h, T) : ling_mla_layer(e, c, gf, L, h, in, p0, T, n_kv);
            ggml_format_name(h, L.kind == 'k' ? "kda_out-%d" : "mla_out-%d", il);
            x = ggml_add(c, h, x);
            ggml_tensor * f = norm(c, x, L.ffn_norm, m.rms_eps);
            ggml_format_name(f, "ffn_norm-%d", il);
            if (il < m.n_dense) {
                f = ggml_mul_mat(c, L.ffn_down, glm_swiglu(c, ggml_mul_mat(c, L.ffn_gate, f), ggml_mul_mat(c, L.ffn_up, f),
                                                           m.clamp_sh[il]));
            } else {                                         // the next blocks' routers on their FFN norm of this residual
                ggml_tensor * pred = e.predict && il + 1 < m.n_layer ? norm(c, x, m.layers[il + 1].ffn_norm, m.rms_eps) : nullptr;
                ggml_tensor * pred2 = pred && e.predict2 && il + 2 < m.n_layer
                                    ? norm(c, x, m.layers[il + 2].ffn_norm, m.rms_eps) : nullptr;
                f = glm_moe_layer(e, c, gf, il, f, T, pred, pred2);
            }
            ggml_format_name(f, "ffn_out-%d", il);
            x = ggml_add(c, f, x);
            ggml_format_name(x, "l_out-%d", il);
        }
    }
    if (m.qwen4) {                                           // the hyper-connection streams start as hc copies
        ggml_tensor * res = ggml_repeat_4d(c, ggml_reshape_3d(c, x, m.n_embd, 1, T), m.n_embd, m.hc, T, 1);
        for (int il = 0; il < m.n_layer; ++il) {
            Layer & L = m.layers[il];
            if (il == m.ple_layer) res = ple_layer(e, c, gf, L, ple_in, res, T);
            ggml_tensor * inject = nullptr;
            ggml_tensor * h = hc_mix(e, c, res, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inject);
            ggml_format_name(h, "hc_mixed-%d", il);
            h = L.recurrent ? delta_net_layer(e, c, gf, L, h, T, false) : attn_layer(e, c, gf, L, h, in, p0, T, n_kv);
            ggml_format_name(h, L.recurrent ? "linear_attn_out-%d" : "attn_out-%d", il);
            res = hc_combine(e, c, res, h, inject);
            ggml_tensor * f = hc_mix(e, c, res, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject);
            // the predictions: the next layers' MoE input mixers over this residual
            ggml_tensor * pred = nullptr, * pred2 = nullptr;
            if (e.predict && il + 1 < m.n_layer) {
                Layer & N = m.layers[il + 1];
                pred = hc_mix(e, c, res, N.hc_ffn_norm, N.hc_ffn_down, N.hc_ffn_up, nullptr, nullptr);
            }
            if (e.predict2 && il + 2 < m.n_layer) {
                Layer & N = m.layers[il + 2];
                pred2 = hc_mix(e, c, res, N.hc_ffn_norm, N.hc_ffn_down, N.hc_ffn_up, nullptr, nullptr);
            }
            f = moe_layer(e, c, gf, il, f, T, pred, pred2);
            ggml_format_name(f, "ffn_out-%d", il);
            res = hc_combine(e, c, res, f, inject);
            ggml_format_name(res, "l_last-%d", il);
        }
        x = res;                                             // [n_embd, hc, T]
    }
    for (int il = 0; il < m.n_layer && m.gemma4; ++il) x = gemma_block(e, c, gf, il, x, in, p0, T, n_kv);
    for (int il = 0; il < m.n_layer && !m.qwen4 && !m.nemo && !m.glm5 && !m.lfm && !m.gemma4 && !m.ling && !m.ds41; ++il) {
        Layer & L = m.layers[il];
        ggml_tensor * h = norm(c, x, L.attn_norm, m.rms_eps);
        ggml_format_name(h, "attn_norm-%d", il);
        h = L.recurrent ? delta_net_layer(e, c, gf, L, h, T, out && out->snap)
                        : attn_layer(e, c, gf, L, h, in, p0, T, n_kv);
        ggml_format_name(h, L.recurrent ? "linear_attn_out-%d" : "attn_out-%d", il);
        x = ggml_add(c, h, x);
        ggml_format_name(x, "attn_residual-%d", il);
        ggml_tensor * pn = norm(c, x, L.post_norm, m.rms_eps);
        ggml_format_name(pn, "attn_post_norm-%d", il);
        ggml_tensor * pred = e.predict && il + 1 < m.n_layer ? norm(c, x, m.layers[il + 1].post_norm, m.rms_eps) : nullptr;
        ggml_tensor * pred2 = e.predict2 && il + 2 < m.n_layer ? norm(c, x, m.layers[il + 2].post_norm, m.rms_eps) : nullptr;
        ggml_tensor * f = moe_layer(e, c, gf, il, pn, T, pred, pred2);
        ggml_format_name(f, "ffn_out-%d", il);
        x = ggml_add(c, f, x);
        ggml_format_name(x, "l_out-%d", il);
    }
    ggml_tensor * hid = nullptr;
    if (out && m.has_mtp) {                                  // every position's normalized state: the MTP layer's input
        // (qwen4exp: the hc streams before the final mixer, as llama.cpp's h_nextn)
        if (m.glm5) {                                        // (h_nextn: the streams' mean, output-normed)
            ggml_tensor * acc = nullptr;
            for (int64_t s4 = 0; s4 < m.hc; ++s4) {
                ggml_tensor * v = ggml_view_2d(c, x, m.n_embd, T, x->nb[2], s4 * x->nb[1]);
                acc = acc ? ggml_add(c, acc, v) : v;
            }
            hid = norm(c, ggml_scale(c, acc, 1.0f / m.hc), m.out_norm, m.rms_eps);
        } else
        hid = m.qwen4 ? ggml_cont_2d(c, x, m.h_dim(), T) : norm(c, x, m.out_norm, m.rms_eps);
        ggml_set_output(hid);
        ggml_build_forward_expand(gf, hid);
    }
    if (m.qwen4) {                                           // the final mixer is the output norm
        if (!(out && out->all)) x = ggml_view_3d(c, x, m.n_embd, m.hc, 1, x->nb[1], x->nb[2], (size_t) (T - 1) * x->nb[2]);
        x = hc_mix(e, c, x, m.hc_head_norm, m.hc_head_down, m.hc_head_up, nullptr, nullptr);
    } else if (m.glm5 && hid && out->all) {
        x = hid;
    } else if (m.glm5) {                                     // the streams' mean (build_hc_mean), then the output norm
        const int64_t To = out && out->all ? T : 1;
        ggml_tensor * xs = ggml_view_3d(c, x, m.n_embd, m.hc, To, x->nb[1], x->nb[2], (size_t) (T - To) * x->nb[2]);
        ggml_tensor * acc = nullptr;
        for (int64_t s4 = 0; s4 < m.hc; ++s4) {
            ggml_tensor * v = ggml_view_2d(c, xs, m.n_embd, To, xs->nb[2], s4 * xs->nb[1]);
            acc = acc ? ggml_add(c, acc, v) : v;
        }
        x = norm(c, ggml_scale(c, acc, 1.0f / m.hc), m.out_norm, m.rms_eps);
    } else if (m.ds41) {                                     // (the streams collapsed above: the wanted positions)
        x = norm(c, x, m.out_norm, m.rms_eps);
    } else {
        if (!(out && out->all)) x = ggml_view_2d(c, x, m.n_embd, 1, x->nb[1], (size_t) (T - 1) * x->nb[1]);   // the last
        x = hid && out->all ? hid : norm(c, x, m.out_norm, m.rms_eps);
    }
    ggml_set_name(x, "result_norm");
    ggml_tensor * logits = ggml_mul_mat(c, m.output, x);
    if (m.softcap > 0)                                       // (gemma4: the logits soft-capped)
        logits = ggml_scale(c, ggml_tanh(c, ggml_scale(c, logits, 1.0f / m.softcap)), m.softcap);
    ggml_set_name(logits, "result_output");
    static const bool show_top = getenv("MOMOE_TOP") != nullptr;   // the last position's 5 most likely tokens
    if ((out && out->logits) || show_top) ggml_set_output(logits);
    ggml_tensor * best = ggml_argmax(c, logits);
    ggml_set_output(best);
    ggml_build_forward_expand(gf, best);
    b.c = c;
    b.gf = gf;
    b.in = in;
    b.seq0 = seq0;
    b.ple_in = ple_in;
    b.on_cpu = on_cpu;
    b.on_gpu = on_gpu;
    b.hid = hid;
    b.logits = logits;
    b.best = best;
}

// the DSpark drafter's graph (graph_dsv41's token batch and build_dspark_markov_head): `block` positions [the last token,
// the mask token...] through its blocks - the window's ring and the whole block seen at once (the block's window is its
// first position's), mHC with the pre-mix one sublayer late, its MoE (experts whole) - the model's output head, then
// position by position the Markov head's bias from the token before (the last token, then each position's choice) and
// the confidence head.  Built once: the inputs (tokens, positions, ring rows, mask) are set for each draft
static void ds_draft_build(Engine & e) {
    Model & m = e.m;
    auto & D = e.ds;
    const int B = D.block, W = m.swa_rows;
    const int64_t hd = m.ds_hd;
    ggml_init_params ip = {ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true};
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    D.tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    D.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    D.idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, B);
    D.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, W, B);
    for (ggml_tensor * t : {D.tok, D.pos, D.idx, D.mask}) ggml_set_input(t);
    ggml_tensor * x = ggml_get_rows(c, m.tok_embd, D.tok);
    if (!kHostMem && m.tok_embd->buffer && ggml_backend_buffer_is_host(m.tok_embd->buffer)) {   // (momoe-cuda: the
        D.on_cpu = x;                                        // embeddings in RAM, as build_forward's)
        x = D.on_gpu = ggml_scale(c, x, 1.0f);
    }
    ggml_tensor * res = ggml_repeat_4d(c, ggml_reshape_3d(c, x, m.n_embd, 1, B), m.n_embd, m.hc, B, 1);
    ggml_tensor * pre_prev = nullptr;
    for (size_t il = 0; il < D.layers.size(); ++il) {
        Layer & L = D.layers[il];
        ggml_tensor * apre, * apost, * acomb, * fpre, * fpost, * fcomb;
        ds_hc_mixes(e, c, res, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, &apre, &apost, &acomb);
        ggml_tensor * h = pre_prev ? ggml_dsv4_hc_pre(c, res, pre_prev)
                                   : ggml_cont(c, ggml_view_2d(c, res, m.n_embd, B, res->nb[2], 0));
        h = norm(c, h, L.attn_norm, m.rms_eps);
        {                                                    // the attention (window only, plain RoPE)
            ggml_tensor * qr = norm(c, ggml_mul_mat(c, L.wq_a, h), L.q_a_norm, m.rms_eps);
            ggml_tensor * q = ds_rope(m, c, ggml_reshape_3d(c, ggml_mul_mat(c, L.wq_b, qr), hd, m.n_head, B), D.pos, false);
            ggml_tensor * kv = norm(c, ggml_mul_mat(c, L.wkv_a, h), L.kv_a_norm, m.rms_eps);
            kv = ds_rope(m, c, ggml_reshape_3d(c, kv, hd, 1, B), D.pos, false);
            ggml_build_forward_expand(gf, ggml_set_rows(c, L.k_cache, ggml_reshape_2d(c, kv, hd, B), D.idx));
            ggml_tensor * kp = ggml_permute(c, ggml_view_3d(c, L.k_cache, hd, 1, W, L.k_cache->nb[1], L.k_cache->nb[1], 0), 0, 2, 1, 3);
            ggml_tensor * o = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), kp, kp, D.mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(o, L.sinks);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            o = ds_rope(m, c, ggml_reshape_3d(c, o, hd, m.n_head, B), D.pos, false, true);
            o = ggml_permute(c, ggml_reshape_3d(c, o, m.n_head / m.o_groups * hd, m.o_groups, B), 0, 2, 1, 3);
            ggml_tensor * oa = ggml_cont_2d(c, ggml_permute(c, ggml_mul_mat(c, L.wo_a, o), 0, 2, 1, 3), (int64_t) m.o_lora * m.o_groups, B);
            h = ggml_mul_mat(c, L.wo, oa);
        }
        res = ggml_dsv4_hc_post(c, h, res, apost, acomb);
        ds_hc_mixes(e, c, res, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, &fpre, &fpost, &fcomb);
        ggml_tensor * f = norm(c, ggml_dsv4_hc_pre(c, res, apre), L.ffn_norm, m.rms_eps);
        {                                                    // the MoE: sqrt(softplus) router, experts whole
            const int k = D.n_used;
            ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, f);
            ggml_prec_set_acc(lg, GGML_PREC_F32);
            ggml_tensor * probs = ggml_sqrt(c, ggml_softplus(c, lg));                     // [n_expert, B]
            ggml_tensor * top = ggml_argsort_top_k(c, ggml_add(c, probs, L.exp_probs_b), k);   // [k, B]
            ggml_tensor * w = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, D.n_expert, B), top), k, B);
            if (k > 1) w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
            w = ggml_scale(c, ggml_reshape_3d(c, w, 1, k, B), D.scale);
            ggml_tensor * f3 = ggml_reshape_3d(c, f, m.n_embd, 1, B);
            ggml_tensor * up = ggml_mul_mat_id(c, L.exps[1], f3, top);
            ggml_tensor * gt = ggml_mul_mat_id(c, L.exps[0], f3, top);
            ggml_tensor * dn = ggml_mul(c, ggml_mul_mat_id(c, L.exps[2], glm_swiglu(c, gt, up, D.clamp_exp[il]), top), w);
            ggml_tensor * moe = ggml_view_2d(c, dn, m.n_embd, B, dn->nb[2], 0);
            for (int i = 1; i < k; ++i) moe = ggml_add(c, moe, ggml_view_2d(c, dn, m.n_embd, B, dn->nb[2], i * dn->nb[1]));
            ggml_tensor * sh = ggml_mul_mat(c, L.down_sh, glm_swiglu(c, ggml_mul_mat(c, L.gate_sh, f), ggml_mul_mat(c, L.up_sh, f), D.clamp_sh[il]));
            f = ggml_add(c, moe, sh);
        }
        res = ggml_dsv4_hc_post(c, f, res, fpost, fcomb);
        pre_prev = fpre;
    }
    ggml_tensor * hs = ggml_dsv4_hc_pre(c, res, pre_prev);                                 // [n_embd, B], un-normed
    ggml_tensor * logits = ggml_mul_mat(c, m.output, norm(c, hs, D.out_norm, m.rms_eps));  // [V, B]
    ggml_tensor * prev = ggml_view_1d(c, D.tok, 1, 0);
    D.best.clear();
    D.cf.clear();
    for (int i = 0; i < B; ++i) {
        ggml_tensor * w1p = ggml_get_rows(c, D.mk_w1, prev);                                // [R, 1]
        ggml_tensor * col = ggml_add(c, ggml_view_2d(c, logits, logits->ne[0], 1, logits->nb[1], i * logits->nb[1]),
                                     ggml_mul_mat(c, D.mk_w2, w1p));
        ggml_tensor * hi = ggml_cont(c, ggml_view_2d(c, hs, m.n_embd, 1, hs->nb[1], i * hs->nb[1]));
        ggml_tensor * ci = ggml_sigmoid(c, ggml_mul_mat(c, D.conf, ggml_concat(c, hi, w1p, 0)));   // [1, 1]
        prev = ggml_argmax(c, col);
        for (ggml_tensor * t : {prev, ci}) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
        D.best.push_back(prev);
        D.cf.push_back(ci);
    }
    D.c = c;
    D.gf = gf;
    D.sched = ggml_backend_sched_new(e.sched_backends.data(), nullptr, (int) e.sched_backends.size(), 8192, false, true);
    if (D.on_gpu) {
        ggml_backend_sched_set_tensor_backend(D.sched, D.on_cpu, e.cpu);
        ggml_backend_sched_set_tensor_backend(D.sched, D.on_gpu, e.metal);
    }
    if (!ggml_backend_sched_alloc_graph(D.sched, gf)) die("cannot allocate the DSpark drafter's graph");
}

// a draft after `tok` (not yet read by the model) at position p0: the drafter's tokens for p0+1.. while each is at least
// conf_min sure (anchor-first: its position 0 is the token after `tok`)
static std::vector<int32_t> ds_draft(Engine & e, int32_t tok, int p0) {
    Model & m = e.m;
    auto & D = e.ds;
    const double t0 = now_s();
    if (!D.gf) ds_draft_build(e);
    const int B = D.block, W = m.swa_rows, last = p0 + B - 1;
    std::vector<int32_t> toks(B, D.mask_tok), pos(B);
    std::vector<int64_t> idx(B);
    toks[0] = tok;
    for (int i = 0; i < B; ++i) {
        pos[i] = p0 + i;
        idx[i] = (p0 + i) % W;
    }
    std::vector<ggml_fp16_t> mask((size_t) W * B);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int r = 0; r < W; ++r) {                            // the position ring row r holds (as flint_swa_inputs): seen
        const int p = last - (((last - r) % W) + W) % W;     // by every position when within the first one's window
        const ggml_fp16_t v = p >= 0 && p0 - p < m.n_swa ? zero : ninf;
        for (int i = 0; i < B; ++i) mask[(size_t) i * W + r] = v;
    }
    ggml_backend_tensor_set(D.tok, toks.data(), 0, toks.size() * sizeof(int32_t));
    ggml_backend_tensor_set(D.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(D.idx, idx.data(), 0, idx.size() * sizeof(int64_t));
    ggml_backend_tensor_set(D.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(D.sched, D.gf) != GGML_STATUS_SUCCESS) die("the DSpark drafter's graph failed");
    std::vector<int32_t> best(B);
    std::vector<float> cf(B);
    for (int i = 0; i < B; ++i) {
        ggml_backend_tensor_get(D.best[i], &best[i], 0, sizeof(int32_t));
        ggml_backend_tensor_get(D.cf[i], &cf[i], 0, sizeof(float));
    }
    std::vector<int32_t> out;
    for (int i = 0; i < B && cf[i] >= D.conf_min; ++i) out.push_back(best[i]);
    D.t += now_s() - t0;
    ++D.calls;
    return out;
}

static int g_built = 0;                                      // forward graphs built and the time it took (the stats
static double g_built_s = 0;                                 // line's)

// momoe-cuda: not for Nemotron with experts on the CPU (--hybrid: its decode graph is kept by forward_decode, the
// experts through mailboxes), nor across two cards (--layer-split, --helper-tier: not measured)
static bool reuse_backend_ok(const Engine & e) {
#if defined(MOMOE_CUDA)
    return !(e.m.nemo && e.hybrid) && !e.helper && e.split.empty();
#else
    (void) e;
    return true;
#endif
}

// graphs kept and computed again, as llama.cpp reuses its graph (momoe-metal, momoe-cuda).  Building a graph a token kept the GPU
// waiting (LFM2.5-8B on the M1 Ultra: 0.85 ms of a token's 8.3), and macOS then ran it at ~850 MHz, not ~1,250 (120
// tok/s; reused: 186, llama.cpp 175-185).  Only with every expert in its slot (no stops, no callback); each kept graph
// has its own scheduler, so a drafting window, its rollback and the next window keep theirs.  A graph is found by
// what shapes it: the tokens, the attention window, the outputs read back, and the k-pool's blocks (how many a batch
// finishes; their rows are inputs).  MOMOE_NO_GRAPH_REUSE=1: built every time.
struct GraphCache {
    std::array<int, 14> key{};
    bool ready = false;                                      // built and allocated
    Built b;
    ggml_backend_sched_t sched = nullptr;
    uint64_t used = 0;
};
static GraphCache g_gc[16];

static GraphCache * graph_cache(Engine & e, int T, int p0, int n_kv, const Out * out) {
    static const bool off = getenv("MOMOE_NO_GRAPH_REUSE") || getenv("MOMOE_TOP") || getenv("MOMOE_FORCE_CB");
    if (off || !e.preloaded || e.pause || e.nostall || e.reserve_only || debug_on() || T > e.max_window + 1 ||
        !reuse_backend_ok(e)) return nullptr;
    Inputs sh{};
    kpool_shape(e, sh, p0, T);
    const std::array<int, 14> key = {T, n_kv, sh.n_new, sh.n_new && e.m.glm5 ? sh.j_first : -1, sh.n_pool, sh.sel, sh.n_sel,
                                     out && out->logits, out && e.m.has_mtp, out && out->all, out && out->snap,
                                     e.predict, e.predict2, e.img_now};
    static uint64_t clock = 0;
    GraphCache * lru = &g_gc[0];
    for (GraphCache & g : g_gc) {
        if (g.ready && g.key == key) {
            g.used = ++clock;
            return &g;
        }
        if (g.used < lru->used) lru = &g;
    }
    if (lru->b.c) ggml_free(lru->b.c);
    lru->b = Built();
    lru->ready = false;
    lru->key = key;
    lru->used = ++clock;
    if (!lru->sched)
        lru->sched = ggml_backend_sched_new(e.sched_backends.data(), nullptr, (int) e.sched_backends.size(), 32768, false, true);
    return lru;
}

static int forward_once(Engine & e, const int32_t * toks, int T, int p0, Out * out);

// the rope positions of cells p0.., section-major [t.., h.., w.., e..]: text [p, p, p, 0]; a picture's cells mtmd's
// M-RoPE grid (t = q, h = q + y, w = q + x, q its first cell's position), the text after it at q + max(nx, ny); a
// model without M-RoPE (gemma4) numbers a picture's cells as text
static void rope_positions(const Engine & e, int p0, int T, std::vector<int32_t> & pos) {
    pos.assign((size_t) T * 4, 0);
    int delta = 0;
    size_t k = 0;
    const bool mrope = e.m.sections[0] > 0;
    for (int i = 0; i < T; ++i) {
        const int cell = p0 + i;
        while (mrope && k < e.imgs.size() && e.imgs[k].s + e.imgs[k].n <= cell) {
            delta += std::max(e.imgs[k].nx, e.imgs[k].ny) - e.imgs[k].n;
            ++k;
        }
        int t = cell + delta, h = t, w = t;
        if (mrope && k < e.imgs.size() && cell >= e.imgs[k].s) {
            const auto & I = e.imgs[k];
            const int r = cell - I.s;
            t = I.s + delta;
            h = t + r / I.nx;
            w = t + r % I.nx;
        }
        pos[i] = t;
        pos[(size_t) T + i] = h;
        pos[(size_t) 2 * T + i] = w;
    }
}

// qwen35moe with the experts in a pool (momoe-cuda): which expert path is faster depends on the model, the cards, the CPU
// and what else runs on the machine (see hybrid_model), so the engine measures both on decode calls - a few warm-up
// calls, then blocks of 6 calls on the copies (every missing expert read into the pool, the read-ahead on) and on the
// CPU's experts (every missing one on the CPU), twice in turn - and keeps the one with the least time per token.  A
// graph is built for every call here, so switching costs nothing.  It measures again 256 tokens after the first trial (a
// page cache still filling at load, measured wrong once), when the chosen path's pace over two windows of 64 tokens in a
// row is 25% off what its trial measured (a program taking the RAM or the CPU, or giving them back), and every 4096
// tokens.  A trial costs about 12 tokens at the slower path's pace (Ornith-1.5-122B on the RTX 4070 Ti: 0.35 s).
// MOMOE_QWEN_HYBRID=0: the copies; =1: the CPU's experts; MOMOE_HYB_FRAC=x: the split with that share copied in (the
// most used first, Strata's way: slower than the CPU's experts alone with both Ornith-1.5 models on the 4070 Ti +
// Ryzen 9 5950X: 72.1 and 80.3 ms per token for 25% and 50% with the 122B, 60.9 without); none of these measure
static const char * hyb_name(int c) { return c ? "the CPU's experts" : "the copies"; }
static void hyb_set(Engine & e, int c) {
    e.hyb_active = c == 1;
    e.hyb_frac = 0.0f;
    e.predict = c == 1 ? false : e.predict_copy;
}

static int forward(Engine & e, const int32_t * toks, int T, int p0, Out * out = nullptr) {
    const bool pool = e.hybrid && !e.m.nemo && hybrid_model(e.m) && !e.hyb_forced && !e.preloaded && T <= HYB_MAX_T;
    if (!pool) return forward_once(e, toks, T, p0, out);
    const double t0 = now_s();
    if (e.hyb_done) {                                        // the watch on the chosen path
        const int r = forward_once(e, toks, T, p0, out);
        e.hyb_ws += now_s() - t0;
        e.hyb_wt += T;
        e.hyb_since += T;
        const char * why = nullptr;
        static char buf[160];
        if (e.hyb_wt >= 64) {
            const double ms = 1000 * e.hyb_ws / e.hyb_wt;
            e.hyb_off = std::fabs(ms / e.hyb_ms - 1) > 0.25 ? e.hyb_off + 1 : 0;
            if (e.hyb_off >= 2) {
                snprintf(buf, sizeof buf, "its pace moved from %.1f to %.1f ms per token", e.hyb_ms, ms);
                why = buf;
            }
            e.hyb_ws = 0;
            e.hyb_wt = 0;
        }
        if (!why && e.hyb_since >= (e.hyb_trials == 1 ? 256 : 4096))
            why = e.hyb_trials == 1 ? "256 tokens after the first" : "4096 tokens since the last";
        if (why) {                                           // a new trial from the next call
            e.hyb_done = false;
            e.hyb_why = why;
        }
        return r;
    }
    const int WARM = e.hyb_trials ? 2 : 4, B = 6, ROUNDS = 2;
    const int n = e.hyb_trial++;
    const int blk = n < WARM ? -1 : (n - WARM) / B;
    const int c = blk < 0 ? e.hyb_active : blk % 2;          // the warm-up on the path in use
    hyb_set(e, c);
    const int r = forward_once(e, toks, T, p0, out);
    if (blk < 0) return r;
    e.hyb_s[c] += now_s() - t0;
    e.hyb_tok[c] += T;
    if (n == WARM + B * 2 * ROUNDS - 1) {                    // the trial's last call: choose
        double ms[2];
        for (int i = 0; i < 2; ++i) ms[i] = 1000 * e.hyb_s[i] / std::max(1, e.hyb_tok[i]);
        const int best = ms[1] < ms[0];
        hyb_set(e, best);
        fprintf(stderr, "momoe-cuda: expert path: %s (trial %d, %s; ms per token: the copies %.1f, the CPU's experts "
                        "%.1f; %d tokens each; MOMOE_QWEN_HYBRID=0/1 sets it)\n", hyb_name(best), e.hyb_trials + 1,
                e.hyb_why, ms[0], ms[1], e.hyb_tok[0]);
        e.hyb_ms = ms[best];
        e.hyb_done = true;
        ++e.hyb_trials;
        e.hyb_trial = 0;
        e.hyb_s[0] = e.hyb_s[1] = 0;
        e.hyb_tok[0] = e.hyb_tok[1] = 0;
        e.hyb_ws = 0;
        e.hyb_wt = e.hyb_off = e.hyb_since = 0;
    }
    return r;
}

static int forward_once(Engine & e, const int32_t * toks, int T, int p0, Out * out) {
    Model & m = e.m;
    e.img_now = false;
    for (const auto & I : e.imgs) e.img_now |= I.s < p0 + T && I.s + I.n > p0;
#if defined(MOMOE_CUDA)
    if (!e.img_now && decode_cache_ok(e, T, out)) return forward_decode(e, toks, p0, out);
#endif
    const double tb0 = now_s();
    const int n_kv = std::min(e.n_ctx, ((p0 + T + 255) / 256) * 256);
    Built lb;
    GraphCache * gc = graph_cache(e, T, p0, n_kv, out);     // a graph kept for these shapes (else built here)
    Built & b = gc ? gc->b : lb;
    const bool built = gc && gc->ready;
    if (!built) {
        build_forward(e, T, p0, n_kv, out, b, gc != nullptr);
        ++g_built;
    }
    ggml_context * c = b.c;
    ggml_cgraph * gf = b.gf;
    Inputs & in = b.in;
    ggml_tensor * seq0 = b.seq0, * ple_in = b.ple_in, * on_cpu = b.on_cpu, * on_gpu = b.on_gpu;
    ggml_tensor * hid = b.hid, * logits = b.logits, * best = b.best;
    ggml_backend_sched_t sched = gc ? gc->sched : e.sched;
    static const bool show_top = getenv("MOMOE_TOP") != nullptr;   // the last position's 5 most likely tokens

    if (!built) {
        ggml_backend_sched_reset(sched);
        if (!gc) ++g_sched_gen;                              // the reused decode graph is no longer allocated
        if (on_gpu) {
            ggml_backend_sched_set_tensor_backend(sched, on_cpu, e.cpu);
            ggml_backend_sched_set_tensor_backend(sched, on_gpu, e.metal);
        }
#if defined(MOMOE_CUDA)
        for (ggml_tensor * t : e.force_gpu) ggml_backend_sched_set_tensor_backend(sched, t, e.metal);
#endif
        if (e.reserve_only) {                                // fit_pool: the compute buffers of the largest graph
            if (!ggml_backend_sched_reserve(sched, gf)) die("cannot reserve the compute buffers: too little VRAM free");
            ggml_free(c);
            return 0;
        }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) die("cannot allocate the graph");
        if (gc) gc->ready = true;
        g_built_s += now_s() - tb0;
    }
    ggml_backend_tensor_set(in.tokens, toks, 0, (size_t) T * sizeof(int32_t));
    std::vector<int32_t> pos;
    rope_positions(e, p0, T, pos);
    if (in.img) {
        std::vector<float> rows((size_t) m.n_embd * T, 0.0f), keep(T, 1.0f);
        for (const auto & I : e.imgs)
            for (int i = std::max(0, I.s - p0); i < T && p0 + i < I.s + I.n; ++i) {
                const size_t r = (size_t) (p0 + i - I.s);
                memcpy(&rows[(size_t) i * m.n_embd], &e.img_data[I.off + r * m.n_embd], sizeof(float) * m.n_embd);
                keep[i] = 0.0f;
            }
        ggml_backend_tensor_set(in.img, rows.data(), 0, rows.size() * sizeof(float));
        ggml_backend_tensor_set(in.img_keep, keep.data(), 0, keep.size() * sizeof(float));
    }
    if (in.pos->buffer) ggml_backend_tensor_set(in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));   // (nemo: NoPE)
    std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < T; ++i) {
        int lo, hi;
        img_span(e, p0 + i, lo, hi);
        for (int j = 0; j < n_kv; ++j) mask[(size_t) i * n_kv + j] = j <= p0 + i || (j >= lo && j < hi) ? zero : ninf;
    }
    if (in.mask->buffer) ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));   // (deepseek41:
    if (in.mask_swa) flint_swa_inputs(e, in.mask_swa, in.kv_idx_swa, p0, T);                                   // not read)
    if (m.ds41) {
        if ((int) e.tok_at.size() < p0) die("the Engram token history has a gap");
        ds_set(e, in, toks, p0, T);
        e.tok_at.resize(p0);
        e.tok_at.insert(e.tok_at.end(), toks, toks + T);
    }
    if (in.kv_idx) {
        std::vector<int64_t> rows(T);
        for (int i = 0; i < T; ++i) rows[i] = p0 + i;
        ggml_backend_tensor_set(in.kv_idx, rows.data(), 0, rows.size() * sizeof(int64_t));
    }
    if (seq0) {
        const int32_t zero_id = 0;
        ggml_backend_tensor_set(seq0, &zero_id, 0, sizeof zero_id);
    }
    kpool_set(e, in, p0, T, n_kv);
    if (ple_in) {
        std::vector<float> rows;
        ple_rows(e, toks, T, p0, rows);
        ggml_backend_tensor_set(ple_in, rows.data(), 0, rows.size() * sizeof(float));
        if ((int) e.tok_at.size() < p0) die("the PLE token history has a gap");
        e.tok_at.resize(p0);
        e.tok_at.insert(e.tok_at.end(), toks, toks + T);
    }
    e.batch_tokens = T;
    const double tc0 = now_s();
    e.t_build += tc0 - tb0;
#if defined(MOMOE_CUDA)
    uint64_t mb_job = 0;
    // experts moved without the mailboxes (Nemotron too: Lightning's 17-31-token batches go through the pool, and its
    // next mailbox graph read slots since given to other experts: wrong tokens)
    if (e.mailbox && !e.mb_graph) e.mb_stale = true;
    if (e.mb_graph) {                                        // the worker follows this graph's MoE layers
        if (e.mb_stale) mb_rebuild(e);
        ++e.mb_graphs;
        mb_boundary(e);
        std::lock_guard<std::mutex> lk(e.mb.mu);
        ++e.mb.cur;
        *e.mb.epoch = e.mb.cur;
        e.mb.T = T;
        mb_job = ++e.mb.job;
        e.mb.cv.notify_all();
    }
#endif
    if (!e.pause || (e.ns_graph && e.ns_every == 0)) {
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) die("the graph failed");
#if defined(MOMOE_CUDA)
        if (mb_job) {
            if (*e.mb.timeout) die("the GPU waited for the CPU's experts and gave up (the mailbox worker stopped)");
            std::unique_lock<std::mutex> lk(e.mb.mu);
            e.mb.cv.wait(lk, [&] { return e.mb.done_job == mb_job; });
        }
#endif
        if (e.ns_graph) ns_after(e, T);
    } else {
#if defined(MOMOE_CUDA)
        die("pause points need Metal");
#else
        // one submission: the GPU pauses after each layer's router (signal 2k-1, wait for 2k); this thread makes
        // the chosen experts resident and lets it go on
        if (ggml_backend_sched_get_n_splits(e.sched) != 1) die("an op of the graph runs on the CPU: the pauses need one Metal graph");
        const uint64_t base = ggml_backend_metal_pause_counter(e.metal);
        if (ggml_backend_sched_graph_compute_async(e.sched, gf) != GGML_STATUS_SUCCESS) die("the graph failed");
        const uint64_t n_pause = (ggml_backend_metal_pause_counter(e.metal) - base) / 2;
        uint64_t want_pause = 0;
        for (int il = 0; il < m.n_layer; ++il) want_pause += e.topk[il] && ns_paused(e, il);   // (a dense block: none)
        if (n_pause != want_pause) die("the graph has a pause point per layer missing");
        uint64_t at = base + 1;
        int done = 0;                                        // --nostall-every: the layers seen to so far
        for (int il = 0; il < m.n_layer; ++il) {
            if (!e.topk[il] || !ns_paused(e, il)) continue;
            const double w0 = now_s();
            while (ggml_backend_metal_pause_signaled(e.metal) < at) { }
            e.t_wait += now_s() - w0;
            const double t0 = now_s();
            ++e.pool.calls;
            const int32_t * top = (const int32_t *) e.topk[il]->data;
            make_resident(e, il, top, T);
            // the predicted layers: the next two; nemo: the next MoE block only (the layers between have no experts:
            // predictions filed under il + 1 were never used, and filled a 282-slot group on the M4 Air)
            const int nx1 = m.nemo ? m.layers[il].next_moe : il + 1 < m.n_layer ? il + 1 : -1;
            const int nx2 = !m.nemo && il + 2 < m.n_layer ? il + 2 : -1;
            const bool p1 = e.predict && nx1 >= 0, p2 = p1 && e.predict2 && nx2 >= 0;
            // the predictions are copied before the GPU goes on: after it, the allocator reuses this tensor's memory
            const size_t n_pred = (size_t) m.n_used * T * (p1 + p2);
            e.pred_host.assign(top + (size_t) m.n_used * T, top + (size_t) m.n_used * T + n_pred);
            if (e.ns_graph) {
                // the layers since the last pause (computed by now): their wanted experts read; the predictions; then
                // what has been read is published for the layers after this one. Every slot is taken before the
                // publishing: nothing a published layer reads changes while the GPU goes on
                for (; done < il; ++done)
                    if (!ns_paused(e, done)) ns_layer(e, done, T);
                done = il + 1;
                if (p1) prefetch(e, nx1, e.pred_host.data(), T);
                if (p2) prefetch(e, nx2, e.pred_host.data() + (size_t) m.n_used * T, T);
                ns_publish(e, il + 1);
            }
            e.pool.cb_s += now_s() - t0;
            ggml_backend_metal_pause_release(e.metal, at + 1);
            if (!e.ns_graph) {
                if (p1) prefetch(e, nx1, e.pred_host.data(), T);
                if (p2) prefetch(e, nx2, e.pred_host.data() + (size_t) m.n_used * T, T);
            }
            at += 2;
        }
        ggml_backend_sched_synchronize(e.sched);
        if (e.ns_graph) {                                    // the layers after the last pause
            ++e.ns_tokens;
            for (; done < m.n_layer; ++done)
                if (!ns_paused(e, done)) ns_layer(e, done, T);
        }
#endif
    }
    e.t_compute += now_s() - tc0;
    if (show_top) {
        const int64_t V = logits->ne[0];
        std::vector<float> lv(V);
        ggml_backend_tensor_get(logits, lv.data(), (size_t) (logits->ne[1] - 1) * V * sizeof(float), V * sizeof(float));
        const float mx = *std::max_element(lv.begin(), lv.end());
        double z = 0;
        for (float v : lv) z += exp(v - mx);
        std::vector<int> ix(V);
        for (int i = 0; i < V; ++i) ix[i] = i;
        std::partial_sort(ix.begin(), ix.begin() + 5, ix.end(), [&](int a, int b) { return lv[a] > lv[b]; });
        fprintf(stderr, "TOP pos %d:", p0 + T - 1);
        for (int i = 0; i < 5; ++i) fprintf(stderr, " %d=%.4f", ix[i], exp(lv[ix[i]] - mx) / z);
        fprintf(stderr, "\n");
    }
    const int n_best = (int) ggml_nelements(best);
    std::vector<int32_t> nb(n_best);
    ggml_backend_tensor_get(best, nb.data(), 0, n_best * sizeof(int32_t));
    if (out) {
        out->next = nb;
        if (hid) {
            out->hidden.resize((size_t) T * m.h_dim());
            ggml_backend_tensor_get(hid, out->hidden.data(), 0, out->hidden.size() * sizeof(float));
        }
        if (out->logits && out->all) {
            const int64_t V = logits->ne[0];
            out->all_logits.resize((size_t) V * logits->ne[1]);
            ggml_backend_tensor_get(logits, out->all_logits.data(), 0, out->all_logits.size() * sizeof(float));
        }
        if (out->logits) {
            const int64_t V = logits->ne[0];
            out->last_logits.resize(V);
            ggml_backend_tensor_get(logits, out->last_logits.data(), (size_t) (logits->ne[1] - 1) * V * sizeof(float),
                                    V * sizeof(float));
        }
    }
    if (!gc) ggml_free(c);
    return nb.back();
}

// momoe-cuda: the compute buffers of the largest graph (a full prompt chunk at the end of the context) reserved first,
// then the pool takes the VRAM left, less a margin (MOMOE_VRAM_MARGIN_MIB, default 512: other programs, fragmentation)
// the memory this process may still take on a unified-memory machine (an integrated GPU: the DGX Spark): the system's
// MemAvailable, and no more than the room left under the memory limits of its cgroup (v2) and the cgroups above it
// (systemd-run -p MemoryMax=...: what is charged there less its page cache, which is given back first)
static double uma_free_bytes() {
    double avail = 0;
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char k[64];
        unsigned long long v;
        while (fscanf(f, "%63s %llu kB\n", k, &v) == 2)
            if (!strcmp(k, "MemAvailable:")) avail = v * 1024.0;
        fclose(f);
    }
    std::string cg;
    if (FILE * f = fopen("/proc/self/cgroup", "r")) {
        char line[4096];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "0::", 3)) { cg = line + 3; while (!cg.empty() && cg.back() == '\n') cg.pop_back(); }
        fclose(f);
    }
    for (std::string p = cg; !cg.empty();) {
        const std::string d = "/sys/fs/cgroup" + p;
        double mx = -1, cur = 0, file = 0;
        if (FILE * f = fopen((d + "/memory.max").c_str(), "r")) {
            char s[64] = {0};
            if (fscanf(f, "%63s", s) == 1 && strcmp(s, "max")) mx = atof(s);
            fclose(f);
        }
        if (mx >= 0) {
            if (FILE * f = fopen((d + "/memory.current").c_str(), "r")) { if (fscanf(f, "%lf", &cur) != 1) cur = 0; fclose(f); }
            if (FILE * f = fopen((d + "/memory.stat").c_str(), "r")) {
                char k[64];
                double v;
                while (fscanf(f, "%63s %lf\n", k, &v) == 2)
                    if (!strcmp(k, "file")) file = v;
                fclose(f);
            }
            avail = std::min(avail, mx - std::max(0.0, cur - file));
        }
        if (p == "/" || p.empty()) break;
        const size_t s = p.rfind('/');
        p = s == 0 ? "/" : p.substr(0, s);
    }
    return avail;
}

#if defined(MOMOE_CUDA)
// unified memory: what is kept for the rest of the system (MOMOE_UMA_RESERVE_GIB), what must be free to start on top
// of it (MOMOE_UMA_MIN_GIB) and the smallest useful pool: 10, 24 and 4 GiB (set on the DGX Spark, 121 GiB), or on a
// machine under 16 GiB 20%, 15% and 15% of its memory (a Jetson Orin Nano, 7.3 GiB: about 1.5, 1.1 and 1.1; there the
// engine itself, its dense weights and buffers, took 0.7 GB beside a 2 GiB pool)
static double mem_total_gib() {
    double t = 0;
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char k[64];
        unsigned long long v;
        while (fscanf(f, "%63s %llu kB\n", k, &v) == 2)
            if (!strcmp(k, "MemTotal:")) t = v / 1048576.0;
        fclose(f);
    }
    return t;
}
static double uma_gib(const char * env, double cap, double share) {
    if (getenv(env)) return atof(getenv(env));
    const double t = mem_total_gib();
    return t > 0 && t < 16 ? std::min(cap, share * t) : cap;   // (the share only under 16 GiB: the Spark keeps 10, 24)
}
static double uma_reserve_gib() { return uma_gib("MOMOE_UMA_RESERVE_GIB", 10, 0.20); }
static double uma_min_gib() { return uma_gib("MOMOE_UMA_MIN_GIB", 24, 0.15); }
// a unified-memory machine under 16 GiB (a Jetson): its experts are copied from the mapped file, not read past the page
// cache, and the page cache keeps most of the memory - with a model larger than the memory it is the bigger store, read
// at RAM speed.  Measured on a Jetson Orin Nano Super (7.3 GiB, LFM2.5-24B-A2B UD-IQ3_XXS, 400 tokens): direct reads 1.0
// tok/s, copies 8.1 with a 1.5-2.5 GiB pool, 7.3 with 3.0
static bool uma_small() { const double t = mem_total_gib(); return t > 0 && t < 16; }
#endif

static void fit_pool(Engine & e) {
#if defined(MOMOE_CUDA)
    const int T = std::min(e.max_batch, e.n_ctx);
    std::vector<int32_t> toks(T, 0);
    e.reserve_only = true;
    forward(e, toks.data(), T, e.n_ctx - T);
    e.reserve_only = false;
    size_t fr = 0, tot = 0;
    cudaSetDevice(e.device);
    cudaMemGetInfo(&fr, &tot);
    double margin = (getenv("MOMOE_VRAM_MARGIN_MIB") ? atof(getenv("MOMOE_VRAM_MARGIN_MIB")) : 512) * 1048576.0;
    if (e.uma) {                                             // the "VRAM" is the machine's RAM: what the system and
        const double room = uma_free_bytes();               // the cgroup leave, less a reserve for everything else
        margin = uma_reserve_gib() * 1073741824.0;
        fprintf(stderr, "momoe-cuda: unified memory: %.2f GiB free for this process (the GPU reports %.2f), %.0f GiB "
                "kept for the rest of the system\n", room / 1073741824.0, fr / 1073741824.0, margin / 1073741824.0);
        // (not the GPU's figure: it leaves out the page cache, which the system gives back as the pool takes memory;
        // the cgroup does not count the GPU's allocations: MemAvailable does)
        fr = (size_t) std::max(0.0, room);
    }
    double want = (double) e.pool.pool_bytes + (double) fr - margin;
    if (e.uma && !g_direct && uma_small())                   // (see uma_small: MOMOE_UMA_POOL_SHARE, default 0.3)
        want = std::min(want, (getenv("MOMOE_UMA_POOL_SHARE") ? atof(getenv("MOMOE_UMA_POOL_SHARE")) : 0.3)
                              * mem_total_gib() * 1073741824.0);
    fprintf(stderr, "momoe-cuda: %.2f GiB of VRAM free after the compute buffers: the pool gets %.2f GiB\n",
            fr / 1073741824.0, want / 1073741824.0);
    if (n_cards(e) > 1) {                                    // --layer-split: each card's groups get what that card has
        Pool & P = e.pool;
        P.frac.assign(n_cards(e), 0.0);
        for (int d = 0; d < n_cards(e); ++d) {
            size_t f = 0, t = 0;
            cudaSetDevice(d);
            cudaMemGetInfo(&f, &t);
            double have = 0, all = 0;
            for (auto & G : P.g)
                if (G.dev == d) { have += (double) G.bytes * G.n_slots; all += (double) G.bytes * G.n_all; }
            const double w = std::max(0.0, have + (double) f - margin);
            P.frac[d] = all > 0 ? w / all : 0.0;
            fprintf(stderr, "momoe-cuda: card %d: %.2f GiB free after the compute buffers: its pool gets %.2f GiB of %.2f\n",
                    d, f / 1073741824.0, std::min(w, all) / 1073741824.0, all / 1073741824.0);
        }
        cudaSetDevice(0);
        make_pool(e, 0.0);
        return;
    }
    if (e.uma && want < std::min(4.0, uma_min_gib()) * 1073741824.0)   // (unified memory: what is left after the reserve cannot
        die("too little memory left for the expert pool: is another program holding it?");   // hold a useful pool)
    make_pool(e, std::max(0.0, want / (double) e.pool.all_bytes));
#else
    (void) e;
#endif
}

// momoe-metal: the GPU's working set (recommendedMaxWorkingSetSize: 5.7 GB on an 8 GB Mac) holds the dense weights,
// the pool, the state and the compute buffers; past it the GPU's command buffers fail ("Insufficient Memory") and the
// stops read choices the GPU never wrote (Gemma 4 on the 8 GB M1: setup's 3 GiB pool crashed, 2 GiB ran).  So the
// largest graph's buffers are reserved and the pool shrinks to what is left of it less a margin (MOMOE_METAL_MARGIN_MIB,
// default 512: the server needs more than the engine alone.  Measured on that Mac, Flint's server with two chats and two
// 2,651-token prompts: margin 0 ran out of the GPU's memory on the first long prompt, 256 and 512 ran; the engine alone
// ran with 5.50 GiB allocated and failed at 5.74, against a working set of 5.33).  On a large working set 512 MiB is too
// little: 3% of it when that is more (the M1 Ultra, 48.0 GiB: Ling-3.0-flash IQ3_XXS ran with 46.65 GiB allocated and
// ran out at 47.45; Swift IQ3_XXS's 46.30 fits either way).  Nothing changes where it fits (the M4 Air's configs:
// 20.8-22.2 GiB of 25.0).  MOMOE_NO_METAL_FIT=1: not done; MOMOE_METAL_FIT_DEBUG=1: the figures.
static void fit_pool_metal(Engine & e) {
    if (!kHostMem || !e.metal || getenv("MOMOE_NO_METAL_FIT")) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(e.metal);
    size_t fr = 0, tot = 0;
    if (dev) ggml_backend_dev_memory(dev, &fr, &tot);
    if (!tot) return;
    const int T = std::min(e.max_batch, e.n_ctx);
    std::vector<int32_t> toks(T, 0);
    // on a scheduler of its own, freed after: e.sched's buffers stay as small as the graphs it has run (a server's kept
    // graphs have their own; with the largest graph's buffers held here as well, Flint's server on the 8 GB M1 ran out)
    ggml_backend_sched_t keep = e.sched;
    e.sched = ggml_backend_sched_new(e.sched_backends.data(), nullptr, (int) e.sched_backends.size(), 32768, false, true);
    e.reserve_only = true;
    forward(e, toks.data(), T, e.n_ctx - T);
    e.reserve_only = false;
    // the margin left free: 3% (whatever fits with it is left as it is: setup's configs, measured so); a pool nobody
    // asked for that has to be cut down is cut to leave 15% - one fitted to 3% of the 64 GB M1 Ultra's 48 GiB working
    // set (46.6 GiB in all, DeepSeek-V4.1) froze the machine
    const bool env_margin = getenv("MOMOE_METAL_MARGIN_MIB") != nullptr;
    double margin = env_margin ? atof(getenv("MOMOE_METAL_MARGIN_MIB")) * 1048576.0 : std::max(512.0 * 1048576.0, 0.03 * (double) tot);
#if !defined(MOMOE_CUDA)
    const double used = (double) ggml_backend_metal_allocated(e.metal);
#else
    const double used = 0;
#endif
    ggml_backend_sched_free(e.sched);
    e.sched = keep;
    ++g_sched_gen;
    if (!env_margin && !e.pool_asked && used + margin > (double) tot) margin = std::max(margin, 0.15 * (double) tot);
    const double over = used + margin - (double) tot;
    if (getenv("MOMOE_METAL_FIT_DEBUG"))
        fprintf(stderr, "momoe-metal: GPU working set %.3f GiB, allocated %.3f GiB (pool %.3f GiB)\n", tot / 1073741824.0,
                used / 1073741824.0, e.pool.pool_bytes / 1073741824.0);
    if (over <= 0) return;
    Pool & P = e.pool;
    const double want = (double) P.pool_bytes - over;
    fprintf(stderr, "momoe-metal: the GPU's working set is %.2f GiB and this needs %.2f GiB with the %.2f GiB pool: the "
            "pool gets %.2f GiB\n", tot / 1073741824.0, (used + margin) / 1073741824.0, P.pool_bytes / 1073741824.0,
            std::max(0.0, want) / 1073741824.0);
    if (want < 0.25 * 1073741824.0) die("too little of the GPU's working set is left for the expert pool");
    double f = want / (double) P.all_bytes;
    make_pool(e, f);
    for (int i = 0; i < 12 && (double) P.pool_bytes > want; ++i) {   // (each group keeps its share and a floor of
        f *= 0.98 * want / (double) P.pool_bytes;            // slots: the bytes come out above the share asked)
        make_pool(e, f);
    }
    if ((double) P.pool_bytes > want) die("the expert pool cannot be made small enough for the GPU's working set");
}

// --helper-tier: the helper card's pool, a group for each of the main pool's (the same formats), each the same share
// of its experts, with a slot of zeros; and each layer's expert -> helper slot table on that card.  Its size: the card's
// free VRAM less --helper-reserve-mib (its compute buffers), at most --helper-gib
static void make_hpool(Engine & e) {
#if defined(MOMOE_CUDA)
    Model & m = e.m;
    Pool & P = e.pool, & H = e.hpool;
    size_t fr = 0, tot = 0;
    cudaSetDevice(1);
    cudaMemGetInfo(&fr, &tot);
    cudaSetDevice(e.device);
    double room = (double) fr - e.helper_reserve_mib * 1048576.0;
    if (e.helper_gib > 0) room = std::min(room, e.helper_gib * 1073741824.0);
    const double frac = std::max(0.0, room) / (double) P.all_bytes;
    ggml_init_params ip = {ggml_tensor_overhead() * (64 * (P.g.size() + 1) + m.n_layer + 4), nullptr, true};
    ggml_context * ch = ggml_init(ip);
    H.g.clear();
    H.g.resize(P.g.size());
    H.n_slots = 0;
    H.pool_bytes = 0;
    for (size_t gi = 0; gi < P.g.size(); ++gi) {
        const Group & G = P.g[gi];
        Group & K = H.g[gi];
        for (int w = 0; w < 3; ++w) { K.type[w] = G.type[w]; K.ne[w][0] = G.ne[w][0]; K.ne[w][1] = G.ne[w][1]; }
        K.bytes = G.bytes;
        K.n_layers = G.n_layers;
        K.n_all = G.n_all;
        K.dev = 1;
        K.n_slots = std::max(2, std::min((int) (frac * G.n_all), G.n_all) + 1);   // (+ the slot of zeros)
        for (int w = m.part0(); w < 3; ++w) K.t[w] = ggml_new_tensor_3d(ch, K.type[w], K.ne[w][0], K.ne[w][1], K.n_slots);
        K.owner.assign(K.n_slots, -1);
        K.used.assign(K.n_slots, 0);
        K.left.reset(new std::atomic<int>[K.n_slots]);
        for (int i = 0; i < K.n_slots; ++i) K.left[i] = 0;
        K.by_prefetch.assign(K.n_slots, 0);
        K.zero = K.n_slots - 1;
        K.used[K.zero] = UINT64_MAX;                         // never taken
        H.n_slots += K.n_slots;
        H.pool_bytes += K.bytes * K.n_slots;
    }
    for (auto & L : m.layers)
        if (L.slot_map) L.hslot_map = ggml_new_tensor_2d(ch, GGML_TYPE_I32, 1, m.n_expert);
    H.buf = ggml_backend_alloc_ctx_tensors(ch, e.helper_gpu);
    if (!H.buf) die("--helper-tier: cannot allocate the helper's pool (lower --helper-gib)");
    card_buf(H.buf, 1);
    for (auto & K : H.g)
        for (int w = m.part0(); w < 3; ++w) {
            std::vector<char> z(K.t[w]->nb[2], 0);
            tset(K.t[w], z.data(), (size_t) K.zero * K.t[w]->nb[2], z.size());
        }
    for (auto & L : m.layers) {
        if (!L.hslot_map) continue;
        L.hmap_host.assign(m.n_expert, H.g[L.grp].zero);
        tset(L.hslot_map, L.hmap_host.data(), 0, ggml_nbytes(L.hslot_map));
    }
    const int all = m.n_layer * m.n_expert;
    H.slot_of.assign(all, -1);
    H.freq.assign(all, 0);
    H.lfu = P.lfu;
    H.all_bytes = P.all_bytes;
    fprintf(stderr, "momoe-cuda: helper tier: its pool %d slots = %.2f GiB (%.0f%% of the experts; %.2f GiB free on the "
            "card, %.0f MiB kept)\n", H.n_slots - (int) H.g.size(), H.pool_bytes / 1073741824.0,
            100.0 * H.pool_bytes / P.all_bytes, fr / 1073741824.0, e.helper_reserve_mib);
#else
    (void) e;
#endif
}

// the recurrent state (the delta net's recurrent and convolution states, the PLE convolution's history), copied aside
// before a verify window and put back after a failed draft (Apple Silicon: the Metal buffers are host memory; the GPU
// is idle between graphs)
typedef std::vector<std::string> Saved;

static std::vector<ggml_tensor *> state_tensors(Engine & e) {
    std::vector<ggml_tensor *> r;
    for (auto & L : e.m.layers) {
        if (L.ssm_state) r.push_back(L.ssm_state);
        if (L.conv_state) r.push_back(L.conv_state);
        if (L.ple_state) r.push_back(L.ple_state);
        if (L.swa && L.k_cache) {                            // flint's ring: the state at a position, kept whole
            r.push_back(L.k_cache);
            if (L.v_cache) r.push_back(L.v_cache);           // (deepseek41: the key is the value)
        }
        if (L.cst_kv) r.push_back(L.cst_kv);                 // deepseek41: the unfinished group's rows
        if (L.cst_sc) r.push_back(L.cst_sc);
    }
    for (auto & L : e.ds.layers) {                           // deepseek41's DSpark drafter: its rings
        if (L.k_cache) r.push_back(L.k_cache);
    }
    return r;
}

static void save_state(Engine & e, Saved & to) {
    const auto ts = state_tensors(e);
    to.resize(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        to[i].resize(ggml_nbytes(ts[i]));
        tget(ts[i], &to[i][0], 0, to[i].size());
    }
}

static void restore_state(Engine & e, const Saved & from) {
    const auto ts = state_tensors(e);
    for (size_t i = 0; i < ts.size(); ++i) tset(ts[i], from[i].data(), 0, from[i].size());
}

// after a verify window of T tokens of which the first n_keep stay: the delta net's state and convolution window as
// they were after token n_keep-1 (the attention caches need nothing: later positions are written again)
static void rollback(Engine & e, int T, int n_keep) {
    Model & m = e.m;
    if (n_keep >= T) return;
    const int conv_ch = m.ssm_inner + 2 * m.ssm_groups * m.ssm_state;
    const int cw = m.ssm_conv - 1;
    for (auto & L : m.layers) {
        if (!L.recurrent) continue;
        const size_t D = ggml_nbytes(L.ssm_state);
        std::vector<char> sv(D);                                                // slot s: s tokens back
        tget(L.ssm_snap, sv.data(), (size_t) (T - n_keep) * D, D);
        tset(L.ssm_state, sv.data(), 0, D);
        std::vector<float> w((size_t) (cw + T) * conv_ch), st((size_t) cw * conv_ch);   // [cw + T, conv_ch], [cw, conv_ch]
        tget(L.conv_snap, w.data(), 0, w.size() * sizeof(float));
        for (int ch = 0; ch < conv_ch; ++ch)
            for (int t = 0; t < cw; ++t) st[(size_t) ch * cw + t] = w[(size_t) ch * (cw + T) + n_keep + t];
        tset(L.conv_state, st.data(), 0, st.size() * sizeof(float));
    }
}

// the MTP layer over T positions: hidden states h (the main model's, or its own when chaining) and the tokens that
// follow them; returns the greedy token after the last, and its own hidden state for the next draft
// the MTP layer's graphs, kept as forward()'s are (see GraphCache): by the window's tokens and the attention window
struct MtpGraph {
    int T = -1, n_kv = -1;
    bool ready = false;
    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    Inputs in{};
    ggml_tensor *hin = nullptr, *hl = nullptr, *best = nullptr, *bp = nullptr;
    ggml_backend_sched_t sched = nullptr;
    uint64_t used = 0;
};
static MtpGraph g_mtpg[4];
static MtpGraph * mtp_graph(Engine & e, int T, int n_kv) {
    static const bool off = getenv("MOMOE_NO_GRAPH_REUSE") || getenv("MOMOE_FORCE_CB");
    if (off || e.m.glm5 || !e.preloaded || e.pause || e.nostall || debug_on() || T > e.max_window + 1 ||
        !reuse_backend_ok(e)) return nullptr;
    static uint64_t clock = 0;
    MtpGraph * lru = &g_mtpg[0];
    for (MtpGraph & g : g_mtpg) {
        if (g.ready && g.T == T && g.n_kv == n_kv) {
            g.used = ++clock;
            return &g;
        }
        if (g.used < lru->used) lru = &g;
    }
    if (lru->c) ggml_free(lru->c);
    ggml_backend_sched_t sc = lru->sched;
    *lru = MtpGraph();
    lru->sched = sc ? sc : ggml_backend_sched_new(e.sched_backends.data(), nullptr, (int) e.sched_backends.size(), 8192,
                                                   false, true);
    lru->T = T;
    lru->n_kv = n_kv;
    lru->used = ++clock;
    return lru;
}

static int mtp_forward(Engine & e, const float * h, const int32_t * toks, int T, int p0, std::vector<float> & h_out,
                       float * prob = nullptr) {
    Model & m = e.m;
    const double tm0 = now_s();
    Layer & L = m.mtp;
    const int n_kv = std::min(e.n_ctx, ((p0 + T + 255) / 256) * 256);
    MtpGraph * G = mtp_graph(e, T, n_kv);                   // a graph kept for these shapes (else built here)
    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    Inputs in{};
    ggml_tensor * hin = nullptr, * hl = nullptr, * head_in = nullptr, * best = nullptr, * bp = nullptr;
    if (G && G->ready) {
        c = G->c;
        gf = G->gf;
        in = G->in;
        hin = G->hin;
        hl = G->hl;
        best = G->best;
        bp = G->bp;
    } else {
    ggml_init_params ip = {ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true};
    c = ggml_init(ip);
    gf = ggml_new_graph_custom(c, 8192, false);
    in.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(in.tokens);
    hin = ggml_new_tensor_2d(c, GGML_TYPE_F32, m.h_dim(), T);
    ggml_set_input(hin);
    in.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) T * 4);
    ggml_set_input(in.pos);
    in.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, T);
    ggml_set_input(in.mask);
    if (G) {                                                 // the rows written: an input (a reused graph)
        in.kv_idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, T);
        ggml_set_input(in.kv_idx);
    }
    // its MoE: the experts whole (no pool), and the shared expert
    auto moe_of = [&](ggml_tensor * pn) {
        // its own expert count: the checkpoint's 512, also under the Coder's pruned 256
        const int64_t n_exp = L.gate_inp->ne[1];
        ggml_tensor * probs = ggml_soft_max(c, ggml_mul_mat(c, L.gate_inp, pn));
        ggml_tensor * ids = ggml_cont(c, ggml_argsort_top_k(c, probs, m.n_used));
        ggml_set_name(ids, "mtp-ids");                       // not a pause point ("topk-" only)
        ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, n_exp, T), ids);
        w = ggml_reshape_2d(c, w, m.n_used, T);
        w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        w = ggml_reshape_3d(c, w, 1, m.n_used, T);
        ggml_tensor * x3 = ggml_reshape_3d(c, pn, m.n_embd, 1, T);
        ggml_tensor * hh = ggml_swiglu_split(c, ggml_mul_mat_id(c, L.exps[0], x3, ids), ggml_mul_mat_id(c, L.exps[1], x3, ids));
        ggml_tensor * dn = ggml_mul(c, ggml_mul_mat_id(c, L.exps[2], hh, ids), w);
        ggml_tensor * moe = nullptr;
        for (int i = 0; i < m.n_used; ++i) {
            ggml_tensor * v = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
            moe = moe ? ggml_add(c, moe, v) : v;
        }
        ggml_tensor * sh = ggml_swiglu_split(c, ggml_mul_mat(c, L.gate_sh, pn), ggml_mul_mat(c, L.up_sh, pn));
        sh = ggml_mul(c, ggml_mul_mat(c, L.down_sh, sh), ggml_sigmoid(c, ggml_mul_mat(c, L.gate_inp_sh, pn)));
        return ggml_add(c, moe, sh);
    };
    ggml_tensor * te = ggml_get_rows(c, m.tok_embd, in.tokens);
    ggml_tensor * te_cpu = nullptr, * te_gpu = nullptr;      // momoe-cuda: the rows looked up on the CPU, as forward()'s
    if (!kHostMem && m.tok_embd->buffer && ggml_backend_buffer_is_host(m.tok_embd->buffer)) {   // (else the scheduler
        te_cpu = te;                                         // copied the whole table to the GPU every draft: 540 MB,
        te = te_gpu = ggml_scale(c, te, 1.0f);               // ~10 ms on the Spark for Qwen3.6)
    }
    ggml_tensor * en = norm(c, te, L.enorm, m.rms_eps);
                                                             // hl: the next draft's input; head_in: the head's
    if (m.glm5) kpool_new(e, c, in, p0, T);                  // (its MLA block's own k-pool selection)
    if (m.glm5 || m.ling) {
        // llama.cpp's glm5-next graph_mtp: e and h normed, eh_proj of [e ; h], one MLA block (a full indexer of its
        // own) and the MoE (the sigmoid router; its experts whole) with the shared expert; the shared head norm
        // (bailingmoe3's graph_mtp: the same with its own MLA block and the group selection)
        const int il = m.n_layer;
        ggml_tensor * hn = norm(c, hin, L.hnorm, m.rms_eps);
        ggml_tensor * x = ggml_mul_mat(c, L.eh_proj, ggml_concat(c, en, hn, 0));   // [n_embd, T]
        e.cur_layer = il;
        ggml_tensor * xa = norm(c, x, L.attn_norm, m.rms_eps);
        x = ggml_add(c, m.ling ? ling_mla_layer(e, c, gf, L, xa, in, p0, T, n_kv) : glm_mla_layer(e, c, gf, L, xa, in, p0, T, n_kv), x);
        ggml_tensor * f = norm(c, x, L.ffn_norm, m.rms_eps);
        const int k = m.n_used;
        ggml_tensor * probs = ggml_sigmoid(c, ggml_mul_mat(c, L.gate_inp, f));
        ggml_tensor * ids = ggml_cont(c, ggml_argsort_top_k(c, group_select(m, c, ggml_add(c, probs, L.exp_probs_b)), k));
        ggml_set_name(ids, "mtp-ids");                       // not a pause point ("topk-" only)
        ggml_tensor * w = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), ids), k, T);
        w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
        w = ggml_reshape_3d(c, ggml_scale(c, w, m.exp_scale), 1, k, T);
        ggml_tensor * x3 = ggml_reshape_3d(c, f, m.n_embd, 1, T);
        ggml_tensor * hh = glm_swiglu(c, ggml_mul_mat_id(c, L.exps[0], x3, ids), ggml_mul_mat_id(c, L.exps[1], x3, ids),
                                      m.clamp_exp[il]);
        ggml_tensor * dn = ggml_mul(c, ggml_mul_mat_id(c, L.exps[2], hh, ids), w);
        ggml_tensor * moe = nullptr;
        for (int i = 0; i < k; ++i) {
            ggml_tensor * v = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
            moe = moe ? ggml_add(c, moe, v) : v;
        }
        ggml_tensor * sh = ggml_mul_mat(c, L.down_sh, glm_swiglu(c, ggml_mul_mat(c, L.gate_sh, f), ggml_mul_mat(c, L.up_sh, f),
                                                                 m.clamp_sh[il]));
        x = ggml_add(c, ggml_add(c, moe, sh), x);
        hl = ggml_view_2d(c, x, m.n_embd, 1, x->nb[1], (size_t) (T - 1) * x->nb[1]);
        hl = norm(c, hl, L.head_norm ? L.head_norm : m.out_norm, m.rms_eps);  // the next draft's input
        head_in = hl;
    } else if (m.qwen4) {
        // llama.cpp's qwen4exp graph_mtp: each hc stream of h normalized on its own, eh_proj ([e ; h_s]) per stream
        // (as fc_embedding(e) + fc_hidden(h_s)), then one layer with hyper-connections and dense attention; the next
        // draft reads the streams, the head their mix (the layer's own final mixer)
        const int64_t N = m.n_embd, H = m.hc;
        ggml_tensor * hn = ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, hin, N, H, T), m.rms_eps),
                                    ggml_reshape_2d(c, L.hnorm, N, H));
        ggml_tensor * res = ggml_add(c, ggml_mul_mat(c, L.fc_h, hn), ggml_reshape_3d(c, ggml_mul_mat(c, L.fc_e, en), N, 1, T));
        ggml_tensor * inject = nullptr;
        ggml_tensor * a = hc_mix(e, c, res, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inject);
        a = attn_layer(e, c, gf, L, a, in, p0, T, n_kv);
        res = hc_combine(e, c, res, a, inject);
        ggml_tensor * f = hc_mix(e, c, res, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inject);
        res = hc_combine(e, c, res, moe_of(f), inject);     // [N, H, T]
        hl = ggml_cont(c, ggml_view_3d(c, res, N, H, 1, res->nb[1], res->nb[2], (size_t) (T - 1) * res->nb[2]));
        head_in = hc_mix(e, c, hl, m.mtp_head_norm, m.mtp_head_down, m.mtp_head_up, nullptr, nullptr);
        hl = ggml_reshape_2d(c, hl, N * H, 1);
    } else {
        ggml_tensor * hn = norm(c, hin, L.hnorm, m.rms_eps);
        ggml_tensor * x = ggml_mul_mat(c, L.eh_proj, ggml_concat(c, en, hn, 0));   // [n_embd, T]
        ggml_tensor * a = attn_layer(e, c, gf, L, norm(c, x, L.attn_norm, m.rms_eps), in, p0, T, n_kv);
        x = ggml_add(c, a, x);
        x = ggml_add(c, moe_of(norm(c, x, L.post_norm, m.rms_eps)), x);
        hl = ggml_view_2d(c, x, m.n_embd, 1, x->nb[1], (size_t) (T - 1) * x->nb[1]);
        hl = norm(c, hl, L.head_norm ? L.head_norm : m.out_norm, m.rms_eps);  // the next draft's input
        head_in = hl;
    }
    ggml_set_output(hl);
    ggml_tensor * probs_v = ggml_soft_max(c, ggml_mul_mat(c, m.dhead ? m.dhead : m.output, head_in));   // [n_vocab or subset]
    best = ggml_argmax(c, probs_v);
    ggml_set_output(best);
    bp = ggml_get_rows(c, ggml_reshape_2d(c, probs_v, 1, probs_v->ne[0]), best);   // its probability
    ggml_set_output(bp);
    ggml_build_forward_expand(gf, best);
    ggml_build_forward_expand(gf, bp);
    ggml_build_forward_expand(gf, hl);
    ggml_backend_sched_t sc = G ? G->sched : e.sched;
    ggml_backend_sched_reset(sc);
    if (!G) ++g_sched_gen;
    if (te_gpu) {
        ggml_backend_sched_set_tensor_backend(sc, te_cpu, e.cpu);
        ggml_backend_sched_set_tensor_backend(sc, te_gpu, e.metal);
    }
    if (!ggml_backend_sched_alloc_graph(sc, gf)) die("cannot allocate the MTP graph");
    if (G) {
        G->c = c;
        G->gf = gf;
        G->in = in;
        G->hin = hin;
        G->hl = hl;
        G->best = best;
        G->bp = bp;
        G->ready = true;
    }
    }
    ggml_backend_sched_t sched = G ? G->sched : e.sched;
    ggml_backend_tensor_set(in.tokens, toks, 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(hin, h, 0, (size_t) T * m.h_dim() * sizeof(float));
    std::vector<int32_t> pos;
    rope_positions(e, p0, T, pos);                           // (after a picture: the model's positions)
    if (in.pos->buffer) ggml_backend_tensor_set(in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));   // (glm5-next: NoPE)
    std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < n_kv; ++j) mask[(size_t) i * n_kv + j] = j <= p0 + i ? zero : ninf;
    ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (in.kv_idx) {
        std::vector<int64_t> rows(T);
        for (int i = 0; i < T; ++i) rows[i] = p0 + i;
        ggml_backend_tensor_set(in.kv_idx, rows.data(), 0, rows.size() * sizeof(int64_t));
    }
    if (m.glm5) kpool_set(e, in, p0, T, n_kv);
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) die("the MTP graph failed");
    int32_t r = 0;
    ggml_backend_tensor_get(best, &r, 0, sizeof(r));
    if (m.dhead) r = m.dvocab[r];
    if (prob) ggml_backend_tensor_get(bp, prob, 0, sizeof(float));
    e.t_mtp += now_s() - tm0;
    h_out.resize(m.h_dim());
    ggml_backend_tensor_get(hl, h_out.data(), 0, h_out.size() * sizeof(float));
    if (!G) ggml_free(c);
    return r;
}


// ------------------------------------------------------------------------------------------------ --serve
// The engine side of Momoe's server (serve/server.py), for one conversation at a time: "INFO ..." and "READY <ctx> stop"
// at start; "GEN <max_new>[ key=value...] <ids,...>" (the whole prompt every time) answered by "RESUME <reused>",
// "PP <read> <total> <ms> <tok/s>" while the prompt is read, "T <id>" per token and "DONE <generated> <prompt>
// <prompt_ms> <decode_ms> <stop|length|cancel> <drafts_accepted> <drafts_offered> <reused>"; "STOP" (any time; a
// stale one between requests is ignored), "VRAM" (not supported: ERR), "QUIT" or the end of stdin.
#include <queue>
#include <random>

struct Sampling {
    float temp = 0, top_p = 1, min_p = 0;
    int top_k = 20;
    uint32_t seed = 0;
};

static int sample(const std::vector<float> & logits, const Sampling & sp, std::mt19937 & rng) {
    const int V = (int) logits.size();
    if (sp.temp <= 0) return (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    const int k = sp.top_k > 0 ? std::min(sp.top_k, V) : V;
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(k);
    std::vector<double> pr(k);
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += pr[i] = exp((logits[idx[i]] - logits[idx[0]]) / sp.temp);
    for (auto & x : pr) x /= sum;
    int n = k;
    if (sp.min_p > 0)                                        // min_p: relative to the most likely
        while (n > 1 && pr[n - 1] < sp.min_p * pr[0]) --n;
    if (sp.top_p < 1) {
        double c = 0;
        for (int i = 0; i < n; ++i) { c += pr[i]; if (c >= sp.top_p) { n = i + 1; break; } }
    }
    double tot = 0;
    for (int i = 0; i < n; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0, tot);
    double r = u(rng);
    for (int i = 0; i < n; ++i) { r -= pr[i]; if (r <= 0) return idx[i]; }
    return idx[n - 1];
}

// macOS judges a thread that mostly waits for the GPU as light work: it keeps it on the efficiency cores and lowers the
// GPU's clock with it, and the two then hold each other there (the M1 Ultra, LFM2.5-8B through the server: 62-68 tok/s
// on every request; llama.cpp's server 176-182, its CPU threads polling between graphs).  QoS classes and launchd's
// Interactive did not change it; a thread that stays runnable does.  So while a request runs, one thread waits on
// the ARM wait-for-event instruction (the core idles until the next event): 121-122 tok/s, ~0.7 W more on the CPU
// (a counting loop: the same speed for ~2.5 W; a thread asleep even half the time: back to the slow state)
struct Awake {
#if defined(__APPLE__) && defined(__aarch64__)
    static std::mutex & mu() { static std::mutex m; return m; }
    static std::condition_variable & cv() { static std::condition_variable c; return c; }
    static std::atomic<int> & on() { static std::atomic<int> n{0}; return n; }
    Awake() {
        static std::once_flag once;
        std::call_once(once, [] {
            std::thread([] {
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(mu());
                        cv().wait(lk, [] { return on().load() > 0; });
                    }
                    while (on().load(std::memory_order_relaxed) > 0)
                        for (int i = 0; i < 64; ++i) __asm__ volatile("wfe");
                }
            }).detach();
        });
        { std::lock_guard<std::mutex> lk(mu()); ++on(); }
        cv().notify_all();
    }
    ~Awake() { --on(); }
#else
    Awake() = default;
#endif
    Awake(const Awake &) = delete;
    Awake & operator=(const Awake &) = delete;
};

struct Serve {
    std::mutex mu;
    std::condition_variable cv;
    std::queue<std::string> lines;
    std::atomic<bool> busy{false}, stop{false};
    bool eof = false;
};

static void out_line(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    fflush(stdout);
}

static void reset_state(Engine & e) {
    for (ggml_tensor * t : state_tensors(e)) {
        if (kHostMem) memset(t->data, 0, ggml_nbytes(t));
        else { std::vector<char> z(ggml_nbytes(t), 0); ggml_backend_tensor_set(t, z.data(), 0, z.size()); }
    }
}

// the caches kept per position: the attention keys and values (the MTP layer's too) and QSA's indexer keys, each
// with the dimension that counts positions and how many positions one of its rows holds (QSA's pooled keys: a block)
struct PosTensor { ggml_tensor * t; int dim, per_row; };

static std::vector<PosTensor> position_tensors(Engine & e) {
    std::vector<PosTensor> r;
    auto add = [&](ggml_tensor * t, int dim, int per_row) { if (t) r.push_back({t, dim, per_row}); };
    for (auto & L : e.m.layers) {
        if (!L.swa) add(L.k_cache, 2, 1);                    // (flint's ring: state_tensors)
        if (!L.swa) add(L.v_cache, 2, 1);
        add(L.idx_raw, 1, 1);
        add(L.idx_pool, 1, e.m.kpool);
        add(L.lat_cache, 1, 1);                              // glm5-next's MLA latents
        add(L.comp_cache, 1, L.ratio);                       // deepseek41's compressed rows (one per ratio positions)
    }
    add(e.m.mtp.k_cache, 2, 1);
    add(e.m.mtp.v_cache, 2, 1);
    add(e.m.mtp.lat_cache, 1, 1);
    add(e.m.mtp.idx_raw, 1, 1);
    add(e.m.mtp.idx_pool, 1, e.m.kpool);
    return r;
}

// the bytes of positions [a, b) in one of them: (offset, size); a block of QSA's pooled keys counts when it touches them
static std::pair<size_t, size_t> position_rows(const PosTensor & pt, int a, int b) {
    const size_t rb = pt.t->nb[pt.dim];
    const int64_t r0 = a / pt.per_row, r1 = std::min<int64_t>(pt.t->ne[pt.dim], (b + pt.per_row - 1) / pt.per_row);
    return {(size_t) r0 * rb, r1 > r0 ? (size_t) (r1 - r0) * rb : 0};
}

// positions [a, b) of every per-position cache, copied aside (appended to `to`, one string per cache) or put back
// (Apple Silicon: the Metal buffers are host memory; the GPU is idle between requests)
static void save_positions(Engine & e, int a, int b, std::vector<std::string> & to) {
    const auto ts = position_tensors(e);
    to.resize(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        const auto [off, n] = position_rows(ts[i], a, b);
        to[i].resize(n);
        if (n) tget(ts[i].t, &to[i][0], off, n);
    }
}

static void restore_positions(Engine & e, int a, const std::vector<std::string> & from) {
    const auto ts = position_tensors(e);
    for (size_t i = 0; i < ts.size(); ++i)
        if (!from[i].empty()) tset(ts[i].t, from[i].data(), position_rows(ts[i], a, a + 1).first, from[i].size());
}

static size_t position_bytes(Engine & e, int a, int b) {
    size_t n = 0;
    for (const auto & pt : position_tensors(e)) n += position_rows(pt, a, b).second;
    return n;
}

static int serve_loop(Engine & e, int mtp_k, float spec_min_p) {
    Model & m = e.m;
    const int ne = m.h_dim();                               // the MTP layer's hidden input per position
    const std::vector<int32_t> eos = m.eos_ids;             // Qwen: 248044, 248046; Nemotron: 11 (and </s> if eos)
    // never freed: after QUIT the stdin reader may still wake (end of input) and lock it before the process ends
    Serve & sv = *new Serve;
    std::thread reader([&sv = sv] {                         // stdin on its own thread: a STOP reaches a running request
        char buf[1 << 16];
        std::string acc;
        for (;;) {
            if (!fgets(buf, sizeof buf, stdin)) break;
            acc += buf;
            if (acc.empty() || acc.back() != '\n') continue;
            acc.pop_back();
            if (acc == "STOP") {
                if (sv.busy) sv.stop = true;                // between requests: stale, ignored
            } else {
                std::lock_guard<std::mutex> lk(sv.mu);
                sv.lines.push(acc);
                sv.cv.notify_all();
            }
            acc.clear();
        }
        std::lock_guard<std::mutex> lk(sv.mu);
        sv.eof = true;
        sv.cv.notify_all();
    });
    out_line("INFO engine=momoe-metal-0.1 expert_slots=%d", e.pool.n_slots);
    out_line("READY %d stop", e.n_ctx);

    std::vector<int32_t> live;                              // the tokens the state holds (positions 0..)
    std::vector<float> last_h;                              // the normalized hidden state of the last one
    std::vector<int32_t> kv;                                // the tokens whose positions the attention caches hold
    // a restart point: the recurrent state after `toks` (their positions stay in the attention caches as long as kv
    // starts with them).  One per prompt, in front of its last <|im_start|>: the reply's opening is written back
    // differently in the next request (Claude Code: the thinking block), so a checkpoint at the prompt's very end was
    // two tokens too far and every agent turn read its whole prompt again.
    struct Ckpt { std::vector<int32_t> toks; Saved st; std::vector<float> h; };
    std::vector<Ckpt> cks;                                  // oldest first
    const size_t max_cks = 4;
    const int32_t im_start = m.im_start;                    // <|im_start|> (Qwen3.5/3.6, Ornith, Flash-Next: 248045)
    // a conversation set aside while another request overwrites its positions (Claude Code's short side requests,
    // the session's title): its checkpoint and its caches' positions [from, toks.size()) copied aside; those before
    // `from` are the other request's too and stay in place
    struct Parked { Ckpt ck; int from = 0; std::vector<std::string> pos; } parked;
    const size_t park_min = 1024;                           // tokens: shorter conversations are read again instead
    const size_t park_max_bytes = (size_t) 4 << 30;
    auto common = [](const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
        const size_t n = std::min(a.size(), b.size());
        return (size_t) (std::mismatch(a.begin(), a.begin() + n, b.begin()).first - a.begin());
    };
    std::mt19937 rng(1234);
    std::vector<float> hc, hc2;
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(sv.mu);
            sv.cv.wait(lk, [&] { return sv.eof || !sv.lines.empty(); });
            if (sv.lines.empty()) break;
            line = sv.lines.front();
            sv.lines.pop();
        }
        if (line == "QUIT") break;
        if (line.rfind("VRAM", 0) == 0) { out_line("ERR VRAM is not supported by momoe-metal"); continue; }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) { out_line("ERR unknown command"); continue; }
        sv.stop = false;
        sv.busy = true;
        // GEN <max_new>[ key=value...] <ids>; GENI <max_new>[ key=value...] <embeddings file> <ids>
        std::vector<std::string> w;
        std::string emb_file;
        for (size_t a = geni ? 5 : 4, b; a < line.size(); a = b + 1) {
            b = line.find(' ', a);
            if (b == std::string::npos) b = line.size();
            if (b > a) w.push_back(line.substr(a, b - a));
        }
        Sampling sp;
        std::vector<int32_t> prompt;
        int max_new = w.empty() ? 0 : atoi(w[0].c_str());
        for (size_t i = 1; i < w.size(); ++i) {
            const size_t eq = w[i].find('=');
            if (eq == std::string::npos && geni && emb_file.empty()) { emb_file = w[i]; continue; }
            if (eq == std::string::npos) {
                for (size_t a = 0, b; a < w[i].size(); a = b + 1) {
                    b = w[i].find(',', a);
                    if (b == std::string::npos) b = w[i].size();
                    prompt.push_back(atoi(w[i].substr(a, b - a).c_str()));
                }
                break;
            }
            const std::string k = w[i].substr(0, eq), v = w[i].substr(eq + 1);
            if (k == "temperature") sp.temp = (float) atof(v.c_str());
            else if (k == "top_p") sp.top_p = (float) atof(v.c_str());
            else if (k == "top_k") sp.top_k = atoi(v.c_str());
            else if (k == "min_p") sp.min_p = (float) atof(v.c_str());
            else if (k == "seed") sp.seed = (uint32_t) atoll(v.c_str());
        }
        if (sp.seed) rng.seed(sp.seed);
        if (prompt.empty() || max_new <= 0) { sv.busy = false; out_line("ERR empty prompt or max_new"); continue; }
        if ((int) prompt.size() + max_new + 8 > e.n_ctx) {
            sv.busy = false;
            out_line("ERR prompt (%zu tokens) + max_new (%d) exceeds the context (%d)", prompt.size(), max_new, e.n_ctx);
            continue;
        }
        bool bad = false;
        for (int32_t t : prompt) bad |= t < 0 || t >= m.n_vocab;
        if (bad) { sv.busy = false; out_line("ERR a token id is outside the vocabulary"); continue; }
        // pictures: the file's records (strata-vision's: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) go
        // to the prompt's runs of image cells in order.  `ptoks` is what the model reads; in `prompt` (the key the
        // caches are matched by) a picture's cells become negative numbers from its rows, so a cached state is
        // reused only up to the first picture that differs
        std::vector<int32_t> ptoks = prompt;
        e.imgs.clear();
        e.img_data.clear();
        if (geni) {
            std::string err;
            FILE * f = fopen(emb_file.c_str(), "rb");
            if (!f) err = "cannot open the embeddings file " + emb_file;
            for (size_t i = 0; err.empty() && i < prompt.size();) {
                if (prompt[i] != m.img_pad || m.img_pad < 0) { ++i; continue; }
                size_t n = 0;
                while (i + n < prompt.size() && prompt[i + n] == m.img_pad) ++n;
                int32_t hdr[5];
                if (fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != 0x31455653) { err = "the embeddings file has fewer "
                    "pictures than the prompt"; break; }
                if (hdr[4] != m.n_embd) { err = "the pictures' rows are " + std::to_string(hdr[4]) + " wide, the model's "
                    + std::to_string(m.n_embd) + " (an mmproj file of another model?)"; break; }
                if ((size_t) hdr[1] != n || hdr[2] * hdr[3] != hdr[1]) { err = "a picture has " + std::to_string(hdr[1]) +
                    " rows for " + std::to_string(n) + " image cells in the prompt"; break; }
                Engine::Image I{(int) i, (int) n, hdr[2], hdr[3], e.img_data.size()};
                e.img_data.resize(I.off + n * (size_t) m.n_embd);
                if (fread(&e.img_data[I.off], sizeof(float) * m.n_embd, n, f) != n) { err = "the embeddings file is "
                    "cut short"; break; }
                uint64_t h = 1469598103934665603ull;                 // FNV-1a over the rows
                const unsigned char * bp = (const unsigned char *) &e.img_data[I.off];
                for (size_t b = 0; b < n * m.n_embd * sizeof(float); ++b) h = (h ^ bp[b]) * 1099511628211ull;
                for (size_t r = 0; r < n; ++r)
                    prompt[i + r] = -1 - (int32_t) ((h + r * 0x9E3779B97F4A7C15ull) >> 34);
                e.imgs.push_back(I);
                i += n;
            }
            if (f) fclose(f);
            if (err.empty() && e.imgs.empty()) err = "GENI without image cells in the prompt";
            for (const auto & I : e.imgs) {                  // (cells that see each other are read in one batch)
                int lo, hi;
                img_span(e, I.s, lo, hi);
                if (err.empty() && hi > lo && I.n > e.max_batch)
                    err = "a picture of " + std::to_string(I.n) + " rows is more than the engine's batch (" +
                          std::to_string(e.max_batch) + "): fewer image tokens (the vision section's max_tokens)";
            }
            if (!err.empty()) {
                e.imgs.clear();
                e.img_data.clear();
                sv.busy = false;
                out_line("ERR %s", err.c_str());
                continue;
            }
        }
        Awake awake;                                         // (macOS: see Awake)
        // where to start: the state continues when the prompt extends it; else the longest checkpoint the prompt
        // starts with whose positions the caches still hold (an agent's next turn); else the parked conversation's,
        // its positions put back; else from the start.  At least the last prompt token is run again (its logits are
        // not kept).
        auto starts_with = [&](const std::vector<int32_t> & a) {
            return a.size() < prompt.size() && std::equal(a.begin(), a.end(), prompt.begin());
        };
        auto in_caches = [&](const std::vector<int32_t> & a) { return common(a, kv) == a.size(); };
        auto main_ck = [&]() -> Ckpt * {                    // the longest checkpoint of what the caches hold
            Ckpt * b = nullptr;
            for (auto & c : cks)
                if (in_caches(c.toks) && (!b || c.toks.size() > b->toks.size())) b = &c;
            return b;
        };
        auto park = [&](const Ckpt & c, size_t from) {      // c's positions from `from` on, copied aside
            parked = Parked();
            if (c.toks.size() < park_min || from >= c.toks.size()) return;
            const size_t bytes = position_bytes(e, (int) from, (int) c.toks.size());
            if (bytes > park_max_bytes) return;
            const double tp = now_s();
            save_positions(e, (int) from, (int) c.toks.size(), parked.pos);
            parked.ck = c;
            parked.from = (int) from;
            fprintf(stderr, "momoe serve: conversation cache: set aside a conversation of %zu tokens (positions %zu.., "
                    "%.0f MiB in %.0f ms)\n", c.toks.size(), from, bytes / 1048576.0, 1000 * (now_s() - tp));
        };
        std::vector<float> h_before;                        // the hidden state of the position before the new tokens
        Ckpt * best = nullptr;
        for (auto & c : cks)
            if (starts_with(c.toks) && in_caches(c.toks) && (!best || c.toks.size() > best->toks.size())) best = &c;
        if (!live.empty() && starts_with(live) && (!best || live.size() >= best->toks.size())) {
            h_before = last_h;
        } else if (best) {
            restore_state(e, best->st);
            live = best->toks;
            last_h = h_before = best->h;
        } else if (!parked.ck.toks.empty() && starts_with(parked.ck.toks) &&
                   common(kv, parked.ck.toks) >= (size_t) parked.from) {
            Parked back = std::move(parked);
            parked = Parked();
            if (Ckpt * c = main_ck()) park(*c, common(c->toks, back.ck.toks));   // the two swap places
            const double tp = now_s();
            restore_positions(e, back.from, back.pos);
            restore_state(e, back.ck.st);
            live = kv = back.ck.toks;
            last_h = h_before = back.ck.h;
            fprintf(stderr, "momoe serve: conversation cache: put back a conversation of %zu tokens in %.0f ms\n",
                    live.size(), 1000 * (now_s() - tp));
            cks.push_back(std::move(back.ck));
        } else {
            size_t best_ck = 0;
            for (auto & c : cks) best_ck = std::max(best_ck, common(c.toks, prompt));
            if (!live.empty() || !cks.empty())
                fprintf(stderr, "momoe serve: no reuse: the prompt shares %zu of the last state's %zu tokens and %zu "
                        "with a checkpoint\n", common(live, prompt), live.size(), best_ck);
            if (Ckpt * c = main_ck()) park(*c, common(c->toks, prompt));
            reset_state(e);
            live.clear();
            last_h.clear();
        }
        const int reused = (int) live.size();
        // the parked conversation's positions this request writes (from `reused` on) are copied aside first
        if (!parked.ck.toks.empty() && reused < parked.from) {
            if (common(kv, parked.ck.toks) >= (size_t) parked.from) {
                const auto ts = position_tensors(e);
                for (size_t i = 0; i < ts.size(); ++i) {
                    const size_t rb = ts[i].t->nb[ts[i].dim];
                    const size_t r0 = reused / ts[i].per_row, r1 = parked.from / ts[i].per_row;
                    std::string head((size_t) (r1 - r0) * rb, '\0');
                    if (!head.empty()) tget(ts[i].t, &head[0], r0 * rb, head.size());
                    parked.pos[i].insert(0, head);
                }
                parked.from = reused;
            } else {
                parked = Parked();
            }
        }
        out_line("RESUME %d", reused);
        // the n-gram models' token history (Flash-Next's PLE, DeepSeek's Engram) is the prompt's own up to `reused`: a
        // conversation put back from the cache left the history of the one that ran in between
        e.tok_at.assign(ptoks.begin(), ptoks.begin() + reused);
        // the prompt
        const bool greedy = sp.temp <= 0;
        const bool draft = mtp_k > 0;                       // sampled: a draft is kept when it is the token sampled
        const double t0 = now_s();
        int p = reused, next = 0;
        std::vector<float> H;                                // the new positions' hidden states (the MTP layer)
        std::vector<float> lg;
        bool cancelled = false;
        size_t ck_pos = prompt.size();                      // this prompt's checkpoint: in front of its last
        for (size_t j = prompt.size(); j-- > (size_t) reused + 1;)   // <|im_start|>, else at its end
            if (prompt[j] == im_start) { ck_pos = j; break; }
        for (size_t i = reused; i < prompt.size();) {
            if (sv.stop) { cancelled = true; break; }
            size_t end = std::min(prompt.size(), i + (size_t) e.max_batch);
            for (const auto & I : e.imgs) {                  // a picture whose cells see each other is not cut
                int lo, hi;
                img_span(e, I.s, lo, hi);
                if (hi > lo && (size_t) I.s > i && (size_t) I.s < end && (size_t) (I.s + I.n) > end) end = I.s;
            }
            if (i < ck_pos && end > ck_pos) end = ck_pos;
            const int T = (int) (end - i);
            Out o;
            o.logits = !greedy && end == prompt.size();
            next = forward(e, ptoks.data() + i, T, p, &o);
            if (m.has_mtp) H.insert(H.end(), o.hidden.begin(), o.hidden.end());
            if (o.logits) lg = o.last_logits;
            p += T;
            i = end;
            if (end == ck_pos) {
                Ckpt c;
                c.toks.assign(prompt.begin(), prompt.begin() + ck_pos);
                save_state(e, c.st);
                if (m.has_mtp) c.h.assign(H.end() - ne, H.end());
                cks.erase(std::remove_if(cks.begin(), cks.end(), [&](const Ckpt & o) { return o.toks == c.toks; }),
                          cks.end());
                cks.push_back(std::move(c));
                if (cks.size() > max_cks) cks.erase(cks.begin());
            }
            const double ms = 1000 * (now_s() - t0);
            out_line("PP %d %zu %.0f %.1f", p - reused, prompt.size() - reused, ms, (p - reused) / (ms / 1000));
        }
        const double prompt_ms = 1000 * (now_s() - t0);
        int generated = 0, accepted = 0, offered = 0;
        const char * finish = "cancel";
        if (!cancelled) {
            live.assign(prompt.begin(), prompt.end());
            if (m.has_mtp) last_h.assign(H.end() - ne, H.end());
            if (!greedy) next = sample(lg, sp, rng);
            finish = "length";
            const double td0 = now_s();
            auto emit = [&](int t) -> bool {                // false: stop
                out_line("T %d", t);
                ++generated;
                if (std::find(eos.begin(), eos.end(), t) != eos.end()) { finish = "stop"; return false; }
                if (generated >= max_new) { finish = "length"; return false; }
                if (sv.stop) { finish = "cancel"; return false; }
                return true;
            };
            if (draft) {
                // the MTP layer over the new positions: (state of position i, token i+1); the first pair uses the
                // state before them when there is one
                std::vector<float> hh;
                std::vector<int32_t> tt;
                const size_t n_new = prompt.size() - reused;
                int q0 = reused + 1;                         // the MTP position of a pair: its token's
                if (reused > 0 && (int) h_before.size() == ne && !e.ds.on) {
                    hh = h_before;                           // (state before the new tokens, the first new token)
                    tt.push_back(ptoks[reused]);
                    q0 = reused;
                }
                for (size_t i = 0; i < n_new && !e.ds.on; ++i) {   // (DSpark: no MTP layer, no hidden states)
                    hh.insert(hh.end(), H.begin() + i * ne, H.begin() + (i + 1) * ne);
                    tt.push_back(i + 1 < n_new ? ptoks[reused + i + 1] : next);
                }
                float pb = 0;
                int d = 0;
                for (size_t i = 0; i < tt.size() && !e.ds.on; i += e.max_batch) {   // (DSpark: no MTP layer to run)
                    const int T = (int) std::min(tt.size() - i, (size_t) e.max_batch);
                    d = mtp_forward(e, hh.data() + i * ne, tt.data() + i, T, q0 + (int) i, hc, &pb);
                }
                auto chain = [&](int dd, int q) {
                    std::vector<int32_t> dr;
                    if (pb < spec_min_p) return dr;
                    dr.push_back(dd);
                    for (int j = 1; j < mtp_k; ++j) {
                        const int32_t tok = dr.back();
                        const int32_t nx = mtp_forward(e, hc.data(), &tok, 1, q + j, hc2, &pb);
                        hc.swap(hc2);
                        if (pb < spec_min_p) break;
                        dr.push_back(nx);
                    }
                    return dr;
                };
                std::vector<int32_t> drafts = e.ds.on ? ds_draft(e, next, p) : chain(d, p);
                int t = next;
                bool go = emit(t);
                while (go) {
                    std::vector<int32_t> W = {t};
                    W.insert(W.end(), drafts.begin(), drafts.end());
                    const int T = (int) W.size();
                    offered += T - 1;
                    Out ov;
                    ov.all = true;
                    ov.logits = !greedy;
                    if (T > 1) save_state(e, e.saved);
                    forward(e, W.data(), T, p, &ov);
                    std::vector<int32_t> nx;                 // the next token after each kept position: the greedy
                    int n = 0;                               // one, or sampled position by position
                    for (int j = 0; j < T; ++j) {
                        if (greedy) nx.push_back(ov.next[j]);
                        else {
                            const size_t V = ov.all_logits.size() / T;
                            lg.assign(ov.all_logits.begin() + j * V, ov.all_logits.begin() + (j + 1) * V);
                            nx.push_back(sample(lg, sp, rng));
                        }
                        if (j == T - 1 || nx[j] != W[j + 1]) break;
                        ++n;
                    }
                    accepted += n;
                    if (n + 1 < T) {
                        restore_state(e, e.saved);
                        forward(e, W.data(), n + 1, p);
                    }
                    live.insert(live.end(), W.begin(), W.begin() + n + 1);
                    if (m.has_mtp) last_h.assign(ov.hidden.begin() + (size_t) n * ne, ov.hidden.begin() + (size_t) (n + 1) * ne);
                    for (int j = 0; j <= n && go; ++j) go = emit(nx[j]);
                    if (!go) break;
                    if (e.ds.on) drafts = ds_draft(e, nx[n], p + n + 1);
                    else {
                    const int dn = mtp_forward(e, ov.hidden.data(), nx.data(), n + 1, p + 1, hc, &pb);
                    drafts = chain(dn, p + n + 1);
                    }
                    p += n + 1;
                    t = nx[n];
                }
            } else {
                int t = next;
                while (emit(t)) {
                    Out o;
                    o.logits = !greedy;
                    const int g = forward(e, &t, 1, p, &o);
                    live.push_back(t);
                    if (m.has_mtp && !o.hidden.empty()) last_h = o.hidden;
                    ++p;
                    t = greedy ? g : sample(o.last_logits, sp, rng);
                }
            }
            const double decode_ms = 1000 * (now_s() - td0);
            fprintf(stderr, "momoe serve: prompt %zu tokens = %d reused + %zu read in %.0f ms (%.1f tok/s), %d generated in %.0f "
                    "ms (%.1f tok/s), drafts accepted %d of %d\n", prompt.size(), reused, prompt.size() - reused,
                    prompt_ms, (prompt.size() - reused) / (prompt_ms / 1000 + 1e-9), generated, decode_ms,
                    generated / (decode_ms / 1000 + 1e-9), accepted, offered);
            out_line("DONE %d %zu %.0f %.0f %s %d %d %d", generated, prompt.size(), prompt_ms, decode_ms, finish, accepted,
                     offered, reused);
            kv = live;
        } else {
            live.clear();                                    // part of the prompt was read: the state is not a prefix
            kv.assign(prompt.begin(), prompt.begin() + p);   // (its positions are: its checkpoint stays usable)
            out_line("DONE 0 %zu %.0f 0 cancel 0 0 %d", prompt.size(), prompt_ms, reused);
        }
        cks.erase(std::remove_if(cks.begin(), cks.end(), [&](const Ckpt & c) { return !in_caches(c.toks); }),
                  cks.end());
        sv.busy = false;
    }
    reader.detach();                                         // (closing stdin here would wait on the reader's lock)
    return 0;
}

// ------------------------------------------------------------------------------------------------ main
int main(int argc, char ** argv) {
    ggml_log_set([](ggml_log_level level, const char * text, void *) {   // (ggml's debug lines: one per token with
        if (level != GGML_LOG_LEVEL_DEBUG || getenv("MOMOE_GGML_DEBUG")) fputs(text, stderr);   // CUDA graphs)
    }, nullptr);
    const char * model = nullptr;
    std::vector<int32_t> prompt;
    int n_gen = 32, slots = 0, mtp_k = 0;
    float spec_min_p = 0.5f;                                 // a draft only while the MTP layer is this sure
    bool pre = false, nocache = false, bench_window = false, serve = false;
    bool cuda_graphs_off = false;                            // (momoe-cuda: GGML_CUDA_DISABLE_GRAPHS set here)
    const char * nostall_arg = nullptr;                      // --nostall sub|drop (see Engine::nostall)
    int nostall_every = 0;                                   // --nostall-every N
    const char * profile = nullptr;
    double pool_gib = 0;
    Engine e;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("%s needs a value", a.c_str()); return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-n") n_gen = atoi(next());
        else if (a == "--slots") slots = atoi(next());
        else if (a == "--pool-gib") pool_gib = atof(next());
        else if (a == "--ctx") e.n_ctx = atoi(next());
        else if (a == "--batch" || a == "--chunk") e.max_batch = atoi(next());   // (--chunk: the server reads --batch as its own)
        else if (a == "--preload") pre = true;
        else if (a == "--nocache") nocache = true;
        else if (a == "--bench-window") bench_window = true;
        else if (a == "--mtp") mtp_k = atoi(next());
        else if (a == "--serve") serve = true;
        else if (a == "--vision") {}                         // (the server's flag for pictures: GENI always works)
        else if (a == "--draft-vocab") e.draft_vocab = next();
        else if (a == "--mtp-gguf") e.mtp_gguf = next();
        else if (a == "--engram") e.engram = next();
        else if (a == "--dspark") e.ds.path = next();        // (deepseek41: its DSpark drafter's GGUF)
        else if (a == "--dspark-conf") e.ds.conf_min = (float) atof(next());
        else if (a == "--nostall") nostall_arg = next();
        else if (a == "--nostall-every") nostall_every = std::max(0, atoi(next()));
        else if (a == "--spec-min-p") spec_min_p = (float) atof(next());
        else if (a == "--profile") profile = next();
        else if (a == "--device") e.device = atoi(next());
        else if (a == "--layer-split") e.split = next();     // (see Engine::split; momoe-cuda only)
        else if (a == "--helper-tier") e.helper = true;      // (see Engine::helper; momoe-cuda only)
        else if (a == "--helper-gib") e.helper_gib = atof(next());
        else if (a == "--helper-reserve-mib") e.helper_reserve_mib = atof(next());
        else if (a == "--direct-io") g_direct = true;     // (see g_direct)
        else if (a == "--tokens" || a == "--tokens-file") {   // (a file: a long prompt, past the 128 KB of an argument)
            std::string s = next();
            if (a == "--tokens-file") {
                FILE * f = fopen(s.c_str(), "rb");
                if (!f) die("cannot open %s", s.c_str());
                std::string all;
                char buf[65536];
                for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) all.append(buf, n);
                fclose(f);
                while (!all.empty() && (all.back() == '\n' || all.back() == ' ')) all.pop_back();
                s = all;
            }
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                prompt.push_back(atoi(s.substr(p, q - p).c_str()));
                p = q == std::string::npos ? s.size() : q + 1;
            }
        } else die("unknown option %s", a.c_str());
    }
    if (!model || (prompt.empty() && !serve)) die("usage: momoe-metal -m MODEL.gguf (--tokens 1,2,3 [-n N] | --serve) [--slots N | --pool-gib G]");
    if (!e.ds.path.empty()) {                                // DSpark: windows of the last token and up to its block (5)
        if (mtp_k > 0) die("--dspark and --mtp: one drafter at a time");
        mtp_k = 8;                                           // (the windows' room until the file's block size is read)
    }
    e.load_mtp = mtp_k > 0 && e.ds.path.empty();
    e.max_window = std::max(2, mtp_k + 1);
    if (getenv("MOMOE_DIRECT")) g_direct = true;
    bool small_uma = false;                                  // a unified-memory machine under 16 GiB (uma_small)
#if defined(MOMOE_CUDA)
    {                                                        // unified memory: experts read past the page cache unless
        cudaDeviceProp prop;                                 // MOMOE_NO_DIRECT (paging the mapped file in: 0.1-0.25 of the
        if (cudaGetDeviceProperties(&prop, e.device) == cudaSuccess && prop.integrated) {   // SSD there; measured)
            e.uma = true;
            small_uma = uma_small();
            // enough hardware queues that the readers' copy stream does not share the compute stream's (before the
            // context is made: the resident mailboxes' waits would hold it)
            setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 0);
            if (!getenv("MOMOE_NO_DIRECT") && !small_uma) g_direct = true;   // (a small one: see uma_small)
            fprintf(stderr, "momoe-cuda: %s shares the system's memory: experts %s\n", prop.name,
                    g_direct ? "read from the SSD directly (--direct-io; MOMOE_NO_DIRECT=1: from the mapped file)"
                             : "copied from the mapped file");
            // a machine whose memory another program holds (another momoe-cuda, llama.cpp): refused before anything is
            // loaded.  The GPU's allocations come out of the RAM and are charged to no cgroup, so two engines at once
            // ran the DGX Spark out of memory and it stopped answering (2026-10-17)
            const double reserve = uma_reserve_gib() * 1073741824.0;
            const double need = reserve + uma_min_gib() * 1073741824.0
                                + std::max(0.0, pool_gib) * 1073741824.0;   // (--pool-gib: that pool on top)
            const double room = uma_free_bytes();
            if (room < need) {
                const std::string why = std::to_string((int) (room / 1073741824.0)) + " GiB are free, " +
                                        std::to_string((int) (need / 1073741824.0)) + " GiB are needed (MOMOE_UMA_RESERVE_GIB + MOMOE_UMA_MIN_GIB)";
                die("not enough free memory to start: %s; is another momoe-cuda or llama.cpp running?", why.c_str());
            }
        }
    }
#endif
    // (--direct-io: the CPU's experts read the mapping; nemo only, for now: hybrid_on; --layer-split: the stops)
    // (--helper-tier: with qwen35moe's CPU experts through the stops, the helper card computing the missing experts it
    // holds: Ornith-1.5-122B, RTX 4070 Ti + RTX 3060, 12.1 tok/s against 8.7 with the copies; the helper saves RAM, not
    // time: 18-19 without it.  Nemotron: the copies)
    e.hybrid = !kHostMem && !g_direct && !getenv("MOMOE_NO_HYBRID") && e.split.empty();
    load(e, model, pool_gib, slots);
    if (!e.ds.path.empty()) {                                // (before the pool takes what is left)
        load_dspark(e);
        if (e.ds.block > mtp_k) die("--dspark: a drafter block longer than 8 (not done)");
        mtp_k = e.ds.block;
    }
    if (e.helper && e.m.nemo) e.hybrid = false;
    if (mtp_k > 0 && e.m.nemo) die("--mtp: not yet for Nemotron models");
    if (!kHostMem && pool_gib <= 0 && slots <= 0) fit_pool(e);
    e.pool_asked = pool_gib > 0 || slots > 0;
    if (kHostMem) fit_pool_metal(e);
    if (e.helper && kHostMem) die("--helper-tier: NVIDIA cards only");
    if (e.helper && e.pool.n_slots + (int) e.pool.g.size() >= e.m.n_exp_all) {
        fprintf(stderr, "momoe-cuda: helper tier: not used (the main card holds every expert)\n");
        e.helper = false;
    }
    if (e.helper) make_hpool(e);
    if (e.hybrid && hybrid_model(e.m))                       // the CPU's experts: the physical cores less the readers'
        e.cpu_x.start(getenv("MOMOE_CPU_THREADS") ? atoi(getenv("MOMOE_CPU_THREADS"))
                                                  : std::max(2, (int) std::thread::hardware_concurrency() / 2));
#if defined(MOMOE_CUDA)
    if (e.hybrid && !e.m.nemo && hybrid_model(e.m) && !e.preloaded && !e.helper && !getenv("MOMOE_NO_MAILBOX")) {
        bool cpu_dn = false;                                 // qwen35moe: the CPU's experts through mailboxes too (a
        for (auto & L : e.m.layers) cpu_dn = cpu_dn || L.cpu_dn;   // graph per token, built again each time: CUDA
        if (cpu_dn) mb_init(e);                              // graphs stay off below)
    }
    if (e.hybrid && e.m.nemo && !getenv("MOMOE_NO_MAILBOX") && e.pool.n_slots < e.m.n_exp_all) mb_init(e);
    else if (e.m.glm5 && g_direct && getenv("MOMOE_RMB") && e.pool.host && e.pool.n_slots < e.m.n_exp_all) {
        // (opt-in, MOMOE_RMB=1 with MOMOE_POOL_HOST=1: with a device pool the readers' copies to the GPU did not run
        // while a mailbox's wait held it, on the DGX Spark; with the host pool they work, measured 8.9 against 8.5
        // tok/s, but the device pool with the stops is faster still: 9.2)
        rmb_init(e);                                         // (a graph per token, built again each time: no CUDA graphs)
        if (!getenv("MOMOE_CUDA_GRAPHS")) setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
    } else if (e.m.lfm && e.pool.host && (lfm_rmb(e, e.pool.n_slots) || getenv("MOMOE_RMB"))
               && e.pool.n_slots < e.m.n_exp_all) {         // (a device pool: the readers' copies would wait behind
        rmb_init(e);                                         // the mailbox's wait, measured: never with one)
        if (!getenv("MOMOE_CUDA_GRAPHS")) setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
    } else if ((e.m.flint || e.m.gemma4 || e.m.ling) && e.mailbox) {
        // flint through its mailboxes: the kept decode graph (forward_decode) replayed as a CUDA graph (RTX 4070 Ti,
        // UD-IQ3_XXS: 79-82 -> 87 tok/s, the same tokens); the trial's few graphs with stops fit beside it
    } else if (!getenv("MOMOE_CUDA_GRAPHS") && !getenv("GGML_CUDA_DISABLE_GRAPHS")) {   // CUDA graphs for the
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);         // mailbox's one-graph token only: with the stops' many
        cuda_graphs_off = true;                             // splits, instantiating them ran out of VRAM (12 GB card;
    }                                                       // MOMOE_CUDA_GRAPHS=1: on anyway); and for kept graphs
#endif
    if (mtp_k > 0 && !e.m.has_mtp && !e.ds.on) die("--mtp: this model file has no MTP layer");
    e.readers.device = e.device;
    e.fillers.device = e.device;
    if (e.hybrid) e.fillers.start(getenv("MOMOE_FILLERS") ? atoi(getenv("MOMOE_FILLERS")) : 2);
    e.readers.start(getenv("MOMOE_READERS") ? atoi(getenv("MOMOE_READERS"))   // (CUDA: measured; a small unified-memory
                    : kHostMem ? 8 : small_uma ? 4 : 16);   // machine: as fast with 4, a quarter of the pinned buffers)
    e.demand.device = e.device;
    if (getenv("MOMOE_DEMAND_READERS")) e.demand.start(atoi(getenv("MOMOE_DEMAND_READERS")));
    e.predict = !getenv("MOMOE_NO_PREDICT") && !(e.hybrid && e.m.nemo);   // (hybrid: measured faster without)
    e.predict_copy = e.predict;
    if (e.hybrid && hybrid_model(e.m) && !e.m.nemo) {     // a forced expert path: no trial
        const char * q = getenv("MOMOE_QWEN_HYBRID"), * f = getenv("MOMOE_HYB_FRAC");
        if (f) {
            e.hyb_active = e.hyb_done = e.hyb_forced = true;   // MOMOE_HYB_FRAC=x: the split with that share copied in
            e.hyb_frac = std::clamp((float) atof(f), 0.0f, 1.0f);
            e.predict = false;
        } else if (q && q[0] == '1') {
            e.hyb_active = e.hyb_done = e.hyb_forced = true;   // MOMOE_QWEN_HYBRID=1: the CPU's experts
            e.predict = false;
        }
    }
    e.predict2 = e.predict && getenv("MOMOE_PREDICT2");      // two layers ahead: as fast, a third more read (M4)
#if !defined(MOMOE_CUDA)
    if (nocache)                                             // expert reads from the SSD, as on a Mac without the RAM
        for (int fd : e.fds)
            if (fd != e.m.ple_fd) fcntl(fd, F_NOCACHE, 1);
#else
    if (nocache) die("--nocache: Apple Silicon only");
#endif
    if (e.pool.n_slots == e.m.n_exp_all && !getenv("MOMOE_NO_PRELOAD")) pre = true;   // all fit: no pauses
    if (pre) {
        preload(e);
        e.predict = e.predict2 = false;                      // (nothing to read ahead: the graphs
    }                                                        // skip the next layers' routers)
#if defined(MOMOE_CUDA)
    if (cuda_graphs_off && e.preloaded && reuse_backend_ok(e) && !getenv("MOMOE_NO_GRAPH_REUSE") && !debug_on())
        unsetenv("GGML_CUDA_DISABLE_GRAPHS");               // kept graphs (no stops: one split each): replayed as CUDA
#endif                                                       // graphs
    else if (profile) pin_profile(e, profile, getenv("MOMOE_PIN_FRAC") ? atof(getenv("MOMOE_PIN_FRAC")) : 0.6);
    e.pause = !kHostMem ? false : !debug_on() && !getenv("MOMOE_NO_PAUSE") && (!pre || getenv("MOMOE_FORCE_PAUSE"));
#if !defined(MOMOE_CUDA)
    if (e.pause) ggml_backend_metal_set_pause(e.metal, "topk-");
#else
    if (false) {}
#endif
    else if (!pre || debug_on() || getenv("MOMOE_FORCE_CB")) ggml_backend_sched_set_eval_callback(e.sched, eval_cb, &e);
    if (const char * ns = nostall_arg) {
        if (strcmp(ns, "sub") && strcmp(ns, "drop")) die("--nostall takes sub or drop, not %s", ns);
        if (e.m.lfm) die("--nostall: not for lfm2moe models (their dense blocks have no experts)");
        if (!kHostMem) die("--nostall: Apple Silicon only");
        if (!e.pause || pre) fprintf(stderr, "momoe-metal: --nostall: not used (every expert is in memory: no pauses)\n");
        else {
            e.nostall = strcmp(ns, "sub") ? 2 : 1;
            for (auto & G : e.pool.g) {
                G.zero = G.n_slots - 1;
                if (G.owner[G.zero] >= 0) e.pool.slot_of[G.owner[G.zero]] = -1;
                G.owner[G.zero] = -1;
                G.used[G.zero] = UINT64_MAX;                 // never taken
                for (int w = 0; w < 3; ++w)
                    memset((char *) G.t[w]->data + (size_t) G.zero * G.t[w]->nb[2], 0, G.t[w]->nb[2]);
            }
            e.ns_every = nostall_every;
            if (e.ns_every) fprintf(stderr, "momoe-metal: --nostall %s: a pause every %d layers while writing (changes the "
                                    "output)\n", ns, e.ns_every);
            else fprintf(stderr, "momoe-metal: --nostall %s: no pauses while writing (changes the output)\n", ns);
        }
    }

    if (serve) {
        const int r = serve_loop(e, mtp_k, spec_min_p);
#if defined(MOMOE_CUDA)
        mb_stop(e);                                          // (the mailbox worker: a joinable thread at exit aborts)
#endif
        e.readers.stop();
        e.demand.stop();
        ggml_backend_sched_free(e.sched);
        for (auto b : e.bufs) ggml_backend_buffer_free(b);
        ggml_backend_free(e.metal);
        ggml_backend_free(e.cpu);
        fflush(stdout);
        fflush(stderr);
        _exit(r);                                             // the stdin reader may still be blocked in a read
    }
    Awake awake;                                             // (macOS: see Awake)
    double t0 = now_s();
    int p = 0, next = 0;
    std::vector<float> H;                                    // --mtp: the prompt's hidden states (the MTP layer's input)
    for (size_t i = 0; i < prompt.size(); i += e.max_batch) {
        const int T = (int) std::min(prompt.size() - i, (size_t) e.max_batch);
        Out o;
        next = forward(e, prompt.data() + i, T, p, mtp_k ? &o : nullptr);
        if (mtp_k) H.insert(H.end(), o.hidden.begin(), o.hidden.end());
        p += T;
    }
    const double t_prompt = now_s() - t0;
    if (bench_window) {                                       // the cost of a window of T tokens (verify windows)
        std::vector<int32_t> w(8, prompt.back());
        for (int T = 1; T <= 8; ++T) {
            Out o;                                            // MOMOE_BENCH_ALL / _SNAP: as a verify window
            o.all = getenv("MOMOE_BENCH_ALL") != nullptr;
            o.snap = getenv("MOMOE_BENCH_SNAP") != nullptr && T > 1;
            Out * op = (o.all || o.snap) ? &o : nullptr;
            for (int r = 0; r < 3; ++r) forward(e, w.data(), T, p, op);   // warm-up (the state is overwritten: timing only)
            std::vector<double> ts;
            for (int r = 0; r < 15; ++r) { const double a = now_s(); forward(e, w.data(), T, p, op); ts.push_back(now_s() - a); }
            std::sort(ts.begin(), ts.end());
            fprintf(stderr, "momoe-metal: window of %d: %.1f ms (%.1f ms per token)\n", T, 1000 * ts[7], 1000 * ts[7] / T);
        }
        quit(0);
    }
    if (getenv("MOMOE_TF_WINDOW")) {                          // a test: the tokens in MOMOE_TF_FILE after the prompt,
        // fed in windows of MOMOE_TF_WINDOW (as verify windows, all accepted): each window's greedy next tokens against
        // the file's, which a run token by token wrote (a window must give the same tokens, near-ties aside)
        const int W = atoi(getenv("MOMOE_TF_WINDOW"));
        std::vector<int32_t> fol;
        if (FILE * f = fopen(getenv("MOMOE_TF_FILE"), "r")) { int x; while (fscanf(f, "%d%*[, \n]", &x) == 1) fol.push_back(x); fclose(f); }
        if (fol.empty() || fol[0] != next) fprintf(stderr, "TF: the file's first token %d is not the prompt's next %d\n", fol.empty() ? -1 : fol[0], next);
        int bad = 0;
        for (size_t i = 0; i + 1 < fol.size(); i += W) {
            const int T = (int) std::min((size_t) W, fol.size() - 1 - i);
            Out o;
            o.all = true;
            o.logits = getenv("MOMOE_TF_PROBS") != nullptr;   // each position's top 5 probabilities, to that file
            forward(e, fol.data() + i, T, p, &o);
            if (o.logits) {
                static FILE * pf = fopen(getenv("MOMOE_TF_PROBS"), "w");
                const size_t V = o.all_logits.size() / T;
                for (int j = 0; j < T; ++j) {
                    const float * lg = o.all_logits.data() + j * V;
                    const float mx = *std::max_element(lg, lg + V);
                    double z = 0;
                    for (size_t v = 0; v < V; ++v) z += exp(lg[v] - mx);
                    std::vector<int> ix(V);
                    for (size_t v = 0; v < V; ++v) ix[v] = (int) v;
                    std::partial_sort(ix.begin(), ix.begin() + 5, ix.end(), [&](int a2, int b2) { return lg[a2] > lg[b2]; });
                    fprintf(pf, "%zu", i + j + 1);
                    for (int r2 = 0; r2 < 5; ++r2) fprintf(pf, " %d:%.5f", ix[r2], exp(lg[ix[r2]] - mx) / z);
                    fprintf(pf, "\n");
                }
                fflush(pf);
            }
            for (int j = 0; j < T; ++j)
                if (o.next[j] != fol[i + j + 1]) { fprintf(stderr, "TF: position %zu (window at %zu, slot %d): %d, the file %d\n", i + j + 1, i, j, o.next[j], fol[i + j + 1]); ++bad; }
            p += T;
        }
        fprintf(stderr, "TF: windows of %d over %zu tokens: %d differ\n", W, fol.size() - 1, bad);
        fflush(stderr);
        _exit(0);                                            // (the readers' threads are still running)
    }
    const uint64_t pm = e.pool.misses, ph = e.pool.hits;
    const uint64_t pb = e.pool.bytes_read;
    const double pr = e.pool.read_s;
    std::vector<int32_t> out;
    t0 = now_s();
    double t_half = 0, cb_half = 0, b_half = 0, c_half = 0, w_half = 0;
    int built_half = 0;
    double built_s_half = 0;
    if (mtp_k > 0) {                                          // drafts from the MTP layer, checked in one window
        const int ne = e.m.h_dim();
        std::vector<float> hc, hc2;
        float pb = 0;
        auto chain = [&](int d, int q) {                     // drafts from d on (d at position q+1, chained on the MTP's
            std::vector<int32_t> dr;                         // own state) while each is at least spec_min_p sure
            if (pb < spec_min_p) return dr;
            dr.push_back(d);
            for (int j = 1; j < mtp_k; ++j) {
                const int32_t tok = dr.back();
                const int32_t nx = mtp_forward(e, hc.data(), &tok, 1, q + j, hc2, &pb);
                hc.swap(hc2);
                if (pb < spec_min_p) break;
                dr.push_back(nx);
            }
            return dr;
        };
        std::vector<int32_t> drafts;
        if (e.ds.on) drafts = ds_draft(e, next, p);          // (DSpark: its rings were filled with the prompt)
        else {
        // the MTP layer over the prompt: (state i, token i+1), the last with the first generated token
        std::vector<int32_t> mt(prompt.begin() + 1, prompt.end());
        mt.push_back(next);
        int d = 0;
        for (size_t i = 0; i < mt.size(); i += e.max_batch) {
            const int T = (int) std::min(mt.size() - i, (size_t) e.max_batch);
            d = mtp_forward(e, H.data() + i * ne, mt.data() + i, T, 1 + (int) i, hc, &pb);
        }
        drafts = chain(d, p);
        }
        out.push_back(next);
        int t = next;
        uint64_t windows = 0, accepted = 0, drafted = 0, redone = 0;
        double t_win[6] = {0}, t_rb = 0;
        int n_win[6] = {0};
        const double tg0 = now_s();
        double th = 0;                                       // the second half (warmed up): from n_gen/2 tokens on
        size_t nh = 0;
        while ((int) out.size() < n_gen) {
            if (!nh && (int) out.size() >= n_gen / 2) { th = now_s(); nh = out.size(); }
            std::vector<int32_t> W = {t};
            W.insert(W.end(), drafts.begin(), drafts.end());
            const int T = (int) W.size();
            drafted += T - 1;
            Out ov;
            ov.all = true;
            ov.snap = false;             // snapshots in the window cost 19-30 ms (M4): the state is copied aside instead
            if (T > 1) save_state(e, e.saved);   // and the window is run again up to the last accepted token if a draft fails
            const double f0 = now_s();
            forward(e, W.data(), T, p, &ov);
            t_win[std::min(T, 5)] += now_s() - f0;
            ++n_win[std::min(T, 5)];
            int n = 0;
            while (n < T - 1 && ov.next[n] == W[n + 1]) ++n;
            ++windows;
            accepted += n;
            for (int j = 0; j <= n; ++j) out.push_back(ov.next[j]);
            const double r0 = now_s();
            if (n + 1 < T) {                                  // a draft failed: the state as before the window, then the
                restore_state(e, e.saved);                    // accepted tokens again (their K/V are rewritten too)
                forward(e, W.data(), n + 1, p);
                ++redone;
            }
            t_rb += now_s() - r0;
            if (e.ds.on) drafts = ds_draft(e, ov.next[n], p + n + 1);
            else {
            const int dn = mtp_forward(e, ov.hidden.data(), ov.next.data(), n + 1, p + 1, hc, &pb);
            drafts = chain(dn, p + n + 1);              // dn follows the MTP's last position, p+n+1
            }
            p += n + 1;
            t = ov.next[n];
        }
        const double t_end = now_s();
        if (nh) fprintf(stderr, "momoe-metal: second half of the answer: %.1f tok/s\n", (out.size() - nh) / (t_end - th));
        out.resize(n_gen);
        const double tg = now_s() - tg0;
        for (int T = 1; T <= 5; ++T)
            if (n_win[T]) fprintf(stderr, "momoe-metal:   windows of %d%s: %d, %.1f ms each\n", T, T == 5 ? "+" : "", n_win[T],
                                  1000 * t_win[T] / n_win[T]);
        fprintf(stderr, "momoe-metal:   %llu windows run again after a failed draft; rollback %.2f ms per window; MTP %.2f "
                "ms per window; the rest %.2f ms per window\n", (unsigned long long) redone, 1000 * t_rb / windows,
                1000 * e.t_mtp / windows,
                1000 * (tg - t_rb - e.t_mtp - (t_win[1] + t_win[2] + t_win[3] + t_win[4] + t_win[5])) / windows);
        printf("tokens:");
        for (auto x : out) printf(" %d", x);
        printf("\n");
        fprintf(stderr, "momoe-metal: %s up to %d drafts (min p %.2f): %llu windows, %llu drafts, %llu accepted, %.2f tokens "
                "per window; %s %.1f ms per window\n", e.ds.on ? "DSpark" : "MTP", mtp_k, e.ds.on ? e.ds.conf_min : spec_min_p,
                (unsigned long long) windows, (unsigned long long) drafted, (unsigned long long) accepted,
                (double) (accepted + windows) / windows, e.ds.on ? "the drafter" : "MTP layer",
                1000 * (e.ds.on ? e.ds.t : e.t_mtp) / windows);
        fprintf(stderr, "momoe-metal: generated %d tokens in %.2f s (%.1f tok/s): expert hit rate %.1f%%\n", n_gen, tg,
                (n_gen - 1) / tg, 100.0 * e.pool.hits / std::max<uint64_t>(1, e.pool.hits + e.pool.misses));
#if defined(MOMOE_CUDA)
        mb_stop(e);
#endif
        e.readers.stop();
        e.demand.stop();
        ggml_backend_sched_free(e.sched);
        for (auto b : e.bufs) ggml_backend_buffer_free(b);
        ggml_backend_free(e.metal);
        ggml_backend_free(e.cpu);
        quit(0);
    }
    // MOMOE_KL_DUMP=FILE: each step's token and log-probabilities (f16) written; MOMOE_KL_REF=FILE: that run's tokens
    // fed instead of ours, and each step's KL(ref || ours) and top-1 agreement
    FILE * kl_dump = getenv("MOMOE_KL_DUMP") ? fopen(getenv("MOMOE_KL_DUMP"), "wb") : nullptr;
    FILE * kl_ref = getenv("MOMOE_KL_REF") ? fopen(getenv("MOMOE_KL_REF"), "rb") : nullptr;
    if ((getenv("MOMOE_KL_DUMP") && !kl_dump) || (getenv("MOMOE_KL_REF") && !kl_ref)) die("cannot open the KL file");
    double kl_sum = 0, kl_max = 0;
    int kl_n = 0, kl_top1 = 0;
    std::vector<double> kl_all;
    auto logsm = [](const std::vector<float> & l, std::vector<float> & o) {
        const float mx = *std::max_element(l.begin(), l.end());
        double z = 0;
        for (float v : l) z += exp((double) v - mx);
        const float lz = (float) (mx + log(z));
        o.resize(l.size());
        for (size_t j = 0; j < l.size(); ++j) o[j] = l[j] - lz;
    };
    std::vector<float> lp;
    std::vector<ggml_fp16_t> h16;
    for (int i = 0; i < n_gen; ++i) {
        if (i == n_gen / 2) { t_half = now_s(); cb_half = e.pool.cb_s; b_half = e.t_build; built_half = g_built; built_s_half = g_built_s; c_half = e.t_compute; w_half = e.t_wait; }
        int32_t fed = next;
        if (kl_ref && fread(&fed, 4, 1, kl_ref) != 1) die("the KL reference is shorter than -n");
        out.push_back(fed);
        Out o;
        o.logits = kl_dump || kl_ref;
        next = forward(e, &out.back(), 1, p++, o.logits ? &o : nullptr);
        if (!o.logits) continue;
        logsm(o.last_logits, lp);
        const size_t V = lp.size();
        if (kl_dump) {
            h16.resize(V);
            for (size_t j = 0; j < V; ++j) h16[j] = ggml_fp32_to_fp16(std::max(lp[j], -60000.0f));
            fwrite(&fed, 4, 1, kl_dump);
            fwrite(h16.data(), 2, V, kl_dump);
        } else {
            h16.resize(V);
            if (fread(h16.data(), 2, V, kl_ref) != V) die("the KL reference is short");
            double kl = 0;
            size_t ra = 0, oa = 0;
            for (size_t j = 0; j < V; ++j) {
                const float r = ggml_fp16_to_fp32(h16[j]);
                if (r > ggml_fp16_to_fp32(h16[ra])) ra = j;
                if (lp[j] > lp[oa]) oa = j;
                if (r > -30) kl += exp((double) r) * ((double) r - lp[j]);
            }
            kl_sum += kl;
            kl_max = std::max(kl_max, kl);
            kl_all.push_back(kl);
            ++kl_n;
            kl_top1 += ra == oa;
        }
    }
    if (kl_dump) fclose(kl_dump);
    if (kl_ref) {
        fclose(kl_ref);
        std::sort(kl_all.begin(), kl_all.end());
        fprintf(stderr, "momoe-metal: KL against the reference over %d steps: mean %.4f, median %.4f, p99 %.4f, max %.4f; "
                "top-1 agreement %.1f%%\n", kl_n, kl_sum / std::max(1, kl_n), kl_all.empty() ? 0 : kl_all[kl_all.size() / 2],
                kl_all.empty() ? 0 : kl_all[kl_all.size() * 99 / 100], kl_max, 100.0 * kl_top1 / std::max(1, kl_n));
    }
    const double t_gen = now_s() - t0;
    fprintf(stderr, "momoe-metal: second half of the answer: %.1f tok/s (%.1f ms per token, %.1f ms of it in the callback)\n",
            (n_gen - n_gen / 2) / (now_s() - t_half), 1000 * (now_s() - t_half) / (n_gen - n_gen / 2),
            1000 * (e.pool.cb_s - cb_half) / (n_gen - n_gen / 2));
    fprintf(stderr, "momoe-metal:   per token: graph build + alloc %.2f ms, compute %.2f ms (waiting for pauses %.2f ms); "
            "%d graph splits; %d graphs built (%.1f ms)\n", 1000 * (e.t_build - b_half) / (n_gen - n_gen / 2),
            1000 * (e.t_compute - c_half) / (n_gen - n_gen / 2), 1000 * (e.t_wait - w_half) / (n_gen - n_gen / 2),
            ggml_backend_sched_get_n_splits(e.sched), g_built - built_half, 1000 * (g_built_s - built_s_half));
    printf("tokens:");
    for (auto t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr, "momoe-metal: prompt %zu tokens in %.2f s (%.1f tok/s): %llu expert misses, %.0f MiB read in %.2f s\n",
            prompt.size(), t_prompt, prompt.size() / t_prompt, (unsigned long long) pm, pb / 1048576.0, pr);
    if (e.pool.stage_from_pool)
        fprintf(stderr, "momoe-cuda: %llu staged experts copied from the pool instead of read\n",
                (unsigned long long) e.pool.stage_from_pool);
    const uint64_t gh = e.pool.hits - ph, gm = e.pool.misses - pm;
    fprintf(stderr, "momoe-metal: predictions: %llu experts read ahead, %llu of them used (%.0f%%), %llu still being read "
            "when needed\n", (unsigned long long) e.pool.prefetched, (unsigned long long) e.pool.prefetch_hits,
            100.0 * e.pool.prefetch_hits / std::max<uint64_t>(1, e.pool.prefetched), (unsigned long long) e.pool.late);
    if (e.nostall)
        fprintf(stderr, "momoe-metal: --nostall: %llu graphs routed among the experts in the pool; %llu of %llu wanted experts not in it "
                "(%.1f%%, %s)\n", (unsigned long long) e.ns_tokens, (unsigned long long) e.ns_missing,
                (unsigned long long) e.ns_wanted, 100.0 * e.ns_missing / std::max<uint64_t>(1, e.ns_wanted),
                e.nostall == 1 ? "replaced by the best in the pool" : "dropped");
    fprintf(stderr, "momoe-metal: generated %d tokens in %.2f s (%.1f tok/s): hit rate %.1f%%, %.0f MiB read in %.2f s\n",
            n_gen, t_gen, n_gen / t_gen, 100.0 * gh / std::max<uint64_t>(1, gh + gm),
            (e.pool.bytes_read - pb) / 1048576.0, e.pool.read_s - pr);
    if (e.helper)
        fprintf(stderr, "momoe-cuda: helper tier: %llu chosen experts computed on the helper card, %llu evictions copied "
                "down to it\n", (unsigned long long) e.helper_hits, (unsigned long long) e.helper_down);
#if defined(MOMOE_CUDA)
    if (e.cpu_x.n_threads)
        fprintf(stderr, "momoe-cuda: hybrid: %llu of %llu chosen experts computed on the CPU (%d threads), %llu copied in, "
                "%.2f s waited for them, %llu graphs through the mailboxes\n", (unsigned long long) e.hyb_miss,
                (unsigned long long) (e.hyb_hits + e.hyb_miss), e.cpu_x.n_threads, (unsigned long long) e.hyb_copy,
                e.t_cpu_wait, (unsigned long long) e.mb_graphs);
    mb_stop(e);
    if (e.cpu_x.n_threads && getenv("MOMOE_CPU_PROF")) {
        const auto & C = e.cpu_x;
        fprintf(stderr, "momoe-cuda: CPU experts, s in all: all awake %.2f, up %.2f, to down's format %.2f, down %.2f "
                "(thread 0), launch to done %.2f; %llu experts\n", C.prof[0], C.prof[1], C.prof[2], C.prof[3], C.prof[4],
                (unsigned long long) C.experts);
    }
#endif
    if (e.cpu_x.n_threads) e.cpu_x.stop();
    if (!e.fillers.th.empty()) e.fillers.stop();
    e.readers.stop();
    e.demand.stop();
    ggml_backend_sched_free(e.sched);
    for (auto b : e.bufs) ggml_backend_buffer_free(b);
    ggml_backend_free(e.metal);
    ggml_backend_free(e.cpu);
    quit(0);
}

