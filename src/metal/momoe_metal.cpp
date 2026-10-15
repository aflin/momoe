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
#include "ggml-metal.h"
#include "gguf.h"

#include <fcntl.h>
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

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] static void die(const char * fmt, const char * a = "") {
    fprintf(stderr, "momoe-metal: ");
    fprintf(stderr, fmt, a);
    fprintf(stderr, "\n");
    exit(1);
}

// ------------------------------------------------------------------------------------------------ the model
struct Layer {
    bool recurrent = false;
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
    int exp_fd = -1;                                         // the file (shard) its experts are in
    // qwen4exp: hyper-connections (before the token mixer and before the MoE) and the n-gram embedding (PLE, one layer)
    ggml_tensor *hc_attn_norm = nullptr, *hc_attn_down = nullptr, *hc_attn_up = nullptr, *hc_attn_inject = nullptr;
    ggml_tensor *hc_ffn_norm = nullptr, *hc_ffn_down = nullptr, *hc_ffn_up = nullptr, *hc_ffn_inject = nullptr;
    ggml_tensor *ple_key = nullptr, *ple_value = nullptr, *ple_norm_key = nullptr, *ple_norm_query = nullptr,
                *ple_norm_conv = nullptr, *ple_conv1d = nullptr;
    ggml_tensor *ple_state = nullptr;                        // the PLE convolution's history [(kernel-1)*ngram, hc*n_embd]
    // qwen4exp's sparse attention (QSA): the indexer, its raw keys per position and pooled keys per block of positions
    ggml_tensor *idx_q = nullptr, *idx_k = nullptr, *idx_q_norm = nullptr, *idx_k_norm = nullptr;
    ggml_tensor *idx_raw = nullptr, *idx_pool = nullptr;     // [idx_dim, n_ctx], [idx_dim, n_ctx / kpool + 64] f16
};

struct Model {
    int n_embd, n_layer, n_head, n_head_kv, head_dim, n_rot, n_expert, n_used, n_ff_exp, n_vocab;
    int ssm_conv, ssm_state, ssm_groups, ssm_dt_rank, ssm_inner, full_attn_interval;
    int sections[4];
    float rms_eps, rope_base;
    bool qwen4 = false;                                      // qwen4exp (Flash-Next): hyper-connections, PLE, sigmoid
                                                             // gate in the delta net
    int hc = 1, hc_lr = 0;                                   // hyper-connection streams, low rank
    ggml_tensor *hc_head_norm = nullptr, *hc_head_down = nullptr, *hc_head_up = nullptr;   // the final mixer (output norm)
    ggml_tensor *mtp_head_norm = nullptr, *mtp_head_down = nullptr, *mtp_head_up = nullptr;   // the MTP layer's own
    int h_dim() const { return n_embd * hc; }                // the hidden state the MTP layer reads (qwen4exp: all streams)
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
    ggml_tensor *t[3] = {nullptr, nullptr, nullptr};          // gate / up / down: [ne0, ne1, n_slots]
    ggml_tensor *stage[3] = {nullptr, nullptr, nullptr};      // [ne0, ne1, n_expert]: one layer's experts, a big batch
    std::vector<int> owner;                                  // slot -> layer * n_expert + expert, -1
    std::vector<uint64_t> used;                              // slot -> last use (a clock)
    std::unique_ptr<std::atomic<int>[]> left;                // slot -> parts still being read
    std::vector<uint8_t> by_prefetch;                        // slot -> filled by a prediction, not used yet
    int zero = -1;                                           // --nostall: a slot of zeros, never taken
};

struct Pool {
    std::vector<Group> g;
    int n_slots = 0;                                         // all groups
    std::vector<int> slot_of;                                // layer * n_expert + expert -> slot in its group, -1
    uint64_t clock = 0;
    // counters
    uint64_t hits = 0, misses = 0, bytes_read = 0, calls = 0;
    uint64_t prefetched = 0, prefetch_hits = 0, late = 0;    // late: a prefetched expert not read yet when needed
    double read_s = 0, cb_s = 0;
    bool lfu = true;                                         // the least used (counts halved every
    std::vector<uint32_t> freq;                              // 64 tokens) is replaced, recency breaks ties
    uint64_t uses = 0;
};

static void read_at(int fd, void * dst, size_t size, size_t off);

// a few threads that read expert parts from the model file in the background (the SSD gives about twice the bandwidth
// with 4-8 reads in flight as with one: measured on the M4); each job counts down its slot's outstanding parts
#include <deque>
struct Readers {
    struct Job { int fd; void * dst; size_t size, off; std::atomic<int> * left; const void * src = nullptr; };   // src: a copy
    std::vector<std::thread> th;
    std::deque<Job> q;
    std::mutex mu;
    std::condition_variable cv;
    bool quit = false;
    void start(int n) {
        for (int i = 0; i < n; ++i) th.emplace_back([this] { work(); });
    }
    void work() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return quit || !q.empty(); });
                if (quit) return;
                j = q.front();
                q.pop_front();
            }
            if (j.src) memcpy(j.dst, j.src, j.size);
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

struct Engine {
    Model m;
    Pool pool;
    std::vector<int> fds;                                    // the model's files (shards)
    std::vector<int32_t> tok_at;                             // qwen4exp: the token at each position (the PLE n-grams)
    Readers readers;
    int n_ctx = 32768, max_batch = 512;
    bool load_mtp = false;                                   // --mtp K: draft K tokens with the MTP layer
    bool preloaded = false;                                  // every expert in its own slot (preload)
    std::string draft_vocab;                                 // --draft-vocab: the draft head's token subset
    std::string mtp_gguf;                                    // --mtp-gguf: qwen4exp's MTP layer (not in its GGUFs)
    double t_mtp = 0;                                        // time in the MTP layer
    int max_window = 8;                                      // verify windows up to this many tokens
    ggml_backend_t metal = nullptr, cpu = nullptr;
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
};

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
    e.bufs.push_back(ggml_backend_alloc_ctx_tensors(cm, e.metal));
    size_t bytes = 0;
    std::vector<char> raw;
    for (size_t i = 0; i < want.size(); ++i) {
        ggml_tensor * s = src[i].first, * t = *want[i].to;
        bytes += ggml_nbytes(t);
        if (want[i].how == 0) {
            read_at(fd, t->data, ggml_nbytes(s), src[i].second);   // a Metal buffer is host memory on Apple Silicon
            continue;
        }
        raw.resize(ggml_nbytes(s));
        read_at(fd, raw.data(), raw.size(), src[i].second);
        if (want[i].how == 1) {
            const float * r = (const float *) raw.data();
            float * d = (float *) t->data;
            for (int64_t j = 0; j < ggml_nelements(s); ++j) d[j] = r[j] + 1.0f;
        } else {                                             // each expert's first (gate) or second (up) half of rows
            const size_t half = s->nb[1] * (s->ne[1] / 2);
            for (int64_t x = 0; x < s->ne[2]; ++x)
                memcpy((char *) t->data + x * half, raw.data() + x * s->nb[2] + (want[i].how == 3 ? half : 0), half);
        }
    }
    close(fd);
    gguf_free(g);
    ggml_free(meta);
    m.has_mtp = true;
    fprintf(stderr, "momoe-metal: MTP layer from %s: %.2f GiB\n", path, bytes / 1073741824.0);
}

static void load(Engine & e, const char * path, double pool_gib, int slots_arg) {
    Model & m = e.m;
    ggml_context * meta = nullptr;
    gguf_init_params gp = {/*no_alloc*/ true, &meta};
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) die("cannot read %s as GGUF", path);
    std::string arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
    if (arch != "qwen35moe" && arch != "qwen4exp") die("%s: only qwen35moe and qwen4exp models", arch.c_str());
    m.qwen4 = arch == "qwen4exp";
    auto k = [&](const char * s) { return arch + "." + std::string(s); };
    auto has = [&](const char * s) { return gguf_find_key(g, k(s).c_str()) >= 0; };
    m.n_embd = gguf_int(g, k("embedding_length"));
    int nextn = has("nextn_predict_layers") ? gguf_int(g, k("nextn_predict_layers")) : 0;
    m.n_layer = gguf_int(g, k("block_count")) - nextn;     // the MTP layer is not run yet
    m.n_head = gguf_int(g, k("attention.head_count"));
    m.n_head_kv = gguf_int(g, k("attention.head_count_kv"));
    m.head_dim = gguf_int(g, k("attention.key_length"));
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

    // the dense weights: every tensor but the routed experts, the MTP layer and the PLE table, in one Metal buffer
    const int n_split = gguf_find_key(g, "split.count") >= 0 ? gguf_int(g, "split.count") : 1;
    const std::vector<std::string> paths = shard_paths(path, n_split);
    ggml_init_params ip = {ggml_tensor_overhead() * 4096, nullptr, true};
    ggml_context * cw = ggml_init(ip);
    struct ToRead { ggml_tensor * t; int fd; size_t off; };
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
        const size_t data_off = gguf_get_data_offset(g);
        const int n_t = (int) gguf_get_n_tensors(g);
        for (int i = 0; i < n_t; ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * src = ggml_get_tensor(meta, name);
            int il = -1;
            char rest[128] = {0};
            if (sscanf(name, "blk.%d.%127s", &il, rest) == 2 && il > m.n_layer) continue;
            if (il == m.n_layer && !e.load_mtp) continue;      // the MTP layer, loaded only for drafting
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
                ggml_tensor * t = ggml_dup_tensor(cw, src);
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
                    {"nextn.hnorm.weight", &L.hnorm}, {"nextn.shared_head_norm.weight", &L.head_norm}};
                auto it = slot.find(r);
                if (it == slot.end()) fprintf(stderr, "momoe-metal: MTP tensor %s not used\n", name);
                else *it->second = t;
                continue;
            }
            if (il >= 0 && (r == "ffn_gate_exps.weight" || r == "ffn_up_exps.weight" || r == "ffn_down_exps.weight")) {
                int w = r == "ffn_gate_exps.weight" ? 0 : r == "ffn_up_exps.weight" ? 1 : 2;
                Layer & L = m.layers[il];
                L.exp_off[w] = off;
                L.exp_bytes[w] = src->nb[2];
                L.exp_type[w] = src->type;
                L.exp_ne[w][0] = src->ne[0];
                L.exp_ne[w][1] = src->ne[1];
                if (L.exp_fd >= 0 && L.exp_fd != fd) die("a layer's experts are split between files (not supported)");
                L.exp_fd = fd;
                continue;
            }
            ggml_tensor * t = ggml_dup_tensor(cw, src);
            ggml_set_name(t, name);
            to_read.push_back({t, fd, off});
            if (il < 0) {
                if (!strcmp(name, "token_embd.weight")) m.tok_embd = t;
                else if (!strcmp(name, "output_norm.weight")) m.out_norm = t;
                else if (!strcmp(name, "output.weight")) m.output = t;
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
                {"indexer.q_norm.weight", &L.idx_q_norm}, {"indexer.k_norm.weight", &L.idx_k_norm}};
            auto it = slot.find(r);
            if (it == slot.end()) fprintf(stderr, "momoe-metal: tensor %s not used\n", name);
            else *it->second = t;
        }
    }
    if (!m.output) m.output = m.tok_embd;
    m.n_vocab = (int) m.tok_embd->ne[1];
    for (auto & L : m.layers)
        if (L.exp_fd < 0) die("a layer has no routed experts");
    if (m.qwen4 && !m.hc_head_norm) die("the hyper-connection head (output_hc_*) is missing");
    if (m.ple_layer >= 0) {
        if (m.ple_fd < 0) die("the PLE table (per_layer_token_embd.weight) is missing: are all the model's files there?");
        if (!ggml_get_type_traits(m.ple_type)->to_float) die("the PLE table's format cannot be read on the CPU");
        if (!m.layers[m.ple_layer].recurrent) die("the PLE layer is not a delta net layer");
    }

    e.metal = ggml_backend_metal_init();
    if (!e.metal) die("no Metal device");
    e.cpu = ggml_backend_cpu_init();
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors(cw, e.metal);
    e.bufs.push_back(bw);
    size_t dense = 0;
    for (auto & r : to_read) {
        read_at(r.fd, r.t->data, ggml_nbytes(r.t), r.off);  // a Metal buffer is host memory on Apple Silicon
        dense += ggml_nbytes(r.t);
    }
    fprintf(stderr, "momoe-metal: dense weights %.2f GiB loaded\n", dense / 1073741824.0);
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
        ggml_context * cd = ggml_init(ip);
        m.dhead = ggml_new_tensor_2d(cd, m.output->type, m.n_embd, n);
        e.bufs.push_back(ggml_backend_alloc_ctx_tensors(cd, e.metal));
        const size_t row = m.output->nb[1];
        for (long i = 0; i < n; ++i)
            memcpy((char *) m.dhead->data + (size_t) i * row, (const char *) m.output->data + (size_t) m.dvocab[i] * row, row);
        fprintf(stderr, "momoe-metal: draft head over %ld tokens (%.0f MiB)\n", n, (double) n * row / 1048576.0);
    }

    // the expert pool: a group per expert format, each the same share of its experts
    Pool & P = e.pool;
    size_t all_bytes = 0;
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        int gi = 0;
        for (; gi < (int) P.g.size(); ++gi) {
            Group & G = P.g[gi];
            bool same = true;
            for (int w = 0; w < 3; ++w)
                same &= G.type[w] == L.exp_type[w] && G.ne[w][0] == L.exp_ne[w][0] && G.ne[w][1] == L.exp_ne[w][1];
            if (same) break;
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
        }
        L.grp = gi;
        ++P.g[gi].n_layers;
        all_bytes += P.g[gi].bytes * m.n_expert;
    }
    const int all = m.n_layer * m.n_expert;
    // the share of the experts that fits: --slots N (of all), --pool-gib G, or all of them
    const double frac = slots_arg > 0 ? (double) slots_arg / all : pool_gib > 0 ? pool_gib * 1073741824.0 / all_bytes : 1.0;
    ggml_context * cp = ggml_init(ip);
    size_t stage_bytes = 0, pool_bytes = 0;
    for (auto & G : P.g) {
        const int n_all = G.n_layers * m.n_expert;
        G.n_slots = std::min(std::max((int) (frac * n_all), 16 * m.n_used), n_all);   // 16 tokens' worth at least
        for (int w = 0; w < 3; ++w)
            G.t[w] = ggml_new_tensor_3d(cp, G.type[w], G.ne[w][0], G.ne[w][1], G.n_slots);
        ggml_set_name(G.t[0], "pool_gate");
        ggml_set_name(G.t[1], "pool_up");
        ggml_set_name(G.t[2], "pool_down");
        G.owner.assign(G.n_slots, -1);
        G.used.assign(G.n_slots, 0);
        G.left.reset(new std::atomic<int>[G.n_slots]);
        for (int i = 0; i < G.n_slots; ++i) G.left[i] = 0;
        G.by_prefetch.assign(G.n_slots, 0);
        P.n_slots += G.n_slots;
        pool_bytes += G.bytes * G.n_slots;
        stage_bytes = std::max(stage_bytes, G.bytes * m.n_expert);
    }
    e.bufs.push_back(ggml_backend_alloc_ctx_tensors(cp, e.metal));
    // one staging buffer for every group (one layer at a time): each group's staging tensors at its start
    {
        ggml_backend_buffer_t sb = ggml_backend_alloc_buffer(e.metal, stage_bytes + 4096 * 3);
        e.bufs.push_back(sb);
        ggml_context * cg = ggml_init(ip);
        for (auto & G : P.g) {
            char * at = (char *) ggml_backend_buffer_get_base(sb);
            for (int w = 0; w < 3; ++w) {
                G.stage[w] = ggml_new_tensor_3d(cg, G.type[w], G.ne[w][0], G.ne[w][1], m.n_expert);
                if (ggml_backend_tensor_alloc(sb, G.stage[w], at) != GGML_STATUS_SUCCESS) die("cannot place the staging copy");
                at += (ggml_nbytes(G.stage[w]) + 4095) / 4096 * 4096;
            }
        }
    }
    P.slot_of.assign(all, -1);
    P.freq.assign(all, 0);
    P.lfu = !(getenv("MOMOE_POLICY") && !strcmp(getenv("MOMOE_POLICY"), "lru"));   // lfu by default (measured faster)
    fprintf(stderr, "momoe-metal: expert pool %d slots = %.2f GiB of %.2f GiB (%d experts in the model, %.0f%%), %zu "
            "format group%s\n", P.n_slots, pool_bytes / 1073741824.0, all_bytes / 1073741824.0, all,
            100.0 * pool_bytes / all_bytes, P.g.size(), P.g.size() == 1 ? "" : "s");

    // state: the KV cache of the attention layers, the delta net's conv and recurrent states, the slot numbers
    ggml_context * cs = ggml_init(ip);
    const int conv_ch = m.ssm_inner + 2 * m.ssm_groups * m.ssm_state;
    const int head_v = m.ssm_inner / m.ssm_dt_rank;
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        if (L.recurrent) {
            L.conv_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1) * conv_ch);
            L.ssm_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) head_v * head_v * m.ssm_dt_rank);
        } else {
            L.k_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, m.head_dim, m.n_head_kv, e.n_ctx);
            L.v_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, m.head_dim, m.n_head_kv, e.n_ctx);
        }
        if (L.idx_k) {
            L.idx_raw = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.idx_dim, e.n_ctx);
            L.idx_pool = ggml_new_tensor_2d(cs, GGML_TYPE_F16, m.idx_dim, e.n_ctx / m.kpool + 64);
        }
        if (il == m.ple_layer)
            L.ple_state = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ple_kernel - 1) * m.ple_ngram * m.hc * m.n_embd);
        L.slot_map = ggml_new_tensor_2d(cs, GGML_TYPE_I32, 1, m.n_expert);
        L.res_mask = ggml_new_tensor_2d(cs, GGML_TYPE_F32, m.n_expert, 1);
        L.res_keep = ggml_new_tensor_2d(cs, GGML_TYPE_F32, 1, m.n_expert);
        if (L.recurrent && e.load_mtp) {                     // a verify window's per-token states (rollback)
            L.ssm_snap = ggml_new_tensor_2d(cs, GGML_TYPE_F32, (int64_t) head_v * head_v * m.ssm_dt_rank, e.max_window);
            L.conv_snap = ggml_new_tensor_1d(cs, GGML_TYPE_F32, (int64_t) (m.ssm_conv - 1 + e.max_window) * conv_ch);
        }
    }
    if (m.has_mtp) {
        m.mtp.k_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, m.head_dim, m.n_head_kv, e.n_ctx);
        m.mtp.v_cache = ggml_new_tensor_3d(cs, GGML_TYPE_F16, m.head_dim, m.n_head_kv, e.n_ctx);
    }
    ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors(cs, e.metal);
    ggml_backend_buffer_clear(bs, 0);
    e.bufs.push_back(bs);

    ggml_backend_t backends[2] = {e.metal, e.cpu};
    e.sched = ggml_backend_sched_new(backends, nullptr, 2, 32768, false, true);
    gguf_free(g);
}

// ------------------------------------------------------------------------------------------------ making experts resident
static bool debug_on() { static int on = getenv("MOMOE_DEBUG") ? 1 : 0; return on; }

// the least used slot of the group (recency breaking ties) that is not in use from `floor` on (this layer's experts, a
// prediction) and not being read
static int take_slot(Pool & P, Group & G, uint64_t floor) {
    int best = -1;
    auto f = [&](int i) { return G.owner[i] >= 0 ? P.freq[G.owner[i]] : 0u; };
    for (int i = 0; i < G.n_slots; ++i) {
        if (G.used[i] >= floor || G.left[i].load(std::memory_order_relaxed) != 0) continue;
        if (G.by_prefetch[i] && G.used[i] + 12 > P.clock) continue;   // a prediction waiting for its layer (a clock
                                                                         // tick per pause and per prediction: ~3 a layer)
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
    G.left[slot].store(3, std::memory_order_relaxed);
    for (int w = 0; w < 3; ++w) {
        char * dst = (char *) G.t[w]->data + (size_t) slot * G.t[w]->nb[2];
        jobs.push_back({L.exp_fd, dst, L.exp_bytes[w], L.exp_off[w] + (size_t) x * L.exp_bytes[w], &G.left[slot]});
        P.bytes_read += L.exp_bytes[w];
    }
    const int key = il * e.m.n_expert + x;
    G.owner[slot] = key;
    P.slot_of[key] = slot;
}

// called at a layer's pause with its chosen experts ([n_used, n_tokens] i32): every one is made resident (a hit, a
// prediction still being read: waited for, a miss: read now, ahead of the predictions), and its slot is written in
// the layer's expert -> slot table, which the expert matmuls look their slots up in (get_rows on the chosen experts:
// a dependency the Metal backend's reordering keeps)
static void make_resident(Engine & e, int il, const int32_t * ids, int n_tok) {
    Model & m = e.m;
    Pool & P = e.pool;
    Layer & L = m.layers[il];
    Group & G = P.g[L.grp];
    const bool stage = staged(m.n_used, G.n_slots, n_tok);
    int32_t * map = (int32_t *) L.slot_map->data;
    std::vector<Readers::Job> jobs;
    std::vector<int> wait;
    std::vector<std::atomic<int>> sleft(1);                  // staging: the parts still being read or copied
    int n_stage = 0;
    const uint64_t pin = ++P.clock;                          // this layer's experts: not taken by each other
    if (++P.uses % ((uint64_t) 64 * m.n_layer) == 0)       // aging: every 64 tokens' worth of layers
        for (auto & c : P.freq) c >>= 1;
    std::vector<uint8_t> seen(stage ? m.n_expert : 0, 0);
    for (size_t i = 0; i < (size_t) m.n_used * n_tok; ++i) {
        const int x = ids[i];
        ++P.freq[il * m.n_expert + x];
        if (stage) {
            if (seen[x]) continue;
            seen[x] = 1;
        }
        int slot = P.slot_of[il * m.n_expert + x];
        if (slot >= 0) {
            ++P.hits;
            if (G.by_prefetch[slot]) { ++P.prefetch_hits; G.by_prefetch[slot] = 0; }
            if (G.left[slot].load(std::memory_order_acquire)) { ++P.late; wait.push_back(slot); }
        } else {
            ++P.misses;
            slot = take_slot(P, G, pin);
            if (slot < 0) {
                if (!stage) die("the expert pool is too small for this batch");
                for (int w = 0; w < 3; ++w) {               // straight into the staging copy
                    jobs.push_back({L.exp_fd, (char *) G.stage[w]->data + (size_t) x * G.stage[w]->nb[2], L.exp_bytes[w],
                                    L.exp_off[w] + (size_t) x * L.exp_bytes[w], &sleft[0]});
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
    }
    sleft[0].store(n_stage, std::memory_order_relaxed);
    const double t0 = now_s();
    e.readers.submit(jobs, true);
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
            for (int w = 0; w < 3; ++w) {
                Readers::Job j{-1, (char *) G.stage[w]->data + (size_t) x * G.stage[w]->nb[2], L.exp_bytes[w], 0, &left[0],
                               (const char *) G.t[w]->data + (size_t) slot * G.t[w]->nb[2]};
                jobs.push_back(j);
                ++n;
            }
        }
        left[0].store(n, std::memory_order_relaxed);
        e.readers.submit(jobs, true);
        while (left[0].load(std::memory_order_acquire)) { }
        while (sleft[0].load(std::memory_order_acquire)) { }
    }
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
    std::vector<Readers::Job> jobs;
    const uint64_t stamp = ++P.clock;                        // newer than the layer the GPU is computing: kept
    for (size_t i = 0; i < (size_t) m.n_used * n_tok; ++i) {
        const int x = ids[i];
        if (x < 0 || x >= m.n_expert) {
            fprintf(stderr, "momoe-metal: layer %d prediction %zu is %d (tokens %d)\n", il, i, x, n_tok);
            die("a predicted expert is out of range");
        }
        if (P.slot_of[il * m.n_expert + x] >= 0) continue;
        const int slot = take_slot(P, G, stamp - 1);        // not the experts of the layer being computed
        if (slot < 0) break;                                 // a small pool: the rest is read when needed
        fill_slot(e, il, x, slot, jobs);
        G.used[slot] = stamp;
        G.by_prefetch[slot] = 1;
        ++P.prefetched;
    }
    e.readers.submit(jobs);
}

static bool eval_cb(ggml_tensor * t, bool ask, void * user) {
    Engine & e = *(Engine *) user;
    int il = -1;
    if (debug_on() && t->name[0] && strncmp(t->name, "topk-", 5) && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)
            && !strstr(t->name, "(")) {                     // MOMOE_DEBUG: the sums llama-eval-callback prints
        if (ask) return true;
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        double sum = 0;
        for (float f : v) sum += f;
        fprintf(stderr, "DBG %-28s sum = %f\n", t->name, sum);
        return true;
    }
    if (sscanf(t->name, "topk-%d", &il) != 1) return ask ? false : true;
    if (ask) return true;
    const double t0 = now_s();
    ++e.pool.calls;
    const size_t n = ggml_nelements(t);
    e.ids_host.resize(n);
    ggml_backend_tensor_get(t, e.ids_host.data(), 0, n * sizeof(int32_t));
    make_resident(e, il, e.ids_host.data(), e.batch_tokens);
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
    if (P.n_slots < m.n_layer * m.n_expert) die("--preload needs a pool that holds every expert");
    const double t0 = now_s();
    std::vector<int> next(P.g.size(), 0);
    std::vector<Readers::Job> jobs;
    for (int il = 0; il < m.n_layer; ++il) {
        Layer & L = m.layers[il];
        for (int x = 0; x < m.n_expert; ++x) {
            const int slot = next[L.grp]++;
            fill_slot(e, il, x, slot, jobs);
            ((int32_t *) L.slot_map->data)[x] = slot;
        }
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
    // QSA (llama.cpp's llm_graph_input_kpool, one sequence: a position's cell is its position): the blocks completed
    // by this batch (j_first.., n_new) are pooled; with more than top_k / kpool blocks the selection runs (sel)
    ggml_tensor *new_pos = nullptr;                          // I32 [4 * n_new]: rope position of each new block
    ggml_tensor *pool_idxs = nullptr;                        // F32 [kpool, n_pool]: member cells (n_kv: padding)
    ggml_tensor *pool_mask = nullptr;                        // F16 [n_pool, T]: blocks finished at or before a token
    ggml_tensor *tail_idxs = nullptr;                        // F32 [kpool - 1, T]: its unfinished block (n_kv: none)
                                                             // (cell numbers as f32: Metal casts f32 -> i32, not back)
    int j_first = 0, n_new = 0, n_pool = 0;
    bool sel = false;
};

static ggml_tensor * norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
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
    ggml_build_forward_expand(gf, ggml_cpy(c, k_raw, ggml_view_2d(c, L.idx_raw, D, T, rr, rr * p0)));
    if (in.n_new > 0) {
        ggml_tensor * rows = ggml_view_3d(c, L.idx_raw, D, kp, in.n_new, rr, rr * kp, rr * kp * in.j_first);
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < kp; ++i) {
            ggml_tensor * sl = ggml_view_2d(c, rows, D, in.n_new, rows->nb[2], i * rows->nb[1]);
            sl = ggml_cast(c, sl, GGML_TYPE_F32);
            pooled = pooled ? ggml_add(c, pooled, sl) : sl;
        }
        pooled = norm(c, ggml_scale(c, pooled, 1.0f / kp), L.idx_k_norm, m.rms_eps);
        pooled = ggml_reshape_3d(c, pooled, D, 1, in.n_new);
        pooled = ggml_rope_multi(c, pooled, in.new_pos, nullptr, m.n_rot, sections, GGML_ROPE_TYPE_IMROPE, 262144,
                                 m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        const size_t pr = L.idx_pool->nb[1];
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
    ggml_build_forward_expand(gf, ggml_cpy(c, k, ggml_view_3d(c, L.k_cache, hd, m.n_head_kv, T,
                                                               L.k_cache->nb[1], row, row * p0)));
    ggml_build_forward_expand(gf, ggml_cpy(c, v, ggml_view_3d(c, L.v_cache, hd, m.n_head_kv, T,
                                                               L.v_cache->nb[1], row, row * p0)));
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
    ggml_build_forward_expand(gf, ggml_cpy(c, ns, s));
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

static ggml_tensor * moe_layer(Engine & e, ggml_context * c, ggml_cgraph * gf, int il, ggml_tensor * x, int T,
                               ggml_tensor * pred_in, ggml_tensor * pred_in2) {
    Model & m = e.m;
    Layer & L = m.layers[il];
    ggml_tensor * lg = ggml_mul_mat(c, L.gate_inp, x);
    ggml_format_name(lg, "ffn_moe_logits-%d", il);
    ggml_tensor * probs = ggml_soft_max(c, lg);                                         // [n_expert, T]
    // the next layer's router on this layer's hidden state predicts its experts: computed in the same top-k, so the
    // pause has both (the prediction is read while the GPU goes on)
    // (and the one after it, on the same state: less accurate, but twice the time to read)
    ggml_tensor * both = probs;
    if (pred_in) {
        ggml_tensor * p2 = ggml_soft_max(c, ggml_mul_mat(c, m.layers[il + 1].gate_inp, pred_in));
        both = ggml_concat(c, probs, p2, 1);                                            // [n_expert, 2T]
    }
    if (pred_in2) {
        ggml_tensor * p3 = ggml_soft_max(c, ggml_mul_mat(c, m.layers[il + 2].gate_inp, pred_in2));
        both = ggml_concat(c, both, p3, 1);                                             // [n_expert, 3T]
    }
    ggml_tensor * top = ggml_cont(c, ggml_argsort_top_k(c, both, m.n_used));          // [n_used, T or 2T]
    // made resident here: a pause point (or the callback); no-stall: what the layer wanted, read after the graph
    const bool masked = !ns_paused(e, il);
    ggml_format_name(top, masked ? "want-%d" : "topk-%d", il);
    if (e.ns_graph) ggml_set_output(top);
    e.topk[il] = top;
    ggml_build_forward_expand(gf, top);
    ggml_tensor * ids = pred_in ? ggml_view_2d(c, top, m.n_used, T, top->nb[1], 0) : top;   // [n_used, T]
    if (masked && e.nostall == 1)                            // sub: the best of the experts in the pool
        ids = ggml_cont(c, ggml_argsort_top_k(c, ggml_add(c, probs, L.res_mask), m.n_used));
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), ids);   // [1, n_used, T]
    if (masked && e.nostall == 2)                            // drop: the missing ones weigh nothing
        w = ggml_mul(c, w, ggml_reshape_3d(c, ggml_get_rows(c, L.res_keep, ggml_reshape_1d(c, ids, (int64_t) m.n_used * T)),
                                           1, m.n_used, T));
    w = ggml_reshape_2d(c, w, m.n_used, T);
    w = ggml_div(c, w, ggml_clamp(c, ggml_sum_rows(c, w), 6.103515625e-5f, INFINITY));
    w = ggml_reshape_3d(c, w, 1, m.n_used, T);
    ggml_tensor * slots = ggml_get_rows(c, L.slot_map, ggml_reshape_1d(c, ids, (int64_t) m.n_used * T));
    slots = ggml_reshape_2d(c, slots, m.n_used, T);                                     // [n_used, T]
    ggml_tensor * x3 = ggml_reshape_3d(c, x, m.n_embd, 1, T);
    Group & G = e.pool.g[L.grp];
    const bool big = staged(m.n_used, G.n_slots, T);         // the staging copy, by expert number (see STAGE_MIN_TOKENS)
    ggml_tensor ** as = big ? G.stage : G.t;
    ggml_tensor * layer_t[3];
    if (big && e.preloaded) {                                // no pauses fill the staging copy: the layer's experts are
        const int64_t s0 = ((const int32_t *) L.slot_map->data)[0];   // in a row in the pool (preload), a view of them
        for (int k = 0; k < 3; ++k)
            layer_t[k] = ggml_view_3d(c, G.t[k], G.t[k]->ne[0], G.t[k]->ne[1], m.n_expert, G.t[k]->nb[1], G.t[k]->nb[2],
                                      (size_t) s0 * G.t[k]->nb[2]);
        as = layer_t;
    }
    if (big) slots = ids;
    ggml_tensor * up = ggml_mul_mat_id(c, as[1], x3, slots);                            // [n_ff, n_used, T]
    ggml_tensor * gt = ggml_mul_mat_id(c, as[0], x3, slots);
    ggml_tensor * h = ggml_swiglu_split(c, gt, up);
    ggml_tensor * dn = ggml_mul_mat_id(c, as[2], h, slots);                             // [n_embd, n_used, T]
    ggml_format_name(up, "moe_up-%d", il);
    ggml_format_name(dn, "moe_down-%d", il);
    ggml_format_name(w, "moe_w-%d", il);
    dn = ggml_mul(c, dn, w);
    ggml_tensor * moe = nullptr;
    for (int i = 0; i < m.n_used; ++i) {
        ggml_tensor * v = ggml_view_2d(c, dn, m.n_embd, T, dn->nb[2], i * dn->nb[1]);
        moe = moe ? ggml_add(c, moe, v) : v;
    }
    // the shared expert, gated by a sigmoid
    ggml_tensor * sh = ggml_swiglu_split(c, ggml_mul_mat(c, L.gate_sh, x), ggml_mul_mat(c, L.up_sh, x));
    sh = ggml_mul_mat(c, L.down_sh, sh);
    sh = ggml_mul(c, sh, ggml_sigmoid(c, ggml_mul_mat(c, L.gate_inp_sh, x)));
    ggml_format_name(sh, "ffn_shexp_gated-%d", il);
    ggml_format_name(moe, "ffn_moe_out-%d", il);
    return ggml_add(c, moe, sh);
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

static int forward(Engine & e, const int32_t * toks, int T, int p0, Out * out = nullptr) {
    Model & m = e.m;
    const double tb0 = now_s();
    const int n_kv = std::min(e.n_ctx, ((p0 + T + 255) / 256) * 256);
    ggml_init_params ip = {ggml_tensor_overhead() * 40000 + ggml_graph_overhead_custom(32768, false), nullptr, true};
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 32768, false);
    Inputs in;
    e.topk.assign(m.n_layer, nullptr);
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
    ggml_tensor * x = ggml_get_rows(c, m.tok_embd, in.tokens);
    ggml_set_name(x, "model.input_embed");
    if (m.kpool) {                                           // QSA's inputs (see Inputs)
        const int kp = m.kpool, total = p0 + T;
        in.j_first = p0 / kp;
        in.n_new = std::max(0, total / kp - in.j_first);
        in.n_pool = std::max(64, (total / kp + 1 + 63) / 64 * 64);
        static const bool no_qsa = getenv("MOMOE_NO_QSA") != nullptr;   // a test: attention to every position
        in.sel = total / kp > m.idx_top_k / kp && !no_qsa;   // else every finished block is kept: dense attention
        if (in.n_new > 0) {
            in.new_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * (int64_t) in.n_new);
            ggml_set_input(in.new_pos);
        }
        if (in.sel) {
            in.pool_idxs = ggml_new_tensor_2d(c, GGML_TYPE_F32, kp, in.n_pool);
            in.pool_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, in.n_pool, T);
            in.tail_idxs = ggml_new_tensor_2d(c, GGML_TYPE_F32, kp - 1, T);
            ggml_set_input(in.pool_idxs);
            ggml_set_input(in.pool_mask);
            ggml_set_input(in.tail_idxs);
        }
    }
    ggml_tensor * ple_in = nullptr;
    if (m.ple_layer >= 0) {
        if (out && out->snap) die("verify-window snapshots are not supported with qwen4exp");
        ple_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t) m.ple_dim * m.ple_heads, T);
        ggml_set_input(ple_in);
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
    for (int il = 0; il < m.n_layer && !m.qwen4; ++il) {
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
        hid = m.qwen4 ? ggml_cont_2d(c, x, m.h_dim(), T) : norm(c, x, m.out_norm, m.rms_eps);
        ggml_set_output(hid);
        ggml_build_forward_expand(gf, hid);
    }
    if (m.qwen4) {                                           // the final mixer is the output norm
        if (!(out && out->all)) x = ggml_view_3d(c, x, m.n_embd, m.hc, 1, x->nb[1], x->nb[2], (size_t) (T - 1) * x->nb[2]);
        x = hc_mix(e, c, x, m.hc_head_norm, m.hc_head_down, m.hc_head_up, nullptr, nullptr);
    } else {
        if (!(out && out->all)) x = ggml_view_2d(c, x, m.n_embd, 1, x->nb[1], (size_t) (T - 1) * x->nb[1]);   // the last
        x = hid && out->all ? hid : norm(c, x, m.out_norm, m.rms_eps);
    }
    ggml_set_name(x, "result_norm");
    ggml_tensor * logits = ggml_mul_mat(c, m.output, x);
    ggml_set_name(logits, "result_output");
    static const bool show_top = getenv("MOMOE_TOP") != nullptr;   // the last position's 5 most likely tokens
    if ((out && out->logits) || show_top) ggml_set_output(logits);
    ggml_tensor * best = ggml_argmax(c, logits);
    ggml_set_output(best);
    ggml_build_forward_expand(gf, best);

    ggml_backend_sched_reset(e.sched);
    if (!ggml_backend_sched_alloc_graph(e.sched, gf)) die("cannot allocate the graph");
    ggml_backend_tensor_set(in.tokens, toks, 0, (size_t) T * sizeof(int32_t));
    std::vector<int32_t> pos((size_t) T * 4);
    for (int i = 0; i < T; ++i) {                            // text: [p, p, p, 0] per token, section-major
        for (int j = 0; j < 4; ++j) pos[(size_t) j * T + i] = j < 3 ? p0 + i : 0;
    }
    ggml_backend_tensor_set(in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < n_kv; ++j) mask[(size_t) i * n_kv + j] = j <= p0 + i ? zero : ninf;
    ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (in.new_pos) {                                        // a block's keys rotate to its first position
        std::vector<int32_t> np(4 * (size_t) in.n_new);
        for (int j = 0; j < in.n_new; ++j)
            for (int s4 = 0; s4 < 4; ++s4) np[(size_t) s4 * in.n_new + j] = (in.j_first + j) * m.kpool;
        ggml_backend_tensor_set(in.new_pos, np.data(), 0, np.size() * sizeof(int32_t));
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
        ggml_backend_tensor_set(in.tail_idxs, ti.data(), 0, ti.size() * sizeof(float));
    }
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
    if (!e.pause || (e.ns_graph && e.ns_every == 0)) {
        if (ggml_backend_sched_graph_compute(e.sched, gf) != GGML_STATUS_SUCCESS) die("the graph failed");
        if (e.ns_graph) ns_after(e, T);
    } else {
        // one submission: the GPU pauses after each layer's router (signal 2k-1, wait for 2k); this thread makes
        // the chosen experts resident and lets it go on
        if (ggml_backend_sched_get_n_splits(e.sched) != 1) die("an op of the graph runs on the CPU: the pauses need one Metal graph");
        const uint64_t base = ggml_backend_metal_pause_counter(e.metal);
        if (ggml_backend_sched_graph_compute_async(e.sched, gf) != GGML_STATUS_SUCCESS) die("the graph failed");
        const uint64_t n_pause = (ggml_backend_metal_pause_counter(e.metal) - base) / 2;
        uint64_t want_pause = 0;
        for (int il = 0; il < m.n_layer; ++il) want_pause += ns_paused(e, il);
        if (n_pause != want_pause) die("the graph has a pause point per layer missing");
        uint64_t at = base + 1;
        int done = 0;                                        // --nostall-every: the layers seen to so far
        for (int il = 0; il < m.n_layer; ++il) {
            if (!ns_paused(e, il)) continue;
            const double w0 = now_s();
            while (ggml_backend_metal_pause_signaled(e.metal) < at) { }
            e.t_wait += now_s() - w0;
            const double t0 = now_s();
            ++e.pool.calls;
            const int32_t * top = (const int32_t *) e.topk[il]->data;
            make_resident(e, il, top, T);
            // the predictions are copied before the GPU goes on: after it, the allocator reuses this tensor's memory
            const size_t n_pred = (size_t) m.n_used * T * ((e.predict && il + 1 < m.n_layer) + (e.predict2 && il + 2 < m.n_layer));
            e.pred_host.assign(top + (size_t) m.n_used * T, top + (size_t) m.n_used * T + n_pred);
            if (e.ns_graph) {
                // the layers since the last pause (computed by now): their wanted experts read; the predictions; then
                // what has been read is published for the layers after this one. Every slot is taken before the
                // publishing: nothing a published layer reads changes while the GPU goes on
                for (; done < il; ++done)
                    if (!ns_paused(e, done)) ns_layer(e, done, T);
                done = il + 1;
                if (e.predict && il + 1 < m.n_layer) prefetch(e, il + 1, e.pred_host.data(), T);
                if (e.predict2 && il + 2 < m.n_layer) prefetch(e, il + 2, e.pred_host.data() + (size_t) m.n_used * T, T);
                ns_publish(e, il + 1);
            }
            e.pool.cb_s += now_s() - t0;
            ggml_backend_metal_pause_release(e.metal, at + 1);
            if (!e.ns_graph) {
                if (e.predict && il + 1 < m.n_layer) prefetch(e, il + 1, e.pred_host.data(), T);
                if (e.predict2 && il + 2 < m.n_layer) prefetch(e, il + 2, e.pred_host.data() + (size_t) m.n_used * T, T);
            }
            at += 2;
        }
        ggml_backend_sched_synchronize(e.sched);
        if (e.ns_graph) {                                    // the layers after the last pause
            ++e.ns_tokens;
            for (; done < m.n_layer; ++done)
                if (!ns_paused(e, done)) ns_layer(e, done, T);
        }
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
    ggml_free(c);
    return nb.back();
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
    }
    return r;
}

static void save_state(Engine & e, Saved & to) {
    const auto ts = state_tensors(e);
    to.resize(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) to[i].assign((const char *) ts[i]->data, ggml_nbytes(ts[i]));
}

static void restore_state(Engine & e, const Saved & from) {
    const auto ts = state_tensors(e);
    for (size_t i = 0; i < ts.size(); ++i) memcpy(ts[i]->data, from[i].data(), from[i].size());
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
        memcpy(L.ssm_state->data, (char *) L.ssm_snap->data + (size_t) (T - n_keep) * D, D);   // slot s: s tokens back
        const float * w = (const float *) L.conv_snap->data;                   // [cw + T, conv_ch]
        float * st = (float *) L.conv_state->data;                              // [cw, conv_ch]
        for (int ch = 0; ch < conv_ch; ++ch)
            for (int t = 0; t < cw; ++t) st[(size_t) ch * cw + t] = w[(size_t) ch * (cw + T) + n_keep + t];
    }
}

// the MTP layer over T positions: hidden states h (the main model's, or its own when chaining) and the tokens that
// follow them; returns the greedy token after the last, and its own hidden state for the next draft
static int mtp_forward(Engine & e, const float * h, const int32_t * toks, int T, int p0, std::vector<float> & h_out,
                       float * prob = nullptr) {
    Model & m = e.m;
    const double tm0 = now_s();
    Layer & L = m.mtp;
    const int n_kv = std::min(e.n_ctx, ((p0 + T + 255) / 256) * 256);
    ggml_init_params ip = {ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false), nullptr, true};
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 2048, false);
    Inputs in;
    in.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_input(in.tokens);
    ggml_tensor * hin = ggml_new_tensor_2d(c, GGML_TYPE_F32, m.h_dim(), T);
    ggml_set_input(hin);
    in.pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) T * 4);
    ggml_set_input(in.pos);
    in.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, T);
    ggml_set_input(in.mask);
    // its MoE: the experts whole (no pool), and the shared expert
    auto moe_of = [&](ggml_tensor * pn) {
        ggml_tensor * probs = ggml_soft_max(c, ggml_mul_mat(c, L.gate_inp, pn));
        ggml_tensor * ids = ggml_cont(c, ggml_argsort_top_k(c, probs, m.n_used));
        ggml_set_name(ids, "mtp-ids");                       // not a pause point ("topk-" only)
        ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, m.n_expert, T), ids);
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
    ggml_tensor * en = norm(c, ggml_get_rows(c, m.tok_embd, in.tokens), L.enorm, m.rms_eps);
    ggml_tensor * hl, * head_in;                             // the next draft's input; the head's
    if (m.qwen4) {
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
    ggml_tensor * best = ggml_argmax(c, probs_v);
    ggml_set_output(best);
    ggml_tensor * bp = ggml_get_rows(c, ggml_reshape_2d(c, probs_v, 1, probs_v->ne[0]), best);   // its probability
    ggml_set_output(bp);
    ggml_build_forward_expand(gf, best);
    ggml_build_forward_expand(gf, bp);
    ggml_build_forward_expand(gf, hl);
    ggml_backend_sched_reset(e.sched);
    if (!ggml_backend_sched_alloc_graph(e.sched, gf)) die("cannot allocate the MTP graph");
    ggml_backend_tensor_set(in.tokens, toks, 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(hin, h, 0, (size_t) T * m.h_dim() * sizeof(float));
    std::vector<int32_t> pos((size_t) T * 4);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < 4; ++j) pos[(size_t) j * T + i] = j < 3 ? p0 + i : 0;
    ggml_backend_tensor_set(in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < n_kv; ++j) mask[(size_t) i * n_kv + j] = j <= p0 + i ? zero : ninf;
    ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(e.sched, gf) != GGML_STATUS_SUCCESS) die("the MTP graph failed");
    int32_t r = 0;
    ggml_backend_tensor_get(best, &r, 0, sizeof(r));
    if (m.dhead) r = m.dvocab[r];
    if (prob) ggml_backend_tensor_get(bp, prob, 0, sizeof(float));
    e.t_mtp += now_s() - tm0;
    h_out.resize(m.h_dim());
    ggml_backend_tensor_get(hl, h_out.data(), 0, h_out.size() * sizeof(float));
    ggml_free(c);
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
    for (ggml_tensor * t : state_tensors(e)) memset(t->data, 0, ggml_nbytes(t));
}

// the caches kept per position: the attention keys and values (the MTP layer's too) and QSA's indexer keys, each
// with the dimension that counts positions and how many positions one of its rows holds (QSA's pooled keys: a block)
struct PosTensor { ggml_tensor * t; int dim, per_row; };

static std::vector<PosTensor> position_tensors(Engine & e) {
    std::vector<PosTensor> r;
    auto add = [&](ggml_tensor * t, int dim, int per_row) { if (t) r.push_back({t, dim, per_row}); };
    for (auto & L : e.m.layers) {
        add(L.k_cache, 2, 1);
        add(L.v_cache, 2, 1);
        add(L.idx_raw, 1, 1);
        add(L.idx_pool, 1, e.m.kpool);
    }
    add(e.m.mtp.k_cache, 2, 1);
    add(e.m.mtp.v_cache, 2, 1);
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
        to[i].assign((const char *) ts[i].t->data + off, n);
    }
}

static void restore_positions(Engine & e, int a, const std::vector<std::string> & from) {
    const auto ts = position_tensors(e);
    for (size_t i = 0; i < ts.size(); ++i)
        memcpy((char *) ts[i].t->data + position_rows(ts[i], a, a + 1).first, from[i].data(), from[i].size());
}

static size_t position_bytes(Engine & e, int a, int b) {
    size_t n = 0;
    for (const auto & pt : position_tensors(e)) n += position_rows(pt, a, b).second;
    return n;
}

static int serve_loop(Engine & e, int mtp_k, float spec_min_p) {
    Model & m = e.m;
    const int ne = m.h_dim();                               // the MTP layer's hidden input per position
    const std::vector<int32_t> eos = {248044, 248046};
    Serve sv;
    std::thread reader([&] {                                // stdin on its own thread: a STOP reaches a running request
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
    const int32_t im_start = 248045;                        // <|im_start|> (Qwen3.5/3.6, Ornith, Flash-Next)
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
        if (line.rfind("GEN ", 0) != 0) { out_line("ERR unknown command"); continue; }
        sv.stop = false;
        sv.busy = true;
        // GEN <max_new>[ key=value...] <ids>
        std::vector<std::string> w;
        for (size_t a = 4, b; a < line.size(); a = b + 1) {
            b = line.find(' ', a);
            if (b == std::string::npos) b = line.size();
            if (b > a) w.push_back(line.substr(a, b - a));
        }
        Sampling sp;
        std::vector<int32_t> prompt;
        int max_new = w.empty() ? 0 : atoi(w[0].c_str());
        for (size_t i = 1; i < w.size(); ++i) {
            const size_t eq = w[i].find('=');
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
                    parked.pos[i].insert(0, (const char *) ts[i].t->data + r0 * rb, (r1 - r0) * rb);
                }
                parked.from = reused;
            } else {
                parked = Parked();
            }
        }
        out_line("RESUME %d", reused);
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
            if (i < ck_pos && end > ck_pos) end = ck_pos;
            const int T = (int) (end - i);
            Out o;
            o.logits = !greedy && end == prompt.size();
            next = forward(e, prompt.data() + i, T, p, &o);
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
                if (reused > 0 && (int) h_before.size() == ne) {
                    hh = h_before;                           // (state before the new tokens, the first new token)
                    tt.push_back(prompt[reused]);
                    q0 = reused;
                }
                for (size_t i = 0; i < n_new; ++i) {
                    hh.insert(hh.end(), H.begin() + i * ne, H.begin() + (i + 1) * ne);
                    tt.push_back(i + 1 < n_new ? prompt[reused + i + 1] : next);
                }
                float pb = 0;
                int d = 0;
                for (size_t i = 0; i < tt.size(); i += e.max_batch) {
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
                std::vector<int32_t> drafts = chain(d, p);
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
                    last_h.assign(ov.hidden.begin() + (size_t) n * ne, ov.hidden.begin() + (size_t) (n + 1) * ne);
                    for (int j = 0; j <= n && go; ++j) go = emit(nx[j]);
                    if (!go) break;
                    const int dn = mtp_forward(e, ov.hidden.data(), nx.data(), n + 1, p + 1, hc, &pb);
                    drafts = chain(dn, p + n + 1);
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
    const char * model = nullptr;
    std::vector<int32_t> prompt;
    int n_gen = 32, slots = 0, mtp_k = 0;
    float spec_min_p = 0.5f;                                 // a draft only while the MTP layer is this sure
    bool pre = false, nocache = false, bench_window = false, serve = false;
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
        else if (a == "--batch") e.max_batch = atoi(next());
        else if (a == "--preload") pre = true;
        else if (a == "--nocache") nocache = true;
        else if (a == "--bench-window") bench_window = true;
        else if (a == "--mtp") mtp_k = atoi(next());
        else if (a == "--serve") serve = true;
        else if (a == "--draft-vocab") e.draft_vocab = next();
        else if (a == "--mtp-gguf") e.mtp_gguf = next();
        else if (a == "--nostall") nostall_arg = next();
        else if (a == "--nostall-every") nostall_every = std::max(0, atoi(next()));
        else if (a == "--spec-min-p") spec_min_p = (float) atof(next());
        else if (a == "--profile") profile = next();
        else if (a == "--tokens") {
            std::string s = next();
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                prompt.push_back(atoi(s.substr(p, q - p).c_str()));
                p = q == std::string::npos ? s.size() : q + 1;
            }
        } else die("unknown option %s", a.c_str());
    }
    if (!model || (prompt.empty() && !serve)) die("usage: momoe-metal -m MODEL.gguf (--tokens 1,2,3 [-n N] | --serve) [--slots N | --pool-gib G]");
    e.load_mtp = mtp_k > 0;
    e.max_window = std::max(2, mtp_k + 1);
    load(e, model, pool_gib, slots);
    if (mtp_k > 0 && !e.m.has_mtp) die("--mtp: this model file has no MTP layer");
    e.readers.start(getenv("MOMOE_READERS") ? atoi(getenv("MOMOE_READERS")) : 8);
    e.predict = !getenv("MOMOE_NO_PREDICT");
    e.predict2 = e.predict && getenv("MOMOE_PREDICT2");      // two layers ahead: as fast, a third more read (M4)
    if (nocache)                                             // expert reads from the SSD, as on a Mac without the RAM
        for (int fd : e.fds)
            if (fd != e.m.ple_fd) fcntl(fd, F_NOCACHE, 1);
    if (e.pool.n_slots == e.m.n_layer * e.m.n_expert && !getenv("MOMOE_NO_PRELOAD")) pre = true;   // all fit: no pauses
    if (pre) preload(e);
    else if (profile) pin_profile(e, profile, getenv("MOMOE_PIN_FRAC") ? atof(getenv("MOMOE_PIN_FRAC")) : 0.6);
    e.pause = !debug_on() && !getenv("MOMOE_NO_PAUSE") && (!pre || getenv("MOMOE_FORCE_PAUSE"));
    if (e.pause) ggml_backend_metal_set_pause(e.metal, "topk-");
    else if (!pre || debug_on() || getenv("MOMOE_FORCE_CB")) ggml_backend_sched_set_eval_callback(e.sched, eval_cb, &e);
    if (const char * ns = nostall_arg) {
        if (strcmp(ns, "sub") && strcmp(ns, "drop")) die("--nostall takes sub or drop, not %s", ns);
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
        e.readers.stop();
        ggml_backend_sched_free(e.sched);
        for (auto b : e.bufs) ggml_backend_buffer_free(b);
        ggml_backend_free(e.metal);
        ggml_backend_free(e.cpu);
        fflush(stdout);
        fflush(stderr);
        _exit(r);                                             // the stdin reader may still be blocked in a read
    }
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
        return 0;
    }
    const uint64_t pm = e.pool.misses, ph = e.pool.hits;
    const uint64_t pb = e.pool.bytes_read;
    const double pr = e.pool.read_s;
    std::vector<int32_t> out;
    t0 = now_s();
    double t_half = 0, cb_half = 0, b_half = 0, c_half = 0, w_half = 0;
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
        // the MTP layer over the prompt: (state i, token i+1), the last with the first generated token
        std::vector<int32_t> mt(prompt.begin() + 1, prompt.end());
        mt.push_back(next);
        int d = 0;
        for (size_t i = 0; i < mt.size(); i += e.max_batch) {
            const int T = (int) std::min(mt.size() - i, (size_t) e.max_batch);
            d = mtp_forward(e, H.data() + i * ne, mt.data() + i, T, 1 + (int) i, hc, &pb);
        }
        std::vector<int32_t> drafts = chain(d, p);
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
            const int dn = mtp_forward(e, ov.hidden.data(), ov.next.data(), n + 1, p + 1, hc, &pb);
            drafts = chain(dn, p + n + 1);              // dn follows the MTP's last position, p+n+1
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
        fprintf(stderr, "momoe-metal: MTP up to %d drafts (min p %.2f): %llu windows, %llu drafts, %llu accepted, %.2f tokens "
                "per window; MTP layer %.1f ms per window\n", mtp_k, spec_min_p, (unsigned long long) windows,
                (unsigned long long) drafted, (unsigned long long) accepted, (double) (accepted + windows) / windows,
                1000 * e.t_mtp / windows);
        fprintf(stderr, "momoe-metal: generated %d tokens in %.2f s (%.1f tok/s): expert hit rate %.1f%%\n", n_gen, tg,
                (n_gen - 1) / tg, 100.0 * e.pool.hits / std::max<uint64_t>(1, e.pool.hits + e.pool.misses));
        e.readers.stop();
        ggml_backend_sched_free(e.sched);
        for (auto b : e.bufs) ggml_backend_buffer_free(b);
        ggml_backend_free(e.metal);
        ggml_backend_free(e.cpu);
        return 0;
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
        if (i == n_gen / 2) { t_half = now_s(); cb_half = e.pool.cb_s; b_half = e.t_build; c_half = e.t_compute; w_half = e.t_wait; }
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
            "%d graph splits\n", 1000 * (e.t_build - b_half) / (n_gen - n_gen / 2),
            1000 * (e.t_compute - c_half) / (n_gen - n_gen / 2), 1000 * (e.t_wait - w_half) / (n_gen - n_gen / 2),
            ggml_backend_sched_get_n_splits(e.sched));
    printf("tokens:");
    for (auto t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr, "momoe-metal: prompt %zu tokens in %.2f s (%.1f tok/s): %llu expert misses, %.0f MiB read in %.2f s\n",
            prompt.size(), t_prompt, prompt.size() / t_prompt, (unsigned long long) pm, pb / 1048576.0, pr);
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
    e.readers.stop();
    ggml_backend_sched_free(e.sched);
    for (auto b : e.bufs) ggml_backend_buffer_free(b);
    ggml_backend_free(e.metal);
    ggml_backend_free(e.cpu);
    return 0;
}
