"""patch_ggml_pause.py GGML_DIR - Momoe's pause points for ggml's Metal backend.

After a node whose name starts with a set prefix (Momoe: "topk-", a layer's chosen experts), the command buffer
signals a shared event (2k-1 for the k-th pause) and waits for the CPU to signal 2k, all inside one command buffer:
the CPU reads the node's result from shared memory, makes the experts resident and lets the GPU go on, without
ending the command buffer (which cost ~0.5 ms per stop on an M4).  Pauses force single-threaded encoding (n_cb 0),
so the k-th pause is the k-th such node in graph order.

API (ggml-metal.h):
    ggml_backend_metal_set_pause(backend, "topk-")     enable (NULL: off)
    ggml_backend_metal_pause_counter(backend)          the value the last encoded pause waits for
    ggml_backend_metal_pause_signaled(backend)         the event's current value
    ggml_backend_metal_pause_release(backend, value)   signal from the CPU
"""
import sys
from pathlib import Path

G = Path(sys.argv[1]) / "src" / "ggml-metal"
INC = Path(sys.argv[1]) / "include" / "ggml-metal.h"


def edit(path, pairs):
    s = path.read_text()
    if "MOMOE_PAUSE" in s:
        print(f"{path.name}: already patched")
        return
    for old, new in pairs:
        assert s.count(old) == 1, (path.name, s.count(old), old[:80])
        s = s.replace(old, new)
    path.write_text(s)
    print(f"{path.name}: patched")


edit(G / "ggml-metal-ops.h", [(
    "int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx);",
    """int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx);

// MOMOE_PAUSE: the node at idx (for the pause-prefix check), and a pause: the compute encoder is ended,
// `between` encodes into the command buffer (the event signal and wait), and a new encoder is started
struct ggml_tensor * ggml_metal_op_node_at(ggml_metal_op_t ctx, int idx);
void ggml_metal_op_pause(ggml_metal_op_t ctx, void (*between)(ggml_metal_cmd_buf_t cmd_buf, void * user), void * user);""")])

edit(G / "ggml-metal-ops.cpp", [
    ("""        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);""",
     """        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->cmd_buf         = cmd_buf;   // MOMOE_PAUSE"""),
    ("""    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;""",
     """    ggml_metal_encoder_t enc;
    ggml_metal_cmd_buf_t cmd_buf;     // MOMOE_PAUSE: a pause ends the encoder and starts another on it
    ggml_mem_ranges_t    mem_ranges;"""),
    ("""int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}""",
     """int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

// MOMOE_PAUSE
struct ggml_tensor * ggml_metal_op_node_at(ggml_metal_op_t ctx, int idx) {
    return ctx->node(idx);
}

void ggml_metal_op_pause(ggml_metal_op_t ctx, void (*between)(ggml_metal_cmd_buf_t cmd_buf, void * user), void * user) {
    ggml_metal_encoder_end_encoding(ctx->enc);
    ggml_metal_encoder_free(ctx->enc);
    between(ctx->cmd_buf, user);
    ctx->enc = ggml_metal_encoder_init(ctx->cmd_buf, ctx->use_concurrency);
    if (ctx->mem_ranges) {
        ggml_mem_ranges_reset(ctx->mem_ranges);
    }
}"""),
])

edit(G / "ggml-metal-context.h", [(
    "void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);",
    """void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);

// MOMOE_PAUSE
void     ggml_metal_set_pause     (ggml_metal_t ctx, const char * prefix);
uint64_t ggml_metal_pause_counter (ggml_metal_t ctx);
uint64_t ggml_metal_pause_signaled(ggml_metal_t ctx);
void     ggml_metal_pause_release (ggml_metal_t ctx, uint64_t value);""")])

edit(G / "ggml-metal-context.m", [
    ("""    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers""",
     """    // MOMOE_PAUSE: pause points (see ggml_metal_set_pause)
    char pause_prefix[32];
    id<MTLSharedEvent> pause_event;
    uint64_t pause_value;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers"""),
    ("""        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }""",
     """        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            // MOMOE_PAUSE: after a pause node, the GPU signals and waits for the CPU, in this command buffer
            if (ctx->pause_prefix[0]) {
                for (int j = idx; j < idx + res; ++j) {
                    const struct ggml_tensor * t = ggml_metal_op_node_at(ctx_op, j);
                    if (strncmp(t->name, ctx->pause_prefix, strlen(ctx->pause_prefix)) == 0) {
                        ggml_metal_op_pause(ctx_op, ggml_metal_encode_pause, ctx);
                        break;
                    }
                }
            }

            idx += res - 1;
        }"""),
    ("""enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {""",
     """// MOMOE_PAUSE
static void ggml_metal_encode_pause(ggml_metal_cmd_buf_t cmd_buf_raw, void * user) {
    ggml_metal_t ctx = (ggml_metal_t) user;
    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;
    ctx->pause_value += 2;
    [cmd_buf encodeSignalEvent:ctx->pause_event value:ctx->pause_value - 1];
    [cmd_buf encodeWaitForEvent:ctx->pause_event value:ctx->pause_value];
}

void ggml_metal_set_pause(ggml_metal_t ctx, const char * prefix) {
    if (!prefix || !prefix[0]) {
        ctx->pause_prefix[0] = 0;
        return;
    }
    if (!ctx->pause_event) {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        ctx->pause_event = [device newSharedEvent];
        ctx->pause_value = 0;
        ctx->pause_event.signaledValue = 0;
    }
    snprintf(ctx->pause_prefix, sizeof(ctx->pause_prefix), "%s", prefix);
    ggml_metal_set_n_cb(ctx, 0);       // one encoding thread: the k-th pause is the k-th node in graph order
}

uint64_t ggml_metal_pause_counter(ggml_metal_t ctx) {
    return ctx->pause_value;
}

uint64_t ggml_metal_pause_signaled(ggml_metal_t ctx) {
    return ctx->pause_event ? ctx->pause_event.signaledValue : 0;
}

void ggml_metal_pause_release(ggml_metal_t ctx, uint64_t value) {
    ctx->pause_event.signaledValue = value;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {"""),
])

edit(INC, [(
    "GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);",
    """GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

// MOMOE_PAUSE: pause points after nodes whose name starts with prefix (NULL: off); see patch_ggml_pause.py
GGML_BACKEND_API void     ggml_backend_metal_set_pause     (ggml_backend_t backend, const char * prefix);
GGML_BACKEND_API uint64_t ggml_backend_metal_pause_counter (ggml_backend_t backend);
GGML_BACKEND_API uint64_t ggml_backend_metal_pause_signaled(ggml_backend_t backend);
GGML_BACKEND_API void     ggml_backend_metal_pause_release (ggml_backend_t backend, uint64_t value);""")])

edit(G / "ggml-metal.cpp", [(
    """static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {""",
    """// MOMOE_PAUSE
void ggml_backend_metal_set_pause(ggml_backend_t backend, const char * prefix) {
    ggml_metal_set_pause((ggml_metal_t) backend->context, prefix);
}

uint64_t ggml_backend_metal_pause_counter(ggml_backend_t backend) {
    return ggml_metal_pause_counter((ggml_metal_t) backend->context);
}

uint64_t ggml_backend_metal_pause_signaled(ggml_backend_t backend) {
    return ggml_metal_pause_signaled((ggml_metal_t) backend->context);
}

void ggml_backend_metal_pause_release(ggml_backend_t backend, uint64_t value) {
    ggml_metal_pause_release((ggml_metal_t) backend->context, value);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {""")])
