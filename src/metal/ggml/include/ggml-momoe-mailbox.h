#pragma once

// Momoe: mailboxes between a CUDA graph and the CPU (momoe-cuda --hybrid). A mailbox is page-locked host memory the GPU
// maps: a SEND op stores a layer's input and chosen experts there and raises a flag; the CPU, polling the flag, computes
// the experts the GPU's pool does not hold and stores their outputs and its own flag; a WAIT op spins on that flag and
// copies the outputs into the graph. The token runs as one graph: no stream synchronization, no graph split, no copy.
// (After llama.cpp PR #30594's mailbox, with the wait inside the graph instead of between scheduler splits.)
//
// Both are a GGML_OP_CUSTOM whose function is momoe_mailbox_cpu_stub and whose userdata is a momoe_mailbox_op; the CUDA
// backend runs them (src/ggml-cuda/momoe-mailbox.cu), the stub aborts if one reaches the CPU.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOMOE_MAILBOX_MAGIC 0x4d4d4258u // "MMBX"

enum momoe_mailbox_kind {
    MOMOE_MAILBOX_SEND = 1, // src[0] (F32, n_f32) and src[1] (I32, n_i32) into the mailbox, then *flag := *epoch
    MOMOE_MAILBOX_WAIT = 2, // spin until *flag == *epoch, then n_f32 32-bit words from the mailbox into dst (F32 or I32)
};

struct momoe_mailbox_op {
    uint32_t magic;                    // MOMOE_MAILBOX_MAGIC
    int32_t  kind;                     // enum momoe_mailbox_kind
    int32_t  n_f32, n_i32;
    void * dev_f32;                    // the mailbox as the GPU addresses it (cudaHostGetDevicePointer)
    void * dev_i32;
    volatile uint32_t * dev_flag;
    const volatile uint32_t * dev_epoch;
    volatile uint32_t * dev_timeout;   // WAIT: set to 1 when it gave up (the CPU never answered: ~20 s)
};

GGML_API void momoe_mailbox_cpu_stub(struct ggml_tensor * dst, int ith, int nth, void * userdata);

// momoe-cuda: n bytes from `src` (host memory the GPU maps: cudaMallocHost) to device memory `dst` by a kernel on
// `stream` (a cudaStream_t); 0 when launched
GGML_API int momoe_copy_to_device(void * dst, const void * src, size_t n, void * stream);

#ifdef __cplusplus
}
#endif
