// Momoe: the mailbox ops (include/ggml-momoe-mailbox.h) on CUDA.
#include "common.cuh"
#include "momoe-mailbox.cuh"

#include "ggml-impl.h"
#include "ggml-momoe-mailbox.h"

#define MOMOE_MB_BLOCK 256

// store the payload into host memory, make it visible to the host, then raise the flag: a host that sees the flag sees
// the payload
static __global__ void momoe_mb_send(const float * __restrict__ x, const int32_t * __restrict__ ids, float * mb_f32,
        int32_t * mb_i32, volatile uint32_t * flag, const volatile uint32_t * epoch, const int n_f32, const int n_i32) {
    for (int i = threadIdx.x; i < n_f32; i += blockDim.x) {
        mb_f32[i] = x[i];
    }
    for (int i = threadIdx.x; i < n_i32; i += blockDim.x) {
        mb_i32[i] = ids[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *flag = *epoch;
        __threadfence_system();
    }
}

// one block: thread 0 waits for the host's flag (the host stored the payload before it), then the block copies it
static __global__ void momoe_mb_wait(const volatile float * mb_f32, float * __restrict__ dst, volatile uint32_t * flag,
        const volatile uint32_t * epoch, volatile uint32_t * timeout, const int n) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = *epoch;
        const long long t0 = clock64();
        ok = 1;
        while (*flag != want) {
            __nanosleep(200);
            if (clock64() - t0 > 40000000000LL) {          // ~20 s at 2 GHz: the CPU never answered
                *timeout = 1;
                ok = 0;
                break;
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = ok ? mb_f32[i] : 0.0f;
    }
}

static const momoe_mailbox_op * momoe_mailbox_of(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) {
        return nullptr;
    }
    const ggml_custom_op_params * p = (const ggml_custom_op_params *) op->op_params;
    if (p->fun != momoe_mailbox_cpu_stub || p->userdata == nullptr) {
        return nullptr;
    }
    const momoe_mailbox_op * m = (const momoe_mailbox_op *) p->userdata;
    return m->magic == MOMOE_MAILBOX_MAGIC ? m : nullptr;
}

bool ggml_cuda_momoe_mailbox_supported(const ggml_tensor * op) {
    const momoe_mailbox_op * m = momoe_mailbox_of(op);
    if (m == nullptr) {
        return false;
    }
    switch (m->kind) {
        case MOMOE_MAILBOX_SEND:
            return op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                   ggml_nelements(op->src[0]) == m->n_f32 && op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                   ggml_is_contiguous(op->src[1]) && ggml_nelements(op->src[1]) == m->n_i32;
        case MOMOE_MAILBOX_WAIT:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_nelements(op) == m->n_f32;
        default:
            return false;
    }
}

bool ggml_cuda_momoe_mailbox(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const momoe_mailbox_op * m = momoe_mailbox_of(dst);
    if (m == nullptr) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    if (m->kind == MOMOE_MAILBOX_SEND) {
        momoe_mb_send<<<1, MOMOE_MB_BLOCK, 0, stream>>>((const float *) dst->src[0]->data, (const int32_t *) dst->src[1]->data,
                (float *) m->dev_f32, (int32_t *) m->dev_i32, m->dev_flag, m->dev_epoch, m->n_f32, m->n_i32);
    } else {
        momoe_mb_wait<<<1, MOMOE_MB_BLOCK, 0, stream>>>((const volatile float *) m->dev_f32, (float *) dst->data, m->dev_flag,
                m->dev_epoch, m->dev_timeout, m->n_f32);
    }
    return true;
}
