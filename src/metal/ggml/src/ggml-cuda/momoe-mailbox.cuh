#pragma once
// Momoe: the mailbox ops (include/ggml-momoe-mailbox.h)
#include "common.cuh"

bool ggml_cuda_momoe_mailbox_supported(const ggml_tensor * op);
bool ggml_cuda_momoe_mailbox(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
