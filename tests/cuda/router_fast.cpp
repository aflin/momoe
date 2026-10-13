// The fast router (router_top10_fast_kernel, router_top10's default for 64 < n_expert <= 512, k <= 32) against the
// portable kernel on CUDA: the HIP test (tests/hip/router_fast.cpp) unchanged, its runtime calls mapped onto the CUDA
// runtime. Bitwise on ids and weights: normal rows, exact ties, near-ties, wide spreads, all-equal rows, NaN rows, the
// forced serial-sum path, and the 256-expert / k=8 geometry (qwen35moe).
#include <cuda_runtime.h>

#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipError_t cudaError_t
#define hipEventCreate cudaEventCreate
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEvent_t cudaEvent_t
#define hipFree cudaFree
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemset cudaMemset
#define hipSuccess cudaSuccess
#define STRATA_ROUTER_FAST_ON_CUDA 1

#include "../hip/router_fast.cpp"
