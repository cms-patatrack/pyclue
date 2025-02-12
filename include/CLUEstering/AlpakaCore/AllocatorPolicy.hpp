
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue {

  // Which memory allocator to use
  //   - Synchronous:   (device and host) cudaMalloc/hipMalloc and cudaMallocHost/hipMallocHost
  //   - Asynchronous:  (device only)     cudaMallocAsync (requires CUDA >= 11.2)
  //   - Caching:       (device and host) caching allocator
  enum class AllocatorPolicy { Synchronous = 0, Asynchronous = 1, Caching = 2 };

  template <typename TDev>
  constexpr inline AllocatorPolicy allocator_policy = AllocatorPolicy::Synchronous;

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || \
    defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  template <>
  constexpr inline AllocatorPolicy allocator_policy<alpaka::DevCpu> =
#if !defined ALPAKA_DISABLE_CACHING_ALLOCATOR
      AllocatorPolicy::Caching;
#else
      AllocatorPolicy::Synchronous;
#endif
#endif  // defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#if defined ALPAKA_ACC_GPU_CUDA_ENABLED
  template <>
  constexpr inline AllocatorPolicy allocator_policy<alpaka::DevCudaRt> =
#if !defined ALPAKA_DISABLE_CACHING_ALLOCATOR
      AllocatorPolicy::Caching;
#elif CUDA_VERSION >= 11020 && !defined ALPAKA_DISABLE_ASYNC_ALLOCATOR
      AllocatorPolicy::Asynchronous;
#else
          AllocatorPolicy::Synchronous;
#endif
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#if defined ALPAKA_ACC_GPU_HIP_ENABLED
  template <>
  constexpr inline AllocatorPolicy allocator_policy<alpaka::DevHipRt> =
#if !defined ALPAKA_DISABLE_CACHING_ALLOCATOR
      AllocatorPolicy::Caching;
#else
      AllocatorPolicy::Synchronous;
#endif
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#if defined ALPAKA_SYCL_ONEAPI_CPU
  template <>
  constexpr inline AllocatorPolicy allocator_policy<alpaka::DevCpuSycl> = AllocatorPolicy::Synchronous;
#endif

#if defined ALPAKA_SYCL_ONEAPI_GPU
  template <>
  constexpr inline AllocatorPolicy allocator_policy<alpaka::DevGpuSyclIntel> = AllocatorPolicy::Asynchronous;
#endif

}  // namespace clue
