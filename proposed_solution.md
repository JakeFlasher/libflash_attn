Below is a *complete* example illustrating how to:

1. **Keep the exact function signature**:
   ```cpp
   void run_flash_fwd_with_L2(Flash_fwd_params &params, cudaStream_t stream)
   ```
   so we do **not** add an extra pointer parameter.

2. **Automatically discover** a global memory pointer (e.g., `params.q`) by reading it *inside* `run_flash_fwd_with_L2()`. In your real code, you would determine which pointer(s) best correspond to the data your kernel is accessing. For example, if your kernel uses `params.q` as the main buffer, that might be the best pointer to pass into the access policy window.

3. **Use environment variables**:
   - `FLASH_ATTN_SMEM_CARVEOUT` → *L1/Shared* carve-out percentage (0–100).  
   - `FLASH_ATTN_L2_CARVEOUT`   → *L2 persisting cache* percentage (0–100) of the device’s maximum.  
   - (Optional) `FLASH_ATTN_L2_NUMBYTES` → override or specify how many bytes you want to mark as persisting in the access policy window. If unset, we do a naive estimate from `params`.

4. **Set** both `hitProp` and `missProp` to `cudaAccessPropertyPersisting`, so if the region is bigger than the set-aside, you see thrashing—emulating a physically smaller L2.

Below is one **self-contained** code file. Please feel free to adapt it.

---

## Complete Example Code

```cpp
/************************************************************
 * Minimal demonstration of using environment variables to:
 * 1) Set L1 carve-out (FLASH_ATTN_SMEM_CARVEOUT)
 * 2) Set L2 persisting cache as a % of device max 
 *    (FLASH_ATTN_L2_CARVEOUT)
 * 3) Mark the main data pointer from the kernel as 
 *    "persisting" for both hits & misses (thrash if over L2)
 * ---------------------------------------------------------
 *        *** WITHOUT changing the function signature ***
 * 
 * The final "run_flash_fwd_with_L2(Flash_fwd_params&, cudaStream_t)"
 * has no extra pointer parameter. It looks up the data pointer 
 * internally (e.g., from params.q). 
 ************************************************************/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <cassert>

// ------------------------------------------------------------------
// Example parameter struct for your kernel
// (You likely have something similar in your real code.)
// ------------------------------------------------------------------
struct Flash_fwd_params {
    // Some relevant fields
    int b, h;         // batch, heads
    int seqlen_q;     // sequence length (Q)
    int seqlen_k;     // sequence length (K)
    int d;            // head dim
    const void* q;    // pointer to global memory for Q
    const void* k;    // pointer to global memory for K
    const void* v;    // pointer to global memory for V
    // ... Additional fields ...
    int window_size_left, window_size_right;
    const void* alibi_slopes_ptr;
    const void* cu_seqlens_q;
    const void* cu_seqlens_k;
};

// ------------------------------------------------------------------
// 1) Retrieve L1 (Shared Memory) carve-out [0..100]%
//    Uses env var "FLASH_ATTN_SMEM_CARVEOUT"
// ------------------------------------------------------------------
inline int get_smem_carveout() {
    const char* carveout_str = std::getenv("FLASH_ATTN_SMEM_CARVEOUT");
    if (carveout_str) {
        int value = std::stoi(carveout_str);
        // clamp to [0..100]
        return std::min(100, std::max(0, value));
    }
    return 50; // default 50% if not set
}

// ------------------------------------------------------------------
// 2) Sets L2 persisting cache as a percentage of max
//    Uses env var "FLASH_ATTN_L2_CARVEOUT" in [0..100].
// ------------------------------------------------------------------
inline void set_l2_persisting_cache_by_percent() {
    const char* env_str = std::getenv("FLASH_ATTN_L2_CARVEOUT");
    // If not set, default to e.g. 0% or 50%. 
    // Let's pick 50% just for demonstration.
    float carveout_percent = 50.f; 
    if (env_str) {
        try {
            carveout_percent = std::stof(env_str);
        } catch (...) {
            fprintf(stderr, "[Warning] Could not parse FLASH_ATTN_L2_CARVEOUT; using 50.\n");
            carveout_percent = 50.f;
        }
    }
    // clamp [0..100]
    carveout_percent = std::max(0.f, std::min(100.f, carveout_percent));

    // get device's max persisting L2
    int dev_id;
    cudaGetDevice(&dev_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);

    size_t max_persist = prop.persistingL2CacheMaxSize;
    float ratio = carveout_percent / 100.f;
    size_t bytes_to_set = static_cast<size_t>(ratio * static_cast<float>(max_persist));

    cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes_to_set);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Warning] set_l2_persisting_cache_by_percent(%.1f%% => %zu B) failed: %s\n",
                carveout_percent, bytes_to_set, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "[Info] L2 set-aside => %.2f%%, i.e. %zu / %zu bytes.\n",
                carveout_percent, bytes_to_set, max_persist);
    }
}

// ------------------------------------------------------------------
// 3) Decide how many bytes we want to mark as 'persisting' 
//    for the kernel. 
//    If "FLASH_ATTN_L2_NUMBYTES" is set, use that. 
//    Otherwise, do a naive guess from the struct fields.
// ------------------------------------------------------------------
inline size_t get_l2_num_bytes_for_kernel(const Flash_fwd_params& params) {
    const char* env_nb = std::getenv("FLASH_ATTN_L2_NUMBYTES");
    if (env_nb) {
        size_t val = std::stoul(env_nb);
        return val;
    }
    // Otherwise, a naive guess, e.g. b*h*seqlen_q*d*4 bytes 
    // as a demonstration. Adjust if your kernel only 
    // partially accesses Q.
    size_t guess = static_cast<size_t>(params.b) 
                 * static_cast<size_t>(params.h)
                 * static_cast<size_t>(params.seqlen_q)
                 * static_cast<size_t>(params.d)
                 * sizeof(float); 
    return guess;
}

// ------------------------------------------------------------------
// 4) Mark [ptr..ptr+num_bytes) as persisting on hits+misses
//    This forces thrashing if the region is bigger than the
//    actual L2 set-aside. 
// ------------------------------------------------------------------
inline void set_stream_persist_all(cudaStream_t stream, void* base_ptr, size_t num_bytes) {
    if (!base_ptr || num_bytes == 0) {
        // no pointer => do nothing
        return;
    }

    cudaStreamAttrValue stream_attr;
    memset(&stream_attr, 0, sizeof(stream_attr));

    stream_attr.accessPolicyWindow.base_ptr  = base_ptr;
    stream_attr.accessPolicyWindow.num_bytes = num_bytes;
    stream_attr.accessPolicyWindow.hitRatio  = 1.0f;

    // Force lines to remain in L2, whether it’s a “hit” or a “miss.”
    // This is how we emulate a physically smaller L2 
    // if the region is bigger than the set-aside => thrash.
    stream_attr.accessPolicyWindow.hitProp  = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyPersisting;

    cudaError_t err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Warning] set_stream_persist_all(%p, %zu) => %s\n",
                base_ptr, num_bytes, cudaGetErrorString(err));
    }
}

// ------------------------------------------------------------------
// 5) Example kernel stub. Assume your real kernel is named 
//    flash_fwd_kernel<Kernel_traits, ...>
// ------------------------------------------------------------------
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_kernel(const Flash_fwd_params params) {
    // ...
    // your attention compute, etc.
}

// ------------------------------------------------------------------
// 6) The function we do NOT want to change the signature of:
//
//      void run_flash_fwd_with_L2(Flash_fwd_params &params, 
//                                 cudaStream_t stream)
//
//    We'll automatically detect the pointer from 'params' 
//    (e.g., params.q) and do all the L2 manipulations internally.
// ------------------------------------------------------------------
template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_with_L2(Flash_fwd_params &params, cudaStream_t stream) 
{
    // (1) Set L2 persisting cache by percentage 
    //     => from FLASH_ATTN_L2_CARVEOUT
    set_l2_persisting_cache_by_percent();

    // (2) Figure out how many bytes we want to mark as persisting:
    size_t window_bytes = get_l2_num_bytes_for_kernel(params);

    // (3) Because we can't modify the signature to pass a 
    //     separate pointer, we can rely on params.q 
    //     or any other relevant pointer from the struct.
    //     Using const_cast to get a void*:
    void* base_ptr = const_cast<void*>(params.q);

    // (4) Mark it as fully persisting. 
    set_stream_persist_all(stream, base_ptr, window_bytes);

    // *** NOTE *** 
    // If your kernel uses multiple global memory pointers 
    // (e.g., Q, K, V), you can choose the main one you 
    // want to emulate. Or in more advanced usage, you 
    // could mark multiple regions sequentially.

    // (5) Next, do the same approach as your original run_flash_fwd code:
    //     - set L1 carve-out from FLASH_ATTN_SMEM_CARVEOUT
    //     - launch the kernel

    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Example grid dimension logic:
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) 
                                 / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    // Example booleans, as you might have:
    const bool is_even_MN = (params.cu_seqlens_q == nullptr &&
                             params.cu_seqlens_k == nullptr &&
                             (params.seqlen_k % Kernel_traits::kBlockN == 0) &&
                             (params.seqlen_q % Kernel_traits::kBlockM == 0));
    const bool is_even_K = (params.d == Kernel_traits::kHeadDim);

    // Macro to replicate your BOOL_SWITCH pattern:
#define BOOL_SWITCH(COND, NAME, BODY) \
    do {                               \
      if (COND) {                      \
        static const bool NAME = true; \
        BODY();                        \
      } else {                         \
        static const bool NAME = false;\
        BODY();                        \
      }                                \
    } while(0)

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH( (params.window_size_left >= 0 
                          || params.window_size_right >= 0) 
                         && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    // Suggest the correct kernel variant
                    auto kernel = &flash_fwd_kernel<
                        Kernel_traits, 
                        Is_causal,
                        (Is_local && !Is_causal),
                        Has_alibi,
                        (IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128),
                        IsEvenKConst
                    >;

                    // If large dynamic smem is needed:
                    if (smem_size >= 48 * 1024) {
                        cudaFuncSetAttribute(kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             smem_size);
                    }

                    // L1 carve-out:
                    int carveout = get_smem_carveout();
                    cudaError_t err1 = cudaFuncSetAttribute(kernel,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        carveout);
                    if (err1 != cudaSuccess) {
                        fprintf(stderr, "[Warning] L1 carve-out error: %s\n",
                                cudaGetErrorString(err1));
                    }

                    // Launch the chosen kernel
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                });
            });
        });
    });

#undef BOOL_SWITCH
}

```

### Key Takeaways

1. **Unchanged Signature**  
   We have kept:
   ```cpp
   void run_flash_fwd_with_L2(Flash_fwd_params &params, cudaStream_t stream)
   ```
   exactly as is—no new parameters for the pointer or the number of bytes.

2. **Discover the Global Pointer**  
   Inside `run_flash_fwd_with_L2()`, we simply do:
   ```cpp
   void* base_ptr = const_cast<void*>(params.q);
   ```
   If you want to persist the K or V pointer (or multiple pointers), you could similarly call `set_stream_persist_all()` multiple times. Just keep in mind each call configures an access policy window for a separate address range.

3. **Similar to L1**  
   Instead of passing a specific number of L2 bytes, we read from `FLASH_ATTN_L2_CARVEOUT` (0..100) and multiply that fraction by `prop.persistingL2CacheMaxSize`. This closely mirrors the logic for `FLASH_ATTN_SMEM_CARVEOUT`.

4. **Thrashing Emulation**  
   Since your kernel likely uses the data at `params.q` (and possibly more), if the “persisting set-aside” is smaller than the total data in the policy window, the L2 lines will thrash—mimicking a physically smaller L2 on a different GPU (e.g., a 3090 Ti).

5. **Multiple Pointers**  
   If your kernel heavily uses both `params.q` and `params.k`, you might need to do:
   ```cpp
   set_stream_persist_all(stream, const_cast<void*>(params.q), window_bytes_q);
   set_stream_persist_all(stream, const_cast<void*>(params.k), window_bytes_k);
   ```
   But that’s your choice. In many standard attention kernels, the largest data chunk might be Q.

6. **No “Tricks” for Kernel Introspection**  
   There is no standard CUDA API to ask: “Which global memory pointer does a kernel use?” at runtime. Typically, the host code *knows* the pointer used by the kernel and sets an access policy window accordingly. So the easiest approach is to store or retrieve that pointer from `params`.

---

## Final Remarks

- This solution preserves the function signature exactly.  
- It uses the pointer from `params` itself (like `params.q`) to set the access policy window.  
- It reads environment variables for both L1 carve-out (`FLASH_ATTN_SMEM_CARVEOUT`) and L2 carve-out (`FLASH_ATTN_L2_CARVEOUT`).  
- It ensures that both hits and misses are treated as `cudaAccessPropertyPersisting`.  
- If the data region is larger than the set-aside, performance will degrade accordingly, emulating a physically smaller L2 cache.  

This should enable you to continue with your experiments without modifying the existing function prototype.