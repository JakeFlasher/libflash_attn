Below is **one possible** comprehensive, self-contained code example that:

1. **Reads environment variables** to set both *L1 (shared memory)* carve-out and *L2 persisting cache* size (in bytes or as a %).  
2. **Persists multiple pointers** (q_ptr, k_ptr, v_ptr, o_ptr, and oaccum_ptr) in L2 by setting an *access policy window* for each pointer.  
   - *Important Note*: **In practice, multiple calls to `cudaStreamSetAttribute(..., cudaStreamAttributeAccessPolicyWindow, ...)` on the **same** stream often *replace* the previous policy window** rather than add a second one. The CUDA documentation (9.2.2.1) only illustrates *one* region at a time.*  
   - If you truly need multiple pointers simultaneously persisted, **you may need**:  
     1) A single combined region that covers all pointers (if they lie reasonably close in memory).  
     2) Multiple streams, with each stream launching a kernel that persists a different region.  
   - For demonstration, below we simply call `set_stream_access_policy` multiple times in a row. On some GPUs/runtimes, only the *last* call may apply.  
3. **Launches a FlashAttention-style kernel** (placeholder).  
4. **Demonstrates error-checking** and some logs for debug.  

> **Disclaimer**: Because official CUDA docs do *not* confirm multiple access policy windows per stream, you should verify if each subsequent call overrides the previous. If so, you might want to unify your Q, K, V, O, Oaccum pointers into one or two windows, or use separate streams.  

---

## Full Example Code

```cpp
/*************************************************************
 * Example: L1 + L2 Manipulation, Multiple Pointers Persisted
 *
 * This code shows how you might:
 *  - Read env variables to set the L1 carve-out (FLASH_ATTN_SMEM_CARVEOUT).
 *  - Read env variables for L2 carve-out (FLASH_ATTN_L2_CARVEOUT).
 *  - Use cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...) to
 *    artificially reduce the L2 capacity and set aside a portion for persisting data.
 *  - For each pointer (q, k, v, o, oaccum), call cudaStreamSetAttribute to
 *    set an AccessPolicyWindow to "persist" that data in L2.
 *
 * WARNING:
 *   - The GPU might only allow one AccessPolicyWindow at a time per stream.
 *   - The last call may override the earlier calls.
 *   - If you truly need multiple pointers simultaneously, you might:
 *       1) unify them in a single region, or
 *       2) use separate streams, or
 *       3) rely on partial "hitRatio" settings.
 *************************************************************/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <cassert>

// ---------------------------------------------------------------------
// Example parameter struct (like "Flash_fwd_params" that includes Q, K, V, O, Oaccum).
// For illustration, it’s simplified here (some fields omitted).
// ---------------------------------------------------------------------
struct Flash_fwd_params {
    // Q, K, V
    void * __restrict__ q_ptr;
    void * __restrict__ k_ptr;
    void * __restrict__ v_ptr;

    // Output
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    int b, h, seqlen_q, seqlen_k, d;
    // Possibly more fields...
};

// ---------------------------------------------------------------------
// 1) Retrieve L1 (Shared Memory) carve-out [0..100]%
//    from env var "FLASH_ATTN_SMEM_CARVEOUT".
// ---------------------------------------------------------------------
inline int get_smem_carveout() {
    const char* carveout_str = std::getenv("FLASH_ATTN_SMEM_CARVEOUT");
    if (carveout_str) {
        int value = std::stoi(carveout_str);
        // clamp to [0..100]
        return std::min(100, std::max(0, value));
    }
    return 50; // default 50% if not set
}

// ---------------------------------------------------------------------
// 2) Retrieve L2 carve-out [0..100]% from env var "FLASH_ATTN_L2_CARVEOUT".
// ---------------------------------------------------------------------
inline float get_l2_carveout_percent() {
    const char* env_str = std::getenv("FLASH_ATTN_L2_CARVEOUT");
    if (!env_str) return 50.f; // default
    try {
        float val = std::stof(env_str);
        // clamp to [0..100]
        return std::max(0.f, std::min(100.f, val));
    } catch (...) {
        fprintf(stderr, "[Warning] Could not parse FLASH_ATTN_L2_CARVEOUT; default to 50.\n");
        return 50.f;
    }
}

// ---------------------------------------------------------------------
// 3) Set L2 persisting cache size (in bytes).
//    We multiply get_l2_carveout_percent() by the device's max possible set-aside.
// ---------------------------------------------------------------------
inline void set_l2_persisting_cache_by_percent() {
    int dev_id;
    cudaGetDevice(&dev_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);

    size_t max_persist = prop.persistingL2CacheMaxSize; 
    float carveout_percent = get_l2_carveout_percent();
    size_t bytes_to_set = static_cast<size_t>(carveout_percent / 100.f * (float)max_persist);

    cudaError_t err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes_to_set);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Warning] set_l2_persisting_cache_by_percent(%.1f%% => %zu B) failed: %s\n",
                carveout_percent, bytes_to_set, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "[Info] L2 set-aside => %.2f%% => %zu / %zu bytes.\n",
                carveout_percent, bytes_to_set, max_persist);
    }
}

// ---------------------------------------------------------------------
// 4) Mark [base_ptr..base_ptr+num_bytes) as persisting. We also specify
//    how hits/misses are treated. E.g. "hitProp=Persisting" / "missProp=Persisting" or "Streaming".
//    For demonstration, we do hit=Persisting and miss=Persisting => all in L2 => thrash if > set-aside.
//    Alternatively, set miss=Streaming if you only want the frequently accessed subset to remain.
// ---------------------------------------------------------------------
inline void set_stream_access_policy(
    cudaStream_t stream, void* base_ptr, size_t num_bytes,
    float hit_ratio    = 1.0f,
    cudaAccessProperty hit_prop  = cudaAccessPropertyPersisting,
    cudaAccessProperty miss_prop = cudaAccessPropertyPersisting)
{
    if (!base_ptr || num_bytes == 0) {
        // skip if pointer is null or size is zero
        return;
    }

    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));

    attr.accessPolicyWindow.base_ptr  = base_ptr;
    attr.accessPolicyWindow.num_bytes = num_bytes;
    attr.accessPolicyWindow.hitRatio  = hit_ratio;

    attr.accessPolicyWindow.hitProp   = hit_prop;
    attr.accessPolicyWindow.missProp  = miss_prop;

    cudaError_t err = cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "[Warning] set_stream_access_policy(%p, %zu B) => %s\n",
                base_ptr, num_bytes, cudaGetErrorString(err));
    } else {
        fprintf(stderr, "[Info] Marking region %p..%p as persisting. size=%zu\n",
                base_ptr, (void*)((char*)base_ptr + num_bytes), num_bytes);
    }
}

// ---------------------------------------------------------------------
// Dummy kernel to represent FlashAttention or something similar
// We'll just do an empty kernel for demonstration
// ---------------------------------------------------------------------
__global__ void flash_fwd_kernel(Flash_fwd_params params)
{
    // Here is where you'd implement your GPU logic, e.g. a matrix multiply or attention.
    // We'll leave it blank as an example.
}

// ---------------------------------------------------------------------
// The main function that manipulates L1 & L2, persists Q,K,V,O,Oaccum,
// and then launches the kernel. 
//
// If you truly want to set multiple L2 windows simultaneously, note that
// multiple calls to cudaStreamSetAttribute() on the same stream may override
// each other. Check the actual GPU behavior or consider a combined region.
//
// ---------------------------------------------------------------------
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream)
{
    // 1) (L2) Decide how much L2 to set aside:
    set_l2_persisting_cache_by_percent();

    // 2) Mark each pointer as persisting. 
    //    WARNING: multiple calls may override each other in practice.
    //    If you do want them all truly persisted, 
    //    consider a single combined region if they are contiguous or use multiple streams.
    {
        // Q
        size_t q_size_bytes = static_cast<size_t>(params.b) *
                              static_cast<size_t>(params.h) *
                              static_cast<size_t>(params.seqlen_q) *
                              static_cast<size_t>(params.d) * 
                              sizeof(float); // or half/bfloat if needed
        set_stream_access_policy(stream, params.q_ptr, q_size_bytes);

        // K 
        size_t k_size_bytes = /* some logic, e.g. b*h*seqlen_k*d*sizeof(float) */;
        set_stream_access_policy(stream, params.k_ptr, k_size_bytes);

        // V 
        size_t v_size_bytes = /* b*h*seqlen_k*d*sizeof(float) or similar */;
        set_stream_access_policy(stream, params.v_ptr, v_size_bytes);

        // O 
        size_t o_size_bytes = /* b*h*seqlen_q*d*sizeof(float)? */;
        set_stream_access_policy(stream, params.o_ptr, o_size_bytes);

        // Oaccum
        size_t oaccum_size_bytes = /* b*h*seqlen_q*d*sizeof(float)? or some dimension */;
        set_stream_access_policy(stream, params.oaccum_ptr, oaccum_size_bytes);

        // NOTE: Check if the GPU or driver overwrote the previous calls. 
        // Some GPUs only keep the "last" region. 
        // If so, you might want to unify Q,K,V,O,Oaccum in one big region or 
        // use separate streams or partial hitRatios.
    }

    // 3) (L1) Set L1 carve-out, read from "FLASH_ATTN_SMEM_CARVEOUT"
    {
        int carveout = get_smem_carveout();
        // If your kernel uses dynamic shared memory, you might set the dynamic size as well
        // in a real scenario. For demonstration, we do a function attribute:
        cudaFuncSetAttribute(flash_fwd_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            carveout);
    }

    // 4) Launch the kernel (dummy example)
    dim3 block(128);
    dim3 grid(1);
    // In real code, you'd do something like:
    //   dim3 grid((params.seqlen_q + BLOCK_M - 1)/BLOCK_M, params.b, params.h);
    //   ...
    flash_fwd_kernel<<<grid, block, 0, stream>>>(params);

    // 5) Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

//
// End of example
//
```

### Key Points & Caveats

1. **One AccessPolicyWindow per Stream**  
   - Officially, **the documentation** (CUDA Toolkit 12.x, section 9.2.2.1) only presents one window. Empirically, multiple calls on the *same* stream typically override the previous calls, so only the last pointer region truly persists.  
   - If you *really* want Q, K, V, O, Oaccum all persisted **simultaneously**, you can either:  
     a) Use a single combined region `[base_ptr_min.. base_ptr_max)` if they’re mostly contiguous, or  
     b) Use multiple streams (each stream with one window), with each portion of your kernel. But that complicates your code.  
   - Alternatively, set a smaller pointer region for each call with a partial `hitRatio < 1.0f`, so that *some fraction* of accesses are “persisting” (see the “Tuning the Access Window Hit-Ratio” in the CUDA manual).  

2. **Data Size and Thrashing**  
   - If the total region is bigger than the set-aside L2 capacity, you may see *thrashing* and a performance penalty. That might be *desired* if your experiment is to emulate a physically smaller L2.  
   - If you do want to physically limit L2 to a fraction (like 2 MB out of 4 MB), call `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, …)` with that number of bytes.  

3. **Shared Memory (L1) Carve-Out**  
   - Using `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, X)` simply *requests* that a fraction of SM on-chip memory be used for shared memory rather than L1. The device might not always honor it exactly. Large static or dynamic shared allocations might also cause a fallback.  
   - You can check final usage with occupancy tests or query the device after setting this attribute.  

4. **Check Return Codes**  
   - In real code, check the return values of CUDA calls (e.g., `cudaFuncSetAttribute`, `cudaDeviceSetLimit`) to handle driver or hardware constraints gracefully.  

---

## Quick Summary

- **Yes**, you can attempt to persist multiple pointers (Q, K, V, O, Oaccum), *but* you must be aware that **multiple calls** to `cudaStreamSetAttribute(..., cudaStreamAttributeAccessPolicyWindow, ...)` on the *same* stream often *override* each other in practice.  
- If you truly need all of them persisted concurrently, consider combining them into a single large region or using multiple streams.  
- The provided code snippet above demonstrates **one** approach to get around the problem of artificially reducing L2 capacity and controlling which data is “persisted” vs. “streamed.”  
- For “L1 manipulation,” setting `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout)` is typically enough (with the environment variable logic).  

With these guidelines, you have a **starting template** that manipulates both L1 and L2—and attempts to mark multiple pointers as persisting. Adjust as needed for your research experiments!