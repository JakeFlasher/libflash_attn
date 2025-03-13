#include <argparse/argparse.hpp>
#include "tensor.h"
#include "flash.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

// Helper function to measure execution time
template<typename Func>
double measureExecutionTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    cudaDeviceSynchronize(); // Ensure CUDA operations are complete
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int main(int argc, char **argv) {
    // Create argument parser
    argparse::ArgumentParser program("flash_attention_demo");
    program.add_description("Flash Attention implementation for transformer models");
    
    // Common parameters
    program.add_argument("--attention_type")
        .default_value(std::string("basic"))
        .help("Type of attention: 'basic', 'var_len', or 'splitkv_paged'");
    
    program.add_argument("--num_heads")
        .default_value(32)
        .scan<'i', int>()
        .help("Number of attention heads");
    
    program.add_argument("--head_dim")
        .default_value(128)
        .scan<'i', int>()
        .help("Head dimension (embedding size per head)");
    
    program.add_argument("--num_kv_heads")
        .default_value(8)
        .scan<'i', int>()
        .help("Number of key/value heads for GQA/MQA (should be <= num_heads)");
    
    program.add_argument("--batch_size")
        .default_value(4)
        .scan<'i', int>()
        .help("Batch size");
    
    program.add_argument("--seq_q")
        .default_value(1024)
        .scan<'i', int>()
        .help("Query sequence length");
    
    program.add_argument("--seq_k")
        .default_value(1024)
        .scan<'i', int>()
        .help("Key/value sequence length");
    
    program.add_argument("--softmax_scale")
        .default_value(0.0f)
        .scan<'g', float>()
        .help("Softmax scale factor (default: 1/sqrt(head_dim))");
    
    program.add_argument("--causal")
        .default_value(false)
        .implicit_value(true)
        .help("Whether to use causal attention");
    
    program.add_argument("--window_size_left")
        .default_value(-1)
        .scan<'i', int>()
        .help("Left window size for local attention (-1 for unlimited)");
    
    program.add_argument("--window_size_right")
        .default_value(-1)
        .scan<'i', int>()
        .help("Right window size for local attention (-1 for unlimited)");
    
    program.add_argument("--stress_test")
        .default_value(false)
        .implicit_value(true)
        .help("Run with large dimensions to stress test GPU");
    
    program.add_argument("--iterations")
        .default_value(5)
        .scan<'i', int>()
        .help("Number of iterations to run (for performance measurement)");
    
    // Parameters for var_len attention
    program.add_argument("--max_seqlen_q")
        .default_value(0)
        .scan<'i', int>()
        .help("Maximum query sequence length for var_len attention");
    
    program.add_argument("--max_seqlen_k")
        .default_value(0)
        .scan<'i', int>()
        .help("Maximum key sequence length for var_len attention");
    
    // Parameters for splitkv_paged attention
    program.add_argument("--num_blocks")
        .default_value(0)
        .scan<'i', int>()
        .help("Number of blocks for splitkv_paged attention");
    
    program.add_argument("--block_size")
        .default_value(0)
        .scan<'i', int>()
        .help("Block size for splitkv_paged attention");
    
    program.add_argument("--max_num_blocks_per_seq")
        .default_value(0)
        .scan<'i', int>()
        .help("Maximum number of blocks per sequence for splitkv_paged attention");
    
    program.add_argument("--num_splits")
        .default_value(0)
        .scan<'i', int>()
        .help("Number of splits for splitkv_paged attention");
    
    // Parse command-line arguments
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    
    // Get parameters from command-line arguments
    std::string attention_type = program.get<std::string>("--attention_type");
    size_t num_heads = program.get<int>("--num_heads");
    size_t dim = program.get<int>("--head_dim");
    size_t num_kv_heads = program.get<int>("--num_kv_heads");
    int batch = program.get<int>("--batch_size");
    int seq_q = program.get<int>("--seq_q");
    int seq_k = program.get<int>("--seq_k");
    float softmax_scale = program.get<float>("--softmax_scale");
    bool is_causal = program.get<bool>("--causal");
    int window_size_left = program.get<int>("--window_size_left");
    int window_size_right = program.get<int>("--window_size_right");
    int iterations = program.get<int>("--iterations");
    bool stress_test = program.get<bool>("--stress_test");
    
    // Modify parameters for stress test mode
    if (stress_test) {
        std::cout << "Running in stress test mode with large dimensions" << std::endl;
        
        // Use larger values for stress testing
        num_heads = std::max(num_heads, static_cast<size_t>(32));
        dim = std::max(dim, static_cast<size_t>(128));
        num_kv_heads = std::max(num_kv_heads, static_cast<size_t>(8));
        batch = std::max(batch, 16);
        seq_q = std::max(seq_q, 2048);
        seq_k = std::max(seq_k, 2048);
        is_causal = true; // Force causal attention for stress test
    }
    
    // Set specific parameters for different attention types
    int max_seqlen_q = program.get<int>("--max_seqlen_q");
    int max_seqlen_k = program.get<int>("--max_seqlen_k");
    int num_blocks = program.get<int>("--num_blocks");
    int block_size = program.get<int>("--block_size");
    int max_num_blocks_per_seq = program.get<int>("--max_num_blocks_per_seq");
    int num_splits = program.get<int>("--num_splits");
    
    if (stress_test) {
        if (attention_type == "var_len") {
            max_seqlen_q = std::max(max_seqlen_q, seq_q);
            max_seqlen_k = std::max(max_seqlen_k, seq_k);
        }
        
        if (attention_type == "splitkv_paged") {
            num_blocks = std::max(num_blocks, 256);
            block_size = std::max(block_size, 64);
            max_num_blocks_per_seq = std::max(max_num_blocks_per_seq, 32);
            num_splits = std::max(num_splits, 4);
        }
    }
    
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int seq_v = seq_k;
    
    // Print memory usage information for large tensors
    size_t total_q = batch * seq_q;
    size_t total_k = batch * seq_k;
    size_t total_elements_q = total_q * num_heads * dim;
    size_t total_elements_k = total_k * num_kv_heads * dim;
    size_t total_elements_v = total_k * num_kv_heads * dim;
    size_t total_elements_o = total_q * num_heads * dim;
    
    size_t total_memory_bytes = (total_elements_q + total_elements_k + total_elements_v + total_elements_o) * sizeof(cutlass::half_t);
    double total_memory_gb = total_memory_bytes / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "Total tensor memory: " << total_memory_bytes << " bytes (" << total_memory_gb << " GB)" << std::endl;
    
    // Validate parameters
    if (num_kv_heads > num_heads) {
        std::cerr << "Warning: num_kv_heads (" << num_kv_heads << ") is greater than num_heads (" 
                  << num_heads << "). Usually num_kv_heads <= num_heads in GQA/MQA." << std::endl;
    }
    
    // Set default softmax_scale if not specified
    if (softmax_scale <= 0.0f) {
        softmax_scale = 1.0f / std::sqrt(static_cast<float>(dim));
        std::cout << "Using default softmax_scale: " << softmax_scale << std::endl;
    }
    
    // Common stride calculations
    int q_row_stride = dim;
    int k_row_stride = dim;
    int v_row_stride = dim;
    int o_row_stride = dim;
    int q_head_stride = q_row_stride * seq_q;
    int k_head_stride = k_row_stride * seq_k;
    int v_head_stride = v_row_stride * seq_k;
    int o_head_stride = o_row_stride * seq_q;
    int q_batch_stride = q_head_stride * num_heads;
    int k_batch_stride = k_head_stride * num_kv_heads;
    int v_batch_stride = v_head_stride * num_kv_heads;
    int o_batch_stride = o_head_stride * num_heads;
    
    using half = cutlass::half_t;
    
    // Determine which attention function to call based on the attention_type argument
    if (attention_type == "basic") {
        std::cout << "Using basic flash attention" << std::endl;
        std::cout << "Parameters: num_heads=" << num_heads << ", num_kv_heads=" << num_kv_heads 
                  << ", head_dim=" << dim << ", batch_size=" << batch 
                  << ", seq_q=" << seq_q << ", seq_k=" << seq_k 
                  << ", causal=" << (is_causal ? "true" : "false") << std::endl;
        
        // Create tensors
        std::cout << "Allocating tensors..." << std::endl;
        Tensor<half> *Q = new Tensor<half>({total_q, num_heads, dim}, "Tensor Q");
        Tensor<half> *K = new Tensor<half>({total_k, num_kv_heads, dim}, "Tensor K");
        Tensor<half> *V = new Tensor<half>({total_k, num_kv_heads, dim}, "Tensor V");
        Tensor<half> *O = new Tensor<half>({total_q, num_heads, dim}, "Tensor O");
        
        half *q_ptr = Q->getDevPtr();
        half *k_ptr = K->getDevPtr();
        half *v_ptr = V->getDevPtr();
        half *output_ptr = O->getDevPtr();
        
        // Run multiple iterations to measure performance
        std::cout << "Running " << iterations << " iterations..." << std::endl;
        double total_time = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            // Capture flash attention computation time
            double iter_time = measureExecutionTime([&]() {
                flash_attn::flash_attention_forward(
                    q_ptr, k_ptr, v_ptr, output_ptr, 
                    batch, seq_q, seq_k, num_heads, num_kv_heads, 
                    dim, q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride, 
                    q_head_stride, k_head_stride, v_head_stride, o_head_stride, 
                    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
                    softmax_scale, is_causal, window_size_left, window_size_right, stream
                );
            });
            
            total_time += iter_time;
            std::cout << "Iteration " << (i+1) << " completed in " << iter_time << " seconds" << std::endl;
        }
        
        // Explicitly synchronize before cleanup
        cudaDeviceSynchronize();
        
        // Print performance metrics
        double avg_time = total_time / iterations;
        double tokens_per_second = (batch * seq_q) / avg_time;
        std::cout << "Average execution time: " << avg_time << " seconds" << std::endl;
        std::cout << "Tokens per second: " << tokens_per_second << std::endl;
        
        // Clean up
        delete Q;
        delete K;
        delete V;
        delete O;
    } 
    else if (attention_type == "var_len") {
        std::cout << "Using variable length flash attention" << std::endl;
        
        // Check if additional required parameters are provided
        if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
            std::cerr << "Error: max_seqlen_q and max_seqlen_k must be specified for var_len attention" << std::endl;
            return 1;
        }
        
        std::cout << "Parameters: num_heads=" << num_heads << ", num_kv_heads=" << num_kv_heads 
                  << ", head_dim=" << dim << ", batch_size=" << batch 
                  << ", seq_q=" << seq_q << ", seq_k=" << seq_k 
                  << ", max_seqlen_q=" << max_seqlen_q << ", max_seqlen_k=" << max_seqlen_k 
                  << ", causal=" << (is_causal ? "true" : "false") << std::endl;
        
        // For var_len attention, we need cumulative sequence lengths
        int *cu_seqlens_q = nullptr;
        int *cu_seqlens_k = nullptr;
        cudaMalloc(&cu_seqlens_q, (batch + 1) * sizeof(int));
        cudaMalloc(&cu_seqlens_k, (batch + 1) * sizeof(int));
        
        // For simplicity, we'll use equal sequence lengths for all items in batch
        std::vector<int> h_cu_seqlens_q(batch + 1, 0);
        std::vector<int> h_cu_seqlens_k(batch + 1, 0);
        for (int i = 1; i <= batch; ++i) {
            h_cu_seqlens_q[i] = h_cu_seqlens_q[i-1] + seq_q;
            h_cu_seqlens_k[i] = h_cu_seqlens_k[i-1] + seq_k;
        }
        cudaMemcpy(cu_seqlens_q, h_cu_seqlens_q.data(), (batch + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_seqlens_k, h_cu_seqlens_k.data(), (batch + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        int total_q = batch * seq_q;
        int total_k = batch * seq_k;
        
        // Create tensors
        std::cout << "Allocating tensors..." << std::endl;
        Tensor<half> *Q = new Tensor<half>({static_cast<size_t>(total_q), num_heads, dim}, "Tensor Q");
        Tensor<half> *K = new Tensor<half>({static_cast<size_t>(total_k), num_kv_heads, dim}, "Tensor K");
        Tensor<half> *V = new Tensor<half>({static_cast<size_t>(total_k), num_kv_heads, dim}, "Tensor V");
        Tensor<half> *O = new Tensor<half>({static_cast<size_t>(total_q), num_heads, dim}, "Tensor O");
        
        half *q_ptr = Q->getDevPtr();
        half *k_ptr = K->getDevPtr();
        half *v_ptr = V->getDevPtr();
        half *output_ptr = O->getDevPtr();
        
        // Run multiple iterations to measure performance
        std::cout << "Running " << iterations << " iterations..." << std::endl;
        double total_time = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            // Capture flash attention computation time
            double iter_time = measureExecutionTime([&]() {
                flash_attn::flash_attention_var_len_forward(
                    q_ptr, k_ptr, v_ptr, cu_seqlens_q, cu_seqlens_k, output_ptr,
                    batch, max_seqlen_q, max_seqlen_k, num_heads, num_kv_heads,
                    dim, q_head_stride, k_head_stride, v_head_stride, o_head_stride,
                    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
                    softmax_scale, is_causal, total_q, total_k,
                    window_size_left, window_size_right, stream
                );
            });
            
            total_time += iter_time;
            std::cout << "Iteration " << (i+1) << " completed in " << iter_time << " seconds" << std::endl;
        }
        
        // Explicitly synchronize before cleanup
        cudaDeviceSynchronize();
        
        // Print performance metrics
        double avg_time = total_time / iterations;
        double tokens_per_second = (batch * seq_q) / avg_time;
        std::cout << "Average execution time: " << avg_time << " seconds" << std::endl;
        std::cout << "Tokens per second: " << tokens_per_second << std::endl;
        
        // Clean up
        delete Q;
        delete K;
        delete V;
        delete O;
        cudaFree(cu_seqlens_q);
        cudaFree(cu_seqlens_k);
    } 
    else if (attention_type == "splitkv_paged") {
        std::cout << "Using splitkv paged flash attention" << std::endl;
        
        // Check if additional required parameters are provided
        if (num_blocks <= 0 || block_size <= 0 || max_num_blocks_per_seq <= 0) {
            std::cerr << "Error: num_blocks, block_size, and max_num_blocks_per_seq must be specified for splitkv_paged attention" << std::endl;
            return 1;
        }
        
        std::cout << "Parameters: num_heads=" << num_heads << ", num_kv_heads=" << num_kv_heads 
                  << ", head_dim=" << dim << ", batch_size=" << batch 
                  << ", seq_q=" << seq_q << ", seq_k=" << seq_k 
                  << ", num_blocks=" << num_blocks << ", block_size=" << block_size
                  << ", max_num_blocks_per_seq=" << max_num_blocks_per_seq 
                  << ", num_splits=" << num_splits 
                  << ", causal=" << (is_causal ? "true" : "false") << std::endl;
        
        // For splitkv_paged attention, we need additional data structures
        int32_t *block_table_ptr = nullptr;
        int32_t *seqlens_k_ptr = nullptr;
        float *softmax_lse_accum_ptr = nullptr;
        float *output_accum_ptr = nullptr;
        
        cudaMalloc(&block_table_ptr, batch * max_num_blocks_per_seq * sizeof(int32_t));
        cudaMalloc(&seqlens_k_ptr, batch * sizeof(int32_t));
        cudaMalloc(&softmax_lse_accum_ptr, batch * num_heads * seq_q * sizeof(float));
        cudaMalloc(&output_accum_ptr, batch * num_heads * seq_q * dim * sizeof(float));
        
        // Initialize sequence lengths (all equal for simplicity)
        std::vector<int32_t> h_seqlens_k(batch, seq_k);
        cudaMemcpy(seqlens_k_ptr, h_seqlens_k.data(), batch * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        // Initialize block table (linear mapping for simplicity)
        std::vector<int32_t> h_block_table(batch * max_num_blocks_per_seq, -1);
        for (int i = 0; i < batch; ++i) {
            int num_blocks_for_seq = (seq_k + block_size - 1) / block_size;
            for (int j = 0; j < num_blocks_for_seq && j < max_num_blocks_per_seq; ++j) {
                h_block_table[i * max_num_blocks_per_seq + j] = j;  // Simple sequential block assignment
            }
        }
        cudaMemcpy(block_table_ptr, h_block_table.data(), batch * max_num_blocks_per_seq * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        size_t total_q = batch * seq_q;
        int block_table_batch_stride = max_num_blocks_per_seq;
        
        // Create tensors
        std::cout << "Allocating tensors..." << std::endl;
        Tensor<half> *Q = new Tensor<half>({total_q, num_heads, dim}, "Tensor Q");
        
        // For paged KV cache, linearize the dimensions
        Tensor<half> *KCache = new Tensor<half>({static_cast<size_t>(num_blocks * block_size), num_kv_heads, dim}, "Tensor KCache");
        Tensor<half> *VCache = new Tensor<half>({static_cast<size_t>(num_blocks * block_size), num_kv_heads, dim}, "Tensor VCache");
        Tensor<half> *O = new Tensor<half>({total_q, num_heads, dim}, "Tensor O");
        
        half *q_ptr = Q->getDevPtr();
        half *kcache_ptr = KCache->getDevPtr();
        half *vcache_ptr = VCache->getDevPtr();
        half *output_ptr = O->getDevPtr();
        
        // Run multiple iterations to measure performance
        std::cout << "Running " << iterations << " iterations..." << std::endl;
        double total_time = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            // Capture flash attention computation time
            double iter_time = measureExecutionTime([&]() {
                flash_attn::flash_attention_splitkv_paged_forward(
                    q_ptr, kcache_ptr, vcache_ptr, block_table_ptr, seqlens_k_ptr,
                    softmax_lse_accum_ptr, output_accum_ptr, output_ptr,
                    batch, seq_q, num_heads, num_kv_heads, dim,
                    q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride,
                    q_head_stride, k_head_stride, v_head_stride, o_head_stride,
                    q_row_stride, k_row_stride, v_row_stride, o_row_stride,
                    num_blocks, block_size, max_num_blocks_per_seq, block_table_batch_stride,
                    softmax_scale, is_causal, window_size_left, window_size_right, num_splits, stream
                );
            });
            
            total_time += iter_time;
            std::cout << "Iteration " << (i+1) << " completed in " << iter_time << " seconds" << std::endl;
        }
        
        // Explicitly synchronize before cleanup
        cudaDeviceSynchronize();
        
        // Print performance metrics
        double avg_time = total_time / iterations;
        double tokens_per_second = (batch * seq_q) / avg_time;
        std::cout << "Average execution time: " << avg_time << " seconds" << std::endl;
        std::cout << "Tokens per second: " << tokens_per_second << std::endl;
        
        // Clean up
        delete Q;
        delete KCache;
        delete VCache;
        delete O;
        cudaFree(block_table_ptr);
        cudaFree(seqlens_k_ptr);
        cudaFree(softmax_lse_accum_ptr);
        cudaFree(output_accum_ptr);
    } 
    else {
        std::cerr << "Error: Unknown attention type '" << attention_type 
                  << "'. Use 'basic', 'var_len', or 'splitkv_paged'." << std::endl;
        return 1;
    }
    
    cudaStreamDestroy(stream);
    return 0;
}
