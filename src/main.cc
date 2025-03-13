#include "gflags/gflags.h"
#include "tensor.h"
#include "flash.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t num_heads = 32;
    size_t dim = 128;
    size_t num_kv_heads = 64;
    int batch = 2;
    int seq_q = 1;
    int seq_k = 128;
    int seq_v = 128;
    
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

    float softmax_scale = 1;
    bool is_causal = false;

    size_t total_q = batch * seq_q;
    size_t total_k = batch * seq_k;
    size_t total_v = batch * seq_v;
    using half = cutlass::half_t;

    Tensor<half> * Q = new Tensor<half>({total_q, num_heads, dim}, "Tensor Q");
    Tensor<half> * K = new Tensor<half>({total_k, num_kv_heads, dim}, "Tensor K");
    Tensor<half> * V = new Tensor<half>({total_k, num_kv_heads, dim}, "Tensor V");
    Tensor<half> * O = new Tensor<half>({total_q, num_heads, dim}, "Tensor O");

    half *q_ptr = Q->getDevPtr();
    half *k_ptr = K->getDevPtr();
    half *v_ptr = V->getDevPtr();
    half *output_ptr = O->getDevPtr();

    flash_attn::flash_attention_forward(q_ptr, k_ptr, v_ptr, output_ptr, batch, seq_q, seq_k, num_heads, num_kv_heads, 
            dim, q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride, q_head_stride, k_head_stride,
            v_head_stride, o_head_stride, q_row_stride, k_row_stride, v_row_stride, o_row_stride,
            softmax_scale, is_causal, -1, -1, stream);


    cudaStreamDestroy(stream);
    return 0;
}
