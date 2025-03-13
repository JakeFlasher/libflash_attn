#!/bin/bash
export cuda_visible_devices=1
for X in {0..100..10}
do
  FLASH_ATTN_SMEM_CARVEOUT=$X ncu --set full \
    --export ncu_reports/test_large_dim/l1config-$X.ncu-rep \
./build/bench --attention_type=basic --batch_size=8 --seq_q=4096 --seq_k=4096 --num_heads=256 --head_dim=128 --num_kv_heads 128 --iterations=50 --softmax_scale 1
done
