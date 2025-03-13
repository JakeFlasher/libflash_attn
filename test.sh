#!/bin/bash
for X in {0..100..10}
do
  FLASH_ATTN_SMEM_CARVEOUT=$X ncu --set full \
    --export ncu_reports/test/l1config-$X.ncu-rep \
./build/bench --attention_type=basic --batch_size=2 --seq_q=4096 --seq_k=4096 --num_heads=256 --head_dim=64 --num_kv_heads 8 --iterations=50 --softmax_scale 1
done
