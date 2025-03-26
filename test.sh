#!/bin/bash
export cuda_visible_devices=4
# Create directory if it doesn't exist
mkdir -p ncu_reports/test

# Loop through L1 and L2 configurations (5x5 grid)
for L1 in {10..100..15}
do
  for L2 in {10..100..15}
  do
    echo "Testing with L1 carveout = $L1%, L2 carveout = $L2%"
    FLASH_ATTN_SMEM_CARVEOUT=$L1 FLASH_ATTN_L2_CARVEOUT=$L2 ncu -f --set full \
      --export ncu_reports/test/l1config-${L1}_l2config-${L2}.ncu-rep \
    ./build/bench --attention_type=basic --batch_size=2 --seq_q=4096 --seq_k=4096 \
      --num_heads=256 --head_dim=64 --num_kv_heads 8 --iterations=50 --softmax_scale 1
  done
done

# Run analysis scripts automatically
echo "Processing NCU reports..."
python3 read_ncu_rep.py --reports-dir ncu_reports
# echo "Generating plots..."
# python3 plot.py --root-dir ncu_reports
