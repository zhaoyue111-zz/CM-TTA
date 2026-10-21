#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=outputs/HBP_nonoverlap_original

COMMON_ARGS=(
  --data_dir /data/hilab/datasets/WCH_MRI145/HBP
  --split_file /data/hilab/datasets/WCH_MRI145/HBP/split.json
  --voxtell_root /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk
  --model_dir /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model
  --text_model /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model/Qwen3-Embedding-4B
  --prompt "liver cancer"
  --device cuda:0
  --text_device cuda:1
  --fail_if_output_exists
)

python run_voxtell_cmtta.py "${COMMON_ARGS[@]}" \
  --output_dir "$OUTPUT_ROOT/cac" \
  --no_entropy_rank

REPLAY="$OUTPUT_ROOT/cac/results.json"

python run_voxtell_cmtta.py "${COMMON_ARGS[@]}" \
  --output_dir "$OUTPUT_ROOT/cac_entropy" \
  --view_params_replay "$REPLAY"

python run_voxtell_cmtta.py "${COMMON_ARGS[@]}" \
  --output_dir "$OUTPUT_ROOT/tdc" \
  --view_selection_metric tdc \
  --no_entropy_rank \
  --view_params_replay "$REPLAY"

python run_voxtell_cmtta.py "${COMMON_ARGS[@]}" \
  --output_dir "$OUTPUT_ROOT/tdc_entropy" \
  --view_selection_metric tdc \
  --view_params_replay "$REPLAY"
