#!/usr/bin/env bash
set -euo pipefail

# 1. CAC + entropy rank
python run_voxtell_cmtta.py \
  --data_dir /data/zy/CT_MRI_DATA_3D \
  --voxtell_root /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --text_model /home/SENSETIME/yangtingting/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
  --prompt liver \
  --device cuda:0 \
  --output_dir results_/voxtell_cmtta_cac_entropy \
  --lr 0.005 \
  --num_aug_views 9 \
  --view_batch_size 1 \
  --ema_momentum 0.99 \
  --short_memory_length 16 \
  --w_cac 1.0 \
  --w_entropy 0.1 \
  --use_entropy_rank \
  --print_freq 1

# 2. TDC + entropy rank
python run_voxtell_cmtta.py \
  --data_dir /data/zy/CT_MRI_DATA_3D \
  --voxtell_root /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --text_model /home/SENSETIME/yangtingting/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
  --prompt liver \
  --device cuda:0 \
  --output_dir results_/voxtell_cmtta_tdc_entropy \
  --lr 0.005 \
  --num_aug_views 9 \
  --view_batch_size 1 \
  --ema_momentum 0.99 \
  --short_memory_length 16 \
  --w_cac 1.0 \
  --w_entropy 0.1 \
  --view_selection_metric tdc \
  --use_entropy_rank \
  --print_freq 1

# 3. CAC-only rank
python run_voxtell_cmtta.py \
  --data_dir /data/zy/CT_MRI_DATA_3D \
  --voxtell_root /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --text_model /home/SENSETIME/yangtingting/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
  --prompt liver \
  --device cuda:0 \
  --output_dir results_/voxtell_cmtta_cac_only \
  --lr 0.005 \
  --num_aug_views 9 \
  --view_batch_size 1 \
  --ema_momentum 0.99 \
  --short_memory_length 16 \
  --w_cac 1.0 \
  --w_entropy 0.1 \
  --view_selection_metric cac \
  --no_entropy_rank \
  --print_freq 1

# 4. TDC-only rank
python run_voxtell_cmtta.py \
  --data_dir /data/zy/CT_MRI_DATA_3D \
  --voxtell_root /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --text_model /home/SENSETIME/yangtingting/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
  --prompt liver \
  --device cuda:0 \
  --output_dir results_/voxtell_cmtta_tdc_only \
  --lr 0.005 \
  --num_aug_views 9 \
  --view_batch_size 1 \
  --ema_momentum 0.99 \
  --short_memory_length 16 \
  --w_cac 1.0 \
  --w_entropy 0.1 \
  --view_selection_metric tdc \
  --no_entropy_rank \
  --print_freq 1
