#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 DATA_DIR VOXTELL_ROOT MODEL_DIR [PROMPT] [OUTPUT_DIR]" >&2
  exit 2
fi

DATA_DIR=$1
VOXTELL_ROOT=$2
MODEL_DIR=$3
EPOCHS=100
PROMPT=${4:-liver}
OUTPUT_DIR=${5:-results_/voxtell_sfda_tdc}

# Default SFDA TDC experiment: quality-only rank. Add --use_entropy_rank
# explicitly when running the TDC+entropy ablation.
exec python "$SCRIPT_DIR/run_sfda_voxtell.py" \
  --data_dir "$DATA_DIR" \
  --voxtell_root "$VOXTELL_ROOT" \
  --model_dir "$MODEL_DIR" \
  --prompt "$PROMPT" \
  --epochs "$EPOCHS" \
  --output_dir "$OUTPUT_DIR" \
  --quality_metric tdc \
  --quality_mode cac \
  --quality_config "$SCRIPT_DIR/configs/tse.json" \
  --no_entropy_rank \
  --w_quality 0 \
  --w_cac 0
