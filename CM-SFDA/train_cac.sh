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
PROMPT=${4:-prostate}
OUTPUT_DIR=${5:-results/voxtell_sfda_cac}

exec python "$SCRIPT_DIR/run_sfda_voxtell.py" \
  --data_dir "$DATA_DIR" \
  --voxtell_root "$VOXTELL_ROOT" \
  --model_dir "$MODEL_DIR" \
  --prompt "$PROMPT" \
  --output_dir "$OUTPUT_DIR" \
  --quality_mode cac \
  --quality_config "$SCRIPT_DIR/configs/tse.json" \
  --w_quality 0 \
  --w_cac 0
