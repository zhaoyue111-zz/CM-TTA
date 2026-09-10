#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "train.sh is deprecated and intentionally disabled; choose train_cac.sh or train_tse.sh explicitly." >&2
echo "Example: bash \"$SCRIPT_DIR/train_cac.sh\" DATA_DIR VOXTELL_ROOT MODEL_DIR [PROMPT] [OUTPUT_DIR]" >&2
exit 2
