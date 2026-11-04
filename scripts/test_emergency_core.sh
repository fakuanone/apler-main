#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${MANIFEST:-data/paper_split/manifest.json}"
: "${CKPT_DIR:?Set CKPT_DIR to the selected paper-v1 checkpoint}"
python test.py --manifest "$MANIFEST" --checkpoint "$CKPT_DIR" --split "${EVAL_SPLIT:-test}" --output "${REPORT_PATH:-outputs/paper_test.json}" "$@"
