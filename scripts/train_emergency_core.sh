#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${MANIFEST:-data/paper_split/manifest.json}"
python train.py --config "${CONFIG:-configs/apler_emergency_data.json}" --manifest "$MANIFEST" --output "${OUTPUT_DIR:-outputs/apler_emergency_data}" "$@"
