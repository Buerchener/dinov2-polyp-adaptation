#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/train.py --arm C_LoRA_LR5e5 "$@"
