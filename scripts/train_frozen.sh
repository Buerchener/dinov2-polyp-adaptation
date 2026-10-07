#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" scripts/train.py --arm Baseline_Fusion "$@"
