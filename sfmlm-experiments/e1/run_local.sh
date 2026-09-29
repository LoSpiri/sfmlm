#!/usr/bin/env bash
# Usage: [STEPS=8] ./run_local.sh [smoke|full|summary]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$ROOT/../s-flm/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
STAGE="${1:-smoke}"
STEPS="${STEPS:-8}"

case "$STAGE" in
  smoke)
    "$PY" "$HERE/run_nmdlm_compare.py" --steps "$STEPS" --subset 128 --K 2 \
      --tag smoke
    "$PY" "$HERE/summarize.py" --pattern '*_smoke.json' \
      --out "$HERE/results/summary_smoke.md"
    ;;
  full)
    "$PY" "$HERE/run_nmdlm_compare.py" --steps "$STEPS" --subset 1319 \
      --K 1,2,4,8
    "$PY" "$HERE/summarize.py" --pattern '*_n1319.json' \
      --out "$HERE/results/summary_full.md"
    ;;
  summary)
    "$PY" "$HERE/summarize.py" --out "$HERE/results/summary_all.md"
    ;;
  *)
    echo "unknown stage: $STAGE (smoke|full|summary)" >&2
    exit 1
    ;;
esac
