#!/usr/bin/env bash
# Overnight RunPod run: N-SFLM vs N-MDLM full-GSM8K sweep (sfmlm e1).
#
# Tuned for RTX PRO 4000 (24 GB, 12 vCPU): batch 16 + 12 scoring workers.
# Run from the sfmlm clone on the CONTAINER disk (e.g. /root/sfmlm); results
# go to the volume ($PERSIST_DIR) so they survive a stop / spot preemption.
#
# The script detaches itself (nohup), so closing the web terminal is fine.
# The pod is stopped automatically when the sweep ends, when any step fails,
# and at the latest after MAX_HOURS (watchdog), whichever comes first.
#
# Env overrides:
#   PERSIST_DIR    results dir              (default /workspace/results/sfmlm)
#   SFLM_ROOT      s-flm checkout           (default /root/s-flm, cloned if missing)
#   MAX_HOURS      watchdog hard limit      (default 10)
#   AUTOSTOP       stop | terminate | none  (default stop)
#   WORKERS/BATCH  scoring workers / batch  (default 12 / 16)
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PERSIST_DIR="${PERSIST_DIR:-/workspace/results/sfmlm}"
if ! mkdir -p "$PERSIST_DIR" 2>/dev/null || ! touch "$PERSIST_DIR/.wtest" 2>/dev/null; then
  echo "==> ${PERSIST_DIR} not writable; falling back to ${REPO}/sfmlm-experiments/e1/results"
  PERSIST_DIR="${REPO}/sfmlm-experiments/e1/results"
  mkdir -p "$PERSIST_DIR"
fi
rm -f "$PERSIST_DIR/.wtest"
LOG="${PERSIST_DIR}/night.log"

if [ -z "${SFMLM_DETACHED:-}" ]; then
  SFMLM_DETACHED=1 PERSIST_DIR="$PERSIST_DIR" nohup bash "$0" "$@" >> "$LOG" 2>&1 &
  echo "started (pid $!) — follow with:  tail -f ${LOG}"
  exit 0
fi

# /workspace is geesefs (S3-backed FUSE): Hugging Face snapshot links end up
# as 0-byte files there, so all caches stay on the container disk.
export XDG_CACHE_HOME=/root/.cache
export HF_HOME=/root/.cache/huggingface
unset TRANSFORMERS_CACHE HF_HUB_CACHE

export SFLM_ROOT="${SFLM_ROOT:-/root/s-flm}"
MAX_HOURS="${MAX_HOURS:-10}"
AUTOSTOP="${AUTOSTOP:-stop}"
WORKERS="${WORKERS:-12}"
BATCH="${BATCH:-16}"

stop_pod() {
  [ "$AUTOSTOP" = "none" ] && { echo "==> AUTOSTOP=none: leaving pod running"; return; }
  echo "==> [$(date +%H:%M:%S)] autostop ($AUTOSTOP) pod ${RUNPOD_POD_ID:-?}"
  sync
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    if [ "$AUTOSTOP" = "terminate" ]; then
      runpodctl remove pod "$RUNPOD_POD_ID" && return
    else
      runpodctl stop pod "$RUNPOD_POD_ID" && return
    fi
  fi
  if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    if [ "$AUTOSTOP" = "terminate" ]; then
      curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" && return
    else
      curl -s -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" && return
    fi
  fi
  echo "!!! autostop FAILED: no runpodctl / RUNPOD_API_KEY. Stop the pod manually."
}

echo "==> [$(date)] sfmlm night run; results -> ${PERSIST_DIR}"
if ! command -v runpodctl >/dev/null 2>&1 && [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "!!! WARNING: neither runpodctl nor RUNPOD_API_KEY available: autostop will not work"
fi
trap stop_pod EXIT
( sleep "$((MAX_HOURS * 3600))"; echo "==> watchdog: ${MAX_HOURS}h limit reached"; stop_pod ) &

fail() { echo "!!! $1"; exit 1; }

if [ ! -f "${SFLM_ROOT}/algo.py" ]; then
  echo "==> cloning s-flm into ${SFLM_ROOT}"
  git clone -q https://github.com/LoSpiri/thesis.git "$SFLM_ROOT" || fail "clone s-flm"
fi

echo "==> installing deps"
pip install -q hydra-core==1.3.2 omegaconf==2.3.0 lightning==2.5.1 \
  einops==0.8.1 fancy-einsum==0.0.3 tqdm==4.67.1 rich==13.9.4 termcolor==3.0.1 \
  pyyaml==6.0.2 fsspec blobfile==3.0.0 transformers==4.45.0 tokenizers==0.20.3 \
  datasets==3.5.0 huggingface-hub==0.30.2 safetensors==0.5.2 sentencepiece==0.2.0 \
  evaluate==0.4.3 matplotlib torchmetrics==1.7.1 pandas==2.2.1 scikit-learn==1.4.0 \
  jinja2==3.1.5 timm==1.0.15 scipy hf_transfer pytest || fail "pip install"
pip install -q --no-deps -e "$REPO" || fail "pip install sfmlm"

echo "==> downloading checkpoints"
python -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<PY || fail "checkpoint download"
from huggingface_hub import hf_hub_download
for f in ["tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt",
          "tinygsm/mdlm.ckpt"]:
    hf_hub_download("jdeschena/s-flm", f, local_dir="${SFLM_ROOT}/checkpoints")
    print("ok", f, flush=True)
PY

echo "==> fetching GSM8K test data (data/gsm8k_test.json)"
mkdir -p "${SFLM_ROOT}/data"
python - <<PY || fail "gsm8k data fetch"
import json, urllib.request
url = "https://raw.githubusercontent.com/LoSpiri/thesis/main/data/gsm8k_test.json"
dst = "${SFLM_ROOT}/data/gsm8k_test.json"
urllib.request.urlretrieve(url, dst)
json.load(open(dst))  # verify it parses (fails loudly if empty/corrupt)
print("ok", dst, flush=True)
PY

echo "==> unit tests"
(cd "$REPO" && python -m pytest -q) || fail "unit tests"

echo "==> GPU check"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" \
  || fail "no CUDA device"

echo "==> preflight (2 examples, 2 steps, both models)"
python -m sfmlm.experiments.run_compare --models sfm,mdlm --modes nmdlm \
  --subset 2 --batch 2 --steps 2 --K 2 --workers 1 \
  --out-dir /root/preflight --overwrite || fail "preflight job"

echo "==> sweep (prioritized, resume-safe)"
python "$REPO/sfmlm-experiments/e1/run_night.py" --out-dir "$PERSIST_DIR" \
  --workers "$WORKERS" --batch "$BATCH" || fail "sweep aborted"

echo "==> [$(date)] DONE; summary: ${PERSIST_DIR}/summary.md"
cat "${PERSIST_DIR}/summary.md" 2>/dev/null || true
