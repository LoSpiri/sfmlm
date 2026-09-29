# Overnight run on RunPod (N-SFLM vs N-MDLM, full GSM8K)

## 1. Deploy a pod

1. runpod.io → **Pods** → **Deploy**.
2. GPU: **RTX PRO 4000 (24 GB)**.
3. Template: **RunPod PyTorch** (2.8 or newer).
4. Container disk **30 GB**; volume disk (mounted at `/workspace`) **10 GB**
   for results. Use a pod volume, not a network volume, if you want the
   default `stop` autostop (pods with a network volume can only be
   terminated; use `AUTOSTOP=terminate` then).
5. Optional but recommended: add a Pod environment variable
   `RUNPOD_API_KEY` (fallback if the in-pod `runpodctl` cannot stop the pod).

## 2. Start (web terminal)

```bash
cd /root
git clone https://github.com/LoSpiri/sfmlm.git
cd sfmlm
bash runpod_night.sh
tail -f /workspace/results/sfmlm/night.log
```

The script detaches itself; you can close the terminal. It clones s-flm into
`/root/s-flm`, installs deps, downloads the two checkpoints (S-FLM
sphere-arch, MDLM), runs the unit tests and a CUDA check, then runs the
prioritized sweep.

## 3. Autostop

The pod is stopped (GPU released, `/workspace` kept) when:

- the sweep finishes,
- any setup step fails (clone, pip, download, tests, no GPU),
- `MAX_HOURS` (default 10) have passed — watchdog, even if the sweep hangs.

Overrides: `MAX_HOURS=8 bash runpod_night.sh`, `AUTOSTOP=terminate` (network
volume), `AUTOSTOP=none` (debugging). The log shows a loud warning at start
if neither `runpodctl` nor `RUNPOD_API_KEY` is available.

## 4. What runs

56 jobs, one (model, mode, steps, K) each, in priority order so a partial
night is still useful:

1. S=16, K=2 — float + `nmdlm`, `delta_bitwise`, `delta_residual`, both models
2. S=16, K=1 and K=4
3. S=8, K=2
4. S=16, K=8; S=8, K=1 and K=4
5. S=32, K=2
6. S=16, K=2 `nmdlm` with the LM head converted

Finished jobs are skipped on re-run (`bash runpod_night.sh` again resumes).
`schedule.log` has per-job status; `summary.md` is refreshed after every job.

## 5. Next morning

Results are in `/workspace/results/sfmlm/` on the stopped pod's volume.
Start the pod (CPU-only is enough) or copy them out, e.g. from the web
terminal `cat /workspace/results/sfmlm/summary.md`, or zip the folder and
download it via the Jupyter file browser.
