"""Unattended, resumable sweep of run_compare jobs in priority order.

Each job is one (model, mode, steps, K) in its own subprocess, so a crash
or timeout loses one job only and a partial night still yields the most
important results first. Finished jobs (result file present) are skipped;
the summary table is refreshed after every job.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from sfmlm.config import Config
from sfmlm.constants.experiment import (
  DEFAULT_BATCH, DEFAULT_SUBSET, DEFAULT_WORKERS, MODEL_IDS)
from sfmlm.constants.neuromorphic import (
  MODE_NMDLM, MODE_FLOAT, SPIKING_MODES)
from sfmlm.experiments.run_compare import result_path

# (steps, K values, extra tag, extra args); float runs once per steps.
# Trimmed to fit ~4-5 h on an RTX PRO 4000: NFE {8,16} x K {1,2,4}, both
# models, all spiking modes + float. (Dropped: K=8, NFE=32, --include-head.)
PRIORITY_TIERS = (
  (16, (2,), '', ()),
  (16, (1, 4), '', ()),
  (8, (2,), '', ()),
  (8, (1, 4), '', ()),
)
JOB_TIMEOUT_BASE_S = 1800
JOB_TIMEOUT_PER_STEP_S = 450
# Consecutive failures usually mean a broken environment, not a bad job.
MAX_CONSECUTIVE_FAILURES = 3


def _jobs(models, subset):
  jobs, seen_float = [], set()
  for steps, Ks, extra_tag, extra in PRIORITY_TIERS:
    for model in models:
      if (model, steps) not in seen_float and not extra:
        seen_float.add((model, steps))
        jobs.append((model, MODE_FLOAT, steps, None, '', ()))
      modes = (MODE_NMDLM,) if extra else SPIKING_MODES
      for K in Ks:
        for mode in modes:
          jobs.append((model, mode, steps, K, f'_K{K}{extra_tag}', extra))
  return jobs


def _log(msg, logfile):
  line = f'[{time.strftime("%H:%M:%S")}] {msg}'
  print(line, flush=True)
  with open(logfile, 'a') as f:
    f.write(line + '\n')


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--out-dir', type=Path, default=Config.RESULTS_DIR)
  p.add_argument('--models', default=','.join(MODEL_IDS))
  p.add_argument('--subset', type=int, default=DEFAULT_SUBSET)
  p.add_argument('--batch', type=int, default=DEFAULT_BATCH)
  p.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
  p.add_argument('--dry-run', action='store_true')
  args = p.parse_args(argv)
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  logfile = out_dir / 'schedule.log'
  job_logs = out_dir / 'logs'
  job_logs.mkdir(exist_ok=True)
  jobs = _jobs(args.models.split(','), args.subset)
  _log(f'=== {len(jobs)} jobs -> {out_dir} ===', logfile)
  failures = 0

  for i, (model, mode, steps, K, tag, extra) in enumerate(jobs, 1):
    out = result_path(out_dir, model, mode, steps, args.subset, tag)
    name = f'[{i}/{len(jobs)}] {out.name}'
    if out.exists():
      _log(f'{name} skip', logfile)
      continue
    cmd = [sys.executable, '-m', 'sfmlm.experiments.run_compare',
           '--models', model, '--modes', mode, '--steps', str(steps),
           '--subset', str(args.subset), '--batch', str(args.batch),
           '--workers', str(args.workers), '--out-dir', str(out_dir),
           *extra]
    if K is not None:
      cmd += ['--K', str(K), '--tag', tag.lstrip('_')]
    if args.dry_run:
      _log(f'{name} DRY {" ".join(cmd[1:])}', logfile)
      continue
    timeout = JOB_TIMEOUT_BASE_S + JOB_TIMEOUT_PER_STEP_S * steps
    job_log = job_logs / f'{out.stem}.log'
    t0 = time.time()
    ok = False
    try:
      r = subprocess.run(cmd, timeout=timeout, capture_output=True,
                         text=True)
      output = r.stdout + r.stderr
      job_log.write_text(output)
      ok = r.returncode == 0 and out.exists()
      _log(f'{name} {"OK" if ok else f"FAIL rc={r.returncode}"} '
           f'({time.time() - t0:.0f}s) log: {job_log}', logfile)
      for line in output.strip().splitlines()[-4:]:
        _log(f'    {line.strip()[:200]}', logfile)
    except subprocess.TimeoutExpired as e:
      job_log.write_text(str(e.stdout or '') + str(e.stderr or ''))
      _log(f'{name} TIMEOUT after {timeout}s log: {job_log}', logfile)
    failures = 0 if ok else failures + 1
    if failures >= MAX_CONSECUTIVE_FAILURES:
      _log(f'=== {failures} consecutive failures: aborting ===', logfile)
      sys.exit(1)
    subprocess.run([sys.executable, '-m', 'sfmlm.experiments.summarize',
                    '--results-dir', str(out_dir),
                    '--out', str(out_dir / 'summary.md')],
                   capture_output=True)
  _log('=== done ===', logfile)


if __name__ == '__main__':
  main()
