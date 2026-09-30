"""Bridge to the s-flm codebase: model recipes, GSM8K data and scoring.

s-flm uses cwd-relative paths for checkpoints and data, and its e1 helpers
import each other as top-level modules, so both the repo root and e1/ go on
sys.path and the process chdirs into the repo.
"""
import os
import sys

import torch

from sfmlm.config import Config

_entered = False


def enter_sflm():
  global _entered
  root = Config.SFLM_ROOT
  if not (root / 'algo.py').is_file():
    raise FileNotFoundError(
      f's-flm not found at {root}; set SFLM_ROOT to the s-flm checkout.')
  if not _entered:
    for p in (root / 'e1', root):
      if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    os.chdir(root)
    _entered = True
  return root


def build_model(name, *, steps, length, temperature, batch):
  root = enter_sflm()
  import loader as sflm_loader
  ckpt = root / sflm_loader.CKPT[name]
  if not ckpt.is_file():
    raise FileNotFoundError(f'Missing checkpoint for {name}: {ckpt}')
  return sflm_loader.build(name, steps=steps, length=length,
                           temperature=temperature,
                           eval_batch_size=batch)


def load_gsm8k(cfg, tokenizer, subset):
  root = enter_sflm()
  # Pin data/cache to absolute paths under the s-flm checkout so loading does
  # not depend on the process cwd (s-flm's config uses cwd-relative paths).
  cfg.data.data_path = str(root / 'data' / 'gsm8k_test.json')
  cfg.data.cache_dir = str(root / 'data_cache')
  import dataloader
  dataset = dataloader.get_dataset(cfg, tokenizer, mode='valid')
  n = min(subset, len(dataset))
  ids = [torch.tensor(dataset[i]['input_ids']) for i in range(n)]
  golds = [dataset[i]['response_ground_truth'] for i in range(n)]
  return ids, golds


def pad_prefix_batch(id_lists, device):
  enter_sflm()
  from run_e2 import pad_prefix_batch as _pad
  return _pad(id_lists, device)


def score_all(pairs, workers):
  enter_sflm()
  from parallel_eval import score_all as _score
  return _score(pairs, workers=workers)


def seed_everything(seed):
  torch.manual_seed(seed)
  if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
