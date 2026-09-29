"""GSM8K comparison of N-SFLM (S-FLM sphere-arch) vs N-MDLM (MDLM).

For each model and mode, generates with every converted nn.Linear input
replaced by its spiking reconstruction, scores accuracy, and records spike
statistics plus roofline throughput / energy. One JSON per (model, mode) so
interrupted sweeps resume.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from sfmlm.config import Config
from sfmlm.constants.experiment import (
  DEFAULT_BATCH, DEFAULT_LENGTH, DEFAULT_STEPS, DEFAULT_SUBSET,
  DEFAULT_TEMPERATURE, DEFAULT_WORKERS, MODEL_IDS, SEED)
from sfmlm.constants.hardware import HARDWARE_PRESETS
from sfmlm.constants.neuromorphic import (
  DEFAULT_K_SWEEP, MODES, MODE_FLOAT, MODE_TIMESTEPS, T_MAX)
from sfmlm.experiments import loader
from sfmlm.neuromorphic.injector import NeuromorphicInjector
from sfmlm.neuromorphic.stats import summarize
from sfmlm.roofline import nmdlm as roofline


def _generate(model, injector, ids, golds, args):
  loader.seed_everything(SEED)
  pairs, gen_tokens, n_seqs = [], 0, 0
  for start in range(0, len(ids), args.batch):
    b_ids = ids[start:start + args.batch]
    padded, lengths = loader.pad_prefix_batch(b_ids, model.device)
    injector.new_sequence()
    with torch.no_grad():
      samples, _ = model.generate_samples(
        num_samples=len(b_ids), num_steps=args.steps,
        prefix_tokens=padded, prefix_lengths=lengths)
    for j, gold in enumerate(golds[start:start + args.batch]):
      pl = int(lengths[j])
      resp = model.tokenizer.decode(samples[j, pl:].cpu().tolist(),
                                    skip_special_tokens=True)
      pairs.append((resp, gold))
    gen_tokens += int((args.length - lengths).sum())
    n_seqs += len(b_ids)
  scores = loader.score_all(pairs, args.workers)
  return sum(scores) / max(len(scores), 1), gen_tokens, n_seqs


def _roofline(injector, cfg, stats, gen_tokens, n_seqs, args):
  dims = injector.layer_dims
  out = {'attention_float_ops_per_token':
         roofline.attention_float_ops_per_token(
           cfg.model.n_blocks, cfg.model.hidden_size, args.length,
           args.steps, n_seqs, gen_tokens)}
  for hw_name, hw in HARDWARE_PRESETS.items():
    workloads = {
      'dense': roofline.dense_costs(dims, args.length, args.steps, n_seqs),
      'ar': roofline.ar_costs(dims, gen_tokens),
    }
    if stats:
      workloads['spiking'] = roofline.spiking_costs(stats, dims)
    out[hw_name] = {k: roofline.evaluate(v, gen_tokens, hw)
                    for k, v in workloads.items()}
  return out


def result_path(out_dir, model_id, mode, steps, n, tag=''):
  return Path(out_dir) / f'{model_id}_{mode}_S{steps}_n{n}{tag}.json'


def run_model(args, model_id, modes, Ks):
  model, tokenizer, cfg = loader.build_model(
    model_id, steps=args.steps, length=args.length,
    temperature=args.temperature, batch=args.batch)
  ids, golds = loader.load_gsm8k(cfg, tokenizer, args.subset)
  injector = NeuromorphicInjector(
    model.backbone, include_head=args.include_head).attach()
  print(f'[{model_id}] {len(ids)} examples, {len(injector.layers)} '
        f'converted linears, device={model.device}', flush=True)

  for mode in modes:
    out_path = result_path(args.out_dir, model_id, mode, args.steps,
                           len(ids), args.tag)
    if out_path.exists() and not args.overwrite:
      print(f'[{model_id}] skip {mode}: {out_path.name} exists', flush=True)
      continue
    result = {'model': model_id, 'mode': mode, 'steps': args.steps,
              'length': args.length, 'subset': len(ids),
              'temperature': args.temperature, 'batch': args.batch,
              't_max': T_MAX, 'include_head': args.include_head,
              'device': str(model.device),
              'layer_dims': {k: list(v)
                             for k, v in injector.layer_dims.items()},
              'runs': []}
    for K in ([None] if mode == MODE_FLOAT else Ks):
      injector.set_mode(mode, K=K)
      injector.reset_stats()
      t0 = time.time()
      acc, gen_tokens, n_seqs = _generate(model, injector, ids, golds, args)
      stats = injector.stats()
      run = {'K': K, 'accuracy': acc, 'gen_tokens': gen_tokens,
             'n_seqs': n_seqs, 'wall_s': time.time() - t0,
             'roofline': _roofline(injector, cfg, stats, gen_tokens,
                                   n_seqs, args)}
      if stats:
        run['spikes'] = summarize(stats, injector.layer_dims,
                                  MODE_TIMESTEPS[mode], gen_tokens)
      result['runs'].append(run)
      msg = f'[{model_id}] {mode} K={K}: acc={acc:.4f}'
      if stats:
        s = run['spikes']
        msg += (f' spikes/tok={s["spikes_binary_per_token"]:.3g}'
                f' sigma={s["sigma"]:.4f} row_active={s["row_active"]:.3f}'
                f' clip={s["clip_rate"]:.2e}')
      print(f'{msg} ({run["wall_s"]:.0f}s)', flush=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(f'[{model_id}] saved {out_path}', flush=True)
  injector.detach()
  del model
  if torch.backends.mps.is_available():
    torch.mps.empty_cache()


def _csv(s, cast):
  return [cast(x) for x in s.split(',') if x]


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--models', default=','.join(MODEL_IDS))
  p.add_argument('--modes', default=','.join(MODES))
  p.add_argument('--K', default=','.join(map(str, DEFAULT_K_SWEEP)))
  p.add_argument('--steps', type=int, default=DEFAULT_STEPS)
  p.add_argument('--length', type=int, default=DEFAULT_LENGTH)
  p.add_argument('--batch', type=int, default=DEFAULT_BATCH)
  p.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
  p.add_argument('--subset', type=int, default=DEFAULT_SUBSET)
  p.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
  p.add_argument('--include-head', action='store_true')
  p.add_argument('--out-dir', type=Path, default=Config.RESULTS_DIR)
  p.add_argument('--tag', default='')
  p.add_argument('--overwrite', action='store_true')
  args = p.parse_args(argv)
  args.out_dir = args.out_dir.resolve()
  if args.tag:
    args.tag = f'_{args.tag}'
  for m in _csv(args.modes, str):
    if m not in MODES:
      p.error(f'unknown mode {m}')
  return args


def main(argv=None):
  args = parse_args(argv)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  modes = _csv(args.modes, str)
  Ks = _csv(args.K, float)
  for model_id in _csv(args.models, str):
    run_model(args, model_id, modes, Ks)


if __name__ == '__main__':
  main()
