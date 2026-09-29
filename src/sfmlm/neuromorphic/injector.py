"""Neuromorphic conversion of a backbone via nn.Linear forward pre-hooks.

Each converted Linear receives V_th * a_hat instead of its float input,
which equals the output of the bitwise spiking layer (see linear_spike).

Modes:
  float          -- identity
  nmdlm          -- N-MDLM: per-token V_th, stateless across sampler steps;
                    transmits a_hat
  delta_bitwise  -- V_th frozen per (sequence, position) at the first sampler
                    step; transmits a_hat_s - a_hat_{s-1}. Reconstruction is
                    V_th0 * a_hat_s (lossless delta of the quantized value)
  delta_residual -- V_th re-estimated per step; transmits
                    q_s = clip(round((a_s - y_{s-1}) / V_th_s)) and
                    reconstructs y_s = y_{s-1} + V_th_s * q_s

Call new_sequence() before each generate_samples() batch: the step counter
and delta state are per batch, while statistics accumulate until
reset_stats(). Counts stay on device until stats() is called.
"""
import torch

from sfmlm.constants.neuromorphic import (
  EXCLUDED_LAYER_PATTERNS, HEAD_LAYER_NAMES, MODES, MODE_DELTA_BITWISE,
  MODE_DELTA_RESIDUAL, MODE_FLOAT, MODE_NMDLM, STAT_FIELDS, T_MAX)
from sfmlm.neuromorphic.codec import binary_popcount, naf_popcount
from sfmlm.neuromorphic.quantize import adaptive_vth, quantize


def select_layers(backbone, include_head=False):
  layers = {}
  for name, module in backbone.named_modules():
    if not isinstance(module, torch.nn.Linear):
      continue
    if any(p in name for p in EXCLUDED_LAYER_PATTERNS):
      continue
    if not include_head and name in HEAD_LAYER_NAMES:
      continue
    layers[name] = module
  return layers


class NeuromorphicInjector:
  def __init__(self, backbone, mode=MODE_FLOAT, K=2.0, t_max=T_MAX,
               include_head=False):
    self.backbone = backbone
    self.K = K
    self.t_max = t_max
    self.layers = select_layers(backbone, include_head)
    self.layer_dims = {n: (m.in_features, m.out_features)
                       for n, m in self.layers.items()}
    self.hooks = []
    self.set_mode(mode)
    self.reset_stats()

  def set_mode(self, mode, K=None):
    if mode not in MODES:
      raise ValueError(mode)
    self.mode = mode
    if K is not None:
      self.K = K
    self.new_sequence()

  def new_sequence(self):
    self._seq = {}

  def reset_stats(self):
    self._acc = {}
    self.new_sequence()

  def attach(self):
    for name, module in self.layers.items():
      self.hooks.append(module.register_forward_pre_hook(self._hook(name)))
    return self

  def detach(self):
    for h in self.hooks:
      h.remove()
    self.hooks = []

  def _hook(self, name):
    def f(module, args):
      a = args[0]
      st = self._seq.setdefault(name, {'step': 0})
      step = st['step']
      st['step'] = step + 1
      if self.mode == MODE_FLOAT:
        return None
      y, sent, clipped = self._convert(a.float(), st)
      self._accumulate(name, step, sent, clipped)
      return (y.to(a.dtype),)
    return f

  def _convert(self, x, st):
    if self.mode == MODE_NMDLM:
      vth = adaptive_vth(x, self.K)
      q, clipped = quantize(x, vth, self.t_max)
      return vth * q, q, clipped
    if self.mode == MODE_DELTA_BITWISE:
      if 'vth' not in st:
        st['vth'] = adaptive_vth(x, self.K)
        st['prev'] = torch.zeros_like(x, dtype=torch.int16)
      q, clipped = quantize(x, st['vth'], self.t_max)
      qi = q.to(torch.int16)
      sent = qi - st['prev']
      st['prev'] = qi
      return st['vth'] * q, sent, clipped
    if self.mode == MODE_DELTA_RESIDUAL:
      if 'y' not in st:
        st['y'] = torch.zeros_like(x)
      vth = adaptive_vth(x, self.K)
      q, clipped = quantize(x - st['y'], vth, self.t_max)
      st['y'] = st['y'] + vth * q
      return st['y'], q, clipped
    raise ValueError(self.mode)

  def _accumulate(self, name, step, sent, clipped):
    d_in = sent.shape[-1]
    per_seq = sent.reshape(sent.shape[0] if sent.ndim >= 3 else 1, -1, d_in)
    si = per_seq.to(torch.int32)
    vec = torch.stack([
      binary_popcount(si).sum(dtype=torch.int64),
      naf_popcount(si).sum(dtype=torch.int64),
      si.abs().sum(dtype=torch.int64),
      torch.tensor(per_seq.shape[0] * per_seq.shape[1], device=si.device),
      clipped.to(torch.int64),
      (si != 0).any(dim=1).sum(dtype=torch.int64),
      torch.tensor(per_seq.shape[0], device=si.device),
    ]).to(torch.int64)
    key = (name, step)
    self._acc[key] = vec if key not in self._acc else self._acc[key] + vec

  def stats(self):
    """{(layer, step): {field: int}} materialized on CPU."""
    return {key: dict(zip(STAT_FIELDS, vec.cpu().tolist()))
            for key, vec in self._acc.items()}
