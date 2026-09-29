"""Adaptive per-token quantization (N-MDLM Eq. 7) with explicit clipping."""
import torch

from sfmlm.constants.neuromorphic import T_MAX, VTH_EPS


def max_level(t_max=T_MAX):
  return 2 ** t_max - 1


def adaptive_vth(a, K):
  """V_th = mean(|a|) / K over the last (feature) dim, one per token."""
  return (a.abs().mean(dim=-1, keepdim=True) / K).clamp_min(VTH_EPS)


def quantize(a, vth, t_max=T_MAX):
  """Returns (integer-valued float tensor, number of clipped entries)."""
  lim = max_level(t_max)
  r = torch.round(a / vth)
  clipped = (r.abs() > lim).sum()
  return r.clamp(-lim, lim), clipped


def quantize_activations(a, K, t_max=T_MAX):
  vth = adaptive_vth(a, K)
  q, clipped = quantize(a, vth, t_max)
  return q, vth, clipped
