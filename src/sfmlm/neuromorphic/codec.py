"""Ternary bitwise spike codes for integer activations (N-MDLM Eq. 8).

Digit t (0-based) carries weight 2**t, i.e. the paper's s_{i,t+1}.
Sign-magnitude binary is the paper-faithful code; the non-adjacent form
(NAF) uses the minimal number of nonzero ternary digits but may need one
more digit than binary.
"""
import torch

from sfmlm.constants.neuromorphic import POPCOUNT_BITS


def min_timesteps(m):
  """Binary digits needed for |a_hat| <= m: floor(log2 m) + 1 (0 for m=0)."""
  return int(m).bit_length()


def _naf_parts(x):
  """For x >= 0 returns (pos, neg) bitmasks with x = pos - neg in NAF."""
  xh = x >> 1
  x3 = x + xh
  c = xh ^ x3
  return x3 & c, xh & c


def encode_binary(q, t):
  x = q.to(torch.int64)
  sign = torch.sign(x)
  mag = x.abs()
  if bool((mag >= 2 ** t).any()):
    raise ValueError(f'values exceed {t}-digit binary range')
  bits = torch.stack([(mag >> i) & 1 for i in range(t)], dim=-1)
  return bits * sign.unsqueeze(-1)


def encode_naf(q, t):
  x = q.to(torch.int64)
  sign = torch.sign(x)
  pos, neg = _naf_parts(x.abs())
  if bool(((pos | neg) >= 2 ** t).any()):
    raise ValueError(f'values exceed {t}-digit NAF range')
  digits = torch.stack(
    [((pos >> i) & 1) - ((neg >> i) & 1) for i in range(t)], dim=-1)
  return digits * sign.unsqueeze(-1)


def decode(digits):
  t = digits.shape[-1]
  weights = 2 ** torch.arange(t, device=digits.device, dtype=digits.dtype)
  return (digits * weights).sum(dim=-1)


def _popcount(mask, n_bits):
  count = torch.zeros_like(mask)
  for i in range(n_bits):
    count = count + ((mask >> i) & 1)
  return count


def binary_popcount(q, n_bits=POPCOUNT_BITS):
  """Elementwise number of spikes of the sign-magnitude binary code."""
  return _popcount(q.to(torch.int32).abs(), n_bits)


def naf_popcount(q, n_bits=POPCOUNT_BITS):
  """Elementwise number of spikes of the NAF code."""
  x = q.to(torch.int32).abs()
  xh = x >> 1
  return _popcount(xh ^ (x + xh), n_bits + 1)
