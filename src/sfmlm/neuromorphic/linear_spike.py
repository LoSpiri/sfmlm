"""Linear layer driven by bitwise spike trains.

By linearity, W a_hat = sum_t 2**t W s_t exactly, so a spiking linear layer
reproduces V_th * W a_hat with no approximation beyond input quantization.
"""
import torch


def spiking_linear(digits, weight, vth, bias=None):
  """digits: [..., D_in, T] ternary; weight: [D_out, D_in]; vth: [..., 1]."""
  t = digits.shape[-1]
  scales = 2.0 ** torch.arange(t, device=digits.device, dtype=weight.dtype)
  currents = torch.einsum('...it,oi->...ot', digits.to(weight.dtype), weight)
  y = vth * (currents * scales).sum(dim=-1)
  return y if bias is None else y + bias
