import pytest
import torch

from sfmlm.constants.neuromorphic import T_MAX
from sfmlm.neuromorphic.codec import (
  binary_popcount, decode, encode_binary, encode_naf, min_timesteps,
  naf_popcount)
from sfmlm.neuromorphic.injector import NeuromorphicInjector, select_layers
from sfmlm.neuromorphic.linear_spike import spiking_linear
from sfmlm.neuromorphic.quantize import (
  adaptive_vth, max_level, quantize, quantize_activations)

LIM = max_level(T_MAX)
ALL_INTS = torch.arange(-LIM, LIM + 1)


def _reference_naf(n):
  digits, n = [], abs(n)
  while n:
    if n & 1:
      d = 2 - (n % 4)
      n -= d
    else:
      d = 0
    digits.append(d)
    n >>= 1
  return digits


def test_binary_round_trip_exact():
  assert torch.equal(decode(encode_binary(ALL_INTS, T_MAX)), ALL_INTS)


def test_naf_round_trip_exact_and_non_adjacent():
  digits = encode_naf(ALL_INTS, T_MAX + 1)
  assert torch.equal(decode(digits), ALL_INTS)
  nz = digits != 0
  assert not (nz[:, 1:] & nz[:, :-1]).any()


def test_naf_matches_reference_and_is_sparser():
  for n in ALL_INTS.tolist():
    ref = _reference_naf(n)
    assert naf_popcount(torch.tensor(n)).item() == sum(d != 0 for d in ref)
  assert (naf_popcount(ALL_INTS) <= binary_popcount(ALL_INTS)).all()


def test_popcount_matches_digits():
  assert torch.equal(binary_popcount(ALL_INTS),
                     (encode_binary(ALL_INTS, T_MAX) != 0).sum(-1).int())


def test_min_timesteps_is_bit_length():
  assert [min_timesteps(m) for m in (0, 1, 2, 3, 4, 5, 255, 256)] == \
    [0, 1, 2, 2, 3, 3, 8, 9]
  for m in range(1, 600):
    t = min_timesteps(m)
    assert m <= 2 ** t - 1
    assert m > 2 ** (t - 1) - 1
  # The paper's floor(log2 m) cannot represent m = 5 with ternary digits.
  with pytest.raises(ValueError):
    encode_binary(torch.tensor([5]), 2)


def test_spiking_linear_equals_quantized_linear():
  torch.manual_seed(0)
  a = torch.randn(3, 5, 16, dtype=torch.float64)
  w = torch.randn(7, 16, dtype=torch.float64)
  b = torch.randn(7, dtype=torch.float64)
  q, vth, _ = quantize_activations(a, K=4)
  digits = encode_binary(q, T_MAX).to(torch.float64)
  y_spike = spiking_linear(digits, w, vth, b)
  y_ref = torch.nn.functional.linear(vth * q, w, b)
  assert torch.allclose(y_spike, y_ref, rtol=0, atol=1e-10)


def test_clipping_applied_and_counted():
  a = torch.ones(1, 64)
  a[0, 0] = 1e4
  vth = adaptive_vth(a, K=8)
  q, clipped = quantize(a, vth)
  assert q.abs().max().item() == LIM
  assert clipped.item() == 1


def test_vth_is_per_token():
  row = torch.randn(32)
  a = torch.stack([row, 10 * row])
  vth = adaptive_vth(a, K=2)
  assert vth.shape == (2, 1)
  assert torch.allclose(vth[1], 10 * vth[0])
  q, _, _ = quantize_activations(a, K=2)
  assert torch.equal(q[0], q[1])


class _Toy(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(16, 8)
    self.sigma_map = torch.nn.Linear(4, 4)
    self.lm_head = torch.nn.Linear(8, 10)

  def forward(self, x):
    return self.lm_head(self.fc(x))


def test_layer_selection():
  toy = _Toy()
  assert set(select_layers(toy)) == {'fc'}
  assert set(select_layers(toy, include_head=True)) == {'fc', 'lm_head'}


def _run_steps(inj, toy, xs):
  inj.new_sequence()
  return [toy(x) for x in xs]


def test_float_mode_is_identity():
  torch.manual_seed(0)
  toy = _Toy()
  x = torch.randn(2, 5, 16)
  ref = toy(x)
  inj = NeuromorphicInjector(toy, mode='float').attach()
  assert torch.equal(toy(x), ref)
  assert inj.stats() == {}
  inj.detach()


def test_nmdlm_mode_matches_manual_quantization():
  torch.manual_seed(0)
  toy = _Toy()
  x = torch.randn(2, 5, 16)
  q, vth, _ = quantize_activations(x, K=2)
  ref = toy.lm_head(toy.fc(vth * q))
  inj = NeuromorphicInjector(toy, mode='nmdlm', K=2).attach()
  assert torch.allclose(toy(x), ref)
  st = inj.stats()[('fc', 0)]
  assert st['spikes_binary'] == int(binary_popcount(q).sum())
  assert st['rate_count'] == int(q.abs().sum())
  assert st['n_tokens'] == 10 and st['n_seqs'] == 2
  inj.detach()


def test_delta_bitwise_equals_frozen_vth_quantization():
  torch.manual_seed(0)
  toy = _Toy()
  xs = [torch.randn(2, 5, 16)]
  for _ in range(3):
    xs.append(xs[-1] + 0.05 * torch.randn(2, 5, 16))
  vth0 = adaptive_vth(xs[0], K=2)
  inj = NeuromorphicInjector(toy, mode='delta_bitwise', K=2).attach()
  outs = _run_steps(inj, toy, xs)
  inj.detach()
  inj_n = NeuromorphicInjector(toy, mode='nmdlm', K=2).attach()
  first_nmdlm = toy(xs[0])
  inj_n.detach()
  assert torch.allclose(outs[0], first_nmdlm)
  for x, out in zip(xs, outs):
    q, _ = quantize(x, vth0)
    assert torch.allclose(out, toy.lm_head(toy.fc(vth0 * q)))
  st = inj.stats()
  first = st[('fc', 0)]['spikes_binary']
  later = st[('fc', 3)]['spikes_binary']
  assert later < first


def test_delta_residual_tracks_input():
  torch.manual_seed(0)
  toy = _Toy()
  xs = [torch.randn(2, 5, 16)]
  for _ in range(3):
    xs.append(xs[-1] + 0.05 * torch.randn(2, 5, 16))
  captured = []
  inj = NeuromorphicInjector(toy, mode='delta_residual', K=2).attach()
  h = toy.fc.register_forward_pre_hook(
    lambda m, args: captured.append(args[0].clone()))
  _run_steps(inj, toy, xs)
  inj.detach()
  h.remove()
  for x, y in zip(xs, captured):
    vth = adaptive_vth(x, K=2)
    assert ((y - x).abs() <= vth / 2 + 1e-6).all()


def test_new_sequence_resets_step_counter():
  toy = _Toy()
  x = torch.randn(1, 3, 16)
  inj = NeuromorphicInjector(toy, mode='nmdlm', K=2).attach()
  _run_steps(inj, toy, [x, x])
  _run_steps(inj, toy, [x, x])
  steps = sorted(k[1] for k in inj.stats() if k[0] == 'fc')
  assert steps == [0, 1]
  assert inj.stats()[('fc', 0)]['n_seqs'] == 2
  inj.detach()
