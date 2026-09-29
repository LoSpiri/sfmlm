import pytest

from sfmlm.constants.hardware import HARDWARE_PRESETS
from sfmlm.roofline.nmdlm import (
  ar_costs, dense_costs, evaluate, spiking_costs)

DIMS = {'fc': (64, 128)}
L, S, N_SEQS, GEN = 32, 4, 2, 48
T = 8


def _stats(sigma, row_active):
  per_forward_tokens = N_SEQS * L
  spikes = round((1 - sigma) * T * 64 * per_forward_tokens)
  return {('fc', s): {'spikes_binary': spikes, 'active_channels':
                      round(row_active * N_SEQS * 64), 'n_seqs': N_SEQS}
          for s in range(S)}


def test_spiking_ops_match_eq12():
  sigma = 0.75
  res = evaluate(spiking_costs(_stats(sigma, 1.0), DIMS), GEN,
                 HARDWARE_PRESETS['ICMS'])
  expected = 2 * T * 64 * 128 * (1 - sigma) * L * S * N_SEQS / GEN
  assert res['ops_per_token'] == pytest.approx(expected, rel=1e-6)


def test_dense_equals_spiking_with_one_digit_and_no_sparsity():
  dense = evaluate(dense_costs(DIMS, L, S, N_SEQS), GEN,
                   HARDWARE_PRESETS['OCMS'])
  stats = {('fc', s): {'spikes_binary': 64 * N_SEQS * L,
                       'active_channels': 64 * N_SEQS, 'n_seqs': N_SEQS}
           for s in range(S)}
  spk = evaluate(spiking_costs(stats, DIMS), GEN, HARDWARE_PRESETS['OCMS'])
  assert spk['ops_per_token'] == pytest.approx(dense['ops_per_token'])
  assert spk['bits_per_token'] == pytest.approx(dense['bits_per_token'])


def test_row_activity_scales_memory():
  full = evaluate(spiking_costs(_stats(0.5, 1.0), DIMS), GEN,
                  HARDWARE_PRESETS['ICMS'])
  half = evaluate(spiking_costs(_stats(0.5, 0.5), DIMS), GEN,
                  HARDWARE_PRESETS['ICMS'])
  assert half['bits_per_token'] == pytest.approx(0.5 * full['bits_per_token'])


def test_ar_is_memory_bound_on_ocms():
  res = evaluate(ar_costs(DIMS, GEN), GEN, HARDWARE_PRESETS['OCMS'])
  assert res['compute_bound_time_frac'] == 0.0
  assert res['bits_per_token'] == pytest.approx(64 * 128 * 16)
