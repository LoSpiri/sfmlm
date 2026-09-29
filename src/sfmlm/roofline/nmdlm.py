"""Token-level roofline model (N-MDLM Sec. IV), adapted to full-sequence
samplers where every NFE processes all L positions.

A workload is a list of Cost(ops, bits, count): `count` independent
executions of one linear layer, each needing `ops` operations and loading
`bits` of weights. Per-layer latency follows Eq. (14), end-to-end latency
Eq. (15) and energy Eq. (16); both are divided by the generated tokens.

Spiking costs come from measured spikes: 2 * D_out ops per input spike
(Eq. 12 with the popcount in place of T D_in (1 - sigma) per token), and a
weight row is loaded when its input channel spikes anywhere in the sequence
(measured row-active fraction instead of the i.i.d. 1 - sigma^{BT} of Eq. 13).
"""
from dataclasses import dataclass

from sfmlm.constants.hardware import WEIGHT_BITS


@dataclass(frozen=True)
class Cost:
  ops: float
  bits: float
  count: float


def spiking_costs(stats, layer_dims, weight_bits=WEIGHT_BITS):
  costs = []
  for (name, _), s in stats.items():
    d_in, d_out = layer_dims[name]
    n = s['n_seqs']
    row_active = s['active_channels'] / (n * d_in)
    costs.append(Cost(ops=2 * d_out * s['spikes_binary'] / n,
                      bits=d_in * d_out * weight_bits * row_active,
                      count=n))
  return costs


def dense_costs(layer_dims, length, steps, n_seqs, weight_bits=WEIGHT_BITS):
  """Float full-sequence model: every NFE runs all L positions."""
  return [Cost(ops=2 * d_in * d_out * length,
               bits=d_in * d_out * weight_bits,
               count=n_seqs * steps)
          for d_in, d_out in layer_dims.values()]


def ar_costs(layer_dims, gen_tokens, weight_bits=WEIGHT_BITS):
  """AR with KV cache: one position per weight load (prefill ignored)."""
  return [Cost(ops=2 * d_in * d_out,
               bits=d_in * d_out * weight_bits,
               count=gen_tokens)
          for d_in, d_out in layer_dims.values()]


def evaluate(costs, gen_tokens, hw):
  latency = compute_bound = energy = ops = bits = 0.0
  for c in costs:
    t_ops = c.ops / hw['l_ops']
    t_data = c.bits / hw['l_data']
    latency += c.count * max(t_ops, t_data)
    compute_bound += c.count * t_ops if t_ops >= t_data else 0.0
    energy += c.count * (c.ops * hw['e_ops'] + c.bits * hw['e_data'])
    ops += c.count * c.ops
    bits += c.count * c.bits
  return {
    'latency_per_token_s': latency / gen_tokens,
    'throughput_tok_s': gen_tokens / latency if latency else float('inf'),
    'energy_per_token_j': energy / gen_tokens,
    'ops_per_token': ops / gen_tokens,
    'bits_per_token': bits / gen_tokens,
    'compute_bound_time_frac': compute_bound / latency if latency else 0.0,
  }


def attention_float_ops_per_token(n_blocks, dim, length, steps, n_seqs,
                                  gen_tokens):
  """QK^T and AV stay float: 4 L^2 D ops per block per NFE."""
  return n_blocks * 4 * length ** 2 * dim * steps * n_seqs / gen_tokens
