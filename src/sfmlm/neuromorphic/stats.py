"""Aggregate injector statistics into totals and per-layer / per-step views."""
from collections import defaultdict


def _derived(s, d_in, t_digits):
  values = s['n_tokens'] * d_in
  return {
    'sigma': 1.0 - s['spikes_binary'] / (values * t_digits),
    'row_active': s['active_channels'] / (s['n_seqs'] * d_in),
    'clip_rate': s['clipped'] / values,
    'spikes_per_value': s['spikes_binary'] / values,
  }


def summarize(stats, layer_dims, t_digits, gen_tokens):
  totals = defaultdict(int)
  per_layer = defaultdict(list)
  per_step = defaultdict(lambda: defaultdict(int))
  for (name, step), s in sorted(stats.items()):
    d_in = layer_dims[name][0]
    for k, v in s.items():
      totals[k] += v
    totals['values'] += s['n_tokens'] * d_in
    totals['channel_slots'] += s['n_seqs'] * d_in
    per_layer[name].append({'step': step, **_derived(s, d_in, t_digits)})
    agg = per_step[step]
    agg['spikes_binary'] += s['spikes_binary']
    agg['values'] += s['n_tokens'] * d_in
  return {
    'spikes_binary': totals['spikes_binary'],
    'spikes_naf': totals['spikes_naf'],
    'rate_count': totals['rate_count'],
    'spikes_binary_per_token': totals['spikes_binary'] / gen_tokens,
    'spikes_naf_per_token': totals['spikes_naf'] / gen_tokens,
    'rate_count_per_token': totals['rate_count'] / gen_tokens,
    'sigma': 1.0 - totals['spikes_binary'] / (totals['values'] * t_digits),
    'row_active': totals['active_channels'] / totals['channel_slots'],
    'clip_rate': totals['clipped'] / totals['values'],
    'per_step_spikes_per_value': {
      str(k): v['spikes_binary'] / v['values']
      for k, v in sorted(per_step.items())},
    'per_layer': dict(per_layer),
  }
