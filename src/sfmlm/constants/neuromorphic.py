T_MAX = 8
DEFAULT_K_SWEEP = (1, 2, 4, 8)
VTH_EPS = 1e-12

# Enough bit positions to popcount a delta of two clipped values
# (|delta| <= 2 * (2**T_MAX - 1)) in both binary and NAF form.
POPCOUNT_BITS = T_MAX + 2

MODE_FLOAT = 'float'
MODE_NMDLM = 'nmdlm'
MODE_DELTA_BITWISE = 'delta_bitwise'
MODE_DELTA_RESIDUAL = 'delta_residual'
MODES = (MODE_FLOAT, MODE_NMDLM, MODE_DELTA_BITWISE, MODE_DELTA_RESIDUAL)
SPIKING_MODES = (MODE_NMDLM, MODE_DELTA_BITWISE, MODE_DELTA_RESIDUAL)

# Timesteps of the transmitted signal: a delta of clipped values needs one
# extra binary digit.
MODE_TIMESTEPS = {
  MODE_NMDLM: T_MAX,
  MODE_DELTA_BITWISE: T_MAX + 1,
  MODE_DELTA_RESIDUAL: T_MAX,
}

# Per-sequence conditioning layers (timestep embedder, adaLN / alpha
# modulation); their inputs are not per-token activations.
EXCLUDED_LAYER_PATTERNS = (
  'sigma_map',
  'alpha_modulation',
  'adaLN_modulation',
  'time_token_head',
)
# Vocabulary projection (SphereArch / DIT); converted only on request.
HEAD_LAYER_NAMES = ('lm_head', 'output_layer.linear')

# Order of the per-(layer, step) integer accumulators.
STAT_FIELDS = (
  'spikes_binary',
  'spikes_naf',
  'rate_count',
  'n_tokens',
  'clipped',
  'active_channels',
  'n_seqs',
)
