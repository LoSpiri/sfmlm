WEIGHT_BITS = 16

# Table I of the N-MDLM paper, in SI units (OP/s, bit/s, J/OP, J/bit).
HARDWARE_PRESETS = {
  'OCMS': {'l_ops': 300e12, 'l_data': 8e12,
           'e_ops': 0.05e-12, 'e_data': 10e-12},
  'ICMS': {'l_ops': 20e12, 'l_data': 30e12,
           'e_ops': 0.05e-12, 'e_data': 0.5e-12},
}
