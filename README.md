# sfmlm — Spiking Flow Matching Language Models

N-MDLM-style neuromorphic conversion (Wu et al., *Neuromorphic Diffusion
Language Models*, arXiv 2607.24841) applied to a flow matching LM:

- **N-SFLM**: S-FLM (hyperspherical flow matching, `sphere-arch` backbone)
  with spiking linear layers.
- **N-MDLM (ours)**: the same conversion on the tiny GSM8K MDLM from s-flm
  (not the paper's E2D2/WMT model).

Models, checkpoints, samplers and GSM8K scoring come from the sibling
`../s-flm` checkout (override with `SFLM_ROOT`).

## Setup

```bash
../s-flm/.venv/bin/python -m pip install -e '.[test]'
../s-flm/.venv/bin/python -m pytest
```

Checkpoints expected under `../s-flm/checkpoints/tinygsm/`:
`sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt` and `mdlm.ckpt`.
On macOS, run outside any sandbox that blocks Metal (MPS).

## Conversion

Per token (each activation row `a` of a converted `nn.Linear` input), every
forward pass:

- `V_th = mean(|a|) / K`, `a_hat = clip(round(a / V_th), ±(2^T_MAX - 1))`
  (Eq. 7, plus explicit clipping; `T_MAX = 8`).
- `a_hat = sum_t 2^t s_t`, `s_t ∈ {-1, 0, 1}` (Eq. 8). The minimal number
  of digits is `floor(log2 max|a_hat|) + 1`; the paper's
  `floor(log2 max|a_hat|)` is one short. Spikes are counted for
  sign-magnitude binary (paper) and non-adjacent form (NAF, sparser, may need
  one extra digit).
- By linearity `W a ≈ V_th W a_hat = V_th sum_t 2^t W s_t` exactly, so the
  simulation feeds `V_th * a_hat` into the unchanged layer. The paper's
  IF dynamics (Eqs. 9–11) are not simulated: as written they mix the input
  and output thresholds, and an output-side neuron would need its own
  per-token `V_th(y)` (a global reduction).

Converted: per-token linears (attention q/k/v/out, MLP in/out). Excluded:
timestep embedder, adaLN / alpha modulation. The LM head is converted only
with `--include-head`. Attention `QK^T`, `AV`, softmax and normalizations
stay float and are reported as `attention_float_ops_per_token`.

### Modes

| mode | transmits | reconstruction |
|---|---|---|
| `float` | – | identity |
| `nmdlm` | `a_hat` each NFE | `V_th * a_hat` |
| `delta_bitwise` | `a_hat_s - a_hat_{s-1}`, `V_th` frozen per position at the first NFE | `V_th0 * a_hat_s` |
| `delta_residual` | `round((a_s - y_{s-1}) / V_th_s)`, `V_th` per NFE | `y_{s-1} + V_th_s * q_s` |

The delta modes are the flow-specific part: activations move smoothly along
the sampling trajectory, so step-to-step changes need fewer spikes.

### Roofline

`sfmlm.roofline.nmdlm` follows Sec. IV but for full-sequence samplers (every
NFE processes all `L` positions, prompt included). Per generated token:
`ops = 2 * D_out * spikes` (Eq. 12) and weight bits
`D_in * D_out * b_W * row_active` where `row_active` is the measured fraction
of input channels spiking at least once in a sequence (instead of the
i.i.d. `1 - sigma^{BT}` of Eq. 13). Latency / energy use Table I (OCMS, ICMS).
Baselines: the float model (`dense`) and a KV-cached AR model (`ar`).

## Experiments

```bash
cd sfmlm-experiments/e1
STEPS=8 ./run_local.sh smoke     # 128 examples, K=2, all modes, both models
STEPS=16 ./run_local.sh smoke
STEPS=16 ./run_local.sh full     # 1319 examples, K in {1,2,4,8}
./run_local.sh summary
```

For the unattended full sweep on a RunPod GPU (with autostop), see
[RUNPOD.md](RUNPOD.md).

One JSON per (model, mode) in `sfmlm-experiments/e1/results/`; existing files
are skipped (`--overwrite` to redo). Each run holds accuracy, spike totals
(binary, NAF, rate), sparsity `sigma`, row activity, clip rate, per-layer /
per-step statistics and roofline numbers.

## Not done yet

- Post-conversion fine-tuning (paper: 1000 Adam steps, lr 3e-5, STE); for
  `sphere-arch` call `renormalize_weights()` after each step.
- Paper-faithful E2D2 on WMT14 de-en.
