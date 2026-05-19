# Spin-RNN handoff (low-rank-rnn → runpod)

This document is the single source of truth for the next Claude Code instance picking
this project up on a GPU box. Read this end to end before doing anything; everything
else in `docs/` is the upstream scientific plan, while this file pins what is
actually built, what has been measured, and exactly what work is queued.

**Working-directory rule.** The owner of this project (Aniket) requires the agent to
stay in this directory at all times. Run `pwd` periodically to confirm
`…/low-rank-rnn`. Do not `cd` elsewhere. Do not create sibling repos.

---

## 1. Research question (one sentence)

When an RNN is trained on Ising/Glauber spin trajectories, which statistical-mechanical
operator does its learned recurrent geometry align with — the microscopic coupling $A$,
the equilibrium covariance $C_\beta$, or the dynamical slow modes $K_\tau$? Low-rank
structure is treated as one possible emergent outcome, not as the assumed one.

Full theory: `docs/theory_spin_rnn.md`. Minimal implementation plan:
`docs/experiment_spin_rnn.md`.

---

## 2. Repo layout (what was added vs. inherited)

Inherited (old OU-tracking low-rank RNN code, kept untouched):

```
lowrank_rnn/models/rnn.py          # LowRankRNN: J = gW − (b/N)11ᵀ + m u vᵀ
lowrank_rnn/models/low_rank.py     # helpers around the rank-1 structure
lowrank_rnn/data/ou.py             # exact-update OU process
lowrank_rnn/train/trainer.py       # OU-tracking trainer
scripts/train_ou_tracking.py
scripts/visualize_training.py
tests/                             # tests for the OU / RNN baseline
```

New for the spin experiment (added in this branch):

```
lowrank_rnn/data/ising.py                  # IsingConfig, make_couplings, synchronous Glauber
lowrank_rnn/models/vanilla_rnn.py          # leaky tanh RNN with a fully trainable J
lowrank_rnn/analysis/spin_geometry.py      # cov, lagged cov, effective_rank, energy_rank,
                                           # subspace_alignment, effective_spin_operator,
                                           # geometry_report (+ random-control alignments)
lowrank_rnn/train/spin_trainer.py          # SpinTrainConfig + train_spin_prediction
                                           # logs val_loss, zero_loss, Δ_baseline
scripts/run_spin_experiment.py             # CLI entrypoint (single run)
scripts/sweep_spin_e1000.sh                # 10-run extended-matrix sweep
scripts/plot_spin_results.py               # per-run diagnostic figures
```

Output runs live under `runs/<name>/`. They are tracked in git in this project
(the `runs/*` `.gitignore` rule is overridden with `git add -f`). Keep that habit.

---

## 3. Environment

Local Mac (where this was developed) ran on Python 3.11.4, torch 2.8.0 CPU, numpy
1.26.4, scikit-learn 1.7.2. On runpod, install per `pyproject.toml`:

```bash
pip install -e .
# or, minimally:
pip install "torch>=1.12" "numpy>=1.20" "scipy>=1.7" "matplotlib>=3.3" "scikit-learn>=1.0"
```

Use a CUDA-enabled torch wheel on the runpod image. The trainer threads `device`
through, so `--device cuda` should Just Work — all tensor ops in `ising.py`,
`vanilla_rnn.py`, and `spin_geometry.py` are framework-agnostic. If anything
breaks on GPU, check (a) `make_couplings` builds `A` on CPU then `.to(device)` (fine,
small matrix), and (b) `subspace_alignment` casts to `float32` via SVD — should be
fine on GPU.

---

## 4. What has been measured so far

### 4.1 Original 300-epoch matrix (CPU)

Config: `n_spins=64, hidden_dim=64, seq_len=100, batch_size=64, alpha=0.2, lr=1e-3,
seed=0`, MSE on $\{-1,+1\}$ targets. Random baseline `align_random_*` is the same-shape
gaussian matrix's alignment with each operator (≈ 0.08–0.10 at $k=5$).

| run dir | regime | final loss | `eff_rank_G` | `align_G_A` | `align_G_C` | `align_G_lag` |
|---|---|---|---|---|---|---|
| `runs/curie_beta02`     | Curie–Weiss $\beta=0.2$    | 1.000 → 1.000 | 30 → **4.7**  | 0.08 → 0.23 | ~0.07 | 0.10 → 0.18 |
| `runs/lattice_beta02`   | 2D lattice $\beta=0.2$     | 1.019 → 0.900 | 30 → **13.7** | 0.10 → **0.60** | 0.10 → 0.58 | 0.11 → 0.66 |
| `runs/lattice_beta044`  | 2D lattice $\beta=0.44$    | 1.022 → 0.375 | 30 → **1.9**  | 0.11 → 0.46 | 0.11 → **0.51** | 0.11 → **0.51** |
| `runs/block_beta05`     | 2-block $\beta=0.5$        | 1.018 → 0.999 | 30 → **2.9**  | 0.12 → **0.41** | ~0.14 | 0.10 → 0.34 |

### 4.2 One extended-matrix run (already completed locally)

| run dir | epochs | Δ_baseline | `eff_rank_G` | `align_G_A` | `align_G_C` | `align_G_lag` |
|---|---|---|---|---|---|---|
| `runs/curie_beta02_e1000` | 1000 | **+0.001** | **1.46** | 0.265 | 0.094 | 0.202 |

Interpretation: even at 1000 epochs the model is still essentially the zero predictor
on Curie–Weiss $\beta=0.2$ ($L_{\rm val}\approx 0.999\approx L_{\rm zero}$). The eff-rank
collapse and modest alignment growth are *not* prediction-driven — they are weak
statistical bias / optimizer geometry. This confirms ChatGPT's reading that
Curie–Weiss at this temperature is a **weak-signal regime**, not undertraining. Keep
this run as the weak-signal control.

### 4.3 Headline take

- **`lattice_beta044`** is the first scientifically interesting result: near-critical
  collapse to rank≈2, with `align_G_C` and `align_G_lag` slightly above `align_G_A`
  (covariance / slow-mode dominance, as predicted by hypothesis H2 in `theory_spin_rnn.md`).
- **`lattice_beta02`** is a clean high-temperature sanity check: strongest microscopic
  coupling recovery (`align_G_A ≈ 0.60`, ~6× random).
- **Curie–Weiss $\beta=0.2$** and **block $\beta=0.5$** are statistically flat at the
  current coupling scale (couplings are normalized by $1/n$ in `ising.py`). Need
  stronger $\beta$ before their geometry diagnostics are interpretable.

---

## 5. H100 experiment plan (the actual queue)

The earlier CPU sweep (`scripts/sweep_spin_e1000.sh`) is **superseded** by what
follows. That script just ran the same 4 regimes longer plus stronger Curie/block;
useful as a CPU sanity loop, but on an H100 you should spend the compute on
**scale × β × seeds × tasks**, not on repeating the small regime longer. The
ChatGPT review (see `notes/training.tex` and the chat transcript at the top of
the project history) makes the case for this and I agree.

All new runs go under `runs_h100/` (kept distinct from the CPU `runs/` so old
and new results don't collide). Treat the 300-epoch CPU matrix and
`runs/curie_beta02_e1000` as committed controls — do not delete or overwrite them.

Before launching any tier, **smoke-test speed** (see §6.5 for the optimization
order). If even the small smoke runs feel slow, land the §6 optimizations first.

### 5.1 Tier 1 — replicate the original $n=64$ matrix with seeds and longer training

Purpose: rule out seed-dependence and undertraining as confounds before doing
anything fancier. Same scale as the CPU runs, but `seq_len=200`, `batch_size=256`,
and 5 seeds.

```bash
for seed in 0 1 2 3 4; do
  for spec in \
    "lattice_2d 0.2"  \
    "lattice_2d 0.44" \
    "curie_weiss 0.2" \
    "block 0.5"; do
    read -r kind beta <<<"$spec"
    name="${kind}_beta${beta/./}_seed${seed}"
    python scripts/run_spin_experiment.py \
      --graph-kind $kind --beta $beta \
      --n-spins 64 --hidden-dim 64 \
      --epochs 1000 --seq-len 200 --batch-size 256 \
      --eval-every 50 --seed $seed --device cuda \
      --save-dir runs_h100/tier1_${name}
  done
done
```

Outputs: 20 runs. Expectation: `lattice_2d` β=0.2/0.44 reproduce the CPU story
across seeds; `curie_weiss` β=0.2 and `block` β=0.5 stay flat (`Δ_baseline ≈ 0`).

### 5.2 Tier 2 — temperature phase diagram at $n=64$

Purpose: this is the **first real paper figure**. Map the dependence of
$\Delta_{\rm baseline}$, $r_{\rm eff}(G)$, $\mathrm{align}(G,A)$, $\mathrm{align}(G,C)$,
$\mathrm{align}(G,C_\tau)$ on $\beta$ for each graph family.

```bash
# 2D lattice — densely cover the critical region β_c ≈ 0.4407
for beta in 0.1 0.2 0.3 0.38 0.42 0.44 0.46 0.5 0.6 0.8; do
  for seed in 0 1 2; do
    python scripts/run_spin_experiment.py \
      --graph-kind lattice_2d --beta $beta \
      --n-spins 64 --hidden-dim 64 \
      --epochs 1000 --seq-len 200 --batch-size 256 \
      --eval-every 50 --seed $seed --device cuda \
      --save-dir runs_h100/tier2_lattice64_beta${beta}_seed${seed}
  done
done

# Curie–Weiss — sweep through the mean-field transition (couplings are J/n, so
# the interesting regime is β J ~ 1, i.e. β ≳ 1)
for beta in 0.2 0.5 0.8 1.0 1.2 1.5 2.0; do
  for seed in 0 1 2; do
    python scripts/run_spin_experiment.py \
      --graph-kind curie_weiss --beta $beta \
      --n-spins 64 --hidden-dim 64 \
      --epochs 1000 --seq-len 200 --batch-size 256 \
      --eval-every 50 --seed $seed --device cuda \
      --save-dir runs_h100/tier2_curie64_beta${beta}_seed${seed}
  done
done

# Block ferromagnet — same idea, J_in = 1, J_out = 0.2 are 1/n-scaled
for beta in 0.5 0.8 1.0 1.2 1.5 2.0; do
  for seed in 0 1 2; do
    python scripts/run_spin_experiment.py \
      --graph-kind block --beta $beta \
      --n-spins 64 --hidden-dim 64 \
      --epochs 1000 --seq-len 200 --batch-size 256 \
      --eval-every 50 --seed $seed --device cuda \
      --save-dir runs_h100/tier2_block64_beta${beta}_seed${seed}
  done
done
```

Outputs: 30 + 21 + 18 = 69 runs. Plot a $\beta$-by-metric figure per graph family.

### 5.3 Tier 3 — scale $n$

Purpose: test whether the geometry observed at $n=64$ survives at $n=256$ and
$n=1024$. Three βs per size, three seeds per cell.

```bash
for n in 64 256 1024; do
  for beta in 0.2 0.44 0.6; do
    for seed in 0 1 2; do
      python scripts/run_spin_experiment.py \
        --graph-kind lattice_2d --beta $beta \
        --n-spins $n --hidden-dim $n \
        --epochs 1000 --seq-len 200 --batch-size 128 \
        --eval-every 100 --align-k 10 \
        --seed $seed --device cuda \
        --save-dir runs_h100/tier3_lattice${n}_beta${beta}_seed${seed}
    done
  done
done
```

For $n=1024$, if memory or time becomes an issue, drop `--seq-len` to 100 first,
then `--batch-size` to 64. The Python loop in `VanillaRNN.forward` is unrolled
over `seq_len`, so that knob has bigger compute effect than `batch_size`.

### 5.4 Tier 4 — finite-size scaling near criticality (the headline experiment)

If Tier 3 looks healthy, run this. It's the single sweep that, if positive,
directly supports the project's thesis:

> Near criticality, the learned $G=RJB$ becomes lower-rank and aligns more strongly
> with covariance / lagged slow modes than with the microscopic coupling matrix,
> and this effect strengthens with system size.

```bash
for n in 64 256 1024; do
  for beta in 0.36 0.40 0.42 0.44 0.46 0.48 0.52; do
    for seed in 0 1 2; do
      python scripts/run_spin_experiment.py \
        --graph-kind lattice_2d --beta $beta \
        --n-spins $n --hidden-dim $n \
        --epochs 1000 --seq-len 200 --batch-size 128 \
        --eval-every 100 --align-k 10 \
        --seed $seed --device cuda \
        --save-dir runs_h100/tier4_fss_lattice${n}_beta${beta}_seed${seed}
    done
  done
done
```

Outputs: 63 runs. The cross-run plot is $\beta$ on the x-axis, one of
$\{r_{\rm eff}(G),\ \mathrm{align}(G,A),\ \mathrm{align}(G,C),\ \mathrm{align}(G,C_\tau)\}$
on the y-axis, one curve per $n$ (with seed-mean ± std as a band).

### 5.5 Task extensions (worth landing before the n=1024 budget)

The current CLI exposes only **one-step full-state prediction**. That's fine for
the lattice phase-diagram, but the project's strongest novelty claim
("coarse-graining is what the RNN actually represents") needs at least one
harder task. The minimum addition is a `--task` flag in `scripts/run_spin_experiment.py`
that dispatches to one of:

| value | description | code change |
|---|---|---|
| `next_state` (default) | $x_t=s_t$, $y_t=s_{t+1}$, MSE | already implemented |
| `denoise` | $x_t = M_t\odot s_t + (1-M_t)\odot\xi_t$, $y_t=s_t$ (clean current state), MSE | new branch in trainer; corrupt input with per-element Bernoulli mask + ±1 noise |
| `partial` | $x_t = P_\Omega s_t$ with `obs_frac` fraction observed, $y_t=s_{t+1}$ (or $s_t$ full) | apply a fixed-per-run mask; either zero-fill missing coords or concat a mask channel |
| `magnetization` | $y_t = \frac1n\sum_i s_i(t+1)$, scalar | change `output_dim=1` in `VanillaRNN`; MSE on a single scalar per timestep |

Predictions per the theory note (`docs/theory_spin_rnn.md` §"Candidate tasks"):
denoise and partial should shift weight from $\mathrm{align}(G,A)$ toward
$\mathrm{align}(G,C)$ / $\mathrm{align}(G,C_\tau)$; magnetization should collapse
$r_{\rm eff}(G)$ to ~1 across regimes.

A minimal first step: run `--task denoise --mask-frac 0.3` against the lattice
β-sweep and compare alignment ratios.

### 5.6 Interpretation grid (after each tier)

Use this as the read-out template. The CPU runs already filled in the top rows;
the H100 runs should populate the rest.

| Observation | Interpretation |
|---|---|
| Tier-1 lattice runs reproduce the CPU eff_rank / alignment values across all 5 seeds | core result is real, not a seed artifact |
| Tier-1 weak Curie/block stay at $\Delta_{\rm baseline}\approx 0$ | weak-signal confirmed (already evident from `runs/curie_beta02_e1000`); strong-β variants should be the focus |
| Tier-2 alignment $\mathrm{align}(G,A)$ peaks above criticality and drops near $\beta_c$, while $\mathrm{align}(G,C),\ \mathrm{align}(G,C_\tau)$ peak at criticality | direct evidence for H2 — the phase-dependent geometry hypothesis |
| Tier-3 effects of Tier-2 strengthen with $n$ | finite-size scaling consistent with collective-mode dominance at large $n$ |
| Tier-4 lower $r_{\rm eff}(G)$ at fixed $\beta$ as $n$ grows | strongest single signature of emergent coarse-graining; this is the paper figure |
| denoise / partial tasks raise $\mathrm{align}(G,C)$ vs `next_state` | corroborates the coarse-graining interpretation |

### 5.7 After-the-sweeps work

1. `python3 scripts/plot_spin_results.py runs_h100/tier*` — per-run figures.
2. New script `scripts/plot_phase_diagram.py` (not yet written): take a directory
   of runs whose names match `*_beta{β}_seed{s}` and emit per-graph-family curves
   of $\{\Delta_{\rm baseline},\ r_{\rm eff}(G),\ \mathrm{align}_*\}$ vs $\beta$,
   with seed-mean ± std bands. Re-use it for Tier 4 by grouping on $n$.
3. Commit the surviving runs (`git add -f runs_h100/...`), then push.
   `final.pt` files are ~100 KB at $n=64$; at $n=1024$ they balloon to ~tens of
   MB. Consider `final_light.pt` (state dict + small metadata only) for $n≥256$
   if you want to keep many runs in git — see §6.7.

---

## 6. Optimizations to land before big sweeps

The current code is correct but un-tuned. ChatGPT identified three classes of
bottleneck that bite at $n\ge 256$: dense Glauber updates, Python time loops,
and full-matrix SVDs in diagnostics. Land these patches in roughly the order
listed; (6.1) and (6.2) alone get you to $n=1024$ comfortably.

The single most important change is **(6.1)**: stop forming the dense `s @ A.T`
update for lattice / Curie / block. That alone changes Ising sampling from
$O(B n^2)$ to $O(B n)$ for the structured-graph cases.

### 6.1 Local-field updates in `lowrank_rnn/data/ising.py`

For each structured graph, replace `field = s @ A.T` with a kernel that exploits
the structure. Keep `A` as the source of truth (it's still used by analysis), but
add a `graph_kind`-aware `_local_field` helper and call it from
`synchronous_glauber_step` when the kind is known.

```python
def _lattice_field_2d(s: torch.Tensor, L: int, coupling: float) -> torch.Tensor:
    # s: (B, n), n = L * L; returns (B, n)
    x = s.view(s.shape[0], L, L)
    f = (torch.roll(x, 1, dims=1) + torch.roll(x, -1, dims=1)
         + torch.roll(x, 1, dims=2) + torch.roll(x, -1, dims=2))
    return coupling * f.reshape_as(s)

def _curie_field(s: torch.Tensor, coupling: float) -> torch.Tensor:
    n = s.shape[-1]
    total = s.sum(dim=-1, keepdim=True)
    return (coupling / n) * (total - s)

def _block_field(s, labels, j_in, j_out, n_blocks):
    B, n = s.shape
    sums = torch.zeros(B, n_blocks, device=s.device, dtype=s.dtype)
    sums.scatter_add_(1, labels[None].expand(B, -1), s)
    own = sums.gather(1, labels[None].expand(B, -1))  # (B, n)
    total = s.sum(dim=-1, keepdim=True)
    return (j_in / n) * (own - s) + (j_out / n) * (total - own)
```

Plumb the choice through `sample_ising_batch` via the `meta` dict (which already
carries `graph_kind`, `lattice_shape`, `block_labels`). Keep the dense path for
`sk` and as a fallback.

### 6.2 Vectorize input / readout in `VanillaRNN.forward`

Three changes:

- Compute `inp = self.input(x)` once over the whole sequence: `(B, T, input)` →
  `(B, T, H)`.
- Pre-allocate `hs = torch.empty(B, T, H, ...)` instead of appending to a list +
  `torch.cat`.
- Apply `self.readout` once at the end to the full hidden trajectory:
  `y_seq = self.readout(hs)`.

The recurrent dependency still serializes the inner loop, but you cut two
per-step matmul calls and remove all Python-list overhead. This is the second
biggest single win.

### 6.3 Mixed precision on H100

Add to `train_spin_prediction` (gated on `cfg.device.startswith("cuda")`):

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
use_amp = cfg.device.startswith("cuda")
...
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
    y_pred, _ = model(x, return_states=False)
    loss = loss_fn(y_pred, y_true)
```

Cast back to float32 for **all** diagnostic ops (the analysis helpers already do
`.to(torch.float32)` before `linalg.svdvals` — keep that). Geometry trends are
qualitative; bf16 forward + fp32 SVD is safe.

### 6.4 Optional `--compile` flag

Once (6.1)–(6.3) are in and behavior is stable, add a `--compile` flag that wraps
the model with `torch.compile(model, mode="reduce-overhead")`. Off by default —
debugging compiled PyTorch is painful and not worth it during development.

### 6.5 Smoke-test order

Before launching Tier 1, run these in sequence on a single GPU. If the second
one is unhappy, land §6.1 and try again.

```bash
# 1) n=64 — should be < 20 s
python scripts/run_spin_experiment.py \
    --graph-kind lattice_2d --beta 0.44 \
    --n-spins 64 --hidden-dim 64 \
    --epochs 100 --seq-len 200 --batch-size 256 \
    --eval-every 25 --device cuda \
    --save-dir runs_h100/_smoke_lattice64

# 2) n=256 — should be < 1 min after §6.1+§6.2
python scripts/run_spin_experiment.py \
    --graph-kind lattice_2d --beta 0.44 \
    --n-spins 256 --hidden-dim 256 \
    --epochs 100 --seq-len 200 --batch-size 128 \
    --eval-every 25 --device cuda \
    --save-dir runs_h100/_smoke_lattice256

# 3) n=1024 — should be < 5 min after §6.1+§6.2+§6.3
python scripts/run_spin_experiment.py \
    --graph-kind lattice_2d --beta 0.44 \
    --n-spins 1024 --hidden-dim 1024 \
    --epochs 100 --seq-len 100 --batch-size 64 \
    --eval-every 25 --align-k 10 --device cuda \
    --save-dir runs_h100/_smoke_lattice1024
```

If $n=1024$ is still slow, the bottleneck is likely the diagnostic SVDs — land
§6.6 next.

### 6.6 Fast-diagnostic mode for $n\ge 1024$

`effective_rank`, `energy_rank`, and `subspace_alignment` all currently call
`torch.linalg.svd` / `svdvals` on full $n\times n$ matrices. At $n=1024$ that's
~$O(n^3)\approx 10^9$ flops per eval per metric — non-trivial but ok. At
$n=4096$ it dominates training. Add `cfg.diag_mode` with two settings:

- `"fast"` (default for $n\ge 512$): use `torch.svd_lowrank(M.float(), q=k+10, niter=2)`
  for top-$k$ subspaces; estimate `effective_rank` from the top $q$ singular values
  only (truncated participation ratio with a clear note in the JSON that this is
  approximate). Skip random-control alignment except at epoch 1 and the final
  epoch.
- `"full"` (default for $n<512$): current behavior.

Also cache `top_left_singular(A, k)` once at the start of training — $A$ is
fixed per run, no need to re-SVD it every eval. Same goes for the random-control
matrix.

### 6.7 Multi-seed concurrent launcher

For $n\le 256$ a single job under-utilizes an H100. Run 4 seeds in parallel:

```bash
for seed in 0 1 2 3; do
  python scripts/run_spin_experiment.py ... --seed $seed \
    --save-dir runs_h100/.../seed${seed} &
done
wait
```

Pin via `CUDA_VISIBLE_DEVICES=0` if running across multiple GPUs. For $n=1024$
keep concurrency at 1–2 unless you've measured headroom.

### 6.8 Lighter checkpointing for big runs

`runs_h100/tier4_fss_lattice1024_beta*_seed*/final.pt` will be in the tens of
MB each, and there are 63 of them in Tier 4. Two options:

- Write a `final_light.pt` containing `{state_dict, cfg, history, losses, A, J0}`
  but NOT the full optimizer state, and use that for git.
- Or just `git add -f` only the JSON + plots, and keep `final.pt` local /
  optional. The JSON history is enough to redraw every figure.

### 6.9 Optional later: low-rank parameterization of $J$ for $n\ge 4096$

Replace `self.recurrent` with `J = U V.T` where `U,V \in R^{H \times r}`. This
changes the recurrent cost from $O(H^2)$ to $O(H r)$. **Don't** make this the
default — the project's scientific point is **emergent** low rank, and imposing
it would defeat that. But for purely-engineering scaling tests, it's the right
move.

### 6.10 What not to optimize yet

- Don't switch to `torch.nn.RNN` / `RNNCell` — you lose the fully-trainable,
  inspectable $J$ that the analysis depends on.
- Don't add Hydra / W&B / live dashboards. `docs/experiment_spin_rnn.md`
  explicitly forbids them at this stage. Argparse + JSON + matplotlib only.
- Don't unify the OU-tracking and spin code paths. They share nothing
  scientifically; cross-contamination would create surprise.

---

## 7. Code-level notes & gotchas

- **MSE on $\pm 1$ targets.** Zero predictor has MSE ≈ 1. Always read $\Delta_{\rm baseline}$
  before trusting alignment numbers — alignment can drift even when prediction loss
  hasn't budged (this is what happened for curie/block at low $\beta$).
- **`align_random_A` etc.** are computed against the same target operator each eval
  with a fixed-seed gaussian. They drift slightly over runs because the *target*
  $C, C_\tau$ depend on the validation batch, while $A$ is fixed.
- **`subspace_alignment`** requires both matrices to have the same row dimension. The
  effective spin operator $G_{\rm spin} = RJB$ is $n\times n$ and is the apples-to-apples
  thing to compare to $A,C,C_\tau$. Do **not** compare the hidden-space $J\in\mathbb R^{H\times H}$
  to $A\in\mathbb R^{n\times n}$ — that path was deliberately not built.
- **`hidden_dim == n_spins == 64`** in the current configs. That is the simplest case
  per `experiment_spin_rnn.md` "Solution 1". The `G_spin` path works for any
  $H,n$ pair so feel free to break this constraint later (e.g. `hidden_dim=256`).
- **No GPU was available when this was built.** Everything is CPU-tested. The model is
  trivially small, so on GPU consider scaling up: `n_spins=256`, `hidden_dim=512`,
  `seq_len=500`, multiple seeds (`--seed 0/1/2/3/4`).
- **Couplings are scaled by $1/n$** for Curie and block. That's why $\beta$ needs to be
  large to see learning. The 2D lattice does **not** divide by $n$, so it sees signal
  at smaller $\beta$.
- **Synchronous Glauber.** Asynchronous dynamics is a planned extension; see
  `docs/experiment_spin_rnn.md` ("Add asynchronous Glauber").

---

## 8. Reproducing the existing local results on runpod

If you want a sanity check that the GPU box reproduces what's in the repo:

```bash
python3 scripts/run_spin_experiment.py \
    --graph-kind lattice_2d --beta 0.44 \
    --n-spins 64 --hidden-dim 64 \
    --epochs 300 --device cuda \
    --save-dir runs/_repro_lattice044
```

Compare `runs/_repro_lattice044/history.json` against `runs/lattice_beta044/history.json`.
Numbers won't be bit-identical across devices, but `eff_rank_G` should collapse to
~2 and `align_G_C ≈ align_G_lag ≈ 0.5` should reappear.

---

## 9. Git / push hygiene

- Commits so far on `main` (origin = `https://github.com/aniket-desh/low-rank-rnn`):
  - `029fead` — docs: switch LaTeX delimiters to `$` and `$$`
  - `913ba27` — spin-RNN experiment: modules + first 300-epoch matrix + plots
  - `4e3b045` — baseline-loss metric, sweep script, `curie_beta02_e1000`, HANDOFF.md v1
  - (this commit) — HANDOFF.md v2: H100 three-tier plan + optimization queue
- New H100 runs should live under `runs_h100/` so they don't collide with the
  CPU `runs/` controls.
- The owner expects commits + pushes when code lands. Use `git add -f` to override
  the `runs/*` ignore rule, and consider stripping `final.pt` for $n\ge 256$ (see §6.8).

---

## 10. Quick context the owner repeats

- The owner is Aniket (`aniketdeshh@gmail.com`). This is a research project he wants
  to extend rigorously; treat results as something he'll defend in a meeting.
- Don't bloat the codebase. The doc explicitly calls out "what not to build yet"
  (Hydra, W&B, dashboards). Stick to argparse + JSON + matplotlib.
- The `docs/*.md` files use `$…$` and `$$…$$` for math. Don't reintroduce `\(\)` or `\[\]`.
- Stay in `/Users/aniket/Documents/university/research/low-rank-rnn` (or whatever the
  runpod equivalent is). Run `pwd` to confirm before destructive ops.
