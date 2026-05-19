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

## 5. Pending work (the queue)

This is what was about to run when the sweep was cancelled. The runpod CC instance
should pick up here.

### 5.1 Remaining sweep (9 runs)

Already encoded in `scripts/sweep_spin_e1000.sh`. The first run (`curie_beta02_e1000`)
finished locally and is already in the repo. Delete that one line from the script
to skip it, then run:

```bash
./scripts/sweep_spin_e1000.sh
```

The 9 still-pending configs:

| name | graph | $\beta$ | role |
|---|---|---|---|
| `block_beta05_e1000`   | block      | 0.5  | weak-signal control (continuation of `block_beta05`) |
| `lattice_beta02_e1000`  | lattice_2d | 0.2  | longer high-T sanity check |
| `lattice_beta044_e1000` | lattice_2d | 0.44 | longer near-critical run (this one matters most) |
| `curie_beta08_e1000`    | curie_weiss| 0.8  | strong-Curie ladder |
| `curie_beta10_e1000`    | curie_weiss| 1.0  | strong-Curie ladder |
| `curie_beta12_e1000`    | curie_weiss| 1.2  | strong-Curie ladder |
| `block_beta10_e1000`    | block      | 1.0  | strong-block ladder |
| `block_beta15_e1000`    | block      | 1.5  | strong-block ladder |
| `block_beta20_e1000`    | block      | 2.0  | strong-block ladder |

On GPU, expect each run to drop from ~40 s (CPU) to a few seconds. The dominant
overhead is the per-step Python loop in `VanillaRNN.forward`; consider switching
to `torch.compile` or torch-RNN cell APIs if profiling shows it matters.

### 5.2 Interpretation grid (from ChatGPT, encoded as a checklist)

For each (`curie_*`, `block_*`) regime, after the run finishes, answer:

| If… | …then |
|---|---|
| original weak Curie / block improve after 1000 epochs | they were undertrained — but the 1000-epoch `curie_beta02_e1000` says **they aren't**, so this should be `false` |
| original weak regimes stay near $L\approx 1$ AND stronger regimes improve | confirms weak-signal regime, not optimizer failure |
| even strong Curie / block fail | task / model / loss issue — inspect alpha, lr, sequence length |
| strong Curie becomes rank-1-ish and aligns with $A$ / $C$ | sanity check: mean-field collective mode recovered |
| strong block becomes low-rank and block-structured (`same_block_mean_G` ≫ `diff_block_mean_G`) | strong evidence for emergent modular geometry |

### 5.3 After-the-sweep work

1. `python3 scripts/plot_spin_results.py runs/*_e1000` — generates per-run figures.
2. Build a cross-run comparison plot: $\Delta_{\rm baseline}$, `eff_rank_G`, and the
   three alignment scores as a function of $\beta$ for the Curie and block ladders.
   (Not yet implemented — add to `scripts/plot_spin_results.py` or a new
   `scripts/plot_alignment_phase.py`.)
3. Commit the new runs and the updated plots (`git add -f runs/...`), then push.

---

## 6. Code-level notes & gotchas

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

## 7. Reproducing the existing local results on runpod

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

## 8. Git / push hygiene

- Commits so far on `main` (origin = `https://github.com/aniket-desh/low-rank-rnn`):
  - `029fead` — docs: switch LaTeX delimiters to `$` and `$$`
  - `913ba27` — spin-RNN experiment: modules + first 300-epoch matrix + plots
  - (this commit) — baseline-loss metric, sweep script, `curie_beta02_e1000`, HANDOFF.md
- The owner expects commits + pushes when code lands. Use `git add -f runs/<name>` to
  override the `runs/*` ignore rule for completed runs you want preserved.

---

## 9. Quick context the owner repeats

- The owner is Aniket (`aniketdeshh@gmail.com`). This is a research project he wants
  to extend rigorously; treat results as something he'll defend in a meeting.
- Don't bloat the codebase. The doc explicitly calls out "what not to build yet"
  (Hydra, W&B, dashboards). Stick to argparse + JSON + matplotlib.
- The `docs/*.md` files use `$…$` and `$$…$$` for math. Don't reintroduce `\(\)` or `\[\]`.
- Stay in `/Users/aniket/Documents/university/research/low-rank-rnn` (or whatever the
  runpod equivalent is). Run `pwd` to confirm before destructive ops.
