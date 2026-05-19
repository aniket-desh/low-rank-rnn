# Minimal experiment plan: RNNs trained on Ising/Glauber dynamics

This document describes a minimal but sufficient codebase extension for testing whether a trained RNN learns microscopic couplings, covariance modes, or dynamical slow modes from spin-system trajectories.

The priority is not to build a large framework. The priority is to add a small set of files that answer one scientific question cleanly.

## Current repo diagnosis

The existing code already has useful pieces:

- an RNN class with explicit recurrent connectivity,
- diagnostics for spectra and clustering,
- a simple trainer loop,
- OU data generation.

But the current experiment is not enough for the new question. The original model constructs

\[
J = gW - \frac{b}{N}\mathbf 1\mathbf 1^\top + m u v^\top
\]

by design. In the default config, `train_u=False`, `train_v=False`, and `train_readout=True`, so recurrent structure is mostly imposed rather than learned. For the new experiment, the recurrent matrix must be trainable, and the data should come from an Ising/Glauber process.

## Main experimental question

Train an RNN on spin trajectories and measure whether the learned recurrent geometry aligns with:

1. the microscopic Ising coupling matrix \(A\),
2. the empirical covariance \(C_\beta\),
3. time-lagged slow modes \(K_\tau\),
4. block/community structure when present,
5. low-rank structure as measured by singular spectra and effective rank.

Do not assume the answer is low rank. The experiment should discover the geometry.

## GPU necessity

A GPU is useful but not necessary for the first version.

### CPU is sufficient for

- spin dimension `n_spins <= 64`,
- RNN hidden size `hidden_dim <= 128`,
- sequence length `T <= 200`,
- batch size `<= 64`,
- a few graph families and temperatures,
- proof-of-concept diagnostics.

This should run on a laptop CPU if written cleanly.

### GPU becomes useful for

- `n_spins >= 256`,
- hidden size `>= 512`,
- long trajectories `T >= 1000`,
- many temperature sweeps,
- many seeds,
- full SVD/eigendecomposition repeated often.

### Recommendation

Start CPU-first. Design the experiment so it runs with:

```bash
python scripts/run_spin_experiment.py --device cpu --n-spins 64 --hidden-dim 128 --epochs 300
```

Then scale to GPU only after the diagnostics look meaningful. This avoids burning GPU time on a possibly wrong task definition.

## Proposed file layout

Add only the following files:

```text
lowrank_rnn/data/ising.py
lowrank_rnn/models/vanilla_rnn.py
lowrank_rnn/analysis/spin_geometry.py
lowrank_rnn/train/spin_trainer.py
scripts/run_spin_experiment.py
```

Optional later:

```text
scripts/sweep_spin_experiment.py
notebooks/spin_geometry_exploration.ipynb
```

Do not add a large config system yet. Use dataclasses and command-line flags.

## File 1: `lowrank_rnn/data/ising.py`

Purpose: generate Ising coupling matrices and spin trajectories.

Minimal functions:

```python
import torch
from dataclasses import dataclass
from typing import Literal, Optional

GraphKind = Literal["curie_weiss", "lattice_2d", "block", "sk"]

@dataclass
class IsingConfig:
    n_spins: int = 64
    beta: float = 0.5
    field: float = 0.0
    graph_kind: GraphKind = "lattice_2d"
    coupling: float = 1.0
    n_blocks: int = 2
    j_in: float = 1.0
    j_out: float = 0.2
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
```

### `make_couplings(cfg) -> tuple[torch.Tensor, dict]`

Returns:

- `A`: shape `(n_spins, n_spins)`, symmetric, zero diagonal,
- `meta`: dictionary containing graph metadata such as block labels or lattice shape.

Implementation details:

#### Curie--Weiss

```python
A = cfg.coupling / n * torch.ones(n, n)
A.fill_diagonal_(0.0)
```

#### 2D lattice

Require `n_spins` to be a perfect square. Let `L = int(sqrt(n_spins))`. Use periodic boundary conditions for simplicity. Connect four nearest neighbors.

```python
A[i, j] = cfg.coupling
```

for nearest-neighbor pairs only. Symmetrize and zero diagonal.

#### Block model

Assign each spin to one of `n_blocks` blocks. Use

```python
A[i, j] = cfg.j_in / n if same_block else cfg.j_out / n
```

Zero diagonal. Return `block_labels` in metadata.

#### SK model

```python
G = torch.randn(n, n) * cfg.coupling / torch.sqrt(torch.tensor(float(n)))
A = (G + G.T) / 2
A.fill_diagonal_(0.0)
```

### `sample_initial_spins(batch_size, n_spins, device, dtype)`

Returns random spins in `{-1,+1}`:

```python
spins = 2 * torch.randint(0, 2, (batch_size, n_spins), device=device) - 1
return spins.to(dtype)
```

### `synchronous_glauber_step(s, A, beta, h=None, generator=None)`

Input:

- `s`: `(batch, n_spins)`.
- `A`: `(n_spins, n_spins)`.

Compute local fields:

```python
field = s @ A.T
if h is not None:
    field = field + h
p_plus = torch.sigmoid(2.0 * beta * field)
s_next = torch.where(torch.rand_like(p_plus) < p_plus, 1.0, -1.0)
```

Return `s_next`.

### `sample_ising_batch(cfg, batch_size, seq_len, burn_in=100)`

Return:

- `states`: shape `(batch_size, seq_len + 1, n_spins)`, because one-step prediction uses `states[:, :-1]` as input and `states[:, 1:]` as target,
- `A`,
- `meta`.

Pseudo-code:

```python
A, meta = make_couplings(cfg)
s = sample_initial_spins(batch_size, cfg.n_spins, device, dtype)
for _ in range(burn_in):
    s = synchronous_glauber_step(s, A, cfg.beta)

states = [s]
for _ in range(seq_len):
    s = synchronous_glauber_step(s, A, cfg.beta)
    states.append(s)
return torch.stack(states, dim=1), A, meta
```

Keep this vectorized over batch. Do not loop over batch elements.

## File 2: `lowrank_rnn/models/vanilla_rnn.py`

Purpose: train an RNN whose recurrent matrix is actually learned.

Avoid modifying the existing low-rank RNN class at first. Add a clean standalone model.

```python
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn as nn

Nonlinearity = Literal["tanh", "relu"]

@dataclass
class VanillaRNNConfig:
    input_dim: int
    hidden_dim: int = 128
    output_dim: int | None = None
    alpha: float = 0.2
    nonlinearity: Nonlinearity = "tanh"
    train_initial_state: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

class VanillaRNN(nn.Module):
    def __init__(self, cfg: VanillaRNNConfig):
        super().__init__()
        self.cfg = cfg
        out = cfg.output_dim or cfg.input_dim
        self.input = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=True)
        self.recurrent = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.readout = nn.Linear(cfg.hidden_dim, out, bias=True)
        self.activation = torch.tanh if cfg.nonlinearity == "tanh" else torch.relu

        h0 = torch.zeros(cfg.hidden_dim, dtype=cfg.dtype)
        if cfg.train_initial_state:
            self.h0 = nn.Parameter(h0)
        else:
            self.register_buffer("h0", h0)

        self.to(device=torch.device(cfg.device), dtype=cfg.dtype)

    @property
    def J(self):
        return self.recurrent.weight

    def forward(self, x, h0=None, return_states=False):
        # x: (batch, seq_len, input_dim)
        B, T, _ = x.shape
        h = self.h0[None, :].expand(B, -1) if h0 is None else h0
        ys = []
        hs = []
        alpha = self.cfg.alpha
        for t in range(T):
            pre = self.input(x[:, t]) + self.recurrent(h)
            h_new = self.activation(pre)
            h = (1.0 - alpha) * h + alpha * h_new
            y = self.readout(h)
            ys.append(y[:, None, :])
            if return_states:
                hs.append(h[:, None, :])
        y = torch.cat(ys, dim=1)
        h_traj = torch.cat(hs, dim=1) if return_states else None
        return y, h_traj
```

### Why use MSE instead of cross-entropy first?

Spins are in `{-1,+1}`. A minimal first version can use MSE between predicted real-valued spins and target spins:

\[
\mathcal L = \frac{1}{BTN}\sum_{b,t,i}(\hat s_{bti}-s_{bti})^2.
\]

This is simple and lets the output be interpreted as conditional mean. Later, switch to Bernoulli cross-entropy with logits if needed.

## File 3: `lowrank_rnn/analysis/spin_geometry.py`

Purpose: compute all geometry diagnostics in one place.

Minimal functions:

```python
import torch
from typing import Optional, Dict, Any
```

### `centered_cov(states)`

Input `states` shape `(batch, time, n_spins)` or `(samples, n_spins)`.

Flatten batch/time:

```python
X = states.reshape(-1, states.shape[-1])
X = X - X.mean(dim=0, keepdim=True)
C = X.T @ X / max(X.shape[0] - 1, 1)
return C
```

### `lagged_cov(states, lag=1)`

```python
X = states[:, :-lag, :].reshape(-1, n)
Y = states[:, lag:, :].reshape(-1, n)
X = X - X.mean(0, keepdim=True)
Y = Y - Y.mean(0, keepdim=True)
return X.T @ Y / max(X.shape[0] - 1, 1)
```

### `effective_rank(M)`

```python
s = torch.linalg.svdvals(M)
return (s.sum() ** 2 / (s.pow(2).sum() + 1e-12)).item()
```

### `energy_rank(M, eps=0.05)`

```python
s2 = torch.linalg.svdvals(M).pow(2)
frac = torch.cumsum(s2, dim=0) / (s2.sum() + 1e-12)
return int((frac >= 1.0 - eps).nonzero()[0].item() + 1)
```

### `subspace_alignment(M, N, k=5)`

Use top left singular vectors:

```python
UM = torch.linalg.svd(M, full_matrices=False).U[:, :k]
UN = torch.linalg.svd(N, full_matrices=False).U[:, :k]
return (torch.linalg.norm(UM.T @ UN, ord="fro") ** 2 / k).item()
```

This only works when matrices have compatible row dimension. For comparing RNN hidden-space matrices to spin-space matrices, use a projected operator as described below.

### Important: hidden-space/spin-space mismatch

The RNN recurrent matrix `J` lives in hidden space:

\[
J\in\mathbb R^{H\times H}.
\]

The Ising coupling and covariance live in spin space:

\[
A,C\in\mathbb R^{n\times n}.
\]

So do not directly compare `J` to `A` unless `hidden_dim == n_spins` and the hidden state has the same coordinate system as the spins.

There are two minimal solutions.

#### Solution 1: set `hidden_dim = n_spins`

For the first experiment, set hidden dimension equal to spin dimension. This makes direct comparison possible. It is the simplest and recommended first version.

#### Solution 2: analyze the input-output effective operator

For general hidden size, estimate the local effective spin-to-spin Jacobian:

\[
G_t = \frac{\partial y_t}{\partial x_t}.
\]

This is expensive if done exactly. A cheap approximation for one-step tasks is the matrix product

\[
G \approx R\,J\,B,
\]

where `B = model.input.weight`, `J = model.recurrent.weight`, and `R = model.readout.weight`. For a nonlinear RNN, include an average activation derivative later. In code:

```python
def effective_spin_operator(model):
    B = model.input.weight       # H x n
    J = model.recurrent.weight   # H x H
    R = model.readout.weight     # n x H
    return R @ J @ B             # n x n
```

This gives an operator in spin space and can be compared to \(A\), \(C\), and lagged covariance.

For the minimal experiment, compute both when possible:

- `J_hidden = model.J.detach()`
- `G_spin = effective_spin_operator(model).detach()`

Use `G_spin` for physical comparisons.

### `geometry_report(model, J0, A, states, meta, k=5)`

Return a dictionary:

```python
{
    "eff_rank_J": ...,
    "eff_rank_delta_J": ...,
    "energy_rank_delta_J_95": ...,
    "eff_rank_G": ...,
    "align_G_A": ...,
    "align_G_C": ...,
    "align_G_lag": ...,
    "same_block_mean": ...,      # only if block labels exist
    "diff_block_mean": ...,      # only if block labels exist
}
```

Where:

- `J0` is the initial recurrent matrix,
- `A` is the true Ising coupling,
- `states` are validation trajectories,
- `C = centered_cov(states)`.

For lagged modes, start with raw lagged covariance:

```python
Clag = lagged_cov(states, lag=1)
```

Do not implement whitening in v1 unless needed.

## File 4: `lowrank_rnn/train/spin_trainer.py`

Purpose: one compact training function.

```python
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
import torch.nn as nn

from lowrank_rnn.data.ising import IsingConfig, sample_ising_batch
from lowrank_rnn.models.vanilla_rnn import VanillaRNN, VanillaRNNConfig
from lowrank_rnn.analysis.spin_geometry import geometry_report

@dataclass
class SpinTrainConfig:
    n_spins: int = 64
    hidden_dim: int = 64
    beta: float = 0.5
    graph_kind: str = "lattice_2d"
    seq_len: int = 100
    batch_size: int = 64
    epochs: int = 300
    lr: float = 1e-3
    alpha: float = 0.2
    burn_in: int = 100
    eval_every: int = 25
    seed: int = 0
    device: str = "cpu"
    save_dir: str = "runs/spin_rnn_debug"
```

### `train_spin_prediction(cfg)`

Pseudo-code:

```python
def train_spin_prediction(cfg: SpinTrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    ising_cfg = IsingConfig(
        n_spins=cfg.n_spins,
        beta=cfg.beta,
        graph_kind=cfg.graph_kind,
        seed=cfg.seed,
        device=cfg.device,
    )

    model = VanillaRNN(VanillaRNNConfig(
        input_dim=cfg.n_spins,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.n_spins,
        alpha=cfg.alpha,
        nonlinearity="tanh",
        device=cfg.device,
    ))

    J0 = model.J.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history = []
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        states, A, meta = sample_ising_batch(
            ising_cfg,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            burn_in=cfg.burn_in,
        )
        x = states[:, :-1, :]
        y_true = states[:, 1:, :]

        y_pred, _ = model(x, return_states=False)
        loss = loss_fn(y_pred, y_true)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch == 1 or epoch % cfg.eval_every == 0:
            with torch.no_grad():
                val_states, A, meta = sample_ising_batch(
                    ising_cfg,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    burn_in=cfg.burn_in,
                )
                report = geometry_report(model, J0, A, val_states, meta, k=5)
                report["epoch"] = epoch
                report["loss"] = float(loss.item())
                history.append(report)
                print(report)

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "history": history,
    }, save_dir / "final.pt")

    return model, history
```

Keep the trainer intentionally small. No live plotting in v1. Save JSON and plot afterward.

## File 5: `scripts/run_spin_experiment.py`

Purpose: CLI entry point.

Minimal argparse:

```python
import argparse
from lowrank_rnn.train.spin_trainer import SpinTrainConfig, train_spin_prediction

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-spins", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--graph-kind", type=str, default="lattice_2d")
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-dir", type=str, default="runs/spin_rnn_debug")
    args = p.parse_args()

    cfg = SpinTrainConfig(
        n_spins=args.n_spins,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        graph_kind=args.graph_kind,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )
    train_spin_prediction(cfg)

if __name__ == "__main__":
    main()
```

## First experiment matrix

Run only these first:

```bash
python scripts/run_spin_experiment.py --graph-kind curie_weiss --beta 0.2 --n-spins 64 --hidden-dim 64 --epochs 300 --device cpu --save-dir runs/curie_beta02

python scripts/run_spin_experiment.py --graph-kind lattice_2d --beta 0.2 --n-spins 64 --hidden-dim 64 --epochs 300 --device cpu --save-dir runs/lattice_beta02

python scripts/run_spin_experiment.py --graph-kind lattice_2d --beta 0.44 --n-spins 64 --hidden-dim 64 --epochs 300 --device cpu --save-dir runs/lattice_beta044

python scripts/run_spin_experiment.py --graph-kind block --beta 0.5 --n-spins 64 --hidden-dim 64 --epochs 300 --device cpu --save-dir runs/block_beta05
```

Use `n_spins=64` because an `8 x 8` lattice is small but nontrivial.

For the 2D square-lattice Ising model, the infinite-volume critical inverse temperature is

\[
\beta_c = \frac{1}{2}\log(1+\sqrt 2) \approx 0.4406868
\]

when coupling is set to 1 and there is no external field. Finite-size effects will blur this, but `beta=0.44` is a useful near-critical test.

## What plots to generate after v1

Add plotting only after training works.

Required plots:

1. loss versus epoch,
2. singular values of `Delta J`,
3. effective rank versus epoch,
4. `align_G_A`, `align_G_C`, and `align_G_lag` versus epoch,
5. heatmap of `A`, `G_spin`, and `Delta J` for `hidden_dim=n_spins`,
6. block same/different means for block graph.

Put these in a later script:

```text
scripts/plot_spin_results.py
```

Do not add plotting to the trainer.

## Minimal success criteria

The experiment is working if:

1. training loss decreases,
2. `G_spin` has non-random alignment with at least one physical operator,
3. high-temperature full-state prediction shows stronger alignment with `A` than random control,
4. near-critical or block cases show changed alignment with covariance/block modes,
5. `Delta J` has interpretable singular spectrum changes across tasks/regimes.

## Random controls

Add random controls early. Without them, alignment scores are hard to interpret.

For every alignment score, compare against:

```python
M_random = torch.randn_like(G_spin)
```

and report:

```python
align_random_A
align_random_C
align_random_lag
```

A result is meaningful only if learned alignment consistently beats random alignment across seeds.

## Seeds and reproducibility

For v1, use three seeds:

```bash
--seed 0
--seed 1
--seed 2
```

Do not run twenty seeds until the experiment makes sense.

## Extensions after v1

Only after the basic one-step task works:

### Add denoising

Corrupt input spins:

```python
mask = torch.rand_like(states[:, :-1, :]) > p_mask
noise = 2 * torch.randint_like(states[:, :-1, :].long(), 0, 2).float() - 1
x = torch.where(mask, states[:, :-1, :], noise)
y_true = states[:, :-1, :]
```

Prediction target is the clean current state, not the next state.

### Add partial observation

Observe only a subset of spins. Either zero-fill missing coordinates or add a mask channel. Minimal version: zero-fill.

### Add macroscopic target

Predict magnetization:

```python
y_true = states[:, 1:, :].mean(dim=-1, keepdim=True)
```

Set `output_dim=1`.

### Add asynchronous Glauber

This is more faithful but harder to vectorize. Add only after synchronous dynamics works.

## What not to build yet

Do not add:

- Hydra,
- Weights & Biases,
- a giant model zoo,
- complicated plotting dashboards,
- notebook-first logic,
- full inverse-Ising baselines,
- continuous-time generators,
- large-scale GPU sweeps.

The first version should answer one question:

> Does the learned effective spin-space operator align more with the true coupling matrix, the empirical covariance, or lagged dynamical modes?

Everything else is secondary.

## Recommended first commit sequence

1. Add `ising.py` and test that trajectories have shape `(B, T+1, n)`.
2. Add `vanilla_rnn.py` and test a forward pass.
3. Add `spin_geometry.py` and test diagnostics on random matrices.
4. Add `spin_trainer.py` and run one CPU experiment.
5. Add plotting only after the JSON history contains reasonable values.

This keeps the codebase minimal and prevents theory bloat from turning into implementation bloat.