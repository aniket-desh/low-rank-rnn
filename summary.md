# Spin-RNN H100 run summary

Live document — updated as each tier completes on the runpod H100. The science is
laid out in `docs/theory_spin_rnn.md` (theory) and `HANDOFF.md` (queue). This
file is the running readout.

Last update: 2026-05-19, status: **smoke tests passed, Tier 1 launched**.

---

## 0. Environment

- GPU: NVIDIA H100 80GB HBM3 (single device, no other processes)
- Driver: 580.126.09, CUDA 13.0
- torch: 2.12.0+cu130, cuda available
- repo: `/workspace/aniket/low-rank-rnn`, branch `main`
- runs under `runs_h100/`, smoke output is `runs_h100/_smoke_lattice{64,256,1024}/`

---

## 1. Smoke tests (§6.5 of HANDOFF)

| n     | epochs | seq_len | batch | wall   | target  | final Δ_baseline | eff_rank_G | align(G,A) | align(G,C) |
|-------|--------|---------|-------|--------|---------|------------------|------------|------------|------------|
| 64    | 100    | 200     | 256   | 14.4 s | < 20 s  | +0.500           | 5.79       | 0.383      | 0.400      |
| 256   | 100    | 200     | 128   | 13.3 s | < 1 min | +0.546           | 11.75      | 0.455      | 0.418      |
| 1024  | 100    | 100     | 64    | 14.4 s | < 5 min | +0.464           | 33.50      | 0.801      | 0.764      |

Three observations from the smoke alone:

1. **H100 is fast enough** that the §6.1–§6.3 optimizations (local-field Glauber,
   forward vectorization, bf16 autocast) are not needed for any of the planned
   tiers. The dense `s @ A.T` Glauber update at n=1024 is sub-second per epoch.
   Skipping the optimization queue for now.
2. **Effective rank of $G_{\rm spin}=RJB$** collapses fast in all three sizes —
   from `min(n, hidden)` toward a much smaller number within 100 epochs. The
   collapse is more dramatic at smaller n (5.79 at n=64) and more gradual at
   n=1024 (33.50, still trending down).
3. **At n=1024, $\mathrm{align}(G,A)$ peaks at 0.969 around epoch 75** before
   dropping. This is striking — the random baseline at this scale is 0.008
   (k=10), so the learned operator's top-10 left singular subspace lands almost
   inside $A$'s. This is the first piece of evidence that the picture from the
   small-n CPU runs survives, and arguably sharpens, at large $n$.

![smoke: size comparison](figures/summary/smoke_size_comparison.png)

Per-run plots: `runs_h100/_smoke_lattice{64,256,1024}/plots/`.

---

## 2. Tier 1 — n=64 seed replication (20 runs)

5 seeds × {`lattice_2d` β=0.2, β=0.44; `curie_weiss` β=0.2; `block` β=0.5}
at n=hidden=64, seq_len=200, batch=256, 1000 epochs.

Purpose: rule out seed-dependence and undertraining before doing anything fancier.

_Status:_ pending launch / running. Results table will populate here when done.

---

## 3. Tier 2 — temperature phase diagram (69 runs)

3 seeds × β-sweep × {lattice_2d, curie_weiss, block} at n=hidden=64.
This is the first paper-figure-shaped experiment: $\Delta_{\rm baseline}$,
$r_{\rm eff}(G)$, $\mathrm{align}(G,A)$, $\mathrm{align}(G,C)$,
$\mathrm{align}(G,C_\tau)$ vs $\beta$ per graph family.

_Status:_ blocked on Tier 1.

---

## 4. Tier 3 — scale sweep (27 runs)

n ∈ {64, 256, 1024} × β ∈ {0.2, 0.44, 0.6} × 3 seeds, lattice_2d.

_Status:_ blocked on Tier 2.

---

## 5. Tier 4 — finite-size scaling near criticality (63 runs)

n ∈ {64, 256, 1024} × β ∈ {0.36, 0.40, 0.42, 0.44, 0.46, 0.48, 0.52} × 3 seeds.
The single headline experiment.

_Status:_ blocked on Tier 3.

---

## Methodology notes

- Loss is MSE on $\pm 1$ spin targets, so the zero-predictor baseline is $\approx 1$.
  $\Delta_{\rm baseline} = (L_{\rm zero} - L_{\rm val})/L_{\rm zero}$.
- $G_{\rm spin} = R J B$ is the linearised input→output operator in spin space;
  it is the apples-to-apples object to compare with the spin-space operators
  $A$ (true coupling), $C$ (covariance), $C_\tau$ (lag-1 covariance).
- $\mathrm{align}_k(M,N) = \frac{1}{k}\|U_k(M)^\top U_k(N)\|_F^2$ on top-$k$
  left singular subspaces. $k=5$ at n=64, $k=10$ at n≥256.
- The "random baseline" alignments come from a gaussian matrix of the same shape;
  any signal must beat these.
- Plots are rebuilt by `python scripts/build_summary_figures.py {smoke,tier1,…}`.
