# The RNN Learned an Effective Theory of the Ising Model

*Training a recurrent network on Ising dynamics and projecting its recurrence
back into spin space gives a learned operator $G=RJB$ whose dimension, spectrum,
and modes track phase, scale, graph symmetry, and task-induced
coarse-graining.*

---

A recurrent neural network has no reason to organize its hidden units like the
physical degrees of freedom in the data. Hidden unit 17 is not spin 17. The
recurrent matrix $J$ is written in an arbitrary learned basis. If we train an
RNN on a physical system and then stare directly at $J$, we are looking in the
wrong coordinates.

The trick in this project was to put the network back into the physical basis.

I trained a vanilla RNN on trajectories from an Ising model. The data are spin
configurations

$$
s_t\in\{-1,+1\}^n,
$$

evolving under synchronous Glauber dynamics with a known coupling matrix
$A\in\mathbb R^{n\times n}$. The RNN sees $s_t$ and predicts $s_{t+1}$. Its
input map is $B$, its recurrent matrix is $J$, and its readout is $R$. So the
learned recurrent pathway, projected back into spin space, is

$$
G = R\,J\,B.
$$

This is the central object. $J$ lives in hidden space and is not directly
comparable to anything physical. $G$ lives in spin space. That means $G$ can
be compared to the physical operators of the Ising model: the microscopic
coupling $A$, the equilibrium covariance $C$, and the lagged covariance
$C_\tau$.

The question is no longer "did the RNN learn a low-rank operator?"

The better question is:

> Which effective physical degrees of freedom does the RNN use to solve the
> task?

![Singular spectrum of G during training at three temperatures](figures/blog/animations/singular_spectrum_three_betas.gif)

*The singular spectrum of the learned spin-space recurrent operator
$G=RJB$ during training, at high temperature ($\beta=0.2$), near
criticality ($\beta=0.44$), and low temperature ($\beta=0.8$).
[MP4](figures/blog/animations/singular_spectrum_three_betas.mp4).*

The animation above is already most of the story. The high-temperature
operator keeps many modes alive. The low-temperature operator collapses
toward a few dominant directions. Criticality sits between them, and reaches
its final spectrum more slowly. None of these training runs were given any
notion of "phase" — they only saw spin trajectories.

The rest of this post is an attempt to say carefully what just happened.

---

## Loss and geometry are different observables

The RNN is trained with MSE on $\pm 1$ spin targets. The zero predictor has
loss $\approx 1$, so I measure prediction improvement by

$$
\Delta_{\rm baseline} = \frac{L_{\rm zero} - L_{\rm model}}{L_{\rm zero}}.
$$

Prediction gets easier as $\beta$ increases: at high temperature the next
state is genuinely noisy; at low temperature it is mostly determined by the
current one. Nothing surprising there.

The geometry of $G$ tells a different story.

![main result triptych](figures/blog/main_result_triptych.png)

*$n \in \{64, 256, 1024\}$, 21 βs around the critical region × 3 seeds.
Left: effective rank of $G$. Centre: prediction improvement. Right:
alignment of $G$'s top-10 left singular subspace with $A$'s, normalized by
a random baseline. Vertical line: lattice critical inverse temperature
$\beta_c = \tfrac{1}{2}\ln(1+\sqrt 2) \approx 0.4407$.*

The clean separation is

$$
\Delta_{\rm baseline} \approx f(\beta),
\qquad
r_{\rm eff}(G) \approx f(\beta, n).
$$

Prediction difficulty is set by temperature. Learned geometry is set by
temperature *and* scale.

This makes sense at a glance. The conditional law of Glauber dynamics is local,

$$
\mathbb E[s_i(t+1)\mid s_t] = \tanh\!\big(\beta\,\ell_i(s_t)\big),
$$

so the per-site prediction problem has a thermodynamic-limit difficulty set
mostly by $\beta$. But the *global* operator $G$ has to choose a basis of
spatial modes, and how many modes it can usefully resolve depends on the
system size.

Loss asks whether the network predicts. $G$ asks what degrees of freedom it
used.

---

## A learned phase diagram

The most compact way to see the result is to plot the effective rank of $G$,
normalized by system size, as a function of $\beta$ and $n$.

![Effective rank phase map](figures/blog/effective_rank_heatmap.png)

*Phase map of $r_{\rm eff}(G)/n$ from a dense critical zoom (21 βs evenly
spaced from 0.400 to 0.500). The vertical dashed line marks $\beta_c$.*

At high temperature, the operator is broad: the network needs many
microscopic directions because the dynamics are local and noisy. As $\beta$
increases, the operator compresses. The Ising model develops stronger
correlations and eventually an ordered phase, and the network no longer needs
as many independent spin-space directions — it can predict through collective
structure.

The first version of this experiment, on a coarser β grid, made this look
like a sharp collapse at $\beta_c$. The dense critical sweep tells a better
story: finite systems see a smooth crossover. The learned rank does not fall
off a cliff. It flows.

To make that precise I fit, at each $\beta$,

$$
r_{\rm eff}(G; n, \beta) \sim n^{\alpha(\beta)},
$$

using $n \in \{64, 256, 1024\}$ and bootstrap over seeds.

![Scaling exponent α(β)](figures/blog/scaling_exponent_alpha.png)

*Bootstrapped finite-size scaling exponent $\alpha(\beta)$ in
$r_{\rm eff}(G)\sim n^{\alpha(\beta)}$. Bars: 95% interval from 1000
seed-resamples per β.*

$\alpha(\beta)$ rises from about $1.20$ at $\beta = 0.400$, peaks near
$\beta \approx 0.465\text{–}0.470$ at $\alpha \approx 1.42$, then drifts down
to $\approx 1.37$ at $\beta = 0.500$. The maximum is **inside** the critical
window but slightly above the infinite-volume $\beta_c$ — the finite-size
shift you would expect.

I would not call $\alpha(\beta)$ a universal critical exponent. It is an
empirical finite-size scaling slope for the learned operator. But it is the
right *kind* of observable: it asks how the number of useful recurrent
degrees of freedom scales with system size, and it confirms that the
learned representation has a scale-dependent critical window.

The result is not that the RNN detects a mathematically exact $\beta_c$. The
result is that the learned recurrent geometry has a scale-dependent critical
window centered slightly above it.

That's more interesting.

---

## What does the learned operator look like?

The singular values tell us how many directions matter. The singular vectors
tell us what those directions are.

![Singular spectrum of G at three β](figures/blog/singular_values_lattice64.png)

*$\sigma_i(G)$ vs index for the lattice $n=64$ at three temperatures.
At $\beta = 0.2$ the tail stays well above $10^{-1}$; at $\beta = 0.44$
the spectrum kinks after the first $\sim 4$ modes; at $\beta = 0.8$ only
the top two modes survive.*

![Top-4 left singular vectors of G](figures/blog/top_modes_lattice64.png)

*Top-4 left singular vectors of $G$ for the lattice $n=64$, reshaped onto
the $8\times 8$ grid. Each row is one temperature. The modes are spatial.*

This is where the "effective theory" language becomes visually literal. The
learned recurrent pathway is not a black-box matrix with lower rank. It has
*modes*, and the modes look like spatial Fourier components of the
underlying lattice.

There is a subtlety worth flagging. A post-hoc $k$-sweep over alignment
showed that the top *two*-dimensional subspace of $G$ aligns essentially
perfectly with the top-2 of $A$ across every temperature tested. So the
right story is not

> the RNN discovers the physical coupling near criticality

but

> the RNN robustly finds a core physical subspace, and temperature controls
> how many additional modes survive.

The core is found. The tail is pruned.

---

## Not everything becomes low-rank

If I had only studied the lattice, I might have written the wrong post.
Maybe RNNs just like low rank; maybe training always compresses recurrent
dynamics into a few directions; maybe the Ising model is incidental.

So I also trained on Curie–Weiss and block Ising systems. Curie–Weiss has
all-to-all couplings $J/n$ — a mean-field model whose coupling matrix is
rank-one in expectation. Block has two communities with strong within-block
coupling $J_{\rm in}/n$ and weak between-block coupling $J_{\rm out}/n$.

![Graph family comparison](figures/blog/graph_family_comparison.png)

*Phase diagrams across the three graph families at $n=64$. The lattice
shows the crossover discussed above. Curie–Weiss is essentially rank-one
until $\beta$ exceeds the mean-field transition at $\beta \approx 1$;
then $r_{\rm eff}(G)$ **grows**, because the network starts using
additional recurrent directions to track fluctuations around the
magnetization. The block model stays close to $r_{\rm eff}(G)\approx 2$ —
the two-community structure — at every temperature.*

The three graph families behave qualitatively differently. The simple "RNNs
learn low rank" reading does not survive this figure. A better one is:

> The learned operator reflects the symmetry of the data-generating process.

Curie gives a single magnetization direction; block gives two community
modes; lattice gives a scale-dependent spectrum of spatial modes. Low rank
is not the *cause* — it is one possible signature of the relevant effective
degrees of freedom.

---

## Task-induced coarse-graining

The cleanest evidence for the "effective theory" view comes from changing
the task.

At the same critical temperature, I trained three versions:

1. `next_state`: see $s_t$, predict $s_{t+1}$.
2. `denoise`: see a $30\%$-corrupted $s_t$, reconstruct the clean $s_t$.
3. `partial`: see only $50\%$ of the spins, predict the full $s_{t+1}$.

![Task-induced coarse graining](figures/blog/task_induced_coarse_graining.png)

*Same lattice ($n=64$), same temperature ($\beta=0.44$), three tasks.
Right panel: rank-vs-performance scatter. Denoise and partial sit at the
same $\Delta_{\rm baseline}$ as `next_state` but at half the effective
rank.*

At $\beta = 0.44$, all three tasks have nearly identical prediction
performance:

| task         | $\Delta_{\rm baseline}$ | $r_{\rm eff}(G)$ |
|--------------|------------------------:|-----------------:|
| `next_state` |                   0.681 |             3.93 |
| `denoise`    |                   0.680 |             1.82 |
| `partial`    |                   0.661 |             1.84 |

Denoising and partial observation **halve the rank with essentially no
performance cost**.

This is the most direct evidence that the learned geometry is task-dependent.
The same physical system, at the same temperature, induces a different
effective operator when the task changes.

In renormalization language, the task defines the scale at which the model
needs to be predictive. Corrupted or missing microscopic details are *made
irrelevant by construction*, and the network responds by keeping only the
coarse variables. Same physics, different effective theory.

---

## Watching the effective theory form

![Training dashboard](figures/blog/animations/training_dashboard_three_betas.gif)

*Training dashboard at three temperatures.
Top-left: $r_{\rm eff}(G_t)$ vs epoch. Top-right: $\Delta_{\rm baseline}$.
Bottom-left: alignments with $A$, $C$, and $C_\tau$. Bottom-right: full
singular spectrum at the current epoch.
[MP4](figures/blog/animations/training_dashboard_three_betas.mp4).*

Early in training, $G$ is effectively random. As loss falls, the spectrum
reshapes and alignment with physical operators rises. The recurrent pathway
becomes an operator with recognizable spin-space structure — and the *shape*
of that structure depends on temperature.

The point of this is not that RNNs perform an explicit renormalization
algorithm during training. They don't. The point is that the controlled
setting — a known data-generating physical system, plus a recurrent pathway
that can be projected back into the physical basis — makes the dependence
between training and effective description unusually clean.

---

## What I think this shows

The original question was:

> Do RNNs trained on Ising dynamics learn low-rank structure?

The answer is:

> Sometimes, but that was the wrong question.

The right question is:

> What effective operator does the RNN learn?

In this experiment, the answer is that $G=RJB$ tracks:

- **temperature** — phase-dependent rank, alignment, and training dynamics
- **finite-size scale** — $r_{\rm eff}$ scales sublinearly with $n$
- **the critical window** — $\alpha(\beta)$ peaks slightly above $\beta_c$
  with a smooth crossover, not a delta-function collapse
- **graph symmetry** — lattice, Curie–Weiss, and block produce
  qualitatively different rank trends
- **task** — at the same physics, denoise and partial observation halve
  the effective rank

So the result is not

$$
\text{RNNs like low rank.}
$$

The result is

$$
\text{RNNs can learn effective operators whose geometry reflects the relevant degrees of freedom of the data-generating process.}
$$

That's why this feels adjacent to renormalization. Not because I ran an RG
algorithm — I didn't — but because the learned recurrent pathway behaves like
a scale- and task-dependent effective description: it throws away
microscopic detail when the phase and task allow it, and keeps additional
modes when prediction requires them.

In this controlled setting, $G$ is an interpretable observable of what the
network has learned.

---

## Caveats

First, $G = RJB$ is a linear structural proxy for the recurrent pathway. The
RNN is nonlinear, and the true local Jacobian depends on activation
derivatives and current hidden state. So $G$ should be read as the
spin-space *shadow* of the recurrent pathway, not the complete dynamics. I
chose $G$ specifically because it lives in spin space, not because it's the
strongest possible probe. A natural follow-up is to compute the
trajectory-averaged Jacobian
$\bar{\mathcal J} = \mathbb E_t[\partial h_{t+1}/\partial h_t]$ and project
*that*.

Second, the scaling exponent $\alpha(\beta)$ is empirical. It is fit from
$n \in \{64, 256, 1024\}$ — three sizes — and bootstrapped over seeds. I am
not making any claim that it is a universal critical exponent. The point is
that it has the right structural feature (a peak inside the critical
window), not that its value is exactly $1.42$.

Third, this is a clean synthetic system. That's a feature for the
experiment: the data have a known physical operator, the task is closed-form
defined, and the RNN architecture is small enough that the recurrent
pathway has a well-defined low-dimensional projection. None of those
properties transfer for free to language models or other frontier systems.

The broader lesson is methodological:

> If you can identify the right basis and the right projected operator,
> neural network internals can become physical observables.

That's the part I think is worth exporting.

---

## Appendix: what got cut

A few figures I built but kept out of the main narrative:

- [`matrix_heatmaps_lattice64.png`](figures/blog/matrix_heatmaps_lattice64.png) —
  side-by-side heatmaps of $A$, $C$, $C_\tau$, and $G$ at three
  temperatures. Good supporting evidence that $G$'s top modes really do live
  in the same subspace as $A$'s.
- [`relative_alignment_phase_diagram.png`](figures/blog/relative_alignment_phase_diagram.png)
  and [`relative_alignment_A_C_Ctau.png`](figures/blog/relative_alignment_A_C_Ctau.png) —
  alignment normalized by the random-baseline $k/n$. The absolute alignment
  numbers can be misleading at large $n$ because the random reference
  shrinks; these plots correct for that and show the signal is monotonic in
  $n$.
- [`effective_rank_raw_heatmap.png`](figures/blog/effective_rank_raw_heatmap.png)
  and [`effective_rank_log_heatmap.png`](figures/blog/effective_rank_log_heatmap.png) —
  the FSS phase map in absolute and log scales.
- Per-temperature G-heatmap animations:
  [β=0.2](figures/blog/animations/G_heatmap_beta0p200_next_state.gif),
  [β=0.44](figures/blog/animations/G_heatmap_beta0p440_next_state.gif),
  [β=0.8](figures/blog/animations/G_heatmap_beta0p800_next_state.gif), plus
  task variants for [denoise](figures/blog/animations/G_heatmap_beta0p440_denoise.gif)
  and [partial](figures/blog/animations/G_heatmap_beta0p440_partial.gif).
- Top-4 mode animation at criticality:
  [`top_modes_beta0p440.gif`](figures/blog/animations/top_modes_beta0p440.gif).

Total: 388 production training runs on a single H100; the code, full
numerical tables, and per-run JSON histories are on GitHub. Figures are
reproducible via `python scripts/build_blog_figures.py --all` and
`python scripts/build_animations.py`.

---

*One paragraph summary, if you only have time for one: I trained RNNs on
Ising/Glauber trajectories and projected the learned recurrence back into
spin space as $G = RJB$. This made it possible to compare the network's
internal computation to physical operators like the coupling matrix $A$,
covariance $C$, and lagged covariance $C_\tau$. Across 388 training runs,
$G$'s geometry tracked the statistical mechanics of the data: its effective
rank flowed smoothly across the critical window, its finite-size scaling
peaked slightly above $\beta_c$, different graph symmetries produced
different rank trends, and denoising/partial-observation tasks compressed
the same physical content into fewer recurrent directions. The result is
not that RNNs universally learn low-rank structure; it is that, in this
controlled setting, the learned recurrent pathway behaves like a
task-dependent effective theory of the spin dynamics.*
