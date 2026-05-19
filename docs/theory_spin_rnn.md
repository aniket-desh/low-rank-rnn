# Emergent recurrent geometry from learning spin-system dynamics

This note reframes the original low-rank RNN project into a more defensible theoretical problem. The old project imposed a recurrent matrix of the form

$$
J = gW - \frac{b}{N}\mathbf 1\mathbf 1^\top + m u v^\top,
$$

then trained mostly the readout to track an Ornstein--Uhlenbeck process. That is a legitimate low-rank-plus-random RNN model, but it does not strongly support the claim that low-rank recurrent structure emerges from learning. A stronger project is to train an unconstrained or weakly constrained RNN on trajectories from an Ising/Glauber system and ask what recurrent geometry is learned.

The core question should be:

> When an RNN is trained to predict, filter, or coarse-grain stochastic spin dynamics, does its learned recurrent update align with the microscopic coupling matrix, the equilibrium covariance modes, or the dynamical slow modes of the spin system?

This avoids reproducing old results about random neural networks, Hopfield/Ising networks, inverse Ising reconstruction, or low-rank RNNs. The target is not merely "RNNs learn low rank." The target is a phase-dependent operator-identification question.

## Prior-art boundary

The following claims are not novel enough by themselves.

1. Random recurrent networks have mean-field phase transitions and chaos. Sompolinsky, Crisanti, and Sommers studied large random continuous-time networks and derived a transition from stationary dynamics to chaos as gain crosses a critical value.

2. Recurrent networks and Ising/Hopfield systems have a long shared history. Symmetric recurrent networks with binary states are closely related to Ising energy landscapes and associative-memory spin-glass models.

3. Low-rank RNN theory already explains how random-plus-structured connectivity produces low-dimensional dynamics. Mastrogiuseppe and Ostojic analyze recurrent matrices with a random bulk plus low-dimensional structure, and Schuessler et al. show that trained RNNs on low-dimensional tasks can develop low-rank connectivity changes even without explicitly constraining the recurrent matrix.

4. Inverse Ising and graphical-model learning from Glauber trajectories are established. If the project is simply "observe spin trajectories and recover the Ising graph," it becomes an inverse-Ising project.

5. Neural networks have already been used to classify Ising phases and estimate phase transitions from spin configurations.

The novelty should therefore be narrower and sharper: compare the geometry of the learned recurrent matrix to several theoretically meaningful operators of the underlying spin process.

Useful references:

- Sompolinsky, Crisanti, Sommers, *Chaos in Random Neural Networks*, Phys. Rev. Lett. 1988. DOI: `10.1103/PhysRevLett.61.259`.
- Mastrogiuseppe and Ostojic, *Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks*, Neuron 2018. DOI: `10.1016/j.neuron.2018.07.003`.
- Schuessler, Mastrogiuseppe, Dubreuil, Ostojic, Barak, *The Interplay Between Randomness and Structure During Learning in RNNs*, NeurIPS 2020.
- Bresler, Gamarnik, Shah, *Learning Graphical Models from the Glauber Dynamics*, Allerton 2014 / arXiv:1410.7659.

## Ising model setup

Let

$$
s_t \in \{-1,+1\}^n
$$

be a spin configuration. The Ising Hamiltonian is

$$
H(s) = -\frac{1}{2}s^\top A s - h^\top s,
$$

where

- $A \in \mathbb R^{n\times n}$ is symmetric with zero diagonal,
- $h \in \mathbb R^n$ is an external field,
- $\beta \ge 0$ is inverse temperature.

The equilibrium distribution is

$$
\pi_\beta(s) = \frac{1}{Z_\beta}\exp\left(\beta\left[\frac{1}{2}s^\top A s + h^\top s\right]\right).
$$

The local field at spin $i$ is

$$
\ell_i(s) = h_i + \sum_{j\ne i} A_{ij}s_j.
$$

Under heat-bath Glauber dynamics, one index $i_t$ is selected and resampled according to

$$
\mathbb P(s_i(t+1)=+1\mid s_{-i}(t)) = \sigma(2\beta \ell_i(s_t)),
$$

where $\sigma(x)=(1+e^{-x})^{-1}$. Equivalently,

$$
\mathbb E[s_i(t+1)\mid s_{-i}(t)] = \tanh(\beta \ell_i(s_t)).
$$

For synchronous update, all spins are resampled in parallel:

$$
\mathbb E[s(t+1)\mid s(t)] = \tanh\big(\beta(A s(t)+h)\big),
$$

where the nonlinearity is coordinatewise. Synchronous dynamics is easier for neural-network training; asynchronous dynamics is closer to standard Glauber dynamics. For the first experiment, use synchronous or block-synchronous updates because it produces clean `(input state, next state)` pairs.

## High-temperature linearization

At high temperature or weak coupling,

$$
\tanh(x)=x-\frac{x^3}{3}+O(x^5).
$$

Therefore,

$$
\mathbb E[s(t+1)\mid s(t)]
= \tanh\big(\beta(A s(t)+h)\big)
\approx \beta A s(t)+\beta h.
$$

This gives the first baseline prediction:

> In the high-temperature one-step prediction task with full-state observation, a sufficiently expressive predictor should learn an operator aligned with $A$, not necessarily a low-rank operator.

This is important because it prevents overclaiming. If the target is one-step full-state prediction, the natural object is the microscopic interaction matrix $A$. Low-rank emergence is not guaranteed unless $A$ itself has low-rank structure or the task compresses the state.

## Equilibrium covariance and susceptibility modes

Define the magnetization vector

$$
\mu_\beta = \mathbb E_{\pi_\beta}[s]
$$

and the covariance matrix

$$
C_\beta = \mathbb E_{\pi_\beta}\left[(s-\mu_\beta)(s-\mu_\beta)^\top\right].
$$

The susceptibility is the response of magnetization to external field:

$$
\chi_\beta = \frac{\partial \mu_\beta}{\partial h}.
$$

For an Ising model, covariance and susceptibility are closely related by fluctuation-response identities up to temperature conventions. Near criticality, the covariance develops dominant collective modes. For a ferromagnet, the top mode is close to the uniform magnetization direction

$$
\mathbf m = \frac{1}{\sqrt n}\mathbf 1.
$$

For a modular/block Ising model, the top modes correspond to community magnetizations. For spin-glass-like systems, the spectrum can become more distributed and frustrated.

This gives the second prediction:

> If the RNN task is denoising, partial observation, phase classification, or macroscopic observable prediction, the learned recurrent geometry may align more strongly with $C_\beta$'s dominant modes than with $A$'s microscopic entries.

This is the coarse-graining view: the network learns statistical sufficient directions, not necessarily the true physical coupling matrix.

## Dynamical slow modes

Let $P_\beta$ be the Markov transition kernel of the Glauber chain. The slow modes of the dynamics are the leading nontrivial eigenfunctions of $P_\beta$ or of the continuous-time generator $\mathcal L_\beta$. They control relaxation, metastability, and mixing.

For a finite chain,

$$
P_\beta f_k = \lambda_k f_k,
$$

with $1=\lambda_0 > |\lambda_1| \ge |\lambda_2|\ge \cdots$. The eigenfunction $f_0$ is constant. The next eigenfunctions describe slow collective coordinates. Near a ferromagnetic critical point, slow modes are magnetization-like. In low-temperature multimodal regimes, slow modes separate metastable basins.

This gives the third prediction:

> If the task requires multi-step prediction or hidden-state filtering, the RNN may learn modes aligned with the slow eigenspaces of $P_\beta$, rather than with either $A$ or $C_\beta$ alone.

In practice, exactly computing $P_\beta$ is impossible for large $n$, but we can estimate slow modes using trajectory data: time-lagged covariance, dynamic mode decomposition, or principal components of delayed features.

## RNN setup

Use an unconstrained continuous-state RNN:

$$
h_{t+1} = (1-\alpha)h_t + \alpha\phi(Jh_t + Bx_t + b),
$$

$$
y_t = Rh_t + c.
$$

Here

- $x_t$ is the observed spin vector, possibly corrupted or partially observed,
- $h_t\in\mathbb R^N$ is the RNN state,
- $J\in\mathbb R^{N\times N}$ is trainable,
- $B$ is an input map,
- $R$ is a readout,
- $\phi$ can be `tanh` or `relu`, with `tanh` more natural for spin variables,
- $\alpha=\Delta t/\tau$ controls update timescale.

The central learned object is

$$
\Delta J = J_{\mathrm{trained}} - J_0.
$$

The experiment should analyze $J_{\mathrm{trained}}$, $\Delta J$, and the linearized recurrent Jacobian

$$
\mathcal J_t = \frac{\partial h_{t+1}}{\partial h_t}
= (1-\alpha)I + \alpha\operatorname{diag}\left(\phi'(Jh_t+Bx_t+b)\right)J.
$$

The time-averaged Jacobian is

$$
\bar{\mathcal J}=\mathbb E_t[\mathcal J_t].
$$

For nonlinear RNNs, $\bar{\mathcal J}$ may be more meaningful than the raw recurrent matrix $J$, because it includes which units are active in the task regime.

## Candidate tasks

### Task A: full-state one-step prediction

Input:

$$
x_t=s_t.
$$

Target:

$$
y_t=s_{t+1}.
$$

Loss:

$$
\mathcal L = \mathbb E_t\|y_t-s_{t+1}\|_2^2.
$$

Expected geometry: high-temperature alignment with $A$, weaker evidence for low-rank emergence unless $A$ or the dynamics are low-dimensional.

### Task B: denoising

Input:

$$
x_t = M_t \odot s_t + (1-M_t)\odot \xi_t,
$$

where $M_t$ is a random observation mask and $\xi_t$ is noise.

Target:

$$
y_t=s_t.
$$

Expected geometry: alignment with covariance and community modes; more likely low-rank or block structure.

### Task C: partial-observation filtering

Input: subset of spins

$$
x_t = P_\Omega s_t.
$$

Target: full state, future state, or macroscopic observables.

Expected geometry: hidden state should encode collective latent variables; learned recurrent structure may align with slow modes.

### Task D: macroscopic observable prediction

Target options:

$$
m_t=\frac{1}{n}\sum_i s_i(t),
$$

$$
E_t=-\frac{1}{2}s_t^\top A s_t-h^\top s_t,
$$

or phase label / inverse temperature.

Expected geometry: low-dimensional structure is most plausible here, but this also overlaps most with known "low-dimensional task induces low-rank updates" results. The novelty comes from comparing the learned modes to statistical-mechanical observables.

## Graph families

The experiment should not use only one Ising graph. Use families whose theoretical modes differ.

### Curie--Weiss / mean-field ferromagnet

$$
A = \frac{J_0}{n}\mathbf 1\mathbf 1^\top.
$$

This is explicitly rank one. It is good for sanity checks but weak for novelty, because low-rank structure is built into the data-generating process.

### 2D lattice ferromagnet

Spins live on an $L\times L$ grid. $A_{ij}=J_0$ for nearest neighbors and zero otherwise. The coupling matrix is sparse/local, not low-rank. Near criticality, large-scale modes dominate correlations.

This is a strong test of whether the RNN learns local couplings, long-wavelength covariance modes, or coarse magnetization variables.

### Modular/block Ising model

Partition spins into communities. Let

$$
A_{ij}=\begin{cases}
J_{\mathrm{in}}/n, & i,j\text{ same block},\\
J_{\mathrm{out}}/n, & i,j\text{ different blocks}.
\end{cases}
$$

This creates interpretable community modes. Learned connectivity may become block structured rather than purely low rank.

### SK / spin-glass-like model

$$
A_{ij}\sim \mathcal N(0,J_0^2/n).
$$

This is a harder negative control: there may be no clean small number of modes, especially in frustrated regimes.

## Diagnostics

### Singular spectrum and effective rank

For a matrix $M$, let $\sigma_1\ge\cdots\ge\sigma_N$ be singular values. Define participation effective rank:

$$
r_{\mathrm{eff}}(M)=\frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2}.
$$

Also compute energy rank:

$$
r_\epsilon(M)=\min\left\{k:\frac{\sum_{i=1}^k\sigma_i^2}{\sum_i\sigma_i^2}\ge 1-\epsilon\right\}.
$$

Apply this to $J_0$, $J_{\mathrm{trained}}$, $\Delta J$, and $\bar{\mathcal J}$.

### Alignment with coupling matrix

Let $U_k(M)$ denote the top-$k$ left singular subspace of $M$. Define subspace alignment

$$
\mathrm{align}_k(M,N)=\frac{1}{k}\|U_k(M)^\top U_k(N)\|_F^2.
$$

This lies in $[0,1]$. Compute

$$
\mathrm{align}_k(\Delta J,A),\qquad
\mathrm{align}_k(J_{\mathrm{trained}},A).
$$

For symmetric $A$, eigenspaces can be used instead of singular subspaces.

### Alignment with covariance modes

Estimate

$$
\hat C_\beta=\frac{1}{T}\sum_{t=1}^T(s_t-\bar s)(s_t-\bar s)^\top.
$$

Then compute

$$
\mathrm{align}_k(\Delta J,\hat C_\beta),
\qquad
\mathrm{align}_k(\bar{\mathcal J},\hat C_\beta).
$$

If this alignment beats coupling alignment for denoising/filtering tasks, that supports the coarse-graining hypothesis.

### Alignment with slow modes

Estimate a lagged covariance

$$
C_\tau = \mathbb E[(s_t-\mu)(s_{t+\tau}-\mu)^\top].
$$

Use the whitened operator

$$
K_\tau = C_0^{-1/2}C_\tau C_0^{-1/2}.
$$

The top eigenvectors of $K_\tau$ approximate slow predictive directions. Compare learned recurrent geometry with $K_\tau$:

$$
\mathrm{align}_k(\bar{\mathcal J},K_\tau).
$$

This is especially important for multi-step prediction and partial observation.

### Block/modularity diagnostics

For modular Ising graphs, compare within-block and between-block learned weights:

$$
\bar J_{\mathrm{same}} = \frac{1}{|S|}\sum_{(i,j):g_i=g_j} J_{ij},
$$

$$
\bar J_{\mathrm{diff}} = \frac{1}{|D|}\sum_{(i,j):g_i\ne g_j} J_{ij}.
$$

This avoids forcing everything through a low-rank lens. The emergent structure may be block-like.

### Non-normality

Trained recurrent matrices need not be symmetric. Non-normal amplification can matter even when eigenvalues look stable. Compute the Henrici departure from normality:

$$
d_{\mathrm{Henrici}}(J)=\frac{\|J^\top J-JJ^\top\|_F}{\|J\|_F^2}.
$$

This is optional but useful if the learned dynamics are transient-amplifying rather than attractor-like.

## Hypotheses

### H1: high-temperature full-state prediction recovers microscopic couplings

For high temperature and full-state one-step prediction, the learned update should align with $A$, because the conditional expectation is approximately linear in $As_t$.

### H2: near-critical denoising and macroscopic prediction recover covariance/susceptibility modes

Near criticality, covariance develops large collective modes. For denoising, filtering, and macroscopic prediction, $\Delta J$ or $\bar{\mathcal J}$ should align more with $C_\beta$ than with raw $A$.

### H3: low-temperature partial-observation filtering learns metastable basin coordinates

At low temperature, trajectories spend long periods in metastable basins. Learned recurrent dynamics should reflect basin memory and slow switching directions. Alignment with lagged covariance / slow modes should increase.

### H4: low rank is phase- and task-dependent, not universal

Low-rank structure should be strongest when the task asks for compressed collective information. It should be weaker for full-state microscopic prediction on sparse lattices or spin glasses.

## Minimal theoretical contribution

A small but real theoretical result can be included as a proposition.

**Proposition: high-temperature one-step predictor.** Consider synchronous Glauber dynamics with zero field and small $\beta\|A\|$. Then

$$
\mathbb E[s_{t+1}\mid s_t] = \beta A s_t + O(\beta^3\|A s_t\|^3).
$$

Therefore, the population-risk minimizer among linear predictors $y=Ws_t$ satisfies

$$
W^* \approx \beta A
$$

when the input covariance is sufficiently well-conditioned.

Sketch:

$$
\mathbb E[s_{t+1}\mid s_t]=\tanh(\beta A s_t),
$$

and Taylor expansion gives the first statement. The linear least-squares optimum is

$$
W^*=\mathbb E[s_{t+1}s_t^\top]\,\mathbb E[s_ts_t^\top]^{-1}.
$$

Using the conditional expectation,

$$
\mathbb E[s_{t+1}s_t^\top]
=\mathbb E[\tanh(\beta A s_t)s_t^\top]
\approx \beta A\mathbb E[s_ts_t^\top],
$$

hence $W^*\approx \beta A$.

This theorem tells us exactly when the experiment should recover $A$. Deviations from $A$-alignment in other tasks/regimes are then meaningful rather than accidental.

## Main novelty statement

The strongest formulation is:

> We study which statistical-mechanical operator is represented in the recurrent geometry of a trained RNN. By varying the Ising graph, temperature, and task, we distinguish microscopic coupling recovery from covariance-mode learning and dynamical slow-mode learning. Low-rank structure is treated as one possible emergent geometry, not as the assumed outcome.

That is the project.