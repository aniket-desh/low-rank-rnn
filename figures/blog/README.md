# Blog / LessWrong figure suite

Publication-quality figures and animations for the spin-RNN writeup. All
static figures are emitted in three formats (`.png` at dpi 300, `.svg` for
vector, `.pdf` for paper-style use). Animations are MP4 + GIF preview.

| Figure | File | What it shows | Best use |
|---|---|---|---|
| Main result | [`main_result_triptych.png`](main_result_triptych.png) | Tier 4 — r_eff(G), Δ_baseline, relative align(G,A) vs β, one curve per n. β_c marked. | Top-of-post hero figure. |
| FSS phase map | [`effective_rank_heatmap.png`](effective_rank_heatmap.png) | r_eff(G)/n as a (β, n) heatmap. | The "phase diagram of learned dimension". |
| Raw FSS map | [`effective_rank_raw_heatmap.png`](effective_rank_raw_heatmap.png) | r_eff(G) (un-normalized). | Companion to the heatmap; emphasises absolute scale. |
| Log FSS map | [`effective_rank_log_heatmap.png`](effective_rank_log_heatmap.png) | log₁₀ r_eff(G). | Dynamic-range view. |
| Scaling exponent | [`scaling_exponent_alpha.png`](scaling_exponent_alpha.png) | Bootstrapped α(β) in r_eff ~ n^α with 95% intervals. | Demonstrates critical exponent. |
| Scaling exponent data | [`scaling_exponent_alpha.csv`](scaling_exponent_alpha.csv) | α(β) numerical table for reuse. | LaTeX / spreadsheet imports. |
| Geometry — heatmaps | [`matrix_heatmaps_lattice64.png`](matrix_heatmaps_lattice64.png) | A, C, C_τ, G heatmaps at β=0.2/0.44/0.8 for n=64. | Visual identity of the learned operator. |
| Geometry — spectra | [`singular_values_lattice64.png`](singular_values_lattice64.png) | σᵢ(G) vs index for three β. | Rank collapse story. |
| Geometry — top modes | [`top_modes_lattice64.png`](top_modes_lattice64.png) | Top-4 left singular vectors of G reshaped to 8×8. | Show emergent spatial patterns. |
| Task contrast | [`task_induced_coarse_graining.png`](task_induced_coarse_graining.png) | Tier 5 — rank/Δ/alignment by task + rank-vs-performance scatter. | The "denoise halves r_eff with no Δ cost" point. |
| Graph families | [`graph_family_comparison.png`](graph_family_comparison.png) | Tier 2 — Δ, r_eff, align(G,A) vs β for lattice / Curie / block. | Cross-family story. |
| Rel-align phase diagram | [`relative_alignment_phase_diagram.png`](relative_alignment_phase_diagram.png) | align(G,A) / align_rand vs β per n. | Corrects the k/n random-baseline artifact. |
| Rel-align triptych | [`relative_alignment_A_C_Ctau.png`](relative_alignment_A_C_Ctau.png) | Same, for A / C / C_τ side-by-side. | Operator comparison. |

## Animations (regenerate locally; not committed)

| File | What it shows | Best use |
|---|---|---|
| `animations/G_heatmap_beta0p200_next_state.mp4` | G = R J B evolving during training at β=0.2 (high T). | Sub-fig in temperature triptych. |
| `animations/G_heatmap_beta0p440_next_state.mp4` | Same at β=0.44 (critical). | Sub-fig in temperature triptych. |
| `animations/G_heatmap_beta0p800_next_state.mp4` | Same at β=0.8 (low T). | Sub-fig in temperature triptych. |
| `animations/G_heatmap_beta0p440_denoise.mp4` | Same at β=0.44 for the denoise task. | Task-contrast story (Tier 5). |
| `animations/G_heatmap_beta0p440_partial.mp4` | Same at β=0.44 for the partial task. | Task-contrast story (Tier 5). |
| `animations/singular_spectrum_three_betas.mp4` | σ(G) vs index, three β panels, training time. | **Best single explanatory animation.** |
| `animations/top_modes_beta0p440.mp4` | Top-4 left singular vectors of G near criticality, training time. | Visualize "what the network looks at". |
| `animations/training_dashboard_three_betas.mp4` | r_eff, Δ, align, spectrum dashboard, three β. | Full training-dynamics view. |
| `*.gif` | Lower-fps GIF preview of each MP4. | Embeddable preview, autoplay. |

## How to (re)generate

```bash
# Static figures (uses completed Tiers 1–5 + Tier 6 if present):
python scripts/build_blog_figures.py --all

# Tier-specific:
python scripts/build_blog_figures.py --tier tier4
python scripts/build_blog_figures.py --tier tier5
python scripts/build_blog_figures.py --tier tier6

# Animations (requires checkpointed runs):
bash scripts/run_animation_sources.sh        # 5 runs × ~3 min each
python scripts/build_animations.py            # render all
```

## Tier 6 status

`tier6_zoom_*` runs: a 189-job dense β-zoom around β_c at n ∈ {64, 256, 1024}.
See `figures/blog/tier6_zoom_stats.md` (created by the post-run hook) when
complete.
