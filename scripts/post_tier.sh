#!/usr/bin/env bash
# Post-tier hook: regenerate per-run plots, build the aggregate summary figure,
# print a markdown stats table, and commit + push.
#
# Usage: bash scripts/post_tier.sh tier1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

tier="$1"
if [[ ! "$tier" =~ ^tier[1-5]$ ]]; then
    echo "usage: post_tier.sh tier{1,2,3,4,5}"
    exit 1
fi

echo ">> [$tier] per-run plots"
shopt -s nullglob
declare -a plot_targets=()
for d in runs_h100/${tier}_*; do
    [ -d "$d" ] && [ -f "$d/final.pt" ] && plot_targets+=("$d")
done
if [ "${#plot_targets[@]}" -gt 0 ]; then
    python3 scripts/plot_spin_results.py "${plot_targets[@]}" > /dev/null 2>&1 || true
fi

echo ">> [$tier] aggregate summary figure"
python3 scripts/build_summary_figures.py "$tier" || true

echo ">> [$tier] stats table"
python3 scripts/tier_stats.py "$tier" | tee "figures/summary/${tier}_stats.md"

echo ">> [$tier] git add"
# runs_h100/* is NOT gitignored (only runs/* is). The big final.pt files
# are excluded explicitly in .gitignore for n>=256 tier3/tier4 dirs.
# Use plain `git add` so those rules are respected.
git add runs_h100/${tier}_*/history.json \
        runs_h100/${tier}_*/config.json \
        runs_h100/${tier}_*/losses.json \
        runs_h100/${tier}_*/plots \
        runs_h100/${tier}_*/lag_alignment.json \
        runs_h100/${tier}_*/k_sweep.json 2>/dev/null || true
# Include n=64 final.pt where it isn't gitignored. The .gitignore filters
# the n=256 / n=1024 ones automatically.
git add runs_h100/${tier}_*/final.pt 2>/dev/null || true
git add figures/summary/${tier}_*.png \
        figures/summary/${tier}_stats.md \
        figures/summary/lag_sweep.png 2>/dev/null || true
git add summary.md 2>/dev/null || true

if git diff --staged --quiet; then
    echo ">> [$tier] nothing staged, skipping commit"
else
    git commit -m "$(cat <<EOF
$tier results: aggregate plots + per-run histories

Auto-committed by scripts/post_tier.sh. See figures/summary/${tier}_*.png
and figures/summary/${tier}_stats.md for the cross-run summary.
EOF
)"
    git push https://aniket-desh:${GH_TOKEN}@github.com/aniket-desh/low-rank-rnn.git main || \
        git push origin main || true
fi

echo ">> [$tier] done"
