#!/usr/bin/env bash
# Extended experiment matrix (10 runs, 1000 epochs each, CPU).
# Continues the original four regimes longer AND adds stronger Curie/block variants.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=1000
COMMON="--n-spins 64 --hidden-dim 64 --epochs ${EPOCHS} --device cpu --seed 0"

run() {
    local name="$1"; shift
    local args="$*"
    echo "=== [start] ${name} ==="
    python3 scripts/run_spin_experiment.py ${COMMON} --save-dir "runs/${name}" ${args} \
        2>&1 | tee "runs/${name}.log" | tail -n 6
    echo "=== [done ] ${name} ==="
}

# Continue original regimes longer. (curie_beta02_e1000 already ran locally
# and is committed; the line is left in for completeness — set RUN_CURIE02=1
# to re-run it on the new box, otherwise it is skipped.)
if [[ "${RUN_CURIE02:-0}" = "1" ]]; then
    run curie_beta02_e1000   --graph-kind curie_weiss --beta 0.2
fi
run block_beta05_e1000   --graph-kind block       --beta 0.5
run lattice_beta02_e1000 --graph-kind lattice_2d  --beta 0.2
run lattice_beta044_e1000 --graph-kind lattice_2d --beta 0.44

# Stronger Curie regimes.
run curie_beta08_e1000   --graph-kind curie_weiss --beta 0.8
run curie_beta10_e1000   --graph-kind curie_weiss --beta 1.0
run curie_beta12_e1000   --graph-kind curie_weiss --beta 1.2

# Stronger block regimes.
run block_beta10_e1000   --graph-kind block --beta 1.0
run block_beta15_e1000   --graph-kind block --beta 1.5
run block_beta20_e1000   --graph-kind block --beta 2.0

echo "all runs complete"
