#!/usr/bin/env bash
# Animation-source runs — n=64 lattice with `--save-every 25`, single seed.
# Three temperature-only runs (next_state) and three task-only runs at β=0.44.
# Total: 6 runs, each ~3 min on H100. Run 3 in parallel.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="runs_h100"
mkdir -p logs

beta_tag() { echo "${1//./p}"; }

run_one() {
    local beta="$1" task="$2" seed="$3"
    local btag; btag="$(beta_tag "$beta")"
    local save="$OUT_ROOT/animation_lattice64_beta${btag}_${task}"
    if [ -f "$save/history.json" ]; then
        echo "[skip] $save"
        return
    fi
    mkdir -p "$save"
    local extra=()
    case "$task" in
        denoise) extra+=(--task denoise --mask-frac 0.3) ;;
        partial) extra+=(--task partial --obs-frac 0.5) ;;
        next_state) extra+=(--task next_state) ;;
        *) echo "unknown task $task"; exit 1 ;;
    esac
    python3 scripts/run_spin_experiment.py \
        --graph-kind lattice_2d --beta "$beta" \
        --n-spins 64 --hidden-dim 64 \
        --epochs 1000 --seq-len 200 --batch-size 256 \
        --eval-every 25 --save-every 25 \
        --seed "$seed" --device cuda \
        --save-dir "$save" \
        "${extra[@]}" \
        > "$save.log" 2>&1
    echo "[done] $save"
}

# β sweep at next_state
echo "[anim] launching β sweep (next_state) at n=64"
run_one 0.200 next_state 0 &
run_one 0.440 next_state 0 &
run_one 0.800 next_state 0 &
wait

# Task sweep at β=0.44
echo "[anim] launching task sweep (β=0.44) at n=64"
# next_state at β=0.44 is the same run as above — reuse via path; below covers
# denoise and partial which need separate animation sources.
run_one 0.440 denoise 0 &
run_one 0.440 partial 0 &
wait

echo "[anim sources done]"
