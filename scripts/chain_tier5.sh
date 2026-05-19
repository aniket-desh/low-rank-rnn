#!/usr/bin/env bash
# Wait for Tier 4 to finish (visible in runs_h100/master.log), then run Tier 5
# and post-process. Designed to be launched in the background after the master
# chain has already begun running Tiers 2/3/4.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOG="runs_h100/master.log"
echo "[chain_tier5] waiting for tier4 completion marker in $LOG"
# run_all_tiers.sh prints '=== finished tier4; running post-tier hook ==='
# after tier4 itself finishes, and '=== all tiers complete ===' at the very
# end. Trigger on the post-hook marker so tier5 starts asap.
until grep -q "=== finished tier4;" "$LOG" 2>/dev/null; do
    sleep 60
done
echo "[chain_tier5] tier4 detected; launching tier5"

bash scripts/run_tier5.sh
echo "[chain_tier5] tier5 sweep done; running post hook"
bash scripts/post_tier.sh tier5
echo "[chain_tier5] all done"
