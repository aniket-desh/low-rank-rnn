#!/usr/bin/env bash
# Master driver — chain Tier 1 → 2 → 3 → 4, post-processing between each.
# Designed to run in the background overnight; logs to runs_h100/master.log.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ROOT="runs_h100"
mkdir -p "$ROOT"

# Tier 1 is launched separately (already running). This script picks up after
# Tier 1 completes and chains 2/3/4. If invoked with --from-tier1, it also runs
# Tier 1.

FROM="${1:-tier2}"

run_and_post() {
    local tier="$1"
    echo "=== launching $tier ==="
    bash "scripts/run_${tier}.sh"
    echo "=== finished $tier; running post-tier hook ==="
    bash "scripts/post_tier.sh" "$tier"
}

case "$FROM" in
    --from-tier1|tier1)
        run_and_post tier1
        ;& # fallthrough
    tier2)
        run_and_post tier2
        ;& # fallthrough
    tier3)
        run_and_post tier3
        ;& # fallthrough
    tier4)
        run_and_post tier4
        ;;
    *)
        echo "usage: run_all_tiers.sh [tier1|tier2|tier3|tier4|--from-tier1]"
        exit 1
        ;;
esac

echo "=== all tiers complete ==="
