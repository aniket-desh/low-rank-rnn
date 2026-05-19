#!/usr/bin/env python3
"""CLI entrypoint for training an RNN on Ising/Glauber trajectories.

Example:
    python scripts/run_spin_experiment.py \
        --graph-kind lattice_2d --beta 0.44 --n-spins 64 \
        --hidden-dim 64 --epochs 300 --device cpu \
        --save-dir runs/lattice_beta044
"""
from __future__ import annotations
import argparse
import os
import sys

# Make the repo importable when run as a script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lowrank_rnn.train.spin_trainer import SpinTrainConfig, train_spin_prediction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-spins", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--field", type=float, default=0.0)
    p.add_argument(
        "--graph-kind",
        type=str,
        default="lattice_2d",
        choices=["curie_weiss", "lattice_2d", "block", "sk"],
    )
    p.add_argument("--coupling", type=float, default=1.0)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--j-in", type=float, default=1.0)
    p.add_argument("--j-out", type=float, default=0.2)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--burn-in", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--nonlinearity", type=str, default="tanh", choices=["tanh", "relu"])
    p.add_argument("--align-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-dir", type=str, default="runs/spin_rnn_debug")
    p.add_argument(
        "--task",
        type=str,
        default="next_state",
        choices=["next_state", "denoise", "partial"],
        help="prediction task; see docs/theory_spin_rnn.md and HANDOFF §5.5",
    )
    p.add_argument("--mask-frac", type=float, default=0.3,
                   help="denoise: fraction of input coords corrupted to iid ±1")
    p.add_argument("--obs-frac", type=float, default=0.5,
                   help="partial: fraction of coords observed (fixed per run)")
    args = p.parse_args()

    cfg = SpinTrainConfig(
        n_spins=args.n_spins,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        field=args.field,
        graph_kind=args.graph_kind,
        coupling=args.coupling,
        n_blocks=args.n_blocks,
        j_in=args.j_in,
        j_out=args.j_out,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        burn_in=args.burn_in,
        eval_every=args.eval_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        nonlinearity=args.nonlinearity,
        align_k=args.align_k,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        task=args.task,
        mask_frac=args.mask_frac,
        obs_frac=args.obs_frac,
    )
    train_spin_prediction(cfg)


if __name__ == "__main__":
    main()
