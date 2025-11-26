# scripts/train_ou_tracking.py

import sys
from pathlib import Path

# Add parent directory to path so we can import lowrank_rnn
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from lowrank_rnn.train.trainer import TrainConfig, train_ou_tracking

def main():
    import os
    from datetime import datetime
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"runs/ou_tracking_{timestamp}"
    
    cfg = TrainConfig(
        T=10.0,
        dt=1e-2,
        batch_size=16,
        n_epochs=200,          # start small to test
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        N=800,
        g=2.0,
        b=10.0,
        m=1.5,
        train_u=False,
        train_v=False,
        train_readout=True,
        out_dim=1,
        print_every=20,
        diag_every=50,
        plot_every=10,         # recompute example prediction every 10 epochs (plots update every epoch)
        save_every=50,         # save data every 50 epochs
        save_dir=save_dir,     # directory to save training data
        show_plots=True,       # show real-time plots
    )

    results = train_ou_tracking(cfg)
    
    print(f"\nTraining data saved to: {save_dir}")
    print(f"Final model saved to: {save_dir}/final_model.pt")

if __name__ == "__main__":
    main()