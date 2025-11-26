# scripts/visualize_training.py
# Visualize saved training data

import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def visualize_training(save_dir: str):
    """Visualize training data from a saved run."""
    save_path = Path(save_dir)
    
    if not save_path.exists():
        print(f"Error: Directory {save_dir} does not exist")
        return
    
    # Load losses
    losses_file = save_path / "losses.json"
    if losses_file.exists():
        with open(losses_file, 'r') as f:
            losses = json.load(f)
    else:
        print(f"Warning: {losses_file} not found")
        losses = []
    
    # Load diagnostics
    diag_file = save_path / "diagnostics.json"
    diag_history = []
    if diag_file.exists():
        with open(diag_file, 'r') as f:
            diag_history = json.load(f)
    
    # Load example predictions
    examples_dir = save_path / "examples"
    example_files = sorted(examples_dir.glob("example_epoch_*.pt")) if examples_dir.exists() else []
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss curve
    ax = axes[0, 0]
    if losses:
        ax.plot(losses, 'b-', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Diagnostics
    ax = axes[0, 1]
    if diag_history:
        epochs = [d['epoch'] for d in diag_history]
        re_lams = [d['re_lambda_out'] for d in diag_history]
        im_lams = [d.get('im_lambda_out', 0) for d in diag_history]
        ax.plot(epochs, re_lams, 'r-', marker='o', markersize=5, label='Re(λ_out)')
        if any(im != 0 for im in im_lams):
            ax.plot(epochs, im_lams, 'g--', marker='s', markersize=4, label='Im(λ_out)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Leading Eigenvalue')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: First and last example predictions
    ax = axes[1, 0]
    if len(example_files) >= 2:
        # First example
        first_data = torch.load(example_files[0])
        pred_first = first_data['prediction'].numpy().squeeze()
        target_first = first_data['target'].numpy().squeeze()
        epoch_first = first_data['epoch']
        time_steps = range(len(pred_first))
        
        ax.plot(time_steps, target_first, 'b-', label=f'Target (epoch {epoch_first})', alpha=0.7, linewidth=2)
        ax.plot(time_steps, pred_first, 'r--', label=f'Pred (epoch {epoch_first})', alpha=0.7, linewidth=1.5)
        
        # Last example
        last_data = torch.load(example_files[-1])
        pred_last = last_data['prediction'].numpy().squeeze()
        target_last = last_data['target'].numpy().squeeze()
        epoch_last = last_data['epoch']
        
        ax.plot(time_steps, target_last, 'g-', label=f'Target (epoch {epoch_last})', alpha=0.5, linewidth=2)
        ax.plot(time_steps, pred_last, 'm--', label=f'Pred (epoch {epoch_last})', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title('Example OU Tracking: First vs Last Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Error evolution
    ax = axes[1, 1]
    if len(example_files) >= 2:
        errors_first = pred_first - target_first
        errors_last = pred_last - target_last
        ax.plot(time_steps, errors_first, 'r-', label=f'Error (epoch {epoch_first})', alpha=0.7)
        ax.plot(time_steps, errors_last, 'g-', label=f'Error (epoch {epoch_last})', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Tracking Error Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nTraining Summary from {save_dir}:")
    print(f"  Total epochs: {len(losses)}")
    if losses:
        print(f"  Initial loss: {losses[0]:.4e}")
        print(f"  Final loss: {losses[-1]:.4e}")
        print(f"  Best loss: {min(losses):.4e} (epoch {losses.index(min(losses)) + 1})")
    if diag_history:
        print(f"  Final Re(λ_out): {diag_history[-1]['re_lambda_out']:.4f}")
    print(f"  Example predictions: {len(example_files)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_training.py <save_dir>")
        print("Example: python visualize_training.py runs/ou_tracking_20240101_120000")
        sys.exit(1)
    
    visualize_training(sys.argv[1])

