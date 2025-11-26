from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Any, List
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
import matplotlib
# Try to use interactive backend, fall back to default if not available
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend
import matplotlib.pyplot as plt

from lowrank_rnn.data.ou import OUProcess
from lowrank_rnn.models.rnn import LowRankRNN, LowRankRNNConfig
from lowrank_rnn.models.low_rank import jacobian_from_mask, largest_real_eig, effective_m
from sklearn.cluster import KMeans

@dataclass
class TrainConfig:
    # OU / task params
    T: float = 10.0
    dt: float = 1e-2
    batch_size: int = 32
    ou_tau: float = 1.0
    ou_sigma: float = 0.5
    ou_x0: float = 0.0

    # RNN / low-rank params
    N: int = 800
    g: float = 2.0
    b: float = 10.0
    m: float = 1.5
    nonlinearity: str = "relu"
    train_u: bool = False
    train_v: bool = False
    train_readout: bool = True
    out_dim: int = 1

    # optimization
    n_epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0.0

    # device / reproducibility
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0

    print_every: int = 50
    diag_every: int = 200
    use_diags: bool = True
    
    # visualization / saving
    plot_every: int = 10
    save_every: int = 50
    save_dir: Optional[str] = None
    show_plots: bool = True

def make_ou_batch(cfg: TrainConfig,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    # sample a batch of OU trajectories with shape (batch_size, T_steps)
    device = torch.device(cfg.device)
    T_steps = int(cfg.T / cfg.dt)
    
    batch_trajectories = []
    for _ in range(cfg.batch_size):
        ou = OUProcess(
            dim=1,
            tau=cfg.ou_tau,
            sigma=cfg.ou_sigma,
            mu=0.0,
            x0=torch.tensor([cfg.ou_x0], device=device, dtype=cfg.dtype),
            device=device,
            dtype=cfg.dtype,
            generator=generator,
        )
        ou.reset(torch.tensor([cfg.ou_x0], device=device, dtype=cfg.dtype))
        traj = ou.sample(T=T_steps, dt=cfg.dt, burn_in=0).squeeze()  # (T_steps,)
        batch_trajectories.append(traj)
    
    xs = torch.stack(batch_trajectories, dim=0)  # (batch_size, T_steps)
    return xs

def build_model(cfg: TrainConfig) -> LowRankRNN:
    # construct a low rank RNN 
    model_cfg = LowRankRNNConfig(
        N=cfg.N,
        g=cfg.g,
        b=cfg.b,
        m=cfg.m,
        tau=cfg.ou_tau,
        dt=cfg.dt,
        nonlinearity=cfg.nonlinearity,
        device=cfg.device,
        dtype=cfg.dtype,
        train_u=cfg.train_u,
        train_v=cfg.train_v,
        train_readout=cfg.train_readout,
        out_dim=cfg.out_dim,
    )
    model = LowRankRNN(model_cfg)
    return model.to(torch.device(cfg.device))

def rnn_track_batch_loss(
    model: LowRankRNN,
    batch_ou: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    # computing avg loss over a batch of OU trajectories
    device = model.W.device
    B, T = batch_ou.shape
    total_loss = 0.0

    for b in range(B):
        I_t = batch_ou[b]
        I_t = I_t.to(device=device, dtype=model.W.dtype)
        target = I_t.view(T, 1)
        y_t, _ = model(I_t, h0=None, return_states=False)

        loss = loss_fn(y_t, target)
        total_loss = total_loss + loss
    
    return total_loss / B

def compute_clustering_diags(model: LowRankRNN, J_initial: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Compute clustering-based diagnostics similar to Rainer's animation.
    - K-means clustering on u vector (low-rank input direction)
    - Mean connectivity within/between clusters
    - Returns cluster assignments and connectivity statistics
    """
    device = model.W.device
    dtype = model.W.dtype
    N = model.N
    J = model._build_J()
    
    # Use u vector for clustering (low-rank input direction)
    u_vec = model.u.detach().cpu().numpy().reshape(-1, 1)  # (N, 1) for KMeans
    
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=200)
    assignments = kmeans.fit_predict(u_vec)
    
    # Sort by cluster assignments
    sorted_indices = torch.argsort(torch.tensor(assignments))
    J_sorted = J[sorted_indices][:, sorted_indices].cpu()
    
    # Compute mean connectivity within/between clusters
    assignments_tensor = torch.tensor(assignments, device=device)
    n_same = 0
    n_different = 0
    sum_same = 0.0
    sum_different = 0.0
    
    for i in range(N):
        for j in range(N):
            if assignments[i] == assignments[j]:
                n_same += 1
                sum_same += J[i, j].item()
            else:
                n_different += 1
                sum_different += J[i, j].item()
    
    mean_same = sum_same / n_same if n_same > 0 else 0.0
    mean_different = sum_different / n_different if n_different > 0 else 0.0
    mean_all = J.mean().item()
    
    # Compute eigenvalues of J - I
    J_minus_I = J - torch.eye(N, device=device, dtype=dtype)
    eigvals = torch.linalg.eigvals(J_minus_I).cpu()
    
    eigvals_initial = None
    if J_initial is not None:
        J_initial_minus_I = J_initial - torch.eye(N, device=device, dtype=dtype)
        eigvals_initial = torch.linalg.eigvals(J_initial_minus_I).cpu()
    
    return {
        'J_sorted': J_sorted,
        'assignments': assignments,
        'mean_same': mean_same,
        'mean_different': mean_different,
        'mean_all': mean_all,
        'eigvals': eigvals,
        'eigvals_initial': eigvals_initial,
    }

def compute_spectrum_diags(model: LowRankRNN, n_samples: int = 64) -> Dict[str, float]:
    # diag to tie training back to dmft
    # sample a short traj with zero input to estimate an avg ReLU mask
    # constructs A_avg = -I + J diag(Dbar)
    # returns the leading real eigenvalue of A_avg
    device = model.W.device
    dtype = model.W.dtype
    N = model.N
    cfg = model.cfg

    T_diag = 5.0
    dt = cfg.dt
    T_steps = int(T_diag / dt)

    # zero input trajectory
    I_t = torch.zeros(T_steps, device=device, dtype=dtype) 
    h = 0.1 * torch.randn(N, device=device, dtype=dtype)
    J = model._build_J()

    phi = model.phi
    masks = []

    for t in range(T_steps):
        x = phi(h)
        if isinstance(phi, nn.ReLU):
            D = (h > 0.0).to(dtype=dtype)
        else:
            D = 1.0 - torch.tanh(h)**2
        
        masks.append(D)

        dh = (-h + J @ x + 0.0) * (dt / cfg.tau)
        h = h + dh
    
    Dbar = torch.stack(masks, dim=0).mean(dim=0)
    A_avg = jacobian_from_mask(J, Dbar)
    lam_out = largest_real_eig(A_avg)

    return {
        're_lambda_out': float(lam_out.real),
        'im_lambda_out': float(lam_out.imag),
    }

def save_training_data(save_dir: Path, epoch: int, losses: List[float], 
                      diag_history: List[Dict[str, float]], 
                      example_pred: Optional[torch.Tensor] = None,
                      example_target: Optional[torch.Tensor] = None):
    """Save training data incrementally."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save losses
    losses_file = save_dir / "losses.json"
    with open(losses_file, 'w') as f:
        json.dump(losses, f)
    
    # Save diagnostics
    if diag_history:
        diag_file = save_dir / "diagnostics.json"
        with open(diag_file, 'w') as f:
            json.dump(diag_history, f)
    
    # Save example predictions
    if example_pred is not None and example_target is not None:
        example_dir = save_dir / "examples"
        example_dir.mkdir(exist_ok=True)
        torch.save({
            'prediction': example_pred.cpu(),
            'target': example_target.cpu(),
            'epoch': epoch,
        }, example_dir / f"example_epoch_{epoch:05d}.pt")

def plot_training_progress(losses: List[float], diag_history: List[Dict[str, float]], 
                          example_pred: Optional[torch.Tensor] = None,
                          example_target: Optional[torch.Tensor] = None,
                          clustering_data: Optional[Dict[str, Any]] = None,
                          clustering_history: Optional[List[Dict[str, Any]]] = None,
                          fig_axes=None):
    """Create/update real-time training plots with Rainer-style animation."""
    if fig_axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.ion()
        plt.show()
        fig_axes = (fig, axes)
    else:
        fig, axes = fig_axes
    
    # Panel 1: Connectivity matrix sorted by clusters (like Rainer's left plot)
    ax = axes[0, 0]
    ax.clear()
    if clustering_data is not None and 'J_sorted' in clustering_data:
        J_sorted = clustering_data['J_sorted'].numpy()
        vmax = max(abs(J_sorted.min()), abs(J_sorted.max()))
        vmin = -vmax
        im = ax.imshow(J_sorted, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title('J (sorted by clusters)')
        ax.set_xlabel('Neuron j')
        ax.set_ylabel('Neuron i')
        # Add cluster boundary lines
        assignments = clustering_data['assignments']
        cluster_boundary = (assignments == 0).sum()
        if cluster_boundary > 0 and cluster_boundary < len(assignments):
            ax.axhline(y=cluster_boundary-0.5, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axvline(x=cluster_boundary-0.5, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No clustering data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Connectivity Matrix')
    
    # Panel 2: Eigenvalue spectrum (like Rainer's middle plot)
    ax = axes[0, 1]
    ax.clear()
    if clustering_data is not None:
        eigvals = clustering_data['eigvals']
        eigvals_initial = clustering_data.get('eigvals_initial')
        
        if eigvals_initial is not None:
            ax.scatter(eigvals_initial.real, eigvals_initial.imag, c='red', s=10, alpha=0.6, label='pre training', marker='.')
        ax.scatter(eigvals.real, eigvals.imag, c='black', s=10, alpha=0.6, label='post training', marker='.')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel(r'Real($\lambda_i$)')
        ax.set_ylabel(r'Imaginary($\lambda_i$)')
        ax.set_title('Eigenvalues of J - I')
        if eigvals_initial is not None:
            ax.legend(loc='best', fontsize=8)
        ax.set_xlim(-10, 1)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No eigenvalue data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Eigenvalue Spectrum')
    
    # Panel 3: Mean connectivity within/between clusters over time (like Rainer's right plot)
    ax = axes[1, 0]
    ax.clear()
    if clustering_history is not None and len(clustering_history) > 0:
        epochs = [d['epoch'] for d in clustering_history]
        mean_same_all = [d['mean_same'] for d in clustering_history]
        mean_different_all = [d['mean_different'] for d in clustering_history]
        mean_all_all = [d['mean_all'] for d in clustering_history]
        
        ax.plot(epochs, mean_same_all, '.-r', label=r'$\bar W^{same}_{ij}$', markersize=4)
        ax.plot(epochs, mean_different_all, '.-b', label=r'$\bar W^{different}_{ij}$', markersize=4)
        ax.plot(epochs, mean_all_all, '.-k', label=r'$\bar W_{ij}$', markersize=4)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Weight')
        ax.set_title('Mean Connectivity by Cluster')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    elif clustering_data is not None and 'mean_same' in clustering_data:
        # Show current values if no history yet
        mean_same = clustering_data['mean_same']
        mean_different = clustering_data['mean_different']
        mean_all = clustering_data['mean_all']
        ax.scatter([0], [mean_same], c='red', s=100, label=r'$\bar W^{same}_{ij}$', marker='o', zorder=3)
        ax.scatter([0], [mean_different], c='blue', s=100, label=r'$\bar W^{different}_{ij}$', marker='o', zorder=3)
        ax.scatter([0], [mean_all], c='black', s=100, label=r'$\bar W_{ij}$', marker='o', zorder=3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Weight')
        ax.set_title('Mean Connectivity by Cluster')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(-0.5, 0.5)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No clustering stats', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cluster Connectivity')
    
    # Panel 4: Training loss
    ax = axes[1, 1]
    ax.clear()
    ax.plot(losses, 'b-', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # shorter pause for smoother animation
    fig.canvas.flush_events()  # Force immediate update
    
    return (fig, axes)

def get_example_prediction(model: LowRankRNN, cfg: TrainConfig, 
                          generator: Optional[torch.Generator] = None):
    model.eval()
    with torch.no_grad():
        device = torch.device(cfg.device)
        ou = OUProcess(
            dim=1,
            tau=cfg.ou_tau,
            sigma=cfg.ou_sigma,
            mu=0.0,
            x0=torch.tensor([cfg.ou_x0], device=device, dtype=cfg.dtype),
            device=device,
            dtype=cfg.dtype,
            generator=generator,
        )
        ou.reset(torch.tensor([cfg.ou_x0], device=device, dtype=cfg.dtype))
        T_steps = int(cfg.T / cfg.dt)
        I_t = ou.sample(T=T_steps, dt=cfg.dt, burn_in=0).squeeze()  # (T_steps,)
        I_t = I_t.to(device=device, dtype=model.W.dtype)
        
        y_t, _ = model(I_t, h0=None, return_states=False)
        target = I_t.view(T_steps, 1)
        
        return y_t, target

def train_ou_tracking(cfg: TrainConfig) -> Dict[str, Any]:
    # train an RNN to track OU trajectories
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    model = build_model(cfg)
    optimizer: Optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    assert abs(model.cfg.dt - cfg.dt) < 1e-12, "model dt must match cfg.dt"

    losses: List[float] = []
    diag_history: List[Dict[str, float]] = []
    clustering_history: List[Dict[str, Any]] = []

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    
    # store initial J for eigenvalue comparison
    J_initial = model._build_J().detach().clone()
    
    # setup saving
    save_dir = None
    if cfg.save_dir:
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # save config
        with open(save_dir / "config.json", 'w') as f:
            config_dict = asdict(cfg)
            # convert torch.dtype to string for JSON
            if 'dtype' in config_dict:
                config_dict['dtype'] = str(config_dict['dtype'])
            json.dump(config_dict, f, indent=2)
    
    # setup plotting
    fig_axes = None
    example_pred = None
    example_target = None
    clustering_data = None
    if cfg.show_plots:
        plt.ion()
    
    # separate generator for example predictions (deterministic)
    example_gen = torch.Generator(device=device)
    example_gen.manual_seed(42)

    for epoch in range(cfg.n_epochs):
        model.train()
        batch_ou = make_ou_batch(cfg, generator=gen)
        batch_ou = batch_ou.to(device=device, dtype=model.W.dtype)

        loss = rnn_track_batch_loss(model, batch_ou, loss_fn)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        if cfg.use_diags and ((epoch + 1) % cfg.diag_every == 0 or epoch == 0):
            model.eval()
            with torch.no_grad():
                diag = compute_spectrum_diags(model)
            diag['epoch'] = epoch + 1
            diag_history.append(diag)
        
        # get example prediction for visualization (less frequently to save computation)
        if (epoch + 1) % cfg.plot_every == 0 or epoch == 0:
            example_pred, example_target = get_example_prediction(model, cfg, generator=example_gen)
        
        # compute clustering diagnostics (same frequency as example predictions)
        if (epoch + 1) % cfg.plot_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                clustering_data = compute_clustering_diags(model, J_initial=J_initial)
                clustering_history.append({
                    'epoch': epoch + 1,
                    'mean_same': clustering_data['mean_same'],
                    'mean_different': clustering_data['mean_different'],
                    'mean_all': clustering_data['mean_all'],
                })
        
        # update plots every epoch for smooth animation
        if cfg.show_plots:
            fig_axes = plot_training_progress(
                losses, diag_history, example_pred, example_target, 
                clustering_data, clustering_history, fig_axes
            )
        
        # save data
        if save_dir and ((epoch + 1) % cfg.save_every == 0 or epoch == 0):
            save_training_data(save_dir, epoch + 1, losses, diag_history, example_pred, example_target)
            # save model checkpoint
            checkpoint_dir = save_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1:05d}.pt")
        
        if (epoch + 1) % cfg.print_every == 0:
            log_msg = f'[epoch {epoch+1:04d}] loss = {loss_val:.4e}'
            if diag_history and diag_history[-1]['epoch'] == epoch + 1:
                re_lam = diag_history[-1]['re_lambda_out']
                log_msg += f', Re lam_out = {re_lam:.4f}'
            print(log_msg)
    
    # final save
    if save_dir:
        save_training_data(save_dir, cfg.n_epochs, losses, diag_history, example_pred, example_target)
        # save final model
        torch.save({
            'epoch': cfg.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[-1] if losses else None,
        }, save_dir / "final_model.pt")
    
    # keep plot open at end
    if cfg.show_plots and fig_axes is not None:
        print("\nTraining complete! Close the plot window to exit.")
        plt.ioff()
        plt.show()
    
    return {
        'model': model,
        'losses': losses,
        'diag_history': diag_history,
        'clustering_history': clustering_history,
        'cfg': cfg,
        'save_dir': str(save_dir) if save_dir else None,
    }