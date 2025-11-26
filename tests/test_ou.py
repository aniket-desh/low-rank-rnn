import torch
import numpy as np
import matplotlib.pyplot as plt
from lowrank_rnn.data.ou import OUProcess, make_iid_ou, make_correlated_ou

def test_stationary_mean_var():
    # using seed 1 instead of 0 - seed 0 gives particularly low variance due to sampling
    generator = torch.Generator()
    generator.manual_seed(1)
    ou = make_iid_ou(dim=1, tau=1.0, sigma=0.5, mu=1.0, generator=generator)
    ou.reset(torch.tensor([0.0]))
    burn_in = int(50.0 / 1e-3)
    X = ou.sample(T=500_000, dt=1e-3, burn_in=burn_in).squeeze()
    assert abs(X.mean().item() - 1.0) < 0.05
    assert abs(X.var().item() - 0.25) < 0.01

def test_correlated_two_channel():
    generator = torch.Generator()
    generator.manual_seed(0)
    Sigma = torch.tensor([[1.0, 0.8],[0.8, 1.0]]) * 0.25  # stationary covs
    ou = make_correlated_ou(dim=2, tau=0.5, sigma=0.5, Sigma=Sigma, mu=torch.tensor([0.0, 0.0]), generator=generator)
    burn_in = int(10.0 / 5e-4)
    X = ou.sample(T=200_000, dt=5e-4, burn_in=burn_in)
    X_centered = X - X.mean(dim=0, keepdim=True)
    emp = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    assert abs(emp[0,1].item() - Sigma[0,1].item()) < 0.02

def plot_time_series():
    generator = torch.Generator()
    generator.manual_seed(0)
    ou = make_iid_ou(dim=1, tau=1.0, sigma=0.5, mu=1.0, generator=generator)
    ou.reset(torch.tensor([0.0]))
    X = ou.sample(T=10_000, dt=1e-3).squeeze()  # 10s total
    assert abs(X.mean().item() - 1.0) < 5e-3
    assert abs(X.var().item() - 0.25) < 5e-3
    plt.plot(X.cpu().numpy())
    plt.show()

if __name__ == "__main__":
    plot_time_series()
