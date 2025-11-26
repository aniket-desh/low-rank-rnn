# ornstein-uhlenbeck noise generator (indep and correlated)
# with exact updates

from __future__ import annotations
import torch
from typing import Optional, Union

Tensor = torch.Tensor

class OUProcess:

    def __init__(
        self,
        dim: int,
        tau: float,
        sigma: float | Union[float, Tensor],
        mu: float | Union[float, Tensor] = 0.0,
        Sigma: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ):

        assert dim >= 1
        assert tau > 0

        self.dim = int(dim)
        self.tau = float(tau)
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype
        self.generator = generator

        if isinstance(mu, (int, float)):
            self.mu = torch.full((self.dim,), float(mu), dtype=dtype, device=self.device)
        else:
            mu_tensor = torch.as_tensor(mu, dtype=dtype, device=self.device)
            self.mu = mu_tensor.broadcast_to((self.dim,))

        if isinstance(sigma, (int, float)):
            self._sigma_vec = torch.full((self.dim,), float(sigma), dtype=dtype, device=self.device)
        else:
            sigma_tensor = torch.as_tensor(sigma, dtype=dtype, device=self.device)
            self._sigma_vec = sigma_tensor.broadcast_to((self.dim,))

        # correlation structure
        if Sigma is not None:
            Sigma_tensor = torch.as_tensor(Sigma, dtype=dtype, device=self.device)
            assert Sigma_tensor.shape == (self.dim, self.dim), "Sigma must be square"
            
            self._Sigma = Sigma_tensor
            
            # robust Cholesky
            try:
                self._L = torch.linalg.cholesky(Sigma_tensor)
            except RuntimeError:
                eps = 1e-12 * torch.trace(Sigma_tensor) / self.dim
                self._L = torch.linalg.cholesky(Sigma_tensor + eps * torch.eye(self.dim, dtype=dtype, device=self.device))
            self._correlated = True
        else:
            self._L = None
            self._Sigma = None
            self._correlated = False
        
        if x0 is None:
            self.x = self.mu.clone()
        else:
            x0_tensor = torch.as_tensor(x0, dtype=dtype, device=self.device)
            self.x = x0_tensor.clone()
            if self.x.shape != (self.dim,):
                raise ValueError("x0 must be a vector of length dim")

        
    # core math
    def _coeffs(self, dt: float) -> tuple[float, float]:
        # returns a, b for the exact OU update over step dt.
        a = float(torch.exp(torch.tensor(-dt / self.tau, dtype=self.dtype, device=self.device)).item())
        exp_term = float(torch.exp(torch.tensor(-2.0 * dt / self.tau, dtype=self.dtype, device=self.device)).item())
        b = float(torch.sqrt(torch.clamp(torch.tensor(1.0 - exp_term, dtype=self.dtype, device=self.device), min=0.0)).item())
        return a, b
    
    def _noise(self, size: Optional[int] = None) -> Tensor:
        # standard normals shaped appropriately
        if size is None:
            return torch.randn(self.dim, dtype=self.dtype, device=self.device, generator=self.generator)
        return torch.randn(size, self.dim, dtype=self.dtype, device=self.device, generator=self.generator)

    # public api
    def reset(self, x0: Optional[Tensor] = None):
        # reset state to x0 (or mu if not provided)
        if x0 is None:
            self.x = self.mu.clone()
        else:
            x0_tensor = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
            self.x = x0_tensor.clone()

    def step(self, dt: float) -> Tensor:
        # advance one exact OU step of size dt and return the new state
        a, b = self._coeffs(dt)
        z = self._noise()
        if self._correlated:
            inc = b * (self._L @ z)
        else:
            inc = b * (self._sigma_vec * z)
        
        self.x = self.mu + a * (self.x - self.mu) + inc
        return self.x
    
    def sample(self, T: int, dt: float, burn_in: int = 0) -> Tensor:
        # generate a length-T trajectory with constant dt.
        a, b = self._coeffs(dt)
        out = torch.empty((T, self.dim), dtype=self.dtype, device=self.device)
        x = self.x.clone()
        
        if burn_in > 0:
            Z_burn = self._noise(size=burn_in)
            if self._correlated:
                L = self._L
                for t in range(burn_in):
                    inc = b * (L @ Z_burn[t])
                    x = self.mu + a * (x - self.mu) + inc
            else:
                sig = self._sigma_vec
                for t in range(burn_in):
                    inc = b * (sig * Z_burn[t])
                    x = self.mu + a * (x - self.mu) + inc

        Z = self._noise(size = T)
        if self._correlated:
            L = self._L
            for t in range(T):
                inc = b * (L @ Z[t])
                x = self.mu + a * (x - self.mu) + inc
                out[t] = x
        else:
            sig = self._sigma_vec
            for t in range(T):
                inc = b * (sig * Z[t])
                x = self.mu + a * (x - self.mu) + inc
                out[t] = x
        self.x = x
        return out
    
    # theoretical helpers
    def theoretical_autocov(self, lags: Union[float, Tensor]) -> Tensor:
        lags_tensor = torch.as_tensor(lags, dtype=self.dtype, device=self.device)
        base = torch.exp(-torch.abs(lags_tensor) / self.tau)

        var = self.stationary_variance()
        return torch.outer(base, var) # (len(lags), dim)
    
    def stationary_variance(self) -> Tensor:
        if self._correlated and self._Sigma is not None:
            return torch.diag(self._Sigma)
        else:
            return self._sigma_vec ** 2
    
    def theoretical_psd(self, freqs: Union[float, Tensor]) -> Tensor:
        freqs_tensor = torch.as_tensor(freqs, dtype=self.dtype, device=self.device)
        w = 2.0 * torch.pi * freqs_tensor
        denom = 1.0 + (w * self.tau) ** 2
        scale = (self._sigma_vec ** 2) / (2.0 * self.tau)
        return torch.outer(1.0 / denom, scale) # (len(freqs), dim)

# constructors
def make_iid_ou(dim: int, tau: float, sigma: float, mu: float | Union[float, Tensor] = 0.0, x0: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32, generator: Optional[torch.Generator] = None) -> OUProcess:
    return OUProcess(dim, tau, sigma, mu, None, x0, device, dtype, generator)

def make_correlated_ou(dim: int, tau: float, sigma: float | Union[float, Tensor], mu: float | Union[float, Tensor] = 0.0, Sigma: Optional[Tensor] = None, x0: Optional[Tensor] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32, generator: Optional[torch.Generator] = None) -> OUProcess:
    return OUProcess(dim, tau, sigma, mu, Sigma, x0, device, dtype, generator)
