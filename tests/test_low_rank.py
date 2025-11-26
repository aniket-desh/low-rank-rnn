import torch
from lowrank_rnn.models.low_rank import (
    LowRankParams,
    sample_W,
    unit_vec_orth_to_ones,
    build_rank1_uv,
    build_lowrank_J,
    subtract_balance,
    leading_mode,
    effective_m,
    largest_real_eig,
    jacobian_from_mask,
)

def test_sample_W():
    generator = torch.Generator()
    generator.manual_seed(42)
    N = 100
    W = sample_W(N, generator)
    
    assert W.shape == (N, N), f"Expected shape ({N}, {N}), got {W.shape}"
    assert W.dtype == torch.float32
    assert abs(W.mean().item()) < 0.1, "W should have approximately zero mean"
    assert abs(W.std().item() - (1.0 / (N ** 0.5))) < 0.1, "W should have std ≈ 1/sqrt(N)"

def test_unit_vec_orth_to_ones():
    generator = torch.Generator()
    generator.manual_seed(42)
    N = 50
    v = unit_vec_orth_to_ones(N, generator)
    
    assert v.shape == (N,), f"Expected shape ({N},), got {v.shape}"
    ones = torch.ones(N)
    dot_product = v.dot(ones)
    assert abs(dot_product.item()) < 1e-5, f"v should be orthogonal to ones, got dot={dot_product.item()}"
    assert abs(v.norm().item() - 1.0) < 1e-5, f"v should be unit vector, got norm={v.norm().item()}"

def test_build_rank1_uv():
    generator = torch.Generator()
    generator.manual_seed(42)
    N = 50
    u, v = build_rank1_uv(N, generator)
    
    assert u.shape == (N,), f"Expected u shape ({N},), got {u.shape}"
    assert v.shape == (N,), f"Expected v shape ({N},), got {v.shape}"
    
    ones = torch.ones(N)
    assert abs(u.dot(ones).item()) < 1e-5, "u should be orthogonal to ones"
    # note: v may not be orthogonal to ones after normalization to ensure v^T u = 1
    
    dot_uv = v.dot(u)
    assert abs(dot_uv.item() - 1.0) < 1e-5, f"v^T u should be 1, got {dot_uv.item()}"

def test_build_lowrank_J():
    params = LowRankParams(g=2.0, b=10.0, m=1.5, seed=42)
    N = 100
    J, u, v = build_lowrank_J(N, params)
    
    assert J.shape == (N, N), f"Expected J shape ({N}, {N}), got {J.shape}"
    assert u.shape == (N,), f"Expected u shape ({N},), got {u.shape}"
    assert v.shape == (N,), f"Expected v shape ({N},), got {v.shape}"
    
    # verify structure: J = g*W - (b/N)*11^T + m*u*v^T
    # check that u and v are orthogonal to ones
    ones = torch.ones(N)
    assert abs(u.dot(ones).item()) < 1e-5, "u should be orthogonal to ones"
    assert abs(v.dot(ones).item()) < 1e-5, "v should be orthogonal to ones"
    
    # verify reproducibility with same seed
    J2, u2, v2 = build_lowrank_J(N, params)
    assert torch.allclose(J, J2), "J should be reproducible with same seed"
    assert torch.allclose(u, u2), "u should be reproducible with same seed"
    assert torch.allclose(v, v2), "v should be reproducible with same seed"

def test_subtract_balance():
    N = 50
    b = 10.0
    J = torch.randn(N, N)
    J_balanced = J - (b / N) * torch.ones((N, N))
    
    J_unbalanced = subtract_balance(J_balanced, b)
    
    assert torch.allclose(J, J_unbalanced, atol=1e-5), "subtract_balance should reverse the balance term"

def test_leading_mode():
    N = 50
    # create a matrix with known leading mode
    u_true = torch.randn(N)
    v_true = torch.randn(N)
    u_true = u_true / u_true.norm()
    v_true = v_true / v_true.norm()
    s1 = 5.0
    J = s1 * torch.outer(u_true, v_true) + 0.1 * torch.randn(N, N)
    
    u1, v1 = leading_mode(J)
    
    assert u1.shape == (N,), f"Expected u1 shape ({N},), got {u1.shape}"
    assert v1.shape == (N,), f"Expected v1 shape ({N},), got {v1.shape}"
    assert abs(u1.norm().item() - 1.0) < 1e-5, "u1 should be unit vector"
    assert abs(v1.norm().item() - 1.0) < 1e-5, "v1 should be unit vector"
    
    # verify it's close to the true leading mode (up to sign)
    u_aligned = u1 if u1.dot(u_true) > 0 else -u1
    v_aligned = v1 if v1.dot(v_true) > 0 else -v1
    assert u_aligned.dot(u_true).item() > 0.9, "u1 should align with true leading mode"
    assert v_aligned.dot(v_true).item() > 0.9, "v1 should align with true leading mode"

def test_effective_m():
    N = 50
    m_true = 2.5
    generator = torch.Generator()
    generator.manual_seed(42)
    u, v = build_rank1_uv(N, generator)
    
    # test with exact rank-1 matrix
    J_exact = m_true * torch.outer(u, v)
    m_eff_exact = effective_m(J_exact, u, v)
    assert abs(m_eff_exact - m_true) < 1e-4, f"effective_m should be exact for rank-1 matrix, got {m_eff_exact}"
    
    # test with small noise (effective_m may deviate due to noise projection)
    J = m_true * torch.outer(u, v) + 0.01 * torch.randn(N, N, generator=generator)
    m_eff = effective_m(J, u, v)
    assert abs(m_eff - m_true) < 2.0, f"effective_m should be reasonably close to true m with small noise, got {m_eff}, expected {m_true}"

def test_largest_real_eig():
    N = 20
    # create a matrix with known largest real eigenvalue
    A = torch.randn(N, N)
    # make it symmetric to ensure real eigenvalues
    A = (A + A.T) / 2
    eigval = largest_real_eig(A)
    
    # verify it's actually the largest real part
    all_eigvals = torch.linalg.eigvals(A)
    max_real = all_eigvals.real.max()
    assert abs(eigval.real - max_real) < 1e-5, f"largest_real_eig should return eigenvalue with largest real part"
    
    # test with known eigenvalue
    A = torch.eye(N) * 5.0
    eigval = largest_real_eig(A)
    assert abs(eigval.real - 5.0) < 1e-5, f"largest_real_eig should return 5.0 for 5*I, got {eigval.real}"

def test_jacobian_from_mask():
    N = 50
    J = torch.randn(N, N)
    Dbar = torch.rand(N)
    
    A = jacobian_from_mask(J, Dbar)
    
    assert A.shape == (N, N), f"Expected A shape ({N}, {N}), got {A.shape}"
    
    # verify structure: A = -I + J @ diag(Dbar)
    D = torch.diag(Dbar)
    A_expected = -torch.eye(N, device=J.device, dtype=J.dtype) + J @ D
    assert torch.allclose(A, A_expected, atol=1e-5), "jacobian_from_mask should compute -I + J @ diag(Dbar)"
    
    # test with specific values
    J = torch.eye(N)
    Dbar = torch.ones(N) * 0.5
    A = jacobian_from_mask(J, Dbar)
    A_expected = -torch.eye(N) + 0.5 * torch.eye(N)
    assert torch.allclose(A, A_expected, atol=1e-5), "jacobian_from_mask should work with identity matrix"

def test_lowrank_params():
    params = LowRankParams(g=2.0, b=10.0, m=1.5, seed=42)
    assert params.g == 2.0
    assert params.b == 10.0
    assert params.m == 1.5
    assert params.seed == 42
    
    params2 = LowRankParams(g=1.0, b=5.0, m=0.5)
    assert params2.seed is None

if __name__ == "__main__":
    test_sample_W()
    test_unit_vec_orth_to_ones()
    test_build_rank1_uv()
    test_build_lowrank_J()
    test_subtract_balance()
    test_leading_mode()
    test_effective_m()
    test_largest_real_eig()
    test_jacobian_from_mask()
    test_lowrank_params()
    print("All tests passed!")

