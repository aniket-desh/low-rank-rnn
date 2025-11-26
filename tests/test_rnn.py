import torch
from lowrank_rnn.models.rnn import LowRankRNN, LowRankRNNConfig
from lowrank_rnn.data.ou import make_iid_ou

def test_rnn_with_ou_input():
    config = LowRankRNNConfig(
        N=100,
        g=2.0,
        b=10.0,
        m=1.5,
        tau=1.0,
        dt=1e-3,
        nonlinearity="relu",
        device="cpu",
        dtype=torch.float32,
        train_readout=True,
        out_dim=1,
    )
    
    model = LowRankRNN(config)
    model.train() # set to training mode to ensure gradients are enabled
    
    # generate OU input
    generator = torch.Generator()
    generator.manual_seed(42)
    ou = make_iid_ou(dim=1, tau=1.0, sigma=0.5, mu=0.0, generator=generator, device=torch.device('cpu'))
    ou.reset(torch.tensor([0.0]))
    
    T = 100
    I_t = ou.sample(T=T, dt=config.dt, burn_in=10).squeeze()  # (T,)
    
    # run model
    y_t, h_t = model(I_t, return_states=True)
    
    # assert shapes
    assert y_t.shape == (T, config.out_dim), f"Expected y_t shape ({T}, {config.out_dim}), got {y_t.shape}"
    assert h_t is not None, "h_t should not be None when return_states=True"
    assert h_t.shape == (T, config.N), f"Expected h_t shape ({T}, {config.N}), got {h_t.shape}"
    
    # assert gradients are enabled for output
    assert y_t.requires_grad, "y_t.requires_grad should be True when train_readout=True"
    
    # test backward pass works
    loss = y_t.sum()
    loss.backward()
    assert model.readout.weight.grad is not None, "readout.weight should have gradients"
    assert model.readout.bias.grad is not None, "readout.bias should have gradients"

def test_rnn_with_ou_input_no_states():
    # test with return_states=False
    config = LowRankRNNConfig(
        N=50,
        out_dim=2,
        train_readout=True,
    )
    
    model = LowRankRNN(config)
    model.train()
    
    generator = torch.Generator()
    generator.manual_seed(123)
    ou = make_iid_ou(dim=1, tau=1.0, sigma=0.5, mu=0.0, generator=generator)
    ou.reset(torch.tensor([0.0]))
    
    T = 50
    I_t = ou.sample(T=T, dt=config.dt, burn_in=5).squeeze()
    
    y_t, h_t = model(I_t, return_states=False)
    
    assert y_t.shape == (T, config.out_dim)
    assert h_t is None, "h_t should be None when return_states=False"
    assert y_t.requires_grad

if __name__ == "__main__":
    test_rnn_with_ou_input()
    test_rnn_with_ou_input_no_states()
    print("All tests passed!")

