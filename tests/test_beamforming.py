import torch

from llm_fas.beamforming import beamforming_from_pq, sum_rate


def test_beamforming_shape_and_power_constraint():
    H_eff = torch.randn(2, 3, 4, dtype=torch.complex64)
    p = torch.full((2, 3), 0.1 / 3.0)
    q = torch.full((2, 3), 0.1 / 3.0)
    C = beamforming_from_pq(H_eff, p, q, noise_power=1e-13)
    assert C.shape == (2, 4, 3)
    power = (C.abs() ** 2).sum(dim=(1, 2))
    assert torch.allclose(power, torch.full((2,), 0.1), rtol=1e-4, atol=1e-6)


def test_sum_rate_is_finite_and_positive():
    H_eff = torch.randn(2, 3, 4, dtype=torch.complex64)
    p = torch.full((2, 3), 0.1 / 3.0)
    q = torch.full((2, 3), 0.1 / 3.0)
    C = beamforming_from_pq(H_eff, p, q, noise_power=1e-13)
    rates = sum_rate(H_eff, C, noise_power=1e-13)
    assert rates.shape == (2,)
    assert torch.isfinite(rates).all()
    assert (rates > 0).all()
