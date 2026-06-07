import torch

from llm_fas.physics import (
    dbm_to_watt,
    generate_channels,
    noise_power_watt,
    path_loss_beta,
    port_coordinates,
    spatial_correlation_matrix,
)


def test_dbm_to_watt_for_20_dbm():
    assert abs(dbm_to_watt(20.0) - 0.1) < 1e-12


def test_noise_power_for_10_mhz():
    assert abs(noise_power_watt(-174.0, 10_000_000.0) - 10 ** ((-104.0 - 30.0) / 10.0)) < 1e-20


def test_path_loss_beta_default_distance():
    beta = path_loss_beta(0.2)
    expected_db = 128.1 + 37.6 * __import__("math").log10(0.2)
    assert abs(beta - 10 ** (-expected_db / 10.0)) < 1e-18


def test_port_coordinates_use_paper_mapping():
    coords = port_coordinates(4, 4)
    assert coords[0] == (0, 0)
    assert coords[1] == (1, 0)
    assert coords[4] == (0, 1)
    assert coords[15] == (3, 3)


def test_spatial_correlation_matrix_shape_and_diagonal():
    J = spatial_correlation_matrix(Nx=4, Ny=4, W_lambda_x=2.0, W_lambda_y=2.0)
    assert J.shape == (16, 16)
    assert torch.allclose(torch.diag(J), torch.ones(16), atol=1e-6)
    assert torch.allclose(J, J.T, atol=1e-6)


def test_generate_channels_shape_dtype_and_finite_values():
    H = generate_channels(
        num_samples=5,
        K=3,
        Nx=4,
        Ny=4,
        W_lambda_x=2.0,
        W_lambda_y=2.0,
        distance_km=0.2,
        seed=123,
    )
    assert H.shape == (5, 3, 16)
    assert H.is_complex()
    assert torch.isfinite(H.real).all()
    assert torch.isfinite(H.imag).all()
