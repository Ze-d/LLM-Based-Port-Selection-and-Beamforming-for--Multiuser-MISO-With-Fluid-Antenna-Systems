from __future__ import annotations

import math
from functools import lru_cache

import torch


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def noise_power_watt(noise_psd_dBm_per_Hz: float, bandwidth_Hz: float) -> float:
    noise_dbm = noise_psd_dBm_per_Hz + 10.0 * math.log10(bandwidth_Hz)
    return dbm_to_watt(noise_dbm)


def path_loss_beta(distance_km: float) -> float:
    path_loss_db = 128.1 + 37.6 * math.log10(distance_km)
    return 10.0 ** (-path_loss_db / 10.0)


def port_coordinates(Nx: int, Ny: int) -> list[tuple[int, int]]:
    return [(nx, ny) for ny in range(Ny) for nx in range(Nx)]


def spherical_j0(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / x)


def spatial_correlation_matrix(Nx: int, Ny: int, W_lambda_x: float, W_lambda_y: float) -> torch.Tensor:
    coords = port_coordinates(Nx, Ny)
    coord_tensor = torch.tensor(coords, dtype=torch.float32)
    x = coord_tensor[:, 0]
    y = coord_tensor[:, 1]
    dx = (x[:, None] - x[None, :]).abs() / max(Nx - 1, 1) * W_lambda_x
    dy = (y[:, None] - y[None, :]).abs() / max(Ny - 1, 1) * W_lambda_y
    distance = torch.sqrt(dx * dx + dy * dy)
    return spherical_j0(2.0 * math.pi * distance)


@lru_cache(maxsize=32)
def _correlation_factor(Nx: int, Ny: int, W_lambda_x: float, W_lambda_y: float) -> torch.Tensor:
    J = spatial_correlation_matrix(Nx, Ny, W_lambda_x, W_lambda_y)
    eigvals, eigvecs = torch.linalg.eigh(J)
    sort_idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sort_idx].clamp_min(0.0)
    eigvecs = eigvecs[:, sort_idx]
    return torch.sqrt(eigvals).to(torch.complex64).unsqueeze(1) * eigvecs.T.to(torch.complex64)


def generate_channels(
    num_samples: int,
    K: int,
    Nx: int,
    Ny: int,
    W_lambda_x: float,
    W_lambda_y: float,
    distance_km: float,
    seed: int | None = None,
) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    N = Nx * Ny
    scale = math.sqrt(0.5)
    g_real = torch.randn(num_samples, K, N, generator=generator, dtype=torch.float32)
    g_imag = torch.randn(num_samples, K, N, generator=generator, dtype=torch.float32)
    g = (g_real + 1j * g_imag) * scale

    correlated = g.to(torch.complex64) @ _correlation_factor(Nx, Ny, W_lambda_x, W_lambda_y)
    return math.sqrt(path_loss_beta(distance_km)) * correlated
