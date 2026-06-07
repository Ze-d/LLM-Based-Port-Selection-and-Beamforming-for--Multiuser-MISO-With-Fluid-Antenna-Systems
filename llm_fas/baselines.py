from __future__ import annotations

import torch

from llm_fas.beamforming import beamforming_from_pq, sum_rate
from llm_fas.config import ExperimentConfig
from llm_fas.physics import dbm_to_watt, noise_power_watt
from llm_fas.sinkhorn import ports_to_selection_matrix


def evaluate_random_baseline(H: torch.Tensor, cfg: ExperimentConfig, seed: int) -> float:
    system = cfg.system
    B = H.shape[0]
    N = system.Nx * system.Ny
    n_active = system.n_active
    generator = torch.Generator(device="cpu").manual_seed(seed)

    sampled_ports = [torch.randperm(N, generator=generator)[:n_active] for _ in range(B)]
    ports = torch.stack(sampled_ports, dim=0).to(H.device)
    selection = ports_to_selection_matrix(ports, N).to(device=H.device, dtype=H.dtype)
    H_eff = H @ selection.transpose(-1, -2)

    Pmax = dbm_to_watt(system.Pmax_dBm)
    p = torch.full((B, system.K), Pmax / system.K, dtype=H.real.dtype, device=H.device)
    q = torch.full((B, system.K), Pmax / system.K, dtype=H.real.dtype, device=H.device)
    noise_power = noise_power_watt(system.noise_psd_dBm_per_Hz, system.bandwidth_Hz)
    C = beamforming_from_pq(H_eff, p, q, noise_power)
    rates = sum_rate(H_eff, C, noise_power)
    return float(rates.mean().item())
