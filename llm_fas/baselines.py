from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llm_fas.beamforming import beamforming_from_pq, sum_rate
from llm_fas.config import ExperimentConfig
from llm_fas.physics import dbm_to_watt, noise_power_watt
from llm_fas.sinkhorn import ports_to_selection_matrix


def _random_port_selection(
    B: int, N: int, n_active: int, device: torch.device, seed: int | None = None
) -> torch.Tensor:
    """Generate random port indices (uniform without replacement)."""
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator = generator.manual_seed(seed)
    sampled = [torch.randperm(N, generator=generator)[:n_active] for _ in range(B)]
    return torch.stack(sampled, dim=0).to(device)


class RandomBaselineModel(nn.Module):
    """Random port selection + MLP-based power allocation (paper's Random baseline).

    The model performs random port selection (fixed, non-trainable) and uses an MLP
    to learn power allocation factors p and q, which are then used by the optimal
    beamforming structure (paper Eq. 18).
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.K = cfg.system.K
        self.N = cfg.system.Nx * cfg.system.Ny
        self.n_active = cfg.system.n_active
        self.Pmax_W = dbm_to_watt(cfg.system.Pmax_dBm)
        self.noise_power = noise_power_watt(cfg.system.noise_psd_dBm_per_Hz, cfg.system.bandwidth_Hz)

        # MLP: flattened H_eff (real + imag) → p, q logits
        input_dim = 2 * self.K * self.n_active
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.K),
        )

    def forward(self, H: torch.Tensor, seed: int | None = None) -> dict[str, torch.Tensor]:
        B = H.shape[0]
        ports = _random_port_selection(B, self.N, self.n_active, H.device, seed)
        selection = ports_to_selection_matrix(ports, self.N).to(device=H.device, dtype=H.dtype)
        H_eff = H @ selection.transpose(-1, -2)

        # MLP power allocation from H_eff features (paper: MLP maps feature → power)
        feats = torch.cat([H_eff.real.flatten(1), H_eff.imag.flatten(1)], dim=1)
        power_logits = self.mlp(feats).reshape(B, 2, self.K)
        p = F.softmax(power_logits[:, 0, :], dim=-1) * self.Pmax_W
        q = F.softmax(power_logits[:, 1, :], dim=-1) * self.Pmax_W

        C = beamforming_from_pq(H_eff, p, q, self.noise_power)
        rate = sum_rate(H_eff, C, self.noise_power)
        return {
            "ports": ports,
            "selection": selection,
            "H_eff": H_eff,
            "p": p,
            "q": q,
            "C": C,
            "rate": rate,
        }


def evaluate_random_baseline(H: torch.Tensor, cfg: ExperimentConfig, seed: int) -> float:
    """Evaluate random port selection with uniform power allocation.

    This is the original uniform-power fallback. For the MLP-based paper baseline,
    use ``evaluate_random_baseline_mlp`` with a trained checkpoint.
    """
    system = cfg.system
    B = H.shape[0]
    N = system.Nx * system.Ny
    n_active = system.n_active

    ports = _random_port_selection(B, N, n_active, H.device, seed)
    selection = ports_to_selection_matrix(ports, N).to(device=H.device, dtype=H.dtype)
    H_eff = H @ selection.transpose(-1, -2)

    Pmax = dbm_to_watt(system.Pmax_dBm)
    p = torch.full((B, system.K), Pmax / system.K, dtype=H.real.dtype, device=H.device)
    q = torch.full((B, system.K), Pmax / system.K, dtype=H.real.dtype, device=H.device)
    noise_power = noise_power_watt(system.noise_psd_dBm_per_Hz, system.bandwidth_Hz)
    C = beamforming_from_pq(H_eff, p, q, noise_power)
    rates = sum_rate(H_eff, C, noise_power)
    return float(rates.mean().item())


def evaluate_random_baseline_mlp(
    H: torch.Tensor, cfg: ExperimentConfig, checkpoint_path: str, seed: int
) -> float:
    """Evaluate random baseline with trained MLP power allocation."""
    import torch

    model = RandomBaselineModel(cfg).to(H.device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=H.device))
    model.eval()
    with torch.no_grad():
        out = model(H, seed=seed)
    return float(out["rate"].mean().item())
