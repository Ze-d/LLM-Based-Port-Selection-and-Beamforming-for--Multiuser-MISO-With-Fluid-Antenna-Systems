from __future__ import annotations

import torch


def beamforming_from_pq(
    H_eff: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    noise_power: float,
) -> torch.Tensor:
    B, K, n_active = H_eff.shape
    complex_dtype = H_eff.dtype
    real_dtype = H_eff.real.dtype
    device = H_eff.device

    eye = torch.eye(n_active, dtype=complex_dtype, device=device).expand(B, n_active, n_active)
    weighted_outer = torch.einsum("bk,bki,bkj->bij", q.to(real_dtype), H_eff.conj(), H_eff)
    jitter = torch.tensor(1e-8, dtype=real_dtype, device=device)
    system_matrix = eye + weighted_outer / noise_power
    system_matrix = system_matrix + jitter.to(complex_dtype) * eye

    rhs = H_eff.conj().transpose(1, 2)
    directions = torch.linalg.solve(system_matrix, rhs)
    norms = torch.linalg.vector_norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    return torch.sqrt(p.to(real_dtype)).unsqueeze(1).to(complex_dtype) * directions / norms


def sum_rate(H_eff: torch.Tensor, C: torch.Tensor, noise_power: float) -> torch.Tensor:
    gains = torch.einsum("bkn,bnj->bkj", H_eff, C).abs() ** 2
    signal = gains.diagonal(dim1=1, dim2=2)
    interference = gains.sum(dim=2) - signal
    sinr = signal / (interference + noise_power)
    return torch.log2(1.0 + sinr).sum(dim=1)
