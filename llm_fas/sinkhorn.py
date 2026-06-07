from __future__ import annotations

import torch
import torch.nn.functional as F


def gumbel_sinkhorn(
    logits: torch.Tensor,
    tau: float,
    iters: int,
    add_noise: bool,
) -> torch.Tensor:
    scores = logits
    if add_noise:
        uniform = torch.rand_like(scores).clamp(1e-6, 1.0 - 1e-6)
        scores = scores - torch.log(-torch.log(uniform))

    eps = torch.finfo(scores.dtype).eps
    relaxed = torch.softmax(scores / tau, dim=-1)
    for _ in range(iters):
        relaxed = relaxed / relaxed.sum(dim=-1, keepdim=True).clamp_min(eps)
        col_sum = relaxed.sum(dim=-2, keepdim=True).clamp_min(eps)
        relaxed = relaxed / torch.maximum(col_sum, torch.ones_like(col_sum))
    return relaxed / relaxed.sum(dim=-1, keepdim=True).clamp_min(eps)


def hard_topk_ports(scores: torch.Tensor, n_active: int) -> torch.Tensor:
    per_port_scores = scores.max(dim=1).values
    return torch.topk(per_port_scores, k=n_active, dim=-1).indices.to(torch.long)


def ports_to_selection_matrix(ports: torch.Tensor, N: int) -> torch.Tensor:
    return F.one_hot(ports.to(torch.long), num_classes=N).to(torch.float32)
