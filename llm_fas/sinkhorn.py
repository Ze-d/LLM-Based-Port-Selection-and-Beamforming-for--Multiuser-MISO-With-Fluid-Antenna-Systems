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
    """Convert relaxed n x N scores to hard ports using row-wise argmax.

    The paper describes taking argmax over each row of the Sinkhorn output
    during inference. If two rows pick the same port, repair the duplicate by
    choosing that row's next-best unused port so the activation constraint still
    holds.
    """
    if scores.ndim != 3:
        raise ValueError(f"scores must have shape (B, n, N), got {tuple(scores.shape)}")
    if n_active > scores.shape[1] or n_active > scores.shape[2]:
        raise ValueError(f"n_active={n_active} is incompatible with scores shape {tuple(scores.shape)}")

    ranked_ports = torch.argsort(scores[:, :n_active, :], dim=-1, descending=True)
    ports = torch.empty(scores.shape[0], n_active, dtype=torch.long, device=scores.device)
    for batch_index in range(scores.shape[0]):
        used: set[int] = set()
        for row_index in range(n_active):
            for candidate in ranked_ports[batch_index, row_index].tolist():
                if candidate not in used:
                    ports[batch_index, row_index] = candidate
                    used.add(candidate)
                    break
    return ports


def ports_to_selection_matrix(ports: torch.Tensor, N: int) -> torch.Tensor:
    return F.one_hot(ports.to(torch.long), num_classes=N).to(torch.float32)
