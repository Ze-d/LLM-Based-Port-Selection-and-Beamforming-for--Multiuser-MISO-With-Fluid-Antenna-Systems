from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_fas.config import ExperimentConfig
from llm_fas.physics import generate_channels


def build_datasets(cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    system = cfg.system
    common_kwargs = {
        "K": system.K,
        "Nx": system.Nx,
        "Ny": system.Ny,
        "W_lambda_x": system.W_lambda_x,
        "W_lambda_y": system.W_lambda_y,
        "distance_km": system.distance_km,
    }
    train = generate_channels(num_samples=cfg.data.train_samples, seed=cfg.seed, **common_kwargs)
    val = generate_channels(num_samples=cfg.data.val_samples, seed=cfg.seed + 1, **common_kwargs)
    test = generate_channels(num_samples=cfg.data.test_samples, seed=cfg.seed + 2, **common_kwargs)
    return train, val, test


def make_loader(H: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(TensorDataset(H), batch_size=batch_size, shuffle=shuffle, generator=generator)
