from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SystemConfig:
    K: int
    Nx: int
    Ny: int
    n_active: int
    W_lambda_x: float
    W_lambda_y: float
    Pmax_dBm: float
    noise_psd_dBm_per_Hz: float
    bandwidth_Hz: float
    carrier_Hz: float
    distance_km: float


@dataclass(frozen=True)
class DataConfig:
    train_samples: int
    val_samples: int
    test_samples: int


@dataclass(frozen=True)
class ModelConfig:
    d_mha: int
    mha_heads: int
    backbone_name: str
    gpt2_layers: int
    lora_rank: int
    sinkhorn_iters: int


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    tau_min: float
    tau_decay: float
    output_dir: str


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    system: SystemConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        seed=int(raw["seed"]),
        device=str(raw.get("device", "auto")),
        system=SystemConfig(**raw["system"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )
