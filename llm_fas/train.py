from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch

from llm_fas.baselines import evaluate_random_baseline
from llm_fas.config import ExperimentConfig
from llm_fas.data import build_datasets, make_loader
from llm_fas.models import ProposedLLMFASModel


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _tau_for_epoch(cfg: ExperimentConfig, epoch_index: int) -> float:
    return max(cfg.train.tau_min, cfg.train.tau_decay**epoch_index)


def _evaluate_model_rate(
    model: ProposedLLMFASModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tau: float,
    selection_mode: str,
) -> float:
    model.eval()
    total_rate = 0.0
    total_samples = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            out = model(batch, tau=tau, training=False, selection_mode=selection_mode)
            total_rate += out["rate"].sum().item()
            total_samples += batch.shape[0]
    return total_rate / total_samples


def train_proposed(cfg: ExperimentConfig) -> Path:
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = ProposedLLMFASModel(cfg).to(device)
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=cfg.train.lr)
    history: list[dict[str, float | int]] = []

    for epoch in range(cfg.train.epochs):
        model.train()
        tau = _tau_for_epoch(cfg, epoch)
        total_loss = 0.0
        total_samples = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            out = model(batch, tau=tau, training=True)
            loss = -out["rate"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_samples += batch.shape[0]

        train_loss = total_loss / total_samples
        val_rate = _evaluate_model_rate(model, val_loader, device, tau, selection_mode="hard")
        val_loss = -val_rate
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: train={train_loss}, val={val_loss}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    checkpoint_path = output_dir / "proposed.pt"
    torch.save(model.state_dict(), checkpoint_path)
    _write_history(output_dir / "train_history.csv", history)
    return checkpoint_path


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def _result_row(cfg: ExperimentConfig, method: str, selection_mode: str, test_sum_rate: float) -> dict[str, float | int | str]:
    N = cfg.system.Nx * cfg.system.Ny
    return {
        "method": method,
        "selection_mode": selection_mode,
        "K": cfg.system.K,
        "Nx": cfg.system.Nx,
        "Ny": cfg.system.Ny,
        "N": N,
        "n_active": cfg.system.n_active,
        "W_lambda_x": cfg.system.W_lambda_x,
        "W_lambda_y": cfg.system.W_lambda_y,
        "Pmax_dBm": cfg.system.Pmax_dBm,
        "distance_km": cfg.system.distance_km,
        "seed": cfg.seed,
        "train_samples": cfg.data.train_samples,
        "val_samples": cfg.data.val_samples,
        "test_samples": cfg.data.test_samples,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "gpt2_layers": cfg.model.gpt2_layers,
        "d_mha": cfg.model.d_mha,
        "mha_heads": cfg.model.mha_heads,
        "test_sum_rate": test_sum_rate,
    }


def evaluate_checkpoint(cfg: ExperimentConfig, checkpoint_path: str | Path) -> Path:
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_H = build_datasets(cfg)
    test_loader = make_loader(test_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 2)

    model = ProposedLLMFASModel(cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    tau = cfg.train.tau_min
    proposed_rate = _evaluate_model_rate(model, test_loader, device, tau, selection_mode="hard")
    random_rate = evaluate_random_baseline(test_H.to(device), cfg, seed=cfg.seed + 3)

    results_path = output_dir / "results.csv"
    rows = [
        _result_row(cfg, method="random", selection_mode="hard", test_sum_rate=random_rate),
        _result_row(cfg, method="proposed", selection_mode="hard", test_sum_rate=proposed_rate),
    ]
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "selection_mode",
                "K",
                "Nx",
                "Ny",
                "N",
                "n_active",
                "W_lambda_x",
                "W_lambda_y",
                "Pmax_dBm",
                "distance_km",
                "seed",
                "train_samples",
                "val_samples",
                "test_samples",
                "epochs",
                "batch_size",
                "gpt2_layers",
                "d_mha",
                "mha_heads",
                "test_sum_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return results_path
