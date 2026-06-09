from __future__ import annotations

import csv
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from llm_fas.baselines import evaluate_random_baseline
from llm_fas.config import ExperimentConfig
from llm_fas.data import build_datasets, make_loader
from llm_fas.models import JointFASModelBase, ProposedLLMFASModel, TransformerBaselineModel


MODEL_REGISTRY = {
    "proposed": ProposedLLMFASModel,
    "transformer": TransformerBaselineModel,
}

RESULT_FIELDNAMES = [
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
]

EVALUATION_METHOD_ORDER = ["transformer", "proposed"]


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


def build_model(method: str, cfg: ExperimentConfig) -> JointFASModelBase:
    try:
        model_cls = MODEL_REGISTRY[method]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported trainable method {method!r}. Supported methods: {supported}") from exc
    return model_cls(cfg)


def _evaluate_model_rate(
    model: JointFASModelBase,
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


def train_method(cfg: ExperimentConfig, method: str = "proposed") -> Path:
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = build_model(method, cfg).to(device)
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=cfg.train.lr)
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / f"{method}.pt"
    best_val_rate = -float("inf")

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
        if val_rate > best_val_rate:
            best_val_rate = val_rate
            torch.save(model.state_dict(), checkpoint_path)

    _write_history(output_dir / f"{method}_train_history.csv", history)
    if method == "proposed":
        _write_history(output_dir / "train_history.csv", history)
    return checkpoint_path


def train_proposed(cfg: ExperimentConfig) -> Path:
    return train_method(cfg, method="proposed")


def train_transformer(cfg: ExperimentConfig) -> Path:
    return train_method(cfg, method="transformer")


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


def _ordered_methods(methods: list[str]) -> list[str]:
    order = {method: index for index, method in enumerate(EVALUATION_METHOD_ORDER)}
    return sorted(methods, key=lambda method: (order.get(method, len(order)), method))


def _write_results(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoints(cfg: ExperimentConfig, checkpoints: Mapping[str, str | Path]) -> Path:
    if not checkpoints:
        raise ValueError("At least one checkpoint must be provided")

    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_H = build_datasets(cfg)
    test_loader = make_loader(test_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 2)

    random_rate = evaluate_random_baseline(test_H.to(device), cfg, seed=cfg.seed + 3)

    rows = [
        _result_row(cfg, method="random", selection_mode="hard", test_sum_rate=random_rate),
    ]
    tau = cfg.train.tau_min
    for method in _ordered_methods(list(checkpoints.keys())):
        checkpoint_path = Path(checkpoints[method])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model = build_model(method, cfg).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        method_rate = _evaluate_model_rate(model, test_loader, device, tau, selection_mode="hard")
        rows.append(_result_row(cfg, method=method, selection_mode="hard", test_sum_rate=method_rate))

    results_path = output_dir / "results.csv"
    _write_results(results_path, rows)
    return results_path


def evaluate_checkpoint(cfg: ExperimentConfig, checkpoint_path: str | Path) -> Path:
    return evaluate_checkpoints(cfg, {"proposed": checkpoint_path})
