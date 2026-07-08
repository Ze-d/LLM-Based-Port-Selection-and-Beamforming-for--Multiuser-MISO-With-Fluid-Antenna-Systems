from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch

from llm_fas.baselines import RandomBaselineModel, evaluate_random_baseline, evaluate_random_baseline_mlp
from llm_fas.config import ExperimentConfig
from llm_fas.data import build_datasets, make_loader
from llm_fas.models import (
    CNNBaselineModel,
    JointFASModelBase,
    LLMSequentialBaselineModel,
    ProposedLLMFASModel,
    TransformerBaselineModel,
)


MODEL_REGISTRY = {
    "cnn": CNNBaselineModel,
    "llm_sequential": LLMSequentialBaselineModel,
    "proposed": ProposedLLMFASModel,
    "transformer": TransformerBaselineModel,
    "random": RandomBaselineModel,
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

EVALUATION_METHOD_ORDER = ["cnn", "transformer", "llm_sequential", "proposed"]


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


def _device_info(requested_device: str, resolved_device: torch.device) -> dict[str, str | bool | None]:
    info: dict[str, str | bool | None] = {
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": None,
    }
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        index = resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name(index)
    return info


def _log_device(stage: str, requested_device: str, resolved_device: torch.device, method: str | None = None) -> None:
    info = _device_info(requested_device, resolved_device)
    method_text = f" method={method}" if method else ""
    name_text = f" ({info['cuda_device_name']})" if info["cuda_device_name"] else ""
    print(
        f"[llm_fas] {stage}{method_text} "
        f"requested_device={info['requested_device']} resolved_device={info['resolved_device']}{name_text}"
    )


def _write_device_info(path: Path, requested_device: str, resolved_device: torch.device) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_device_info(requested_device, resolved_device), f, indent=2)
        f.write("\n")


def _tau_for_epoch(cfg: ExperimentConfig, epoch_index: int) -> float:
    # Paper: τ = max(0.1, 0.95^epoch) where epoch is 1-indexed
    return max(cfg.train.tau_min, cfg.train.tau_decay**(epoch_index + 1))


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


def _evaluate_forward_rate(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tau: float,
    forward_fn: Callable[[torch.Tensor, float], dict[str, torch.Tensor | str]],
) -> float:
    total_rate = 0.0
    total_samples = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            out = forward_fn(batch, tau)
            total_rate += out["rate"].sum().item()
            total_samples += batch.shape[0]
    return total_rate / total_samples


def _train_stage(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    train_forward: Callable[[torch.Tensor, float], dict[str, torch.Tensor | str]],
    val_forward: Callable[[torch.Tensor, float], dict[str, torch.Tensor | str]],
    stage_name: str,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    best_val_rate = -float("inf")
    patience = 10
    no_improve = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        tau = _tau_for_epoch(cfg, epoch)
        total_loss = 0.0
        total_samples = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            out = train_forward(batch, tau)
            loss = -out["rate"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {stage_name} loss at epoch {epoch + 1}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_samples += batch.shape[0]

        train_loss = total_loss / total_samples
        model.eval()
        val_rate = _evaluate_forward_rate(val_loader, device, tau, val_forward)
        val_loss = -val_rate
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(
                f"Non-finite {stage_name} loss at epoch {epoch + 1}: train={train_loss}, val={val_loss}"
            )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_rate > best_val_rate:
            best_val_rate = val_rate
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[llm_fas] Early stopping {stage_name} at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs)"
                )
                break
    return history


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError(f"{model.__class__.__name__} has no trainable parameters")
    return params


def _offset_history(
    first: list[dict[str, float | int]],
    second: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    offset = len(first)
    return first + [
        {
            "epoch": int(row["epoch"]) + offset,
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
        }
        for row in second
    ]


def train_llm_sequential_baseline(cfg: ExperimentConfig) -> Path:
    """Train LLM-sequential as a strict two-stage baseline.

    Stage 1 optimizes only the LLM port selector with uniform power. Stage 2
    freezes that selector and trains the CNN power allocator on the selected
    effective channel, preventing the baseline from becoming a joint optimizer.
    """
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_device("train", cfg.device, device, method="llm_sequential")
    _write_device_info(output_dir / "device_info.json", cfg.device, device)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = LLMSequentialBaselineModel(cfg).to(device)

    port_checkpoint_path = output_dir / "llm_sequential_port_selector.pt"
    model.configure_port_selection_stage()
    port_optimizer = torch.optim.Adam(_trainable_parameters(model), lr=cfg.train.lr)
    port_history = _train_stage(
        cfg,
        model,
        train_loader,
        val_loader,
        device,
        port_optimizer,
        port_checkpoint_path,
        train_forward=lambda batch, tau: model.forward_port_selection_stage(batch, tau=tau, training=True),
        val_forward=lambda batch, tau: model.forward_port_selection_stage(
            batch, tau=tau, training=False, selection_mode="soft"
        ),
        stage_name="llm_sequential_port_selection",
    )

    model.load_state_dict(torch.load(port_checkpoint_path, map_location=device))
    checkpoint_path = output_dir / "llm_sequential.pt"
    model.configure_power_allocation_stage()
    power_optimizer = torch.optim.Adam(_trainable_parameters(model), lr=cfg.train.lr)
    power_history = _train_stage(
        cfg,
        model,
        train_loader,
        val_loader,
        device,
        power_optimizer,
        checkpoint_path,
        train_forward=lambda batch, tau: model.forward_power_allocation_stage(
            batch, tau=tau, selection_mode="hard"
        ),
        val_forward=lambda batch, tau: model.forward_power_allocation_stage(
            batch, tau=tau, selection_mode="hard"
        ),
        stage_name="llm_sequential_power_allocation",
    )

    _write_history(output_dir / "llm_sequential_port_train_history.csv", port_history)
    _write_history(output_dir / "llm_sequential_power_train_history.csv", power_history)
    _write_history(output_dir / "llm_sequential_train_history.csv", _offset_history(port_history, power_history))
    return checkpoint_path


def train_cnn_baseline(cfg: ExperimentConfig) -> Path:
    """Train CNN as a strict two-stage sequential baseline."""
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_device("train", cfg.device, device, method="cnn")
    _write_device_info(output_dir / "device_info.json", cfg.device, device)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = CNNBaselineModel(cfg).to(device)

    port_checkpoint_path = output_dir / "cnn_port_selector.pt"
    model.configure_port_selection_stage()
    port_optimizer = torch.optim.Adam(_trainable_parameters(model), lr=cfg.train.lr)
    port_history = _train_stage(
        cfg,
        model,
        train_loader,
        val_loader,
        device,
        port_optimizer,
        port_checkpoint_path,
        train_forward=lambda batch, tau: model.forward_port_selection_stage(batch, tau=tau, training=True),
        val_forward=lambda batch, tau: model.forward_port_selection_stage(
            batch, tau=tau, training=False, selection_mode="soft"
        ),
        stage_name="cnn_port_selection",
    )

    model.load_state_dict(torch.load(port_checkpoint_path, map_location=device))
    checkpoint_path = output_dir / "cnn.pt"
    model.configure_power_allocation_stage()
    power_optimizer = torch.optim.Adam(_trainable_parameters(model), lr=cfg.train.lr)
    power_history = _train_stage(
        cfg,
        model,
        train_loader,
        val_loader,
        device,
        power_optimizer,
        checkpoint_path,
        train_forward=lambda batch, tau: model.forward_power_allocation_stage(
            batch, tau=tau, selection_mode="hard"
        ),
        val_forward=lambda batch, tau: model.forward_power_allocation_stage(
            batch, tau=tau, selection_mode="hard"
        ),
        stage_name="cnn_power_allocation",
    )

    _write_history(output_dir / "cnn_port_train_history.csv", port_history)
    _write_history(output_dir / "cnn_power_train_history.csv", power_history)
    _write_history(output_dir / "cnn_train_history.csv", _offset_history(port_history, power_history))
    return checkpoint_path


def train_method(cfg: ExperimentConfig, method: str = "proposed") -> Path:
    if method == "cnn":
        return train_cnn_baseline(cfg)
    if method == "llm_sequential":
        return train_llm_sequential_baseline(cfg)

    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_device("train", cfg.device, device, method=method)
    _write_device_info(output_dir / "device_info.json", cfg.device, device)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = build_model(method, cfg).to(device)
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=cfg.train.lr)
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / f"{method}.pt"
    best_val_rate = -float("inf")
    patience = 10
    no_improve = 0

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
        # Paper validates with soft Gumbel-Sinkhorn (hard is only for final inference)
        val_rate = _evaluate_model_rate(model, val_loader, device, tau, selection_mode="soft")
        val_loss = -val_rate
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: train={train_loss}, val={val_loss}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_rate > best_val_rate:
            best_val_rate = val_rate
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[llm_fas] Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    _write_history(output_dir / f"{method}_train_history.csv", history)
    if method == "proposed":
        _write_history(output_dir / "train_history.csv", history)
    return checkpoint_path


def train_random_baseline(cfg: ExperimentConfig) -> Path:
    """Train the random baseline MLP power allocation (paper Section V-A)."""
    _set_seeds(cfg.seed)
    device = _select_device(cfg.device)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_device("train", cfg.device, device, method="random")
    _write_device_info(output_dir / "device_info.json", cfg.device, device)

    train_H, val_H, _ = build_datasets(cfg)
    train_loader = make_loader(train_H, cfg.train.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 1)

    model = RandomBaselineModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "random.pt"
    best_val_rate = -float("inf")
    patience = 10
    no_improve = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            # Use different random seed per batch for training diversity
            out = model(batch, seed=None)
            loss = -out["rate"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
            total_samples += batch.shape[0]

        train_loss = total_loss / total_samples
        # Evaluate with fixed seed for reproducibility
        val_rate = _evaluate_random_mlp_val(model, val_loader, device, seed=cfg.seed + 1)
        val_loss = -val_rate
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}: train={train_loss}, val={val_loss}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_rate > best_val_rate:
            best_val_rate = val_rate
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[llm_fas] Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    _write_history(output_dir / "random_train_history.csv", history)
    return checkpoint_path


def _evaluate_random_mlp_val(
    model: RandomBaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    total_rate = 0.0
    total_samples = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            out = model(batch, seed=seed)
            total_rate += out["rate"].sum().item()
            total_samples += batch.shape[0]
    return total_rate / total_samples


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
    _log_device("evaluate", cfg.device, device)
    _write_device_info(output_dir / "device_info.json", cfg.device, device)

    _, _, test_H = build_datasets(cfg)
    test_loader = make_loader(test_H, cfg.train.batch_size, shuffle=False, seed=cfg.seed + 2)

    test_device_data = test_H.to(device)
    # Use MLP-based random baseline if a trained checkpoint is provided,
    # otherwise fall back to uniform power allocation.
    random_checkpoint = checkpoints.get("random")
    if random_checkpoint is not None and Path(random_checkpoint).exists():
        random_rate = evaluate_random_baseline_mlp(test_device_data, cfg, str(random_checkpoint), seed=cfg.seed + 3)
    else:
        random_rate = evaluate_random_baseline(test_device_data, cfg, seed=cfg.seed + 3)

    rows = [
        _result_row(cfg, method="random", selection_mode="hard", test_sum_rate=random_rate),
    ]
    tau = cfg.train.tau_min
    trainable_methods = [m for m in _ordered_methods(list(checkpoints.keys())) if m != "random"]
    for method in trainable_methods:
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
