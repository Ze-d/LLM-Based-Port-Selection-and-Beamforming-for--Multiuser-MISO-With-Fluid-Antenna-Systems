import math

import torch

from llm_fas.baselines import evaluate_random_baseline
from llm_fas.config import load_config
from llm_fas.data import build_datasets, make_loader


def test_build_datasets_shapes_and_reproducibility():
    cfg = load_config("configs/mvp.yaml")
    train_a, val_a, test_a = build_datasets(cfg)
    train_b, val_b, test_b = build_datasets(cfg)

    assert train_a.shape == (cfg.data.train_samples, cfg.system.K, cfg.system.Nx * cfg.system.Ny)
    assert val_a.shape == (cfg.data.val_samples, cfg.system.K, cfg.system.Nx * cfg.system.Ny)
    assert test_a.shape == (cfg.data.test_samples, cfg.system.K, cfg.system.Nx * cfg.system.Ny)
    assert train_a.is_complex()
    assert torch.allclose(train_a, train_b)
    assert torch.allclose(val_a, val_b)
    assert torch.allclose(test_a, test_b)


def test_make_loader_batches_complex_channels():
    cfg = load_config("configs/mvp.yaml")
    train, _, _ = build_datasets(cfg)
    loader = make_loader(train, batch_size=16, shuffle=False, seed=cfg.seed)

    (batch,) = next(iter(loader))
    assert batch.shape == (16, cfg.system.K, cfg.system.Nx * cfg.system.Ny)
    assert batch.is_complex()


def test_random_baseline_returns_finite_positive_sum_rate():
    cfg = load_config("configs/mvp.yaml")
    _, _, test = build_datasets(cfg)
    rate = evaluate_random_baseline(test[:8], cfg, seed=123)

    assert isinstance(rate, float)
    assert math.isfinite(rate)
    assert rate > 0.0
