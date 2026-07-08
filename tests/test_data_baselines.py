import math

import torch

from llm_fas.baselines import RandomBaselineModel, _random_port_selection, evaluate_random_baseline
from llm_fas.config import load_config
from llm_fas.data import build_datasets, make_loader
from llm_fas.physics import dbm_to_watt


def test_paper_default_uses_paper_train_validation_split():
    cfg = load_config("configs/paper_default.yaml")

    assert cfg.system.Nx == 30
    assert cfg.system.Ny == 30
    assert cfg.system.Nx * cfg.system.Ny == 900
    assert cfg.data.train_samples == 8000
    assert cfg.data.val_samples == 2000
    assert cfg.data.test_samples == 1000


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


def test_random_port_selection_unseeded_advances_global_rng():
    torch.manual_seed(123)
    first = _random_port_selection(4, 16, 4, torch.device("cpu"), seed=None)
    second = _random_port_selection(4, 16, 4, torch.device("cpu"), seed=None)

    assert not torch.equal(first, second)


def test_random_port_selection_seeded_is_reproducible():
    first = _random_port_selection(4, 16, 4, torch.device("cpu"), seed=123)
    second = _random_port_selection(4, 16, 4, torch.device("cpu"), seed=123)

    assert torch.equal(first, second)


def test_random_baseline_power_head_uses_sigmoid_before_softmax():
    cfg = load_config("configs/mvp.yaml")
    _, _, test = build_datasets(cfg)
    model = RandomBaselineModel(cfg)
    with torch.no_grad():
        for param in model.mlp.parameters():
            param.zero_()
        model.mlp[-1].bias.copy_(
            torch.tensor([-1000.0, 1000.0, 0.0, -1000.0, 1000.0, 0.0], dtype=model.mlp[-1].bias.dtype)
        )

    out = model(test[:1], seed=123)
    p = out["p"]
    q = out["q"]
    Pmax = dbm_to_watt(cfg.system.Pmax_dBm)

    assert torch.allclose(p.sum(dim=1), torch.tensor([Pmax], dtype=p.dtype), rtol=1e-5, atol=1e-7)
    assert torch.allclose(q.sum(dim=1), torch.tensor([Pmax], dtype=q.dtype), rtol=1e-5, atol=1e-7)
    assert p[0, 1] / Pmax < 0.6
    assert q[0, 1] / Pmax < 0.6
