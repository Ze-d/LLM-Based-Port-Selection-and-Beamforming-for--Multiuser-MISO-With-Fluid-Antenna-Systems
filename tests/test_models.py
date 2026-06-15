from dataclasses import replace
import math

import pytest
import torch
from transformers import GPT2Config, GPT2Model

from llm_fas.config import load_config
from llm_fas.models import CNNBaselineModel, LLMSequentialBaselineModel, ProposedLLMFASModel, TransformerBaselineModel
from llm_fas.physics import dbm_to_watt, generate_channels


def _tiny_cfg():
    cfg = load_config("configs/mvp.yaml")
    return replace(
        cfg,
        model=replace(
            cfg.model,
            d_mha=32,
            mha_heads=4,
            gpt2_layers=2,
            lora_rank=2,
            sinkhorn_iters=3,
        ),
    )


def _tiny_gpt2(_name: str) -> GPT2Model:
    config = GPT2Config(
        n_layer=2,
        n_embd=32,
        n_head=4,
        n_positions=64,
        n_ctx=64,
        vocab_size=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2Model(config)


def _assert_joint_model_contract(out, batch_size: int, cfg) -> None:
    N = cfg.system.Nx * cfg.system.Ny
    assert out["port_scores"].shape == (batch_size, cfg.system.n_active, N)
    assert out["selection_soft"].shape == (batch_size, cfg.system.n_active, N)
    assert out["p"].shape == (batch_size, cfg.system.K)
    assert out["q"].shape == (batch_size, cfg.system.K)
    assert out["H_eff"].shape == (batch_size, cfg.system.K, cfg.system.n_active)
    assert out["C"].shape == (batch_size, cfg.system.n_active, cfg.system.K)
    assert out["rate"].shape == (batch_size,)
    assert torch.isfinite(out["rate"]).all()
    assert torch.allclose(
        out["selection_soft"].sum(dim=-1),
        torch.ones(batch_size, cfg.system.n_active),
        atol=1e-5,
    )

    Pmax = dbm_to_watt(cfg.system.Pmax_dBm)
    assert torch.allclose(out["p"].sum(dim=1), torch.full((batch_size,), Pmax), rtol=1e-5, atol=1e-7)
    assert torch.allclose(out["q"].sum(dim=1), torch.full((batch_size,), Pmax), rtol=1e-5, atol=1e-7)


def test_proposed_model_forward_returns_expected_tensors(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=2,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=99,
    )

    model = ProposedLLMFASModel(cfg)
    out = model(H, tau=1.0, training=True)

    _assert_joint_model_contract(out, batch_size=2, cfg=cfg)


def test_proposed_model_hard_inference_returns_unique_ports(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=3,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=100,
    )

    model = ProposedLLMFASModel(cfg)
    out = model(H, tau=0.1, training=False, selection_mode="hard")

    N = cfg.system.Nx * cfg.system.Ny
    assert out["selection_mode"] == "hard"
    assert out["ports"].shape == (3, cfg.system.n_active)
    assert out["selection_hard"].shape == (3, cfg.system.n_active, N)
    assert out["selection"].shape == (3, cfg.system.n_active, N)
    assert torch.equal(out["selection"], out["selection_hard"])
    assert torch.allclose(out["selection_hard"].sum(dim=-1), torch.ones(3, cfg.system.n_active))
    assert torch.all((out["selection_hard"] == 0.0) | (out["selection_hard"] == 1.0))
    for ports in out["ports"].tolist():
        assert len(set(ports)) == cfg.system.n_active
    assert torch.isfinite(out["rate"]).all()


def test_proposed_model_invalid_selection_mode_raises(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=1,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=101,
    )

    model = ProposedLLMFASModel(cfg)
    with pytest.raises(ValueError):
        model(H[:1], tau=0.1, training=False, selection_mode="")


def test_proposed_model_exposes_trainable_lora_and_layernorm(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    model = ProposedLLMFASModel(_tiny_cfg())

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    assert any("lora" in name for name in trainable_names)
    assert any("ln_" in name for name in trainable_names)
    assert any("fc1_real" in name for name in trainable_names)
    assert all(math.isfinite(param.detach().float().abs().mean().item()) for _, param in model.named_parameters())


def test_transformer_baseline_forward_returns_expected_tensors():
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=2,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=102,
    )

    model = TransformerBaselineModel(cfg)
    out = model(H, tau=1.0, training=True)

    _assert_joint_model_contract(out, batch_size=2, cfg=cfg)
    assert out["selection_mode"] == "soft"


def test_transformer_baseline_hard_inference_returns_unique_ports():
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=3,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=103,
    )

    model = TransformerBaselineModel(cfg)
    out = model(H, tau=0.1, training=False, selection_mode="hard")

    N = cfg.system.Nx * cfg.system.Ny
    assert out["selection_mode"] == "hard"
    assert out["ports"].shape == (3, cfg.system.n_active)
    assert out["selection_hard"].shape == (3, cfg.system.n_active, N)
    assert torch.equal(out["selection"], out["selection_hard"])
    assert torch.allclose(out["selection_hard"].sum(dim=-1), torch.ones(3, cfg.system.n_active))
    assert torch.all((out["selection_hard"] == 0.0) | (out["selection_hard"] == 1.0))
    for ports in out["ports"].tolist():
        assert len(set(ports)) == cfg.system.n_active
    assert torch.isfinite(out["rate"]).all()


def test_cnn_baseline_forward_returns_expected_tensors():
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=2,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=104,
    )

    model = CNNBaselineModel(cfg)
    out = model(H, tau=1.0, training=True)

    _assert_joint_model_contract(out, batch_size=2, cfg=cfg)
    assert out["selection_mode"] == "soft"


def test_llm_sequential_baseline_forward_returns_expected_tensors(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg()
    H = generate_channels(
        num_samples=2,
        K=cfg.system.K,
        Nx=cfg.system.Nx,
        Ny=cfg.system.Ny,
        W_lambda_x=cfg.system.W_lambda_x,
        W_lambda_y=cfg.system.W_lambda_y,
        distance_km=cfg.system.distance_km,
        seed=105,
    )

    model = LLMSequentialBaselineModel(cfg)
    out = model(H, tau=1.0, training=True)

    _assert_joint_model_contract(out, batch_size=2, cfg=cfg)
    assert out["selection_mode"] == "soft"
    assert not any(param.requires_grad for param in model.power_head.parameters())
