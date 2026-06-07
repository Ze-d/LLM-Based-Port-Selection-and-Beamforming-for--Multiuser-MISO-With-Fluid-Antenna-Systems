from dataclasses import replace
import math

import torch
from transformers import GPT2Config, GPT2Model

from llm_fas.config import load_config
from llm_fas.models import ProposedLLMFASModel
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

    assert out["port_scores"].shape == (2, cfg.system.n_active, cfg.system.Nx * cfg.system.Ny)
    assert out["selection_soft"].shape == (2, cfg.system.n_active, cfg.system.Nx * cfg.system.Ny)
    assert out["p"].shape == (2, cfg.system.K)
    assert out["q"].shape == (2, cfg.system.K)
    assert out["H_eff"].shape == (2, cfg.system.K, cfg.system.n_active)
    assert out["C"].shape == (2, cfg.system.n_active, cfg.system.K)
    assert out["rate"].shape == (2,)
    assert torch.isfinite(out["rate"]).all()
    assert torch.allclose(out["selection_soft"].sum(dim=-1), torch.ones(2, cfg.system.n_active), atol=1e-5)

    Pmax = dbm_to_watt(cfg.system.Pmax_dBm)
    assert torch.allclose(out["p"].sum(dim=1), torch.full((2,), Pmax), rtol=1e-5, atol=1e-7)
    assert torch.allclose(out["q"].sum(dim=1), torch.full((2,), Pmax), rtol=1e-5, atol=1e-7)


def test_proposed_model_exposes_trainable_lora_and_layernorm(monkeypatch):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    model = ProposedLLMFASModel(_tiny_cfg())

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    assert any("lora" in name for name in trainable_names)
    assert any("ln_" in name for name in trainable_names)
    assert any("real_proj" in name for name in trainable_names)
    assert all(math.isfinite(param.detach().float().abs().mean().item()) for _, param in model.named_parameters())
