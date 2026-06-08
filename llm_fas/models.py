from __future__ import annotations

from typing import Literal

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model

from llm_fas.beamforming import beamforming_from_pq, sum_rate
from llm_fas.config import ExperimentConfig
from llm_fas.physics import dbm_to_watt, noise_power_watt
from llm_fas.sinkhorn import gumbel_sinkhorn, hard_topk_ports, ports_to_selection_matrix


class ProposedLLMFASModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.system.K
        self.N = cfg.system.Nx * cfg.system.Ny
        self.n_active = cfg.system.n_active
        self.sequence_length = self.N * self.n_active
        self.Pmax_W = dbm_to_watt(cfg.system.Pmax_dBm)
        self.noise_power = noise_power_watt(cfg.system.noise_psd_dBm_per_Hz, cfg.system.bandwidth_Hz)

        backbone = GPT2Model.from_pretrained(cfg.model.backbone_name)
        backbone.h = nn.ModuleList(list(backbone.h[: cfg.model.gpt2_layers]))
        backbone.config.n_layer = len(backbone.h)
        for param in backbone.parameters():
            param.requires_grad = False

        self.d_llm = int(backbone.config.n_embd)
        lora_config = LoraConfig(
            r=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_rank,
            target_modules=["c_attn"],
            bias="none",
            fan_in_fan_out=True,
        )
        self.backbone = get_peft_model(backbone, lora_config)
        for name, param in self.backbone.named_parameters():
            if "ln_" in name:
                param.requires_grad = True

        self.real_proj = nn.Linear(self.N, cfg.model.d_mha)
        self.imag_proj = nn.Linear(self.N, cfg.model.d_mha)
        self.real_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.imag_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.embed_proj = nn.Linear(2 * self.K * cfg.model.d_mha, self.sequence_length * self.d_llm)
        self.port_head = nn.Linear(self.sequence_length * self.d_llm, self.n_active * self.N)
        self.power_head = nn.Linear(self.sequence_length * self.d_llm, 2 * self.K)

    def _preprocess(self, H: torch.Tensor) -> torch.Tensor:
        real_tokens = self.real_proj(H.real.float())
        imag_tokens = self.imag_proj(H.imag.float())
        real_attn, _ = self.real_mha(real_tokens, real_tokens, real_tokens, need_weights=False)
        imag_attn, _ = self.imag_mha(imag_tokens, imag_tokens, imag_tokens, need_weights=False)
        features = torch.cat([real_attn, imag_attn], dim=1).reshape(H.shape[0], -1)
        return self.embed_proj(features).reshape(H.shape[0], self.sequence_length, self.d_llm)

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode = ("soft" if training else "hard") if selection_mode is None else selection_mode
        if active_selection_mode not in ("soft", "hard"):
            raise ValueError(f"selection_mode must be 'soft' or 'hard', got {active_selection_mode!r}")

        embeddings = self._preprocess(H)
        llm_out = self.backbone(inputs_embeds=embeddings).last_hidden_state
        z = llm_out.reshape(H.shape[0], -1)

        port_scores = self.port_head(z).reshape(H.shape[0], self.n_active, self.N)
        power_logits = self.power_head(z).reshape(H.shape[0], 2, self.K)
        p = F.softmax(power_logits[:, 0, :], dim=-1) * self.Pmax_W
        q = F.softmax(power_logits[:, 1, :], dim=-1) * self.Pmax_W

        selection_soft = gumbel_sinkhorn(
            port_scores,
            tau=tau,
            iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
        )
        if active_selection_mode == "hard":
            ports = hard_topk_ports(port_scores, self.n_active)
            selection_hard = ports_to_selection_matrix(ports, self.N).to(device=H.device, dtype=H.real.dtype)
            selection = selection_hard
        else:
            selection = selection_soft

        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        C = beamforming_from_pq(H_eff, p, q, self.noise_power)
        rate = sum_rate(H_eff, C, self.noise_power)
        out: dict[str, torch.Tensor | str] = {
            "port_scores": port_scores,
            "selection_soft": selection_soft,
            "selection": selection,
            "selection_mode": active_selection_mode,
            "p": p,
            "q": q,
            "H_eff": H_eff,
            "C": C,
            "rate": rate,
        }
        if active_selection_mode == "hard":
            out["ports"] = ports
            out["selection_hard"] = selection_hard
        return out
