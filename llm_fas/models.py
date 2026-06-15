from __future__ import annotations

import math
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


def _resolve_selection_mode(training: bool, selection_mode: Literal["soft", "hard"] | None) -> Literal["soft", "hard"]:
    active_selection_mode = ("soft" if training else "hard") if selection_mode is None else selection_mode
    if active_selection_mode not in ("soft", "hard"):
        raise ValueError(f"selection_mode must be 'soft' or 'hard', got {active_selection_mode!r}")
    return active_selection_mode


def _power_from_logits(power_logits: torch.Tensor, Pmax_W: float) -> tuple[torch.Tensor, torch.Tensor]:
    power_factors = torch.sigmoid(power_logits)
    p = F.softmax(power_factors[:, 0, :], dim=-1) * Pmax_W
    q = F.softmax(power_factors[:, 1, :], dim=-1) * Pmax_W
    return p, q


def _select_ports_from_scores(
    port_scores: torch.Tensor,
    tau: float,
    sinkhorn_iters: int,
    add_noise: bool,
    selection_mode: Literal["soft", "hard"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    selection_soft = gumbel_sinkhorn(
        port_scores,
        tau=tau,
        iters=sinkhorn_iters,
        add_noise=add_noise,
    )
    if selection_mode == "hard":
        ports = hard_topk_ports(selection_soft, port_scores.shape[1])
        selection_hard = ports_to_selection_matrix(ports, port_scores.shape[-1]).to(
            device=port_scores.device,
            dtype=port_scores.dtype,
        )
        return selection_soft, selection_hard, ports
    return selection_soft, selection_soft, None


class EffectiveChannelPowerCNN(nn.Module):
    def __init__(self, K: int, n_active: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * K * n_active, 2 * K)

    def forward(self, H_eff: torch.Tensor) -> torch.Tensor:
        features = torch.stack([H_eff.real.float(), H_eff.imag.float()], dim=1)
        features = self.net(features).flatten(1)
        return self.fc(features)


class JointFASModelBase(nn.Module):
    def __init__(self, cfg: ExperimentConfig, d_llm: int):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.system.K
        self.N = cfg.system.Nx * cfg.system.Ny
        self.n_active = cfg.system.n_active
        self.sequence_length = self.N * self.n_active
        self.Pmax_W = dbm_to_watt(cfg.system.Pmax_dBm)
        self.noise_power = noise_power_watt(cfg.system.noise_psd_dBm_per_Hz, cfg.system.bandwidth_Hz)
        self.d_llm = int(d_llm)

        # FC1 (paper): flattens K*N → K*d_mha for cross-port interaction
        self.fc1_real = nn.Linear(self.K * self.N, self.K * cfg.model.d_mha)
        self.fc1_imag = nn.Linear(self.K * self.N, self.K * cfg.model.d_mha)
        self.real_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.imag_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.fc2 = nn.Linear(2 * self.K * cfg.model.d_mha, self.sequence_length * self.d_llm)
        self.port_head = nn.Linear(self.sequence_length * self.d_llm, self.n_active * self.N)
        self.power_head = nn.Linear(self.sequence_length * self.d_llm, 2 * self.K)

    @staticmethod
    def _make_sinusoidal_position_encoding(sequence_length: int, d_model: int) -> torch.Tensor:
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, sequence_length, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            encoding[0, :, 1::2] = torch.cos(position * div_term[: encoding[0, :, 1::2].shape[-1]])
        return encoding

    def _preprocess(self, H: torch.Tensor) -> torch.Tensor:
        B = H.shape[0]
        # Paper: flatten CSI into h_real, h_imag ∈ R^{K*N}, then FC1 → R^{K×dmha}
        h_real_flat = H.real.float().reshape(B, self.K * self.N)
        h_imag_flat = H.imag.float().reshape(B, self.K * self.N)
        real_tokens = self.fc1_real(h_real_flat).reshape(B, self.K, self.cfg.model.d_mha)
        imag_tokens = self.fc1_imag(h_imag_flat).reshape(B, self.K, self.cfg.model.d_mha)
        real_attn, _ = self.real_mha(real_tokens, real_tokens, real_tokens, need_weights=False)
        imag_attn, _ = self.imag_mha(imag_tokens, imag_tokens, imag_tokens, need_weights=False)
        features = torch.cat([real_attn, imag_attn], dim=1)  # B, 2*K, d_mha
        h_merged = features.reshape(B, 2 * self.K * self.cfg.model.d_mha)
        embeddings = self.fc2(h_merged).reshape(B, self.sequence_length, self.d_llm)
        # Positional encoding (paper Eq. 13-14): Hem = Hfc2 + Hpe
        position_encoding = self._make_sinusoidal_position_encoding(
            self.sequence_length, self.d_llm
        ).to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings + position_encoding

    def _run_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode = _resolve_selection_mode(training, selection_mode)

        embeddings = self._preprocess(H)
        backbone_out = self._run_backbone(embeddings)
        z = backbone_out.reshape(H.shape[0], -1)

        port_scores = self.port_head(z).reshape(H.shape[0], self.n_active, self.N)
        power_logits = self.power_head(z).reshape(H.shape[0], 2, self.K)
        p, q = _power_from_logits(power_logits, self.Pmax_W)
        selection_soft, selection, ports = _select_ports_from_scores(
            port_scores,
            tau=tau,
            sinkhorn_iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
            selection_mode=active_selection_mode,
        )

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
            out["selection_hard"] = selection
        return out


class ProposedLLMFASModel(JointFASModelBase):
    def __init__(self, cfg: ExperimentConfig):
        backbone = GPT2Model.from_pretrained(cfg.model.backbone_name)
        backbone.h = nn.ModuleList(list(backbone.h[: cfg.model.gpt2_layers]))
        backbone.config.n_layer = len(backbone.h)
        for param in backbone.parameters():
            param.requires_grad = False

        super().__init__(cfg, d_llm=int(backbone.config.n_embd))

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

    def _run_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs_embeds=embeddings).last_hidden_state


class TransformerBaselineModel(JointFASModelBase):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg, d_llm=cfg.model.d_mha)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=cfg.model.mha_heads,
            dim_feedforward=4 * self.d_llm,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.gpt2_layers)

    def _run_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.backbone(embeddings)


class CNNBaselineModel(nn.Module):
    """Sequential CNN baseline: CNN port selection followed by CNN power allocation."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.system.K
        self.N = cfg.system.Nx * cfg.system.Ny
        self.n_active = cfg.system.n_active
        self.Pmax_W = dbm_to_watt(cfg.system.Pmax_dBm)
        self.noise_power = noise_power_watt(cfg.system.noise_psd_dBm_per_Hz, cfg.system.bandwidth_Hz)
        self.port_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.port_head = nn.Linear(32 * self.K * self.N, self.n_active * self.N)
        self.power_cnn = EffectiveChannelPowerCNN(self.K, self.n_active)

    def _port_scores(self, H: torch.Tensor) -> torch.Tensor:
        features = torch.stack([H.real.float(), H.imag.float()], dim=1)
        features = self.port_cnn(features).flatten(1)
        return self.port_head(features).reshape(H.shape[0], self.n_active, self.N)

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode = _resolve_selection_mode(training, selection_mode)
        port_scores = self._port_scores(H)
        selection_soft, selection, ports = _select_ports_from_scores(
            port_scores,
            tau=tau,
            sinkhorn_iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
            selection_mode=active_selection_mode,
        )
        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        power_logits = self.power_cnn(H_eff).reshape(H.shape[0], 2, self.K)
        p, q = _power_from_logits(power_logits, self.Pmax_W)
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
            out["selection_hard"] = selection
        return out


class LLMSequentialBaselineModel(ProposedLLMFASModel):
    """LLM port selection with a sequential CNN power-allocation head."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        for param in self.power_head.parameters():
            param.requires_grad = False
        self.sequential_power_cnn = EffectiveChannelPowerCNN(self.K, self.n_active)

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode = _resolve_selection_mode(training, selection_mode)
        embeddings = self._preprocess(H)
        backbone_out = self._run_backbone(embeddings)
        z = backbone_out.reshape(H.shape[0], -1)
        port_scores = self.port_head(z).reshape(H.shape[0], self.n_active, self.N)
        selection_soft, selection, ports = _select_ports_from_scores(
            port_scores,
            tau=tau,
            sinkhorn_iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
            selection_mode=active_selection_mode,
        )
        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        power_logits = self.sequential_power_cnn(H_eff).reshape(H.shape[0], 2, self.K)
        p, q = _power_from_logits(power_logits, self.Pmax_W)
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
            out["selection_hard"] = selection
        return out
