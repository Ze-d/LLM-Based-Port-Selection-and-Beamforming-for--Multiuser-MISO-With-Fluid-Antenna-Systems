from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model

from llm_fas.beamforming import beamforming_from_pq, sum_rate
from llm_fas.config import ExperimentConfig
from llm_fas.physics import dbm_to_watt, noise_power_watt
from llm_fas.sinkhorn import gumbel_sinkhorn, hard_topk_ports, ports_to_selection_matrix


class QVLoraConv1D(nn.Module):
    """LoRA wrapper for GPT-2 c_attn that updates only Q and V slices."""

    def __init__(self, base_layer: nn.Module, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not hasattr(base_layer, "weight") or not hasattr(base_layer, "bias"):
            raise TypeError("QVLoraConv1D expects a GPT-2 Conv1D-like layer")

        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = int(base_layer.weight.shape[0])
        out_features = int(base_layer.bias.shape[0])
        if out_features % 3 != 0:
            raise ValueError(f"GPT-2 c_attn output features must be divisible by 3, got {out_features}")

        self.hidden_size = out_features // 3
        self.scaling = 1.0
        self.lora_q_A = nn.Linear(in_features, rank, bias=False)
        self.lora_q_B = nn.Linear(rank, self.hidden_size, bias=False)
        self.lora_v_A = nn.Linear(in_features, rank, bias=False)
        self.lora_v_B = nn.Linear(rank, self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.lora_q_A.weight, std=0.02)
        nn.init.normal_(self.lora_v_A.weight, std=0.02)
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        q_delta = self.lora_q_B(self.lora_q_A(x)) * self.scaling
        v_delta = self.lora_v_B(self.lora_v_A(x)) * self.scaling
        h = self.hidden_size
        return torch.cat(
            [
                base_out[..., :h] + q_delta,
                base_out[..., h : 2 * h],
                base_out[..., 2 * h :] + v_delta,
            ],
            dim=-1,
        )


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


def _uniform_power(
    batch_size: int,
    K: int,
    Pmax_W: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    power = torch.full((batch_size, K), Pmax_W / K, dtype=dtype, device=device)
    return power, power.clone()


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
        self.sequence_length = self.n_active
        self.Pmax_W = dbm_to_watt(cfg.system.Pmax_dBm)
        self.noise_power = noise_power_watt(cfg.system.noise_psd_dBm_per_Hz, cfg.system.bandwidth_Hz)
        self.d_llm = int(d_llm)

        # FC1 (paper): flattens K*N → K*d_mha for cross-port interaction
        self.fc1_real = nn.Linear(self.K * self.N, self.K * cfg.model.d_mha)
        self.fc1_imag = nn.Linear(self.K * self.N, self.K * cfg.model.d_mha)
        self.real_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.imag_mha = nn.MultiheadAttention(cfg.model.d_mha, cfg.model.mha_heads, batch_first=True)
        self.fc2 = nn.Linear(2 * self.K * cfg.model.d_mha, self.sequence_length * self.d_llm)
        self.port_head = nn.Linear(self.d_llm, self.N)
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

        port_scores = self.port_head(backbone_out[:, : self.n_active, :])
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

        for block in backbone.h:
            block.attn.c_attn = QVLoraConv1D(block.attn.c_attn, rank=cfg.model.lora_rank)

        self.backbone = backbone
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
        self.port_head = nn.Conv2d(32, self.n_active, kernel_size=1)
        self.power_cnn = EffectiveChannelPowerCNN(self.K, self.n_active)

    def _port_scores(self, H: torch.Tensor) -> torch.Tensor:
        features = torch.stack([H.real.float(), H.imag.float()], dim=1)
        features = self.port_cnn(features)
        return self.port_head(features).mean(dim=2)

    def set_port_selector_trainable(self, trainable: bool) -> None:
        for module in (self.port_cnn, self.port_head):
            for param in module.parameters():
                param.requires_grad = trainable

    def set_power_allocator_trainable(self, trainable: bool) -> None:
        for param in self.power_cnn.parameters():
            param.requires_grad = trainable

    def configure_port_selection_stage(self) -> None:
        self.set_port_selector_trainable(True)
        self.set_power_allocator_trainable(False)

    def configure_power_allocation_stage(self) -> None:
        self.set_port_selector_trainable(False)
        self.set_power_allocator_trainable(True)

    def _cnn_port_selection(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None,
    ) -> tuple[Literal["soft", "hard"], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        active_selection_mode = _resolve_selection_mode(training, selection_mode)
        port_scores = self._port_scores(H)
        selection_soft, selection, ports = _select_ports_from_scores(
            port_scores,
            tau=tau,
            sinkhorn_iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
            selection_mode=active_selection_mode,
        )
        return active_selection_mode, port_scores, selection_soft, selection, ports

    def forward_port_selection_stage(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode, port_scores, selection_soft, selection, ports = self._cnn_port_selection(
            H, tau=tau, training=training, selection_mode=selection_mode
        )
        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        p, q = _uniform_power(H.shape[0], self.K, self.Pmax_W, H.device, H.real.dtype)
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

    def forward_power_allocation_stage(
        self,
        H: torch.Tensor,
        tau: float,
        selection_mode: Literal["soft", "hard"] = "hard",
    ) -> dict[str, torch.Tensor | str]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            active_selection_mode, port_scores, selection_soft, selection, ports = self._cnn_port_selection(
                H, tau=tau, training=False, selection_mode=selection_mode
            )
            H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
            H_eff = H_eff.detach()
            selection_soft = selection_soft.detach()
            selection = selection.detach()
            port_scores = port_scores.detach()
        self.train(was_training)

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

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode, port_scores, selection_soft, selection, ports = self._cnn_port_selection(
            H, tau=tau, training=training, selection_mode=selection_mode
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

    def set_port_selector_trainable(self, trainable: bool) -> None:
        for module in (self.fc1_real, self.fc1_imag, self.real_mha, self.imag_mha, self.fc2, self.port_head):
            for param in module.parameters():
                param.requires_grad = trainable
        for name, param in self.backbone.named_parameters():
            param.requires_grad = trainable and ("lora_" in name or "ln_" in name)
        for param in self.power_head.parameters():
            param.requires_grad = False

    def set_power_allocator_trainable(self, trainable: bool) -> None:
        for param in self.sequential_power_cnn.parameters():
            param.requires_grad = trainable
        for param in self.power_head.parameters():
            param.requires_grad = False

    def configure_port_selection_stage(self) -> None:
        self.set_port_selector_trainable(True)
        self.set_power_allocator_trainable(False)

    def configure_power_allocation_stage(self) -> None:
        self.set_port_selector_trainable(False)
        self.set_power_allocator_trainable(True)

    def _llm_port_selection(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None,
    ) -> tuple[Literal["soft", "hard"], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        active_selection_mode = _resolve_selection_mode(training, selection_mode)
        embeddings = self._preprocess(H)
        backbone_out = self._run_backbone(embeddings)
        port_scores = self.port_head(backbone_out[:, : self.n_active, :])
        selection_soft, selection, ports = _select_ports_from_scores(
            port_scores,
            tau=tau,
            sinkhorn_iters=self.cfg.model.sinkhorn_iters,
            add_noise=training,
            selection_mode=active_selection_mode,
        )
        return active_selection_mode, port_scores, selection_soft, selection, ports

    def forward_port_selection_stage(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode, port_scores, selection_soft, selection, ports = self._llm_port_selection(
            H, tau=tau, training=training, selection_mode=selection_mode
        )
        H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
        p, q = _uniform_power(H.shape[0], self.K, self.Pmax_W, H.device, H.real.dtype)
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

    def forward_power_allocation_stage(
        self,
        H: torch.Tensor,
        tau: float,
        selection_mode: Literal["soft", "hard"] = "hard",
    ) -> dict[str, torch.Tensor | str]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            active_selection_mode, port_scores, selection_soft, selection, ports = self._llm_port_selection(
                H, tau=tau, training=False, selection_mode=selection_mode
            )
            H_eff = H @ selection.transpose(-1, -2).to(H.dtype)
            H_eff = H_eff.detach()
            selection_soft = selection_soft.detach()
            selection = selection.detach()
            port_scores = port_scores.detach()
        self.train(was_training)

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

    def forward(
        self,
        H: torch.Tensor,
        tau: float,
        training: bool,
        selection_mode: Literal["soft", "hard"] | None = None,
    ) -> dict[str, torch.Tensor | str]:
        active_selection_mode, port_scores, selection_soft, selection, ports = self._llm_port_selection(
            H, tau=tau, training=training, selection_mode=selection_mode
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
