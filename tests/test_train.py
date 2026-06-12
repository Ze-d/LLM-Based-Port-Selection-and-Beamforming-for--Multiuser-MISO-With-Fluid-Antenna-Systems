from __future__ import annotations

import csv
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from transformers import GPT2Config, GPT2Model

from llm_fas.config import load_config
from llm_fas.train import evaluate_checkpoint, evaluate_checkpoints, train_method, train_proposed


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


def _tiny_cfg(tmp_path: Path):
    cfg = load_config("configs/mvp.yaml")
    return replace(
        cfg,
        data=replace(cfg.data, train_samples=4, val_samples=2, test_samples=2),
        model=replace(cfg.model, d_mha=32, mha_heads=4, gpt2_layers=2, lora_rank=2, sinkhorn_iters=2),
        train=replace(cfg.train, epochs=1, batch_size=2, output_dir=str(tmp_path / "mvp")),
    )


def test_train_proposed_writes_checkpoint_and_history(monkeypatch, tmp_path):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg(tmp_path)

    checkpoint_path = train_proposed(cfg)
    history_path = Path(cfg.train.output_dir) / "train_history.csv"

    assert checkpoint_path == Path(cfg.train.output_dir) / "proposed.pt"
    assert checkpoint_path.exists()
    assert history_path.exists()

    rows = list(csv.DictReader(history_path.open(newline="")))
    assert len(rows) == 1
    assert rows[0].keys() == {"epoch", "train_loss", "val_loss"}
    assert math.isfinite(float(rows[0]["train_loss"]))
    assert math.isfinite(float(rows[0]["val_loss"]))


def test_evaluate_checkpoint_writes_results(monkeypatch, tmp_path):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg(tmp_path)
    checkpoint_path = train_proposed(cfg)

    results_path = evaluate_checkpoint(cfg, checkpoint_path)

    assert results_path == Path(cfg.train.output_dir) / "results.csv"
    rows = list(csv.DictReader(results_path.open(newline="")))
    expected_columns = {
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
    }
    assert set(rows[0].keys()) == expected_columns
    assert [row["method"] for row in rows] == ["random", "proposed"]
    assert [row["selection_mode"] for row in rows] == ["hard", "hard"]
    for row in rows:
        assert row["K"] == str(cfg.system.K)
        assert row["N"] == str(cfg.system.Nx * cfg.system.Ny)
        assert math.isfinite(float(row["test_sum_rate"]))
        assert float(row["test_sum_rate"]) > 0.0


def test_train_method_writes_transformer_checkpoint_and_history(tmp_path):
    cfg = _tiny_cfg(tmp_path)

    checkpoint_path = train_method(cfg, method="transformer")

    assert checkpoint_path == Path(cfg.train.output_dir) / "transformer.pt"
    assert checkpoint_path.exists()
    history_path = Path(cfg.train.output_dir) / "transformer_train_history.csv"
    assert history_path.exists()
    rows = list(csv.DictReader(history_path.open(newline="")))
    assert len(rows) == 1
    assert rows[0].keys() == {"epoch", "train_loss", "val_loss"}
    assert math.isfinite(float(rows[0]["train_loss"]))
    assert math.isfinite(float(rows[0]["val_loss"]))


def test_train_method_saves_checkpoint_each_time_validation_improves(monkeypatch, tmp_path):
    base_cfg = _tiny_cfg(tmp_path)
    cfg = replace(base_cfg, train=replace(base_cfg.train, epochs=2))
    val_rates = iter([10.0, 11.0])
    saved_paths: list[Path] = []

    def fake_evaluate_model_rate(*args, **kwargs):
        return next(val_rates)

    def fake_save(state_dict, path):
        saved_paths.append(Path(path))
        Path(path).write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr("llm_fas.train._evaluate_model_rate", fake_evaluate_model_rate)
    monkeypatch.setattr("llm_fas.train.torch.save", fake_save)

    checkpoint_path = train_method(cfg, method="transformer")

    assert checkpoint_path == Path(cfg.train.output_dir) / "transformer.pt"
    assert saved_paths == [checkpoint_path, checkpoint_path]
    rows = list(csv.DictReader((Path(cfg.train.output_dir) / "transformer_train_history.csv").open(newline="")))
    assert [float(row["val_loss"]) for row in rows] == [-10.0, -11.0]


def test_evaluate_checkpoints_writes_random_transformer_proposed(monkeypatch, tmp_path):
    monkeypatch.setattr("llm_fas.models.GPT2Model.from_pretrained", _tiny_gpt2)
    cfg = _tiny_cfg(tmp_path)
    transformer_checkpoint = train_method(cfg, method="transformer")
    proposed_checkpoint = train_proposed(cfg)

    results_path = evaluate_checkpoints(
        cfg,
        {
            "transformer": transformer_checkpoint,
            "proposed": proposed_checkpoint,
        },
    )

    rows = list(csv.DictReader(results_path.open(newline="")))
    assert [row["method"] for row in rows] == ["random", "transformer", "proposed"]
    assert [row["selection_mode"] for row in rows] == ["hard", "hard", "hard"]
    assert {tuple(row.keys()) for row in rows} == {tuple(rows[0].keys())}
    for row in rows:
        assert row["K"] == str(cfg.system.K)
        assert row["N"] == str(cfg.system.Nx * cfg.system.Ny)
        assert math.isfinite(float(row["test_sum_rate"]))
        assert float(row["test_sum_rate"]) > 0.0


def test_cli_scripts_expose_expected_arguments():
    train_help = subprocess.run(
        [sys.executable, "scripts/train_mvp.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    eval_help = subprocess.run(
        [sys.executable, "scripts/evaluate_mvp.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert train_help.returncode == 0
    assert "--config" in train_help.stdout
    assert "--device" in train_help.stdout
    assert eval_help.returncode == 0
    assert "--config" in eval_help.stdout
    assert "--checkpoint" in eval_help.stdout
    assert "--device" in eval_help.stdout
