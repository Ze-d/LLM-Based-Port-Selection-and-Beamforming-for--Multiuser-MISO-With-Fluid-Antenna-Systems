from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_fas.config import load_config
from llm_fas.experiments import clone_config_for_seed, parse_seed_list


def test_parse_seed_list_accepts_comma_separated_values():
    assert parse_seed_list("20260606, 20260607,20260608") == [20260606, 20260607, 20260608]


def test_parse_seed_list_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one seed"):
        parse_seed_list(" , ")


def test_clone_config_for_seed_changes_seed_and_output_dir(tmp_path):
    cfg = load_config("configs/mvp.yaml")
    cloned = clone_config_for_seed(cfg, seed=123, output_root=tmp_path)

    assert cloned.seed == 123
    assert cloned.train.output_dir == str(tmp_path / "seed_123")
    assert cfg.seed != cloned.seed
    assert cfg.train.output_dir != cloned.train.output_dir
