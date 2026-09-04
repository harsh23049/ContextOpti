"""Tests for config loading.

The config is the reproducibility surface: an experiment run is supposed to be
recoverable from one file, so the keys the milestone scripts depend on are asserted
to exist rather than left to fail at runtime three hours into a sweep.
"""

from __future__ import annotations

import pytest

from contextopti.config import Config, load_config


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def test_default_config_loads(config):
    assert config.path is not None
    assert config.path.name == "default.yaml"


@pytest.mark.parametrize(
    "key",
    [
        "seed",
        "paths.fixture_repo",
        "paths.outputs",
        "paths.graph",
        "index.exclude_dirs",
        "budgets.T_max",
        "budgets.L_max",
        "structure.hops",
        "structure.tok",
        "semantic.top_k",
        "policy.kind",
        "policy.lambda_cost",
        "policy.actions.hops",
        "policy.actions.tok",
        "generate.provider",
        "generate.frozen",
        "eval.max_eval_n",
        "eval.sweep_tok",
    ],
)
def test_keys_the_milestone_scripts_depend_on_are_present(config, key):
    assert config.get(key) is not None, key


def test_generator_is_frozen_by_default(config):
    """M4/M5 comparability depends on this; flipping it must be deliberate."""
    assert config.get("generate.frozen") is True
    assert config.get("generate.provider") == "mock"


def test_policy_budgets_stay_within_the_hard_ceiling(config):
    t_max = config.get("budgets.T_max")
    assert max(config.get("policy.actions.tok")) <= t_max
    assert max(config.get("eval.sweep_tok")) <= t_max
    assert config.get("structure.tok") <= t_max
    assert config.get("semantic.tok") <= t_max


def test_dotted_get_returns_default_for_missing_keys(config):
    assert config.get("nope.not.here") is None
    assert config.get("nope.not.here", 7) == 7
    # A missing intermediate level must not raise.
    assert config.get("seed.deeper.still", "fallback") == "fallback"


def test_require_raises_on_a_missing_key(config):
    with pytest.raises(KeyError):
        config.require("definitely.absent")


def test_paths_resolve_against_the_repository_root(config):
    fixture = config.path_for("paths.fixture_repo")
    assert fixture.is_absolute()
    assert fixture.is_dir()
    assert (fixture / "shop" / "models.py").is_file()


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_non_mapping_config_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_env_var_overrides_the_default_path(tmp_path, monkeypatch):
    path = tmp_path / "custom.yaml"
    path.write_text("seed: 99\n", encoding="utf-8")
    monkeypatch.setenv("CONTEXTOPTI_CONFIG", str(path))

    assert load_config().get("seed") == 99
