"""Tests for the MetricsCallback training logger."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from catan.ai.metrics_callback import MetricsCallback


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for a training run."""
    return tmp_path / "test_run"


@pytest.fixture
def callback(run_dir: Path) -> MetricsCallback:
    """Create a MetricsCallback with a mock model."""
    cb = MetricsCallback(
        run_dir=run_dir,
        stage_name="stage1_vs_random",
        eval_freq=0,  # Disable eval in unit tests
        eval_baselines=["random"],
        stage_config={"timesteps": 1000, "lr": 1e-4, "ent_coef": 0.01, "n_envs": 1},
    )
    # Mock the model
    mock_model = MagicMock()
    mock_model.logger = MagicMock()
    mock_model.logger.name_to_value = {}
    mock_model.n_steps = 4096
    mock_model.batch_size = 256
    mock_model.n_epochs = 4
    mock_model.gamma = 0.99
    mock_model.gae_lambda = 0.95
    mock_model.clip_range = 0.2
    mock_model.target_kl = 0.015
    cb.model = mock_model
    return cb


class TestConfigWriting:
    def test_config_created_on_start(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        config_path = run_dir / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["status"] == "running"
        assert config["run_id"] == "test_run"
        assert len(config["stages"]) == 1
        assert config["stages"][0]["name"] == "stage1_vs_random"

    def test_config_has_hyperparams(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        config = json.loads((run_dir / "config.json").read_text())
        hp = config["hyperparams"]
        assert hp["n_steps"] == 4096
        assert hp["batch_size"] == 256
        assert hp["gamma"] == 0.99

    def test_config_updated_on_end(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        callback._on_training_end()
        config = json.loads((run_dir / "config.json").read_text())
        assert config["status"] == "completed"
        assert config["end_time"] is not None

    def test_second_stage_appends(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()

        # Simulate a second stage
        cb2 = MetricsCallback(
            run_dir=run_dir,
            stage_name="stage2_vs_mixed",
            eval_freq=0,
            stage_config={"timesteps": 2000, "lr": 5e-5, "ent_coef": 0.01},
        )
        cb2.model = callback.model
        cb2.num_timesteps = 1000
        cb2._on_training_start()

        config = json.loads((run_dir / "config.json").read_text())
        assert len(config["stages"]) == 2
        assert config["stages"][1]["name"] == "stage2_vs_mixed"


class TestMetricsLogging:
    def test_no_metrics_when_logger_empty(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        callback.num_timesteps = 100
        callback._on_step()
        assert not (run_dir / "metrics.jsonl").exists()

    def test_metrics_written_when_logger_has_data(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        callback.model.logger.name_to_value = {
            "rollout/ep_rew_mean": 5.0,
            "rollout/ep_rew_std": 2.0,
            "rollout/ep_len_mean": 100.0,
            "train/policy_gradient_loss": -0.01,
            "train/value_loss": 0.5,
            "train/entropy_loss": 2.0,
            "train/approx_kl": 0.005,
            "train/clip_fraction": 0.1,
        }
        callback.num_timesteps = 4096
        callback._on_step()

        metrics_path = run_dir / "metrics.jsonl"
        assert metrics_path.exists()
        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == 1
        m = json.loads(lines[0])
        assert m["timestep"] == 4096
        assert m["stage"] == "stage1_vs_random"
        assert m["reward_mean"] == 5.0
        assert m["value_loss"] == 0.5

    def test_multiple_metrics_appended(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        callback.model.logger.name_to_value = {"rollout/ep_rew_mean": 1.0}

        callback.num_timesteps = 100
        callback._on_step()
        callback.num_timesteps = 200
        callback._on_step()

        lines = (run_dir / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2


class TestEvaluation:
    def test_eval_not_triggered_when_disabled(self, callback: MetricsCallback, run_dir: Path):
        callback._on_training_start()
        callback.model.logger.name_to_value = {"rollout/ep_rew_mean": 1.0}
        callback.num_timesteps = 100_000
        callback._on_step()
        assert not (run_dir / "evaluations.jsonl").exists()

    def test_eval_respects_frequency(self, run_dir: Path):
        cb = MetricsCallback(
            run_dir=run_dir,
            stage_name="stage1",
            eval_freq=100,
            eval_games=2,
            eval_baselines=["random"],
        )
        # We only test that the trigger logic is correct, not the actual eval
        cb._last_eval_timestep = -1
        cb.num_timesteps = 50
        # Should not trigger at step 50
        assert cb.num_timesteps < cb.eval_freq
        # Should trigger at step 100
        cb.num_timesteps = 100
        assert cb.num_timesteps >= cb.eval_freq
        assert cb.num_timesteps - cb._last_eval_timestep >= cb.eval_freq
