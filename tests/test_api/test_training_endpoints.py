"""Tests for the training dashboard API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from catan.api.server import app

client = TestClient(app)

SAMPLE_CONFIG = {
    "run_id": "20260308_120000_test",
    "run_name": "test",
    "start_time": "2026-03-08T12:00:00+00:00",
    "end_time": "2026-03-08T13:00:00+00:00",
    "status": "completed",
    "stages": [{"name": "stage1_vs_random", "start_timestep": 0, "timesteps": 1000}],
    "hyperparams": {"net_arch": [256, 256], "n_steps": 4096},
    "eval_freq": 500,
    "eval_baselines": ["random"],
}

SAMPLE_METRIC = {
    "timestep": 100,
    "timestamp": "2026-03-08T12:01:00+00:00",
    "stage": "stage1_vs_random",
    "reward_mean": 5.0,
    "reward_std": 2.0,
    "ep_len_mean": 100.0,
    "policy_loss": -0.01,
    "value_loss": 0.5,
    "entropy": 2.0,
    "approx_kl": 0.005,
    "clip_fraction": 0.1,
}

SAMPLE_EVAL = {
    "timestep": 500,
    "timestamp": "2026-03-08T12:05:00+00:00",
    "stage": "stage1_vs_random",
    "baseline": "random",
    "win_rate": 0.75,
    "avg_vp_ppo": 8.0,
    "avg_vp_baseline": 5.0,
    "avg_turns": 150.0,
    "n_games": 20,
}


@pytest.fixture
def training_runs_dir(tmp_path: Path):
    """Create a temporary training_runs directory with fixture data."""
    run_dir = tmp_path / "20260308_120000_test"
    run_dir.mkdir(parents=True)

    (run_dir / "config.json").write_text(json.dumps(SAMPLE_CONFIG))
    (run_dir / "metrics.jsonl").write_text(json.dumps(SAMPLE_METRIC) + "\n")
    (run_dir / "evaluations.jsonl").write_text(json.dumps(SAMPLE_EVAL) + "\n")

    with patch("catan.api.training_router.TRAINING_RUNS_DIR", tmp_path):
        yield tmp_path


@pytest.fixture
def empty_training_dir(tmp_path: Path):
    """Provide an empty training_runs directory."""
    with patch("catan.api.training_router.TRAINING_RUNS_DIR", tmp_path):
        yield tmp_path


class TestListRuns:
    def test_empty_returns_empty_list(self, empty_training_dir):
        resp = client.get("/training/runs")
        assert resp.status_code == 200
        assert resp.json()["runs"] == []

    def test_returns_run_summaries(self, training_runs_dir):
        resp = client.get("/training/runs")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert len(runs) == 1
        assert runs[0]["run_id"] == "20260308_120000_test"
        assert runs[0]["run_name"] == "test"
        assert runs[0]["status"] == "completed"
        assert runs[0]["total_timesteps"] == 100


class TestGetRunMetrics:
    def test_returns_metrics(self, training_runs_dir):
        resp = client.get("/training/runs/20260308_120000_test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "20260308_120000_test"
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["reward_mean"] == 5.0

    def test_not_found(self, empty_training_dir):
        resp = client.get("/training/runs/nonexistent")
        assert resp.status_code == 404

    def test_stage_filter(self, training_runs_dir):
        resp = client.get("/training/runs/20260308_120000_test?stage=nonexistent")
        assert resp.status_code == 200
        assert len(resp.json()["metrics"]) == 0

        resp = client.get("/training/runs/20260308_120000_test?stage=stage1_vs_random")
        assert resp.status_code == 200
        assert len(resp.json()["metrics"]) == 1


class TestGetRunEvaluations:
    def test_returns_evaluations(self, training_runs_dir):
        resp = client.get("/training/runs/20260308_120000_test/evaluations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["evaluations"]) == 1
        assert data["evaluations"][0]["baseline"] == "random"
        assert data["evaluations"][0]["win_rate"] == 0.75

    def test_not_found(self, empty_training_dir):
        resp = client.get("/training/runs/nonexistent/evaluations")
        assert resp.status_code == 404

    def test_baseline_filter(self, training_runs_dir):
        resp = client.get("/training/runs/20260308_120000_test/evaluations?baseline=greedy")
        assert resp.status_code == 200
        assert len(resp.json()["evaluations"]) == 0


class TestCompareRuns:
    def test_compare_single_run(self, training_runs_dir):
        resp = client.get("/training/runs/compare?ids=20260308_120000_test")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert "20260308_120000_test" in runs
        assert len(runs["20260308_120000_test"]["metrics"]) == 1

    def test_compare_no_ids(self, empty_training_dir):
        resp = client.get("/training/runs/compare?ids=")
        assert resp.status_code == 400

    def test_compare_invalid_ids(self, empty_training_dir):
        resp = client.get("/training/runs/compare?ids=fake1,fake2")
        assert resp.status_code == 404
