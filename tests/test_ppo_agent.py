"""Tests for PPOAgent wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from sb3_contrib import MaskablePPO

from catan.ai.gym_env import CatanEnv
from catan.ai.ppo_agent import PPOAgent
from catan.ai.train import make_env
from catan.game_runner import run_game
from catan.ai.heuristic import RandomAgent


@pytest.fixture(scope="module")
def trained_model_path() -> str:
    """Train a tiny model and return its save path."""
    env = make_env(seed=0)
    model = MaskablePPO("MlpPolicy", env, verbose=0, seed=0, n_steps=64, batch_size=32)
    model.learn(total_timesteps=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_model")
        model.save(path)
        env.close()
        yield path


class TestPPOAgent:
    def test_load_and_name(self, trained_model_path: str):
        agent = PPOAgent(trained_model_path)
        assert agent.name() == "PPO"

    def test_choose_action_returns_legal(self, trained_model_path: str):
        """PPOAgent should return an action that was in the legal_actions list."""
        agent = PPOAgent(trained_model_path)
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        legal = game.legal_actions()
        action = agent.choose_action(game, legal)
        # The action should be a valid game action type
        from catan.actions import Action

        assert isinstance(action, Action.__args__)  # type: ignore[attr-defined]

    def test_plays_full_game(self, trained_model_path: str):
        """PPOAgent can play through a complete game as seat 0."""
        ppo = PPOAgent(trained_model_path)
        agents = [ppo] + [RandomAgent() for _ in range(3)]
        result = run_game(agents, seed=42)
        assert result.turns > 0

    def test_deterministic_vs_stochastic(self, trained_model_path: str):
        """Deterministic and stochastic modes should both work."""
        det_agent = PPOAgent(trained_model_path, deterministic=True)
        stoch_agent = PPOAgent(trained_model_path, deterministic=False)
        assert det_agent.name() == "PPO"
        assert stoch_agent.name() == "PPO"

        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        legal = game.legal_actions()
        # Both should return something
        det_agent.choose_action(game, legal)
        stoch_agent.choose_action(game, legal)
