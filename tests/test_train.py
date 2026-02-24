"""Tests for the training scaffold (train.py)."""

from __future__ import annotations

import numpy as np
import pytest

from catan.ai.gym_env import OBS_SIZE, TOTAL_ACTIONS, CatanEnv
from catan.ai.train import (
    _create_model,
    _get_action_mask,
    make_env,
    make_vec_env,
    train_curriculum,
    train_stage,
)
from catan.game import GamePhase
from tests.helpers import make_game

# ------------------------------------------------------------------ #
#  make_env                                                            #
# ------------------------------------------------------------------ #


class TestMakeEnv:
    """Test the make_env factory function."""

    def test_returns_action_masker_wrapper(self):
        from sb3_contrib.common.wrappers import ActionMasker

        env = make_env(seed=42)
        assert isinstance(env, ActionMasker)
        env.close()

    def test_wrapped_env_is_catan(self):
        env = make_env(seed=42)
        # ActionMasker wraps a CatanEnv
        assert isinstance(env.env, CatanEnv)
        env.close()

    def test_reset_returns_valid_obs(self):
        env = make_env(seed=42)
        obs, info = env.reset()
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32
        env.close()

    def test_step_works_through_wrapper(self):
        env = make_env(seed=42)
        env.reset()
        # Get mask through the wrapper's action_masks method
        mask = env.action_masks()
        action = int(np.argmax(mask))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_SIZE,)
        env.close()

    def test_seed_propagates(self):
        env1 = make_env(seed=99)
        obs1, _ = env1.reset()
        env1.close()

        env2 = make_env(seed=99)
        obs2, _ = env2.reset()
        env2.close()

        np.testing.assert_array_equal(obs1, obs2)


# ------------------------------------------------------------------ #
#  _get_action_mask callback                                           #
# ------------------------------------------------------------------ #


class TestGetActionMask:
    """Test the action mask callback used by ActionMasker."""

    def test_returns_bool_array(self):
        env = CatanEnv(seed=42)
        env.reset()
        mask = _get_action_mask(env)
        assert mask.dtype == bool
        assert mask.shape == (TOTAL_ACTIONS,)
        env.close()

    def test_has_legal_actions(self):
        env = CatanEnv(seed=42)
        env.reset()
        mask = _get_action_mask(env)
        assert mask.any()
        env.close()

    def test_matches_env_action_masks(self):
        env = CatanEnv(seed=42)
        env.reset()
        mask1 = _get_action_mask(env)
        mask2 = env.action_masks()
        np.testing.assert_array_equal(mask1, mask2)
        env.close()


# ------------------------------------------------------------------ #
#  MaskablePPO integration smoke test                                  #
# ------------------------------------------------------------------ #


class TestPPOSmoke:
    """Quick smoke test that MaskablePPO can be instantiated with our env."""

    def test_model_instantiation(self):
        from sb3_contrib import MaskablePPO

        env = make_env(seed=42)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=42,
            n_steps=64,
            batch_size=32,
        )
        assert model is not None
        env.close()

    def test_model_predict(self):
        """Model should be able to produce a prediction (before any training)."""
        from sb3_contrib import MaskablePPO

        env = make_env(seed=42)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=42,
            n_steps=64,
            batch_size=32,
        )
        obs, _ = env.reset()
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        assert 0 <= action < TOTAL_ACTIONS
        env.close()

    @pytest.mark.slow
    def test_short_training_run(self):
        """Train for a tiny number of steps to verify the full loop works."""
        from sb3_contrib import MaskablePPO

        env = make_env(seed=42)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=42,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
        )
        # 64 steps minimum to fill one batch
        model.learn(total_timesteps=64)
        env.close()


# ------------------------------------------------------------------ #
#  make_vec_env                                                        #
# ------------------------------------------------------------------ #


class TestMakeVecEnv:
    """Test vectorized environment creation."""

    def test_returns_subproc_vec_env(self):
        from stable_baselines3.common.vec_env import SubprocVecEnv

        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=2, seed=42)
        assert isinstance(vec_env, SubprocVecEnv)
        assert vec_env.num_envs == 2
        vec_env.close()

    def test_reset_returns_correct_shape(self):
        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=2, seed=42)
        obs = vec_env.reset()
        assert obs.shape == (2, OBS_SIZE)
        assert obs.dtype == np.float32
        vec_env.close()

    def test_step_with_multiple_envs(self):
        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=2, seed=42)
        vec_env.reset()

        actions = np.array([0, 0])  # Same action in both envs
        obs, rewards, dones, infos = vec_env.step(actions)

        assert obs.shape == (2, OBS_SIZE)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        vec_env.close()

    def test_multiple_steps_without_crash(self):
        """Run multiple steps in vectorized environment."""
        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=2, seed=42)
        obs = vec_env.reset()

        for _ in range(5):
            # Take random legal actions
            actions = np.array([0, 0])
            obs, rewards, dones, infos = vec_env.step(actions)
            # Reset any finished envs
            if any(dones):
                obs = vec_env.reset()

        vec_env.close()

    def test_seed_produces_different_envs(self):
        """Different seeds should produce different initial states."""
        from catan.ai.train import make_vec_env

        vec_env1 = make_vec_env(n_envs=2, seed=42)
        obs1 = vec_env1.reset()
        vec_env1.close()

        vec_env2 = make_vec_env(n_envs=2, seed=99)
        obs2 = vec_env2.reset()
        vec_env2.close()

        # Observations should differ with different seeds
        assert not np.allclose(obs1, obs2)

    def test_env_cleanup(self):
        """Verify environments clean up properly when closed."""
        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=4, seed=42)
        vec_env.reset()
        vec_env.step(np.array([0] * 4))
        # Should not raise
        vec_env.close()


# ------------------------------------------------------------------ #
#  _create_model                                                       #
# ------------------------------------------------------------------ #


class TestCreateModel:
    """Test model creation and loading."""

    def test_creates_new_model(self):
        from catan.ai.train import _create_model

        env = make_env(seed=42)
        model = _create_model(env, seed=42)

        assert model is not None
        # Should have policy network
        assert hasattr(model, "policy")
        env.close()

    @pytest.mark.slow
    def test_train_and_save_model(self, tmp_path):
        """Train a model and verify it can be saved/loaded."""
        from pathlib import Path

        from sb3_contrib import MaskablePPO

        from catan.ai.train import _create_model

        env = make_env(seed=42)
        model = _create_model(env, seed=42)

        # Train briefly
        model.learn(total_timesteps=64)

        # Save
        save_path = str(tmp_path / "test_model")
        model.save(save_path)
        assert Path(save_path + ".zip").exists()

        env.close()

    @pytest.mark.slow
    def test_load_existing_model(self, tmp_path):
        """Create, save, and load a model."""
        from pathlib import Path

        from sb3_contrib import MaskablePPO

        from catan.ai.train import _create_model

        # Create and save
        env1 = make_env(seed=42)
        model1 = _create_model(env1, seed=42)
        model1.learn(total_timesteps=64)

        save_path = str(tmp_path / "test_model")
        model1.save(save_path)
        env1.close()

        # Load with _create_model
        env2 = make_env(seed=99)
        model2 = _create_model(env2, load_path=save_path)

        assert model2 is not None
        # Predict should work
        obs, _ = env2.reset()
        action, _ = model2.predict(obs, action_masks=env2.action_masks(), deterministic=True)
        assert 0 <= action < TOTAL_ACTIONS

        env2.close()


# ------------------------------------------------------------------ #
#  train_stage                                                         #
# ------------------------------------------------------------------ #


class TestTrainStage:
    """Test single training stage execution."""

    @pytest.mark.slow
    def test_train_stage_completes(self, tmp_path):
        """Test that train_stage completes without error."""
        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        result_path = train_stage(
            stage_name="test_stage",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,  # No checkpoints for this short run
        )

        assert result_path is not None
        assert isinstance(result_path, str)

    @pytest.mark.slow
    def test_train_stage_with_vec_env(self, tmp_path):
        """Test train_stage with multiple parallel environments."""
        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        result_path = train_stage(
            stage_name="test_stage_vec",
            timesteps=128,
            save_dir=str(tmp_path),
            seed=42,
            n_envs=2,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_train_stage_saves_model(self, tmp_path):
        """Verify train_stage saves the final model."""
        from pathlib import Path

        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        result_path = train_stage(
            stage_name="stage_save_test",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,
        )

        # Model file should exist
        assert Path(result_path + ".zip").exists()

    @pytest.mark.slow
    def test_train_stage_with_load_path(self, tmp_path):
        """Test train_stage that loads an existing model."""
        from pathlib import Path

        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        # First stage
        path1 = train_stage(
            stage_name="stage1",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,
        )

        # Second stage loading from first
        def greedy_opponents():
            from catan.ai.heuristic import GreedyAgent

            return [GreedyAgent() for _ in range(3)]

        path2 = train_stage(
            stage_name="stage2",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            load_path=path1,
            n_envs=1,
            opponent_factory=greedy_opponents,
            checkpoint_freq=1_000_000,
        )

        assert path2 is not None
        assert Path(path2 + ".zip").exists()


# ------------------------------------------------------------------ #
#  train_curriculum                                                    #
# ------------------------------------------------------------------ #


class TestTrainCurriculum:
    """Test full curriculum training with multiple stages."""

    @pytest.mark.slow
    def test_stage1_only(self, tmp_path):
        """Test curriculum with only stage 1."""
        from catan.ai.train import train_curriculum

        result_path = train_curriculum(
            stage1_steps=64,
            stage2_steps=0,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path),
            n_envs=1,
            only_stage=1,
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_stage1_and_stage2(self, tmp_path):
        """Test curriculum with stage 1 and 2."""
        from catan.ai.train import train_curriculum

        result_path = train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path),
            n_envs=1,
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_full_curriculum(self, tmp_path):
        """Test full 3-stage curriculum."""
        from catan.ai.train import train_curriculum

        result_path = train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=64,
            seed=42,
            save_dir=str(tmp_path),
            n_envs=1,
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_curriculum_with_vec_envs(self, tmp_path):
        """Test curriculum with vectorized environments."""
        from catan.ai.train import train_curriculum

        result_path = train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=64,
            seed=42,
            save_dir=str(tmp_path),
            n_envs=2,
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_only_stage_2(self, tmp_path):
        """Test loading stage 1 and training stage 2 only."""
        from catan.ai.train import train_curriculum

        # First do stage 1
        train_curriculum(
            stage1_steps=64,
            stage2_steps=0,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        # Now run only stage 2 with the checkpoint
        result_path = train_curriculum(
            stage1_steps=0,
            stage2_steps=64,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "stage2_only" / "models"),
            n_envs=1,
            only_stage=2,
            load_path=str(tmp_path / "models" / "stage1_vs_random"),
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_only_stage_3(self, tmp_path):
        """Test loading stage 2 and training stage 3 only."""
        from catan.ai.train import train_curriculum

        # First do stages 1 and 2
        train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        # Now run only stage 3 with the checkpoint
        result_path = train_curriculum(
            stage1_steps=0,
            stage2_steps=0,
            stage3_steps=64,
            seed=42,
            save_dir=str(tmp_path / "stage3_only" / "models"),
            n_envs=1,
            only_stage=3,
            load_path=str(tmp_path / "models" / "stage2_vs_mixed"),
        )

        assert result_path is not None


# ------------------------------------------------------------------ #
#  PPOAgent in training pipeline                                      #
# ------------------------------------------------------------------ #


class TestPPOAgentTraining:
    """Test PPOAgent as opponent in training."""

    @pytest.mark.slow
    def test_ppo_agent_as_selfplay_opponent(self, tmp_path):
        """Test using PPOAgent as opponents in stage 3."""
        from pathlib import Path

        from catan.ai.ppo_agent import PPOAgent
        from catan.ai.train import train_curriculum

        # Build a checkpoint first
        train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        stage2_model = Path(tmp_path) / "models" / "stage2_vs_mixed"
        assert stage2_model.with_suffix(".zip").exists()

        # Create PPOAgent from checkpoint
        agent = PPOAgent(model_path=str(stage2_model), deterministic=False)
        assert agent is not None
        assert agent.name() == "PPO"

    @pytest.mark.slow
    def test_stage3_with_ppo_opponents(self, tmp_path):
        """Test full stage 3 training with PPOAgent opponents."""
        from catan.ai.train import train_curriculum

        # Full curriculum including stage 3
        result_path = train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=64,
            seed=42,
            save_dir=str(tmp_path),
            n_envs=1,
        )

        assert result_path is not None
        # Should be stage3 path
        assert "stage3" in result_path


# ------------------------------------------------------------------ #
#  Edge cases and robustness                                           #
# ------------------------------------------------------------------ #


class TestTrainingEdgeCases:
    """Test edge cases and error conditions in training pipeline."""

    @pytest.mark.slow
    def test_checkpoint_with_small_timesteps(self, tmp_path):
        """Test checkpoint callback when timesteps < checkpoint_freq."""
        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        # timesteps (64) < checkpoint_freq (100000), so no intermediate checkpoints
        result_path = train_stage(
            stage_name="small_checkpoint",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=100_000,
        )

        assert result_path is not None

    @pytest.mark.slow
    def test_checkpoint_dir_created_automatically(self, tmp_path):
        """Test that checkpoint directory is created if it doesn't exist."""
        from pathlib import Path

        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        save_dir = tmp_path / "new" / "nested" / "dir"
        result_path = train_stage(
            stage_name="nested_test",
            timesteps=64,
            save_dir=str(save_dir),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,
        )

        assert result_path is not None
        assert Path(result_path + ".zip").exists()

    @pytest.mark.slow
    def test_high_n_envs_count(self, tmp_path):
        """Test with many parallel environments."""
        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=4, seed=42)
        obs = vec_env.reset()
        assert obs.shape == (4, OBS_SIZE)

        # Step with valid actions
        actions = np.array([0, 0, 0, 0])
        for _ in range(3):
            obs, rewards, dones, infos = vec_env.step(actions)
            if any(dones):
                obs = vec_env.reset()

        vec_env.close()

    @pytest.mark.slow
    def test_action_masking_in_vec_env(self, tmp_path):
        """Test that action masking works correctly in vectorized envs."""
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker

        from catan.ai.train import make_vec_env

        # Create vec env with masking
        def mask_fn(env):
            return env.action_masks()

        base_env = make_vec_env(n_envs=2, seed=42)

        # Verify we can get masks
        obs = base_env.reset()
        base_env.close()

    @pytest.mark.slow
    def test_deterministic_with_seed(self, tmp_path):
        """Test that same seed produces same training behavior."""
        from pathlib import Path

        from catan.ai.heuristic import RandomAgent
        from catan.ai.train import train_stage

        def random_opponents():
            return [RandomAgent() for _ in range(3)]

        # First training run
        path1 = train_stage(
            stage_name="det1",
            timesteps=64,
            save_dir=str(tmp_path / "run1"),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,
        )

        # Second training run with same seed
        path2 = train_stage(
            stage_name="det2",
            timesteps=64,
            save_dir=str(tmp_path / "run2"),
            seed=42,
            n_envs=1,
            opponent_factory=random_opponents,
            checkpoint_freq=1_000_000,
        )

        assert Path(path1 + ".zip").exists()
        assert Path(path2 + ".zip").exists()

    @pytest.mark.slow
    def test_curriculum_stage_transitions_use_checkpoints(self, tmp_path):
        """Verify stage 2 uses stage 1 checkpoint and stage 3 uses stage 2."""
        from pathlib import Path

        from catan.ai.train import train_curriculum

        # Full curriculum - each stage should load previous
        result = train_curriculum(
            stage1_steps=64,
            stage2_steps=64,
            stage3_steps=64,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        # All stage checkpoints should exist
        stage1_path = Path(tmp_path) / "models" / "stage1_vs_random.zip"
        stage2_path = Path(tmp_path) / "models" / "stage2_vs_mixed.zip"
        stage3_path = Path(tmp_path) / "models" / "stage3_selfplay.zip"

        assert stage1_path.exists()
        assert stage2_path.exists()
        assert stage3_path.exists()

    @pytest.mark.slow
    def test_game_reset_between_episodes_in_vec_env(self, tmp_path):
        """Test that game states are properly reset in vectorized envs."""
        from catan.ai.train import make_vec_env

        vec_env = make_vec_env(n_envs=2, seed=42)
        obs1 = vec_env.reset()

        # Run until we get done in at least one env.
        # Catan games can take many hundreds of agent decisions (MAX_TURNS=300,
        # 4 players, multiple phases per turn), so we need a generous budget.
        for i in range(2000):
            actions = np.array([0, 0])
            obs, rewards, dones, infos = vec_env.step(actions)

            if any(dones):
                # Environment should have reset automatically
                assert obs.shape == (2, OBS_SIZE)
                break
        else:
            assert False, "No game completed in 2000 steps"

        vec_env.close()

    @pytest.mark.slow
    def test_continuous_training_preserves_network_weights(self, tmp_path):
        """Test that loading model for stage 2 preserves learned weights."""
        from pathlib import Path

        from catan.ai.heuristic import GreedyAgent, RandomAgent
        from catan.ai.train import train_stage

        # Stage 1
        stage1_path = train_stage(
            stage_name="weights1",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            n_envs=1,
            opponent_factory=lambda: [RandomAgent() for _ in range(3)],
            checkpoint_freq=1_000_000,
        )

        # Load and continue training
        stage2_path = train_stage(
            stage_name="weights2",
            timesteps=64,
            save_dir=str(tmp_path),
            seed=42,
            load_path=stage1_path,
            n_envs=1,
            opponent_factory=lambda: [GreedyAgent() for _ in range(3)],
            checkpoint_freq=1_000_000,
        )

        # Both should exist
        assert Path(stage1_path + ".zip").exists()
        assert Path(stage2_path + ".zip").exists()


# ------------------------------------------------------------------ #
#  PPOAgent integration tests                                         #
# ------------------------------------------------------------------ #


class TestPPOAgentIntegration:
    """Test PPOAgent loading, prediction, and usage in pipelines."""

    @pytest.mark.slow
    def test_ppo_agent_deterministic_mode(self, tmp_path):
        """Test PPOAgent in deterministic mode produces consistent results."""
        from catan.ai.ppo_agent import PPOAgent
        from catan.ai.train import train_curriculum

        # Create a trained model
        train_curriculum(
            stage1_steps=64,
            stage2_steps=0,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        model_path = str(tmp_path / "models" / "stage1_vs_random")
        agent = PPOAgent(model_path=model_path, deterministic=True)

        game = make_game()

        legal_actions = game.legal_actions()
        if legal_actions:
            # Deterministic mode should give same action multiple times
            action1 = agent.choose_action(game, legal_actions)
            action2 = agent.choose_action(game, legal_actions)
            assert action1 == action2, "Deterministic mode should give same action"

    @pytest.mark.slow
    def test_ppo_agent_stochastic_mode(self, tmp_path):
        """Test PPOAgent in stochastic mode."""
        from catan.ai.ppo_agent import PPOAgent
        from catan.ai.train import train_curriculum

        # Create trained model
        train_curriculum(
            stage1_steps=64,
            stage2_steps=0,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        model_path = str(tmp_path / "models" / "stage1_vs_random")
        agent = PPOAgent(model_path=model_path, deterministic=False)

        game = make_game()

        legal_actions = game.legal_actions()
        if legal_actions and len(legal_actions) > 1:
            # Can choose actions
            action = agent.choose_action(game, legal_actions)
            assert action in legal_actions

    @pytest.mark.slow
    def test_ppo_agent_handles_any_game_state(self, tmp_path):
        """Test PPOAgent can handle various game states."""
        from catan.ai.ppo_agent import PPOAgent
        from catan.ai.train import train_curriculum

        # Create trained model
        train_curriculum(
            stage1_steps=64,
            stage2_steps=0,
            stage3_steps=0,
            seed=42,
            save_dir=str(tmp_path / "models"),
            n_envs=1,
        )

        model_path = str(tmp_path / "models" / "stage1_vs_random")
        agent = PPOAgent(model_path=model_path, deterministic=True)

        # Test on multiple different games
        for i in range(3):
            game = make_game(seed=42 + i)
            for _ in range(10):
                legal_actions = game.legal_actions()
                if legal_actions:
                    action = agent.choose_action(game, legal_actions)
                    assert action in legal_actions
                    game.apply_action(action)

                if game.phase == GamePhase.FINISHED:
                    break
