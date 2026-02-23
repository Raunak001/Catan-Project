"""Tests for the training scaffold (train.py)."""

from __future__ import annotations

import numpy as np
import pytest

from catan.ai.gym_env import OBS_SIZE, TOTAL_ACTIONS, CatanEnv
from catan.ai.train import _get_action_mask, make_env

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
