"""PPOAgent: wraps a trained MaskablePPO model as a game Agent.

Used for self-play (passing trained checkpoints as opponents) and evaluation
via ``run_game`` / ``run_tournament``.
"""

from __future__ import annotations

import random
from pathlib import Path

from sb3_contrib import MaskablePPO

from catan.actions import Action
from catan.ai.agent import Agent
from catan.ai.gym_env import CatanEnv
from catan.game import Game


class PPOAgent(Agent):
    """Agent backed by a trained MaskablePPO model.

    Internally holds a lightweight ``CatanEnv`` instance for encoding
    observations and action masks, and for decoding action IDs back
    to game actions.  The env is *not* used for stepping — only for
    its encoding/decoding helpers.
    """

    def __init__(self, model_path: str | Path, deterministic: bool = True) -> None:
        self._model = MaskablePPO.load(str(model_path))
        self._deterministic = deterministic
        # Helper env for obs/action encoding — never stepped
        self._helper_env = CatanEnv()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        # Point the helper env at the live game so encoding uses current state
        self._helper_env.game = game

        obs = self._helper_env._encode_obs()
        mask = self._helper_env.action_masks()

        try:
            action_id, _ = self._model.predict(
                obs,
                action_masks=mask,
                deterministic=self._deterministic,
            )
            return self._helper_env._action_id_to_game_action(int(action_id))
        except ValueError:
            # Fallback: extreme logits can cause Simplex constraint violation
            # in MaskableCategorical. Pick a random legal action instead.
            return random.choice(legal_actions)

    def name(self) -> str:
        return "PPO"
