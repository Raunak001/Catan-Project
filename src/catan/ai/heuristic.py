"""Heuristic AI agents for baseline comparison."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from catan.ai.agent import Agent

if TYPE_CHECKING:
    from catan.actions import Action
    from catan.game import Game


class RandomAgent(Agent):
    """Picks uniformly at random from legal actions."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "Random"
