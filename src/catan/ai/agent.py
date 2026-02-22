"""Base AI agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catan.actions import Action
    from catan.game import Game


class Agent(ABC):
    """Base class for AI agents that play Catan."""

    @abstractmethod
    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        """Choose an action from the list of legal actions."""

    @abstractmethod
    def name(self) -> str:
        """Return the agent's display name."""
