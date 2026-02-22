"""Player state and actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from catan.dev_cards import DevCardType
from catan.resources import Resource


@dataclass
class Player:
    """A player in the game."""

    name: str
    resources: Counter[Resource] = field(default_factory=Counter)
    victory_points: int = 0

    # Building placements (vertex/edge IDs)
    settlements: list[int] = field(default_factory=list)
    cities: list[int] = field(default_factory=list)
    roads: list[int] = field(default_factory=list)

    # Development cards
    dev_cards: list[DevCardType] = field(default_factory=list)
    played_knights: int = 0
    dev_card_played_this_turn: bool = False
    new_dev_cards: list[DevCardType] = field(default_factory=list)  # bought this turn

    def total_resource_count(self) -> int:
        """Total number of resource cards in hand."""
        return sum(self.resources.values())

    def can_afford(self, cost: Counter[Resource]) -> bool:
        """Check if the player can pay a given cost."""
        for resource, amount in cost.items():
            if self.resources[resource] < amount:
                return False
        return True

    def pay(self, cost: Counter[Resource]) -> None:
        """Subtract resources for a purchase."""
        self.resources -= cost
        # Remove zero/negative entries
        self.resources = +self.resources
