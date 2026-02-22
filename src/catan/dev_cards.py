"""Development card types and deck construction."""

from __future__ import annotations

import random
from enum import Enum


class DevCardType(Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


# Standard deck composition (25 cards total).
DECK_COMPOSITION: dict[DevCardType, int] = {
    DevCardType.KNIGHT: 14,
    DevCardType.VICTORY_POINT: 5,
    DevCardType.ROAD_BUILDING: 2,
    DevCardType.YEAR_OF_PLENTY: 2,
    DevCardType.MONOPOLY: 2,
}

LARGEST_ARMY_THRESHOLD = 3  # minimum knights to claim largest army


def make_deck(rng: random.Random | None = None) -> list[DevCardType]:
    """Create and shuffle a standard 25-card development deck."""
    deck: list[DevCardType] = []
    for card_type, count in DECK_COMPOSITION.items():
        deck.extend([card_type] * count)
    if rng is not None:
        rng.shuffle(deck)
    else:
        random.shuffle(deck)
    return deck
