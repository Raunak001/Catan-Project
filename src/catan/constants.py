"""Game constants: building costs, piece limits, and victory conditions."""

from __future__ import annotations

from collections import Counter

from catan.resources import Resource

# -- Building costs --

SETTLEMENT_COST = Counter(
    {
        Resource.WOOD: 1,
        Resource.BRICK: 1,
        Resource.SHEEP: 1,
        Resource.WHEAT: 1,
    }
)

CITY_COST = Counter(
    {
        Resource.WHEAT: 2,
        Resource.ORE: 3,
    }
)

ROAD_COST = Counter(
    {
        Resource.WOOD: 1,
        Resource.BRICK: 1,
    }
)

DEV_CARD_COST = Counter(
    {
        Resource.SHEEP: 1,
        Resource.WHEAT: 1,
        Resource.ORE: 1,
    }
)

# -- Piece limits per player --

MAX_SETTLEMENTS = 5
MAX_CITIES = 4
MAX_ROADS = 15

# -- Victory --

VICTORY_POINTS_TO_WIN = 10

# -- Trading --

DEFAULT_BANK_TRADE_RATE = 4
GENERIC_PORT_TRADE_RATE = 3
SPECIALIZED_PORT_TRADE_RATE = 2

# -- Robber --

MAX_CARDS_BEFORE_DISCARD = 7  # players with more than this discard on a 7

# -- Game limits (for RL training) --

MAX_TURNS = 300
MAX_TRADES_PER_TURN = 3
