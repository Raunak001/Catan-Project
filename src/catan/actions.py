"""Action types for Catan game moves.

Each action is a frozen dataclass. The union type `Action` covers all possible
moves a player can make during any game phase.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from catan.resources import Resource

# -- Building actions --


@dataclass(frozen=True)
class BuildSettlement:
    vertex_id: int


@dataclass(frozen=True)
class BuildRoad:
    edge_id: int


@dataclass(frozen=True)
class BuildCity:
    vertex_id: int


# -- Trading actions --


@dataclass(frozen=True)
class BankTrade:
    """Trade resources with the bank at the player's best available rate."""

    give: Resource
    receive: Resource


@dataclass(frozen=True)
class ProposeTrade:
    """Propose a trade to another player."""

    offering: Counter[Resource]
    requesting: Counter[Resource]
    target_player_idx: int


@dataclass(frozen=True)
class AcceptTrade:
    pass


@dataclass(frozen=True)
class RejectTrade:
    pass


# -- Development card actions --


@dataclass(frozen=True)
class BuyDevCard:
    pass


@dataclass(frozen=True)
class PlayKnight:
    """Play a knight: move robber to a hex and optionally steal from a player."""

    target_hex: int
    steal_from: int | None  # player index, or None if no adjacent opponents


@dataclass(frozen=True)
class PlayRoadBuilding:
    """Play road building: place up to 2 free roads."""

    edge1: int
    edge2: int | None  # None if only 1 legal placement or 1 road left


@dataclass(frozen=True)
class PlayYearOfPlenty:
    """Take any 2 resources from the bank."""

    resource1: Resource
    resource2: Resource


@dataclass(frozen=True)
class PlayMonopoly:
    """Steal all of one resource type from all other players."""

    resource: Resource


# -- Robber actions (on a 7 roll) --


@dataclass(frozen=True)
class MoveRobber:
    """Move the robber to a new hex and optionally steal."""

    target_hex: int
    steal_from: int | None


@dataclass(frozen=True)
class DiscardResources:
    """Discard resources when a 7 is rolled and player has >7 cards."""

    resources: Counter[Resource]


# -- Turn management --


@dataclass(frozen=True)
class EndTurn:
    pass


Action = (
    BuildSettlement
    | BuildRoad
    | BuildCity
    | BankTrade
    | ProposeTrade
    | AcceptTrade
    | RejectTrade
    | BuyDevCard
    | PlayKnight
    | PlayRoadBuilding
    | PlayYearOfPlenty
    | PlayMonopoly
    | MoveRobber
    | DiscardResources
    | EndTurn
)
