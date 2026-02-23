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


# ------------------------------------------------------------------ #
#  Helper: filter actions by type                                      #
# ------------------------------------------------------------------ #


def _actions_of_type(legal_actions: list[Action], *types: type) -> list[Action]:
    """Return legal actions matching any of the given types."""
    return [a for a in legal_actions if isinstance(a, tuple(types))]


# 2d6 probability table
_TOKEN_PROB: dict[int, float] = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
    7: 6 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}


def _vertex_production_value(game: Game, vertex_id: int) -> float:
    """Sum of 2d6 probabilities for all non-robber hexes adjacent to a vertex."""
    total = 0.0
    for hex_idx in game.topology.vertex_to_hexes[vertex_id]:
        if hex_idx == game.robber_hex:
            continue
        token = game.board.hexes[hex_idx].token
        if token is not None:
            total += _TOKEN_PROB.get(token, 0.0)
    return total


# ------------------------------------------------------------------ #
#  GreedyAgent                                                         #
# ------------------------------------------------------------------ #


class GreedyAgent(Agent):
    """Always builds if possible. Priority: cities > settlements > roads > dev cards.

    For robber/discard/knight actions, picks randomly.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        from catan.actions import (
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        # Priority order: city > settlement > road > dev card
        def _prod(a: Action) -> float:
            return _vertex_production_value(game, a.vertex_id)

        for action_type in (BuildCity, BuildSettlement, BuildRoad, BuyDevCard):
            candidates = _actions_of_type(legal_actions, action_type)
            if candidates:
                if action_type in (BuildSettlement, BuildCity):
                    return max(candidates, key=_prod)
                return self._rng.choice(candidates)

        # End turn if available, otherwise pick randomly
        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "Greedy"


# ------------------------------------------------------------------ #
#  LongestRoadBot                                                      #
# ------------------------------------------------------------------ #


class LongestRoadBot(Agent):
    """Prioritizes road building and settlements to pursue longest road bonus.

    Priority: roads > settlements (extending network) > cities > dev cards.
    Plays road-building dev cards when available.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        from catan.actions import (
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
            PlayRoadBuilding,
        )

        # Play road building dev card first if available
        road_building = _actions_of_type(legal_actions, PlayRoadBuilding)
        if road_building:
            return self._rng.choice(road_building)

        # Build roads aggressively
        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return self._rng.choice(roads)

        # Settlements extend the network
        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(
                settlements, key=lambda a: _vertex_production_value(game, a.vertex_id)
            )

        # Cities for VP
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Buy dev cards (might get road building)
        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "LongestRoad"


# ------------------------------------------------------------------ #
#  DevCardBot                                                          #
# ------------------------------------------------------------------ #


class DevCardBot(Agent):
    """Prioritises buying and playing dev cards. Aims for largest army via knights.

    Priority: play knight > buy dev card > cities > settlements > roads.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        from catan.actions import (
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
            PlayKnight,
            PlayMonopoly,
            PlayYearOfPlenty,
        )

        # Play dev cards first (knights are best for largest army)
        knights = _actions_of_type(legal_actions, PlayKnight)
        if knights:
            return self._rng.choice(knights)

        # Play other dev cards
        dev_plays = _actions_of_type(legal_actions, PlayMonopoly, PlayYearOfPlenty)
        if dev_plays:
            return self._rng.choice(dev_plays)

        # Buy dev cards
        buy = _actions_of_type(legal_actions, BuyDevCard)
        if buy:
            return buy[0]

        # Build for VP: cities > settlements > roads
        for action_type in (BuildCity, BuildSettlement, BuildRoad):
            candidates = _actions_of_type(legal_actions, action_type)
            if candidates:
                if action_type in (BuildCity, BuildSettlement):
                    return max(
                        candidates,
                        key=lambda a: _vertex_production_value(game, a.vertex_id),
                    )
                return self._rng.choice(candidates)

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "DevCard"


# ------------------------------------------------------------------ #
#  ResourceHoarder                                                     #
# ------------------------------------------------------------------ #


class ResourceHoarder(Agent):
    """Targets high-probability hexes and pursues an ore/wheat city strategy.

    Prefers settlements on high-production vertices, upgrades to cities ASAP,
    and uses bank trades to accumulate ore/wheat.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )
        from catan.resources import Resource

        # Cities first — the core strategy
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Bank trade toward ore/wheat if we can
        trades = _actions_of_type(legal_actions, BankTrade)
        ore_wheat_trades = [
            t for t in trades if t.receive in (Resource.ORE, Resource.WHEAT)
        ]
        if ore_wheat_trades:
            return self._rng.choice(ore_wheat_trades)

        # Settlements on high-production vertices
        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(
                settlements, key=lambda a: _vertex_production_value(game, a.vertex_id)
            )

        # Roads to reach better spots
        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return self._rng.choice(roads)

        # Dev cards as fallback
        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "ResourceHoarder"
