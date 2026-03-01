"""Heuristic AI agents for baseline comparison."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from catan.ai.agent import Agent
from catan.constants import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST
from catan.longest_road import compute_longest_road
from catan.resources import Resource

if TYPE_CHECKING:
    from catan.actions import Action, BankTrade, BuildRoad, DiscardResources
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
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #


def _actions_of_type(legal_actions: list[Action], *types: type) -> list[Action]:
    """Return legal actions matching any of the given types."""
    return [a for a in legal_actions if isinstance(a, tuple(types))]


# 2d6 probability table
_TOKEN_PROB: dict[int, float] = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}

# Resource value weights for discard/trade decisions
_RESOURCE_VALUE: dict[Resource, float] = {
    Resource.ORE: 1.2,
    Resource.WHEAT: 1.1,
    Resource.WOOD: 0.8,
    Resource.BRICK: 0.8,
    Resource.SHEEP: 0.6,
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


def _acting_player_idx(game: Game) -> int:
    """Determine which player is currently acting (handles discard phase)."""
    from catan.game import GamePhase

    if game.phase == GamePhase.ROBBER_DISCARD:
        return game.players_to_discard[game._discard_idx]
    return game.current_player_idx


def _hex_production_value(game: Game, hex_idx: int) -> float:
    """Total production blocked by placing robber on this hex.

    Sums token probability × building multiplier for all opponent buildings.
    """
    token = game.board.hexes[hex_idx].token
    if token is None:
        return 0.0
    base_prob = _TOKEN_PROB.get(token, 0.0)
    total = 0.0
    for vid in game.topology.hex_to_vertices[hex_idx]:
        owner = game.vertex_owner.get(vid)
        if owner is not None:
            building = game.vertex_building.get(vid, "settlement")
            multiplier = 2.0 if building == "city" else 1.0
            total += base_prob * multiplier
    return total


def _smart_robber_action(
    game: Game,
    player_idx: int,
    robber_actions: list[Action],
    rng: random.Random,
) -> Action:
    """Pick the best robber placement: target the VP leader's high-production hex."""
    from catan.actions import MoveRobber, PlayKnight

    # Find the VP leader among opponents
    leader_idx = -1
    leader_vp = -1
    for i in range(len(game.players)):
        if i != player_idx and game.players[i].victory_points > leader_vp:
            leader_vp = game.players[i].victory_points
            leader_idx = i

    def _score(action: MoveRobber | PlayKnight) -> float:
        hex_val = _hex_production_value(game, action.target_hex)
        # Bonus for stealing from the VP leader
        steal_bonus = 2.0 if action.steal_from == leader_idx else 1.0
        # Bonus for stealing from someone with resources
        resource_bonus = 0.0
        if action.steal_from is not None:
            resource_bonus = min(game.players[action.steal_from].total_resource_count(), 5) * 0.1
        # Penalty if this hex has our own buildings
        own_penalty = 0.0
        for vid in game.topology.hex_to_vertices[action.target_hex]:
            if game.vertex_owner.get(vid) == player_idx:
                building = game.vertex_building.get(vid, "settlement")
                own_penalty += 0.5 if building == "city" else 0.3
        return hex_val * steal_bonus + resource_bonus - own_penalty

    best_score = -999.0
    best_actions: list[Action] = []
    for action in robber_actions:
        s = _score(action)
        if s > best_score:
            best_score = s
            best_actions = [action]
        elif s == best_score:
            best_actions.append(action)
    return rng.choice(best_actions)


def _smart_discard(
    game: Game,
    player_idx: int,
    discard_actions: list[DiscardResources],
    rng: random.Random,
) -> DiscardResources:
    """Choose a discard that keeps the most valuable resources."""
    player = game.players[player_idx]
    hand = player.resources

    def _kept_value(action: DiscardResources) -> float:
        kept = hand - action.resources
        return sum(_RESOURCE_VALUE[r] * count for r, count in kept.items() if count > 0)

    best_val = -999.0
    best_actions: list[DiscardResources] = []
    for action in discard_actions:
        v = _kept_value(action)
        if v > best_val:
            best_val = v
            best_actions = [action]
        elif v == best_val:
            best_actions.append(action)
    return rng.choice(best_actions)


def _smart_road(
    game: Game,
    player_idx: int,
    road_actions: list[BuildRoad],
    rng: random.Random,
) -> BuildRoad:
    """Pick the road that leads toward the best open settlement vertex."""

    def _can_place_settlement(vid: int) -> bool:
        """Check if a vertex is open for settlement (unowned + distance rule)."""
        if vid in game.vertex_owner:
            return False
        for neighbor in game.topology.vertex_neighbors[vid]:
            if neighbor in game.vertex_owner:
                return False
        return True

    def _road_score(action: BuildRoad) -> float:
        va, vb = game.topology.edge_vertices[action.edge_id]
        score = 0.0
        # Direct: can we place a settlement at either endpoint?
        for v in (va, vb):
            if _can_place_settlement(v):
                score = max(score, _vertex_production_value(game, v))
        # One step further: check neighbors of endpoints
        for v in (va, vb):
            for neighbor_v in game.topology.vertex_neighbors[v]:
                if _can_place_settlement(neighbor_v):
                    # Reduced score for 1-hop away
                    score = max(score, _vertex_production_value(game, neighbor_v) * 0.5)
        return score

    best_score = -1.0
    best_roads: list[BuildRoad] = []
    for action in road_actions:
        s = _road_score(action)
        if s > best_score:
            best_score = s
            best_roads = [action]
        elif s == best_score:
            best_roads.append(action)
    return rng.choice(best_roads)


def _road_extends_longest(
    game: Game,
    player_idx: int,
    road_actions: list[BuildRoad],
) -> list[BuildRoad]:
    """Return road actions that increase the player's longest road length."""
    player = game.players[player_idx]
    current_length = compute_longest_road(
        player.roads, player_idx, game.topology, game.vertex_owner
    )

    extending: list[BuildRoad] = []
    for action in road_actions:
        # Temporarily add road
        test_roads = player.roads + [action.edge_id]
        new_length = compute_longest_road(test_roads, player_idx, game.topology, game.vertex_owner)
        if new_length > current_length:
            extending.append(action)
    return extending


def _need_based_trades(
    game: Game,
    player_idx: int,
    trade_actions: list[BankTrade],
    target_cost: Counter[Resource],
) -> list[BankTrade]:
    """Return bank trades that move toward affording a target, sorted by efficiency."""
    player = game.players[player_idx]

    # What resources do we still need?
    needed: set[Resource] = set()
    for res, cost in target_cost.items():
        if player.resources[res] < cost:
            needed.add(res)

    if not needed:
        return []

    # Filter to trades that give us something we need
    useful = [t for t in trade_actions if t.receive in needed]
    if not useful:
        return []

    # Sort by trade rate (best rate first = fewest cards given away)
    def _efficiency(trade: BankTrade) -> float:
        rate = game._trade_rate(player_idx, trade.give)
        return -rate  # lower rate = better (negated for sort)

    useful.sort(key=_efficiency, reverse=True)
    return useful


def _best_dev_card_play(
    game: Game,
    player_idx: int,
    legal_actions: list[Action],
    rng: random.Random,
) -> Action | None:
    """Pick the best dev card to play, or None."""
    from catan.actions import PlayKnight, PlayMonopoly, PlayRoadBuilding, PlayYearOfPlenty

    # Knight — always good (moves robber + counts toward largest army)
    knights = _actions_of_type(legal_actions, PlayKnight)
    if knights:
        return _smart_robber_action(game, player_idx, knights, rng)

    # Monopoly — play if opponents collectively have >= 3 of one resource
    monopolies = _actions_of_type(legal_actions, PlayMonopoly)
    if monopolies:
        best_monopoly = None
        best_total = 2  # threshold: only play if we'd get 3+
        for action in monopolies:
            total = sum(
                game.players[i].resources[action.resource]
                for i in range(len(game.players))
                if i != player_idx
            )
            if total > best_total:
                best_total = total
                best_monopoly = action
        if best_monopoly is not None:
            return best_monopoly

    # Year of Plenty — pick resources closest to completing a purchase
    yop_actions = _actions_of_type(legal_actions, PlayYearOfPlenty)
    if yop_actions:
        player = game.players[player_idx]
        # Score each YoP by how close it gets us to a purchase
        best_yop = None
        best_yop_score = -1.0
        for action in yop_actions:
            gained = Counter({action.resource1: 1, action.resource2: 1})
            test_hand = player.resources + gained
            score = 0.0
            # Check if we can now afford something we couldn't before
            for cost, value in [(CITY_COST, 3.0), (SETTLEMENT_COST, 2.0), (DEV_CARD_COST, 1.5)]:
                if all(test_hand[r] >= c for r, c in cost.items()):
                    score = max(score, value)
            # Fallback: sum of resource values gained
            if score == 0.0:
                res_pair = [action.resource1, action.resource2]
                score = sum(_RESOURCE_VALUE.get(r, 0.5) for r in res_pair)
                score *= 0.1  # scale down vs purchase-completing score
            if score > best_yop_score:
                best_yop_score = score
                best_yop = action
        if best_yop is not None:
            return best_yop

    # Road Building — play if we have few settlements (need to expand network)
    road_building = _actions_of_type(legal_actions, PlayRoadBuilding)
    if road_building:
        player = game.players[player_idx]
        if len(player.settlements) + len(player.cities) < 4 or len(player.roads) <= 6:
            return road_building[0]

    return None


# ------------------------------------------------------------------ #
#  Shared phase handlers                                               #
# ------------------------------------------------------------------ #


def _handle_special_phases(
    game: Game,
    legal_actions: list[Action],
    rng: random.Random,
) -> Action | None:
    """Handle discard and robber phases smartly. Returns action or None."""
    from catan.actions import DiscardResources, MoveRobber

    pidx = _acting_player_idx(game)

    discards = _actions_of_type(legal_actions, DiscardResources)
    if discards:
        return _smart_discard(game, pidx, discards, rng)

    robber_moves = _actions_of_type(legal_actions, MoveRobber)
    if robber_moves:
        return _smart_robber_action(game, pidx, robber_moves, rng)

    return None


# ------------------------------------------------------------------ #
#  GreedyAgent                                                         #
# ------------------------------------------------------------------ #


class GreedyAgent(Agent):
    """Always builds if possible. Priority: cities > settlements > roads > dev cards.

    Uses smart robber/discard, plays dev cards, and trades toward purchases.
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

        # Handle discard/robber phases
        special = _handle_special_phases(game, legal_actions, self._rng)
        if special is not None:
            return special

        pidx = _acting_player_idx(game)

        # Play dev cards before building
        dev_play = _best_dev_card_play(game, pidx, legal_actions, self._rng)
        if dev_play is not None:
            return dev_play

        # Priority order: city > settlement > road > dev card
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

        dev_buy = _actions_of_type(legal_actions, BuyDevCard)
        if dev_buy:
            return dev_buy[0]

        # Trade toward city cost before ending turn
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, CITY_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, SETTLEMENT_COST)
            if useful:
                return useful[0]

        # End turn
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
    """Prioritizes road building to pursue longest road bonus.

    Uses chain-extending road logic, plays dev cards, and trades toward roads.
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
            PlayRoadBuilding,
        )

        # Handle discard/robber phases
        special = _handle_special_phases(game, legal_actions, self._rng)
        if special is not None:
            return special

        pidx = _acting_player_idx(game)

        # Play road building dev card first
        road_building = _actions_of_type(legal_actions, PlayRoadBuilding)
        if road_building:
            return road_building[0]

        # Play other dev cards (knights, etc.)
        dev_play = _best_dev_card_play(game, pidx, legal_actions, self._rng)
        if dev_play is not None:
            return dev_play

        # Build roads — prefer chain-extending ones
        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            extending = _road_extends_longest(game, pidx, roads)
            if extending:
                return self._rng.choice(extending)
            return _smart_road(game, pidx, roads, self._rng)

        # Settlements extend the network
        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Cities for VP
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Buy dev cards (might get road building)
        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        # Trade toward road materials
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, ROAD_COST)
            if useful:
                return useful[0]

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

    Uses smart targeting for knights/monopoly/YoP, smart robber/discard.
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

        # Handle discard/robber phases
        special = _handle_special_phases(game, legal_actions, self._rng)
        if special is not None:
            return special

        pidx = _acting_player_idx(game)

        # Play dev cards with smart targeting
        dev_play = _best_dev_card_play(game, pidx, legal_actions, self._rng)
        if dev_play is not None:
            return dev_play

        # Buy dev cards
        buy = _actions_of_type(legal_actions, BuyDevCard)
        if buy:
            return buy[0]

        # Build for VP: cities > settlements > roads
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

        # Trade toward dev card cost
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, DEV_CARD_COST)
            if useful:
                return useful[0]

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

    Uses need-based trading, smart robber/discard, and dev card play.
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

        # Handle discard/robber phases
        special = _handle_special_phases(game, legal_actions, self._rng)
        if special is not None:
            return special

        pidx = _acting_player_idx(game)

        # Play dev cards
        dev_play = _best_dev_card_play(game, pidx, legal_actions, self._rng)
        if dev_play is not None:
            return dev_play

        # Cities first — the core strategy
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Need-based trading toward city cost
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, CITY_COST)
            if useful:
                return useful[0]

        # Settlements on high-production vertices
        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Smart roads toward good settlement spots
        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

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


# ------------------------------------------------------------------ #
#  SmartBot                                                            #
# ------------------------------------------------------------------ #


class SmartBot(Agent):
    """Strongest baseline: adaptive strategy based on game state.

    Early game (VP < 5): expand with settlements and roads.
    Mid game (VP 5-7): upgrade to cities, buy dev cards.
    Late game (VP >= 8): sprint to 10 VP by any means.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        # Handle discard/robber phases
        special = _handle_special_phases(game, legal_actions, self._rng)
        if special is not None:
            return special

        pidx = _acting_player_idx(game)
        player = game.players[pidx]
        vp = player.victory_points

        # Always play dev cards first
        dev_play = _best_dev_card_play(game, pidx, legal_actions, self._rng)
        if dev_play is not None:
            return dev_play

        if vp >= 8:
            return self._late_game(game, pidx, legal_actions)
        elif vp >= 5:
            return self._mid_game(game, pidx, legal_actions)
        else:
            return self._early_game(game, pidx, legal_actions)

    def _early_game(self, game: Game, pidx: int, legal_actions: list[Action]) -> Action:
        """Expand: settlements > roads > cities > dev cards."""
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        # Trade toward settlement cost
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, SETTLEMENT_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, ROAD_COST)
            if useful:
                return useful[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def _mid_game(self, game: Game, pidx: int, legal_actions: list[Action]) -> Action:
        """Consolidate: cities > dev cards > settlements > roads."""
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

        # Trade toward city cost
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, CITY_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, DEV_CARD_COST)
            if useful:
                return useful[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def _late_game(self, game: Game, pidx: int, legal_actions: list[Action]) -> Action:
        """Sprint to 10 VP: cities > dev cards > settlements > roads."""
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        # Cities give +1 VP immediately (upgrade from settlement)
        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Dev cards: might draw VP card for instant win
        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            # In late game, prioritize longest road push
            extending = _road_extends_longest(game, pidx, roads)
            if extending:
                return self._rng.choice(extending)
            return _smart_road(game, pidx, roads, self._rng)

        # Trade aggressively toward city or dev card
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, CITY_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, DEV_CARD_COST)
            if useful:
                return useful[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "Smart"

# ------------------------------------------------------------------ #
#  HybridBot                                                           #
# ------------------------------------------------------------------ #


class HybridBot(Agent):
    """Combines aggressive building with adaptive strategy and heavy dev card focus.

    Phase-based priorities with dev card integration:
    - Early (VP < 5):   settlements > roads > dev_cards > cities
    - Mid (VP 5-8):     cities > dev_cards > settlements > roads
    - Late (VP >= 8):   cities > dev_cards > roads > settlements

    Always plays available dev cards (knights, monopolies, etc.) immediately.
    Uses smart helpers for robber placement, discard, road selection, and trading.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def choose_action(self, game: Game, legal_actions: list[Action]) -> Action:
        # Handle discard/robber phases
        special = _handle_special_phases(game, legal_actions, self._rng)
        if special is not None:
            return special

        pidx = _acting_player_idx(game)
        player = game.players[pidx]
        vp = player.victory_points

        # Always play dev cards first (aggressive play)
        dev_play = _best_dev_card_play(game, pidx, legal_actions, self._rng)
        if dev_play is not None:
            return dev_play

        # Phase-based strategy
        if vp >= 8:
            return self._late_game(game, pidx, legal_actions)
        elif vp >= 5:
            return self._mid_game(game, pidx, legal_actions)
        else:
            return self._early_game(game, pidx, legal_actions)

    def _early_game(self, game: Game, pidx: int, legal_actions: list[Action]) -> Action:
        """Expand: settlements > roads > dev_cards > cities."""
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Trade toward settlement cost
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, SETTLEMENT_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, ROAD_COST)
            if useful:
                return useful[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def _mid_game(self, game: Game, pidx: int, legal_actions: list[Action]) -> Action:
        """Consolidate: cities > dev_cards > settlements > roads."""
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            return _smart_road(game, pidx, roads, self._rng)

        # Trade toward city or dev card cost
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, CITY_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, DEV_CARD_COST)
            if useful:
                return useful[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def _late_game(self, game: Game, pidx: int, legal_actions: list[Action]) -> Action:
        """Sprint to 10 VP: cities > dev_cards > roads > settlements."""
        from catan.actions import (
            BankTrade,
            BuildCity,
            BuildRoad,
            BuildSettlement,
            BuyDevCard,
            EndTurn,
        )

        cities = _actions_of_type(legal_actions, BuildCity)
        if cities:
            return max(cities, key=lambda a: _vertex_production_value(game, a.vertex_id))

        dev = _actions_of_type(legal_actions, BuyDevCard)
        if dev:
            return dev[0]

        roads = _actions_of_type(legal_actions, BuildRoad)
        if roads:
            extending = _road_extends_longest(game, pidx, roads)
            if extending:
                return self._rng.choice(extending)
            return _smart_road(game, pidx, roads, self._rng)

        settlements = _actions_of_type(legal_actions, BuildSettlement)
        if settlements:
            return max(settlements, key=lambda a: _vertex_production_value(game, a.vertex_id))

        # Trade aggressively toward city or dev card
        trades = _actions_of_type(legal_actions, BankTrade)
        if trades:
            useful = _need_based_trades(game, pidx, trades, CITY_COST)
            if not useful:
                useful = _need_based_trades(game, pidx, trades, DEV_CARD_COST)
            if useful:
                return useful[0]

        end = _actions_of_type(legal_actions, EndTurn)
        if end:
            return end[0]
        return self._rng.choice(legal_actions)

    def name(self) -> str:
        return "Hybrid"