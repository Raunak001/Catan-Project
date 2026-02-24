"""Game state and turn logic — the core Catan engine.

Manages all game phases, legal action computation, action execution,
resource production, robber mechanics, and victory detection.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from catan.actions import (
    Action,
    BankTrade,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevCard,
    DiscardResources,
    EndTurn,
    MoveRobber,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
)
from catan.board import Board
from catan.constants import (
    CITY_COST,
    DEFAULT_BANK_TRADE_RATE,
    DEV_CARD_COST,
    MAX_CARDS_BEFORE_DISCARD,
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    MAX_TURNS,
    ROAD_COST,
    SETTLEMENT_COST,
    VICTORY_POINTS_TO_WIN,
)
from catan.dev_cards import LARGEST_ARMY_THRESHOLD, DevCardType
from catan.longest_road import compute_longest_road
from catan.player import Player
from catan.ports import PORT_RESOURCE, PORT_TRADE_RATE, PortType
from catan.resources import TERRAIN_TO_RESOURCE, Resource


class GamePhase(Enum):
    PLACEMENT = "placement"
    ROLL = "roll"
    ROBBER_DISCARD = "robber_discard"
    ROBBER_MOVE = "robber_move"
    MAIN = "main"
    FINISHED = "finished"


@dataclass
class Game:
    """Top-level game state and engine."""

    board: Board = field(default_factory=Board.standard)
    players: list[Player] = field(default_factory=list)
    current_player_idx: int = 0
    turn: int = 0
    phase: GamePhase = GamePhase.PLACEMENT
    rng: random.Random = field(default_factory=random.Random)

    # Board ownership
    vertex_owner: dict[int, int | None] = field(default_factory=dict)
    vertex_building: dict[int, str] = field(default_factory=dict)  # "settlement" | "city"
    edge_owner: dict[int, int | None] = field(default_factory=dict)

    # Robber
    robber_hex: int = -1  # initialized in __post_init__

    # Development cards
    dev_card_deck: list[DevCardType] = field(default_factory=list)
    largest_army_player: int | None = None
    longest_road_player: int | None = None

    # Placement phase tracking
    placement_round: int = 0  # 0 = first round, 1 = second round (reverse)
    placement_step: int = 0  # 0 = place settlement, 1 = place road

    # Last dice roll (for display / observation)
    last_roll: int = 0

    # Discard tracking (for 7-roll: which players still need to discard)
    players_to_discard: list[int] = field(default_factory=list)
    _discard_idx: int = 0  # index into players_to_discard

    # Road building tracking
    road_building_remaining: int = 0

    def __post_init__(self) -> None:
        if self.robber_hex == -1:
            self.robber_hex = self.board.desert_hex_index()
        if not self.dev_card_deck:
            from catan.dev_cards import make_deck

            self.dev_card_deck = make_deck(self.rng)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def topology(self):
        return self.board.topology

    # ------------------------------------------------------------------ #
    #  Dice                                                                #
    # ------------------------------------------------------------------ #

    def roll_dice(self) -> int:
        return self.rng.randint(1, 6) + self.rng.randint(1, 6)

    # ------------------------------------------------------------------ #
    #  Port / trade-rate helpers                                           #
    # ------------------------------------------------------------------ #

    def _player_port_types(self, player_idx: int) -> set[PortType]:
        """Return the set of port types accessible to a player."""
        player = self.players[player_idx]
        vertex_set = set(player.settlements) | set(player.cities)
        port_types: set[PortType] = set()
        for port in self.board.ports:
            if port.vertices[0] in vertex_set or port.vertices[1] in vertex_set:
                port_types.add(port.port_type)
        return port_types

    def _trade_rate(self, player_idx: int, resource: Resource) -> int:
        """Best bank trade rate for a specific resource."""
        rate = DEFAULT_BANK_TRADE_RATE
        for pt in self._player_port_types(player_idx):
            port_res = PORT_RESOURCE[pt]
            if port_res == resource:
                rate = min(rate, PORT_TRADE_RATE[pt])
            elif port_res is None:  # generic port
                rate = min(rate, PORT_TRADE_RATE[pt])
        return rate

    # ------------------------------------------------------------------ #
    #  Legal actions                                                       #
    # ------------------------------------------------------------------ #

    def legal_actions(self) -> list[Action]:
        if self.phase == GamePhase.PLACEMENT:
            return self._legal_placement_actions()
        elif self.phase == GamePhase.ROLL:
            return []  # Roll is automatic, not a player action
        elif self.phase == GamePhase.ROBBER_DISCARD:
            return self._legal_discard_actions()
        elif self.phase == GamePhase.ROBBER_MOVE:
            return self._legal_robber_move_actions()
        elif self.phase == GamePhase.MAIN:
            return self._legal_main_actions()
        return []

    def _legal_placement_actions(self) -> list[Action]:
        actions: list[Action] = []

        if self.placement_step == 0:
            # Place settlement: any vertex not occupied and not adjacent to another settlement
            for v in range(self.topology.num_vertices):
                if self._can_place_initial_settlement(v):
                    actions.append(BuildSettlement(v))
        else:
            # Place road: must be adjacent to the last-placed settlement
            last_settlement = self.current_player.settlements[-1]
            for eid in self.topology.vertex_to_edges[last_settlement]:
                if eid not in self.edge_owner:
                    actions.append(BuildRoad(eid))
        return actions

    def _can_place_initial_settlement(self, vertex_id: int) -> bool:
        """Check if a vertex is valid for initial placement (distance rule, unoccupied)."""
        if vertex_id in self.vertex_owner:
            return False
        # Distance rule: no adjacent vertex can have a building
        for neighbor in self.topology.vertex_neighbors[vertex_id]:
            if neighbor in self.vertex_owner:
                return False
        return True

    def _legal_discard_actions(self) -> list[Action]:
        """Generate all valid discard combinations for the current discarding player."""
        pidx = self.players_to_discard[self._discard_idx]
        player = self.players[pidx]
        total = player.total_resource_count()
        discard_count = total // 2

        # Generate combinations of resources to discard
        return _discard_combinations(player.resources, discard_count)

    def _legal_robber_move_actions(self) -> list[Action]:
        actions: list[Action] = []
        pidx = self.current_player_idx
        for hex_idx in range(len(self.board.hexes)):
            if hex_idx == self.robber_hex:
                continue  # must move to a different hex
            # Find adjacent players to steal from
            steal_targets = self._robber_steal_targets(hex_idx, pidx)
            if steal_targets:
                for target in steal_targets:
                    actions.append(MoveRobber(hex_idx, target))
            else:
                actions.append(MoveRobber(hex_idx, None))
        return actions

    def _robber_steal_targets(self, hex_idx: int, robber_player: int) -> list[int]:
        """Players with buildings on this hex (excluding the robber player) who have cards."""
        targets: set[int] = set()
        for vid in self.topology.hex_to_vertices[hex_idx]:
            owner = self.vertex_owner.get(vid)
            if owner is not None and owner != robber_player:
                if self.players[owner].total_resource_count() > 0:
                    targets.add(owner)
        return sorted(targets)

    def _legal_main_actions(self) -> list[Action]:
        actions: list[Action] = []
        pidx = self.current_player_idx
        player = self.current_player

        # If we're in road-building mode, only road placements are legal
        if self.road_building_remaining > 0:
            for eid in range(self.topology.num_edges):
                if self._can_build_road(pidx, eid, free=True):
                    actions.append(BuildRoad(eid))
            if actions:
                return actions
            # No legal road spots — cancel remaining road building so the
            # player isn't stuck with zero legal actions.
            self.road_building_remaining = 0

        # Build settlement
        if len(player.settlements) < MAX_SETTLEMENTS and player.can_afford(SETTLEMENT_COST):
            for v in range(self.topology.num_vertices):
                if self._can_build_settlement(pidx, v):
                    actions.append(BuildSettlement(v))

        # Build city
        if len(player.cities) < MAX_CITIES and player.can_afford(CITY_COST):
            # Verify all settlements in the list have corresponding buildings
            for v in player.settlements:
                building_type = self.vertex_building.get(v)
                owner = self.vertex_owner.get(v)
                if building_type != "settlement" or owner != pidx:
                    raise AssertionError(
                        f"Settlement {v} in player {pidx} settlements list has inconsistent state: "
                        f"vertex_building[{v}]={building_type}, vertex_owner[{v}]={owner}. "
                        f"Player settlements list: {player.settlements}, "
                        f"Player cities list: {player.cities}. "
                        f"This indicates corrupted game state."
                    )
            for v in player.settlements:
                if self._can_build_city(pidx, v):
                    actions.append(BuildCity(v))

        # Build road
        if len(player.roads) < MAX_ROADS and player.can_afford(ROAD_COST):
            for eid in range(self.topology.num_edges):
                if self._can_build_road(pidx, eid):
                    actions.append(BuildRoad(eid))

        # Bank trades
        for give_res in Resource:
            rate = self._trade_rate(pidx, give_res)
            if player.resources[give_res] >= rate:
                for recv_res in Resource:
                    if recv_res != give_res:
                        actions.append(BankTrade(give_res, recv_res))

        # Buy development card
        if player.can_afford(DEV_CARD_COST) and self.dev_card_deck:
            actions.append(BuyDevCard())

        # Play development card (1 per turn, not cards bought this turn)
        if not player.dev_card_played_this_turn:
            playable = set(player.dev_cards)  # cards from previous turns
            if DevCardType.KNIGHT in playable:
                for hex_idx in range(len(self.board.hexes)):
                    if hex_idx == self.robber_hex:
                        continue
                    steal_targets = self._robber_steal_targets(hex_idx, pidx)
                    if steal_targets:
                        for t in steal_targets:
                            actions.append(PlayKnight(hex_idx, t))
                    else:
                        actions.append(PlayKnight(hex_idx, None))

            if DevCardType.ROAD_BUILDING in playable:
                # Check if player has any legal road placements
                has_road_spot = any(
                    self._can_build_road(pidx, eid, free=True)
                    for eid in range(self.topology.num_edges)
                )
                if has_road_spot and len(player.roads) < MAX_ROADS:
                    # We represent this as a single action; the 2 roads are
                    # placed interactively via road_building_remaining state.
                    actions.append(PlayRoadBuilding(edge1=-1, edge2=None))

            if DevCardType.YEAR_OF_PLENTY in playable:
                # Use the same ordering as gym_env.py to ensure action masking matches legal actions
                # Pairs are (r1, r2) where r1 comes before or equals r2 in enum order
                resource_list = list(Resource)
                for i, r1 in enumerate(resource_list):
                    for r2 in resource_list[i:]:
                        actions.append(PlayYearOfPlenty(r1, r2))

            if DevCardType.MONOPOLY in playable:
                for r in Resource:
                    actions.append(PlayMonopoly(r))

        # End turn
        actions.append(EndTurn())
        return actions

    def _can_build_city(self, player_idx: int, vertex_id: int) -> bool:
        """Check if a player can build a city at a vertex (upgrade settlement)."""
        # Must have a settlement at this vertex
        player = self.players[player_idx]
        if vertex_id not in player.settlements:
            return False
        # Settlement must be in vertex_building with the correct player ownership
        if self.vertex_building.get(vertex_id) != "settlement":
            return False
        # Vertex owner must match player
        if self.vertex_owner.get(vertex_id) != player_idx:
            return False
        return True

    def _can_build_settlement(self, player_idx: int, vertex_id: int) -> bool:
        """Check if a player can build a settlement at a vertex (main game phase)."""
        if vertex_id in self.vertex_owner:
            return False
        # Distance rule
        for neighbor in self.topology.vertex_neighbors[vertex_id]:
            if neighbor in self.vertex_owner:
                return False
        # Must be adjacent to one of the player's roads
        player = self.players[player_idx]
        road_set = set(player.roads)
        for eid in self.topology.vertex_to_edges[vertex_id]:
            if eid in road_set:
                return True
        return False

    def _can_build_road(self, player_idx: int, edge_id: int, free: bool = False) -> bool:
        """Check if a player can build a road on an edge."""
        if edge_id in self.edge_owner:
            return False
        if len(self.players[player_idx].roads) >= MAX_ROADS:
            return False
        player = self.players[player_idx]
        a, b = self.topology.edge_vertices[edge_id]
        # Must connect to an existing road or building of this player
        player_vertices = set(player.settlements) | set(player.cities)
        player_roads = set(player.roads)
        for v in (a, b):
            if v in player_vertices:
                return True
            # Check if there's an adjacent road and the vertex isn't blocked
            # by an opponent's building
            owner = self.vertex_owner.get(v)
            if owner is not None and owner != player_idx:
                continue  # opponent building blocks road connection
            for adj_eid in self.topology.vertex_to_edges[v]:
                if adj_eid in player_roads:
                    return True
        return False

    def _validate_settlements_consistency(self) -> None:
        """Validate that settlements and vertex_building are in sync."""
        for pidx, player in enumerate(self.players):
            for vid in player.settlements:
                # Every settlement in player.settlements should be marked as "settlement" in vertex_building
                building_type = self.vertex_building.get(vid)
                if building_type != "settlement":
                    raise AssertionError(
                        f"Settlement {vid} in player {pidx} settlements list "
                        f"but vertex_building[{vid}]={building_type}. "
                        f"Player {pidx} settlements: {player.settlements}, "
                        f"Player {pidx} cities: {player.cities}. "
                        f"This indicates player.settlements is corrupted."
                    )
                # Verify ownership matches
                owner = self.vertex_owner.get(vid)
                if owner != pidx:
                    raise AssertionError(
                        f"Settlement {vid} in player {pidx} settlements list "
                        f"but vertex_owner[{vid}]={owner}. "
                        f"Player {pidx} settlements: {player.settlements}"
                    )
            
            for vid in player.cities:
                # Every city in player.cities should be marked as "city" in vertex_building
                building_type = self.vertex_building.get(vid)
                if building_type != "city":
                    raise AssertionError(
                        f"City {vid} in player {pidx} cities list "
                        f"but vertex_building[{vid}]={building_type}. "
                        f"Player {pidx} settlements: {player.settlements}, "
                        f"Player {pidx} cities: {player.cities}"
                    )
                owner = self.vertex_owner.get(vid)
                if owner != pidx:
                    raise AssertionError(
                        f"City {vid} in player {pidx} cities list "
                        f"but vertex_owner[{vid}]={owner}"
                    )

    # ------------------------------------------------------------------ #
    #  Apply actions                                                       #
    # ------------------------------------------------------------------ #

    def apply_action(self, action: Action) -> None:
        """Execute an action, mutating game state."""
        # Validate invariants before action
        self._validate_settlements_consistency()
        
        match action:
            case BuildSettlement(vertex_id=v):
                self._apply_build_settlement(v)
            case BuildRoad(edge_id=e):
                self._apply_build_road(e)
            case BuildCity(vertex_id=v):
                self._apply_build_city(v)
            case BankTrade(give=give, receive=recv):
                self._apply_bank_trade(give, recv)
            case BuyDevCard():
                self._apply_buy_dev_card()
            case PlayKnight(target_hex=h, steal_from=s):
                self._apply_play_knight(h, s)
            case PlayRoadBuilding():
                self._apply_play_road_building()
            case PlayYearOfPlenty(resource1=r1, resource2=r2):
                self._apply_play_year_of_plenty(r1, r2)
            case PlayMonopoly(resource=r):
                self._apply_play_monopoly(r)
            case MoveRobber(target_hex=h, steal_from=s):
                self._apply_move_robber(h, s)
            case DiscardResources(resources=res):
                self._apply_discard(res)
            case EndTurn():
                self._apply_end_turn()

    def _apply_build_settlement(self, vertex_id: int) -> None:
        pidx = self.current_player_idx
        player = self.current_player
        vertex_id = int(vertex_id)  # Ensure regular Python int, not numpy int

        if self.phase == GamePhase.PLACEMENT:
            # Free during placement
            player.settlements.append(vertex_id)
            self.vertex_owner[vertex_id] = pidx
            self.vertex_building[vertex_id] = "settlement"
            player.victory_points += 1

            # Second round placement: grant starting resources
            if self.placement_round == 1:
                for hex_idx in self.topology.vertex_to_hexes[vertex_id]:
                    h = self.board.hexes[hex_idx]
                    res = TERRAIN_TO_RESOURCE.get(h.terrain)
                    if res:
                        player.resources[res] += 1

            self.placement_step = 1  # next: place road
        else:
            player.pay(SETTLEMENT_COST)
            player.settlements.append(vertex_id)
            self.vertex_owner[vertex_id] = pidx
            self.vertex_building[vertex_id] = "settlement"
            player.victory_points += 1
            self._update_longest_road()

    def _apply_build_road(self, edge_id: int) -> None:
        pidx = self.current_player_idx
        player = self.current_player

        if self.phase == GamePhase.PLACEMENT:
            player.roads.append(edge_id)
            self.edge_owner[edge_id] = pidx
            self._advance_placement()
        elif self.road_building_remaining > 0:
            player.roads.append(edge_id)
            self.edge_owner[edge_id] = pidx
            self.road_building_remaining -= 1
            self._update_longest_road()
        else:
            player.pay(ROAD_COST)
            player.roads.append(edge_id)
            self.edge_owner[edge_id] = pidx
            self._update_longest_road()

    def _apply_build_city(self, vertex_id: int) -> None:
        pidx = self.current_player_idx
        player = self.current_player
        vertex_id = int(vertex_id)  # Ensure regular Python int, not numpy int
        
        # VALIDATE BEFORE PAYING - don't modify game state if invalid
        if vertex_id not in player.settlements:
            # Check if it's in the city list (might be trying to build city on city)
            if vertex_id in player.cities:
                raise ValueError(f"Vertex {vertex_id} already has a city (cannot build city on city)")
            # Check actual state for debugging
            building_type = self.vertex_building.get(vertex_id)
            raise ValueError(
                f"Cannot build city at {vertex_id}: no settlement in player's list. "
                f"vertex_building[{vertex_id}]={building_type}, vertex_owner[{vertex_id}]={self.vertex_owner.get(vertex_id)}, "
                f"player_idx={pidx}, player.settlements={player.settlements}"
            )
        
        # Double-check vertex_building is consistent
        if self.vertex_building.get(vertex_id) != "settlement":
            raise ValueError(
                f"Cannot build city at {vertex_id}: player's settlement list shows it but vertex_building is {self.vertex_building.get(vertex_id)}. "
                f"Game state is corrupted. player.settlements={player.settlements}"
            )
        
        # NOW pay after validation passes
        player.pay(CITY_COST)
        player.settlements.remove(vertex_id)
        player.cities.append(vertex_id)
        self.vertex_building[vertex_id] = "city"
        player.victory_points += 1  # settlement was 1, city is 2 => net +1

    def _apply_bank_trade(self, give: Resource, receive: Resource) -> None:
        pidx = self.current_player_idx
        player = self.current_player
        rate = self._trade_rate(pidx, give)
        player.resources[give] -= rate
        player.resources[receive] += 1
        player.resources = +player.resources

    def _apply_buy_dev_card(self) -> None:
        player = self.current_player
        player.pay(DEV_CARD_COST)
        card = self.dev_card_deck.pop()
        player.new_dev_cards.append(card)
        # VP cards take effect immediately but are hidden
        if card == DevCardType.VICTORY_POINT:
            player.victory_points += 1

    def _apply_play_knight(self, target_hex: int, steal_from: int | None) -> None:
        pidx = self.current_player_idx
        player = self.current_player
        if DevCardType.KNIGHT not in player.dev_cards:
            raise ValueError(
                f"Cannot play KNIGHT: card not in player {pidx} dev_cards={player.dev_cards}, "
                f"new_dev_cards={player.new_dev_cards}"
            )
        player.dev_cards.remove(DevCardType.KNIGHT)
        player.played_knights += 1
        player.dev_card_played_this_turn = True
        self._move_robber(target_hex, steal_from)
        self._update_largest_army()

    def _apply_play_road_building(self) -> None:
        pidx = self.current_player_idx
        player = self.current_player
        if DevCardType.ROAD_BUILDING not in player.dev_cards:
            raise ValueError(
                f"Cannot play ROAD_BUILDING: card not in player {pidx} dev_cards={player.dev_cards}, "
                f"new_dev_cards={player.new_dev_cards}"
            )
        player.dev_cards.remove(DevCardType.ROAD_BUILDING)
        player.dev_card_played_this_turn = True
        roads_left = MAX_ROADS - len(player.roads)
        self.road_building_remaining = min(2, roads_left)

    def _apply_play_year_of_plenty(self, r1: Resource, r2: Resource) -> None:
        player = self.current_player
        pidx = self.current_player_idx
        if DevCardType.YEAR_OF_PLENTY not in player.dev_cards:
            raise ValueError(
                f"Cannot play YEAR_OF_PLENTY: card not in player {pidx} dev_cards={player.dev_cards}, "
                f"new_dev_cards={player.new_dev_cards}"
            )
        player.dev_cards.remove(DevCardType.YEAR_OF_PLENTY)
        player.dev_card_played_this_turn = True
        player.resources[r1] += 1
        player.resources[r2] += 1

    def _apply_play_monopoly(self, resource: Resource) -> None:
        pidx = self.current_player_idx
        player = self.current_player
        if DevCardType.MONOPOLY not in player.dev_cards:
            raise ValueError(
                f"Cannot play MONOPOLY: card not in player {pidx} dev_cards={player.dev_cards}, "
                f"new_dev_cards={player.new_dev_cards}"
            )
        player.dev_cards.remove(DevCardType.MONOPOLY)
        player.dev_card_played_this_turn = True
        total = 0
        for i, p in enumerate(self.players):
            if i != pidx:
                stolen = p.resources[resource]
                p.resources[resource] = 0
                p.resources = +p.resources
                total += stolen
        player.resources[resource] += total

    def _apply_move_robber(self, target_hex: int, steal_from: int | None) -> None:
        self._move_robber(target_hex, steal_from)
        self.phase = GamePhase.MAIN

    def _move_robber(self, target_hex: int, steal_from: int | None) -> None:
        self.robber_hex = target_hex
        if steal_from is not None:
            victim = self.players[steal_from]
            if victim.total_resource_count() > 0:
                resources_list = []
                for res, count in victim.resources.items():
                    resources_list.extend([res] * count)
                stolen = self.rng.choice(resources_list)
                victim.resources[stolen] -= 1
                victim.resources = +victim.resources
                self.current_player.resources[stolen] += 1

    def _apply_discard(self, resources: Counter[Resource]) -> None:
        pidx = self.players_to_discard[self._discard_idx]
        player = self.players[pidx]
        player.resources -= resources
        player.resources = +player.resources
        self._discard_idx += 1
        if self._discard_idx >= len(self.players_to_discard):
            # All players have discarded, move to robber placement
            self.phase = GamePhase.ROBBER_MOVE
            self.players_to_discard = []
            self._discard_idx = 0

    def _apply_end_turn(self) -> None:
        self._end_current_turn()
        self._start_next_turn()

    # ------------------------------------------------------------------ #
    #  Turn management                                                     #
    # ------------------------------------------------------------------ #

    def _end_current_turn(self) -> None:
        player = self.current_player
        # Move newly bought dev cards to playable hand
        player.dev_cards.extend(player.new_dev_cards)
        player.new_dev_cards.clear()
        player.dev_card_played_this_turn = False

    def _start_next_turn(self) -> None:
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self.turn += 1
        self.road_building_remaining = 0

        if self.turn >= MAX_TURNS:
            self.phase = GamePhase.FINISHED
            return

        # Roll dice
        roll = self.roll_dice()
        self.last_roll = roll

        if roll == 7:
            # Check who needs to discard
            self.players_to_discard = [
                i
                for i in range(self.num_players)
                if self.players[i].total_resource_count() > MAX_CARDS_BEFORE_DISCARD
            ]
            if self.players_to_discard:
                self._discard_idx = 0
                self.phase = GamePhase.ROBBER_DISCARD
            else:
                self.phase = GamePhase.ROBBER_MOVE
        else:
            self._produce_resources(roll)
            self.phase = GamePhase.MAIN

    def _produce_resources(self, roll: int) -> None:
        """Distribute resources to all players based on dice roll."""
        for hex_idx in self.board.token_to_hexes.get(roll, []):
            if hex_idx == self.robber_hex:
                continue  # robber blocks production
            h = self.board.hexes[hex_idx]
            resource = TERRAIN_TO_RESOURCE.get(h.terrain)
            if resource is None:
                continue
            for vid in self.topology.hex_to_vertices[hex_idx]:
                owner = self.vertex_owner.get(vid)
                if owner is not None:
                    building = self.vertex_building[vid]
                    amount = 2 if building == "city" else 1
                    self.players[owner].resources[resource] += amount

    # ------------------------------------------------------------------ #
    #  Placement phase                                                     #
    # ------------------------------------------------------------------ #

    def _advance_placement(self) -> None:
        """Advance the placement state machine after placing a road."""
        self.placement_step = 0  # reset for next settlement
        total_players = self.num_players

        if self.placement_round == 0:
            # First round: go forward
            if self.current_player_idx < total_players - 1:
                self.current_player_idx += 1
            else:
                # Start second round (same player goes again)
                self.placement_round = 1
        else:
            # Second round: go backward
            if self.current_player_idx > 0:
                self.current_player_idx -= 1
            else:
                # Placement complete — start normal play
                self.phase = GamePhase.ROLL
                self.current_player_idx = 0
                self._start_next_turn()

    # ------------------------------------------------------------------ #
    #  Longest road / largest army                                         #
    # ------------------------------------------------------------------ #

    def _update_longest_road(self) -> None:
        """Recompute longest road holder."""
        best_length = 4  # must be at least 5 to claim
        best_player = self.longest_road_player

        if best_player is not None:
            best_length = (
                compute_longest_road(
                    self.players[best_player].roads,
                    best_player,
                    self.topology,
                    self.vertex_owner,
                )
                - 1
            )  # current holder keeps it unless beaten

        for pidx in range(self.num_players):
            length = compute_longest_road(
                self.players[pidx].roads, pidx, self.topology, self.vertex_owner
            )
            if length >= 5 and length > best_length:
                best_length = length
                best_player = pidx

        old_holder = self.longest_road_player
        if best_player != old_holder:
            if old_holder is not None:
                self.players[old_holder].victory_points -= 2
            if best_player is not None:
                self.players[best_player].victory_points += 2
            self.longest_road_player = best_player

    def _update_largest_army(self) -> None:
        """Recompute largest army holder."""
        best_knights = LARGEST_ARMY_THRESHOLD - 1
        best_player = self.largest_army_player

        if best_player is not None:
            best_knights = self.players[best_player].played_knights - 1

        for pidx in range(self.num_players):
            k = self.players[pidx].played_knights
            if k >= LARGEST_ARMY_THRESHOLD and k > best_knights:
                best_knights = k
                best_player = pidx

        old_holder = self.largest_army_player
        if best_player != old_holder:
            if old_holder is not None:
                self.players[old_holder].victory_points -= 2
            if best_player is not None:
                self.players[best_player].victory_points += 2
            self.largest_army_player = best_player

    # ------------------------------------------------------------------ #
    #  Victory                                                             #
    # ------------------------------------------------------------------ #

    def check_victory(self) -> int | None:
        """Return the index of the winning player, or None."""
        for i, p in enumerate(self.players):
            if p.victory_points >= VICTORY_POINTS_TO_WIN:
                return i
        return None

    # ------------------------------------------------------------------ #
    #  High-level step (for RL)                                            #
    # ------------------------------------------------------------------ #

    def step(self, action: Action) -> tuple[bool, int | None]:
        """Apply action, return (game_over, winner_idx)."""
        self.apply_action(action)
        winner = self.check_victory()
        if winner is not None:
            self.phase = GamePhase.FINISHED
        game_over = self.phase == GamePhase.FINISHED
        return game_over, winner


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _discard_combinations(
    resources: Counter[Resource], discard_count: int
) -> list[DiscardResources]:
    """Generate all valid ways to discard exactly `discard_count` cards.

    To keep the action space manageable, we limit to a reasonable number
    of combinations. If there are too many, we sample representative ones.
    """
    available: list[tuple[Resource, int]] = [(r, c) for r, c in resources.items() if c > 0]

    results: list[Counter[Resource]] = []
    _generate_discards(available, 0, discard_count, Counter(), results, max_results=50)

    return [DiscardResources(r) for r in results]


def _generate_discards(
    available: list[tuple[Resource, int]],
    idx: int,
    remaining: int,
    current: Counter[Resource],
    results: list[Counter[Resource]],
    max_results: int,
) -> None:
    """Recursively generate discard combinations."""
    if len(results) >= max_results:
        return
    if remaining == 0:
        results.append(Counter(current))
        return
    if idx >= len(available):
        return

    resource, count = available[idx]
    for take in range(min(count, remaining) + 1):
        if take > 0:
            current[resource] = take
        _generate_discards(available, idx + 1, remaining - take, current, results, max_results)
        if take > 0:
            del current[resource]
