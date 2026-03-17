"""Game session management: wraps a Game with agents and action tracking."""

from __future__ import annotations

import random
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from catan.actions import Action
from catan.ai.agent import Agent
from catan.ai.heuristic import (
    DevCardBot,
    GreedyAgent,
    HybridBot,
    LongestRoadBot,
    RandomAgent,
    ResourceHoarder,
    SmartBot,
)
from catan.api.action_tracker import ActionRecord, ActionTracker
from catan.board import Board, Hex
from catan.dev_cards import DevCardType
from catan.game import Game, GamePhase
from catan.player import Player
from catan.resources import Resource, Terrain

AGENT_REGISTRY: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "smartbot": SmartBot,
    "hybridbot": HybridBot,
    "longestroadbot": LongestRoadBot,
    "devcardbot": DevCardBot,
    "resourcehoarder": ResourceHoarder,
}


@dataclass
class GameSession:
    """A live game session with agents, tracking, and autoplay."""

    game_id: str
    game: Game
    agents: list[Agent | None]  # None = human seat
    tracker: ActionTracker
    human_seats: set[int]
    finished: bool = False
    last_actions: list[tuple[int, Action]] | None = None  # (player_idx, action) pairs

    def acting_player_idx(self) -> int:
        """Return the index of the player who must act next."""
        if self.game.phase == GamePhase.ROBBER_DISCARD:
            return self.game.players_to_discard[self.game._discard_idx]
        return self.game.current_player_idx

    def apply_and_advance(self, action: Action) -> None:
        """Apply a human action, then autoplay bots until the next human turn."""
        idx = self.acting_player_idx()
        self.tracker.record(self.game, idx, action)
        self.game.apply_action(action)
        self.last_actions = [(idx, action)]
        if self.game.check_victory() is not None:
            self.game.phase = GamePhase.FINISHED
            self.finished = True
        self._autoplay_bots()

    def _autoplay_bots(self) -> None:
        """Run bot turns until it's a human's turn or the game ends."""
        while self.game.phase != GamePhase.FINISHED:
            if self.game.phase == GamePhase.ROLL:
                self.game._start_next_turn()
                continue

            idx = self.acting_player_idx()
            if idx in self.human_seats:
                break

            legal = self.game.legal_actions()
            if not legal:
                self.game.phase = GamePhase.FINISHED
                break

            agent = self.agents[idx]
            action = agent.choose_action(self.game, legal)
            self.tracker.record(self.game, idx, action)
            self.game.apply_action(action)
            if self.last_actions is not None:
                self.last_actions.append((idx, action))
            if self.game.check_victory() is not None:
                self.game.phase = GamePhase.FINISHED

        if self.game.phase == GamePhase.FINISHED:
            self.finished = True


def create_session(
    seat_configs: list[tuple[int, str]],
    seed: int | None = None,
    shuffle_board: bool = True,
) -> GameSession:
    """Create a new GameSession from seat configurations.

    Parameters
    ----------
    seat_configs : List of (seat_index, agent_name) where agent_name is
                   "human" or a key from AGENT_REGISTRY.
    seed : Random seed for reproducibility.
    shuffle_board : Whether to randomize the board layout.
    """
    num_players = len(seat_configs)
    rng = random.Random(seed)
    board = Board.standard(shuffle=shuffle_board, rng=rng)

    agents: list[Agent | None] = [None] * num_players
    human_seats: set[int] = set()
    player_names: list[str] = []

    for seat_idx, agent_name in seat_configs:
        if agent_name == "human":
            human_seats.add(seat_idx)
            player_names.append("Human")
        else:
            agent = AGENT_REGISTRY[agent_name]()
            agents[seat_idx] = agent
            player_names.append(agent.name())

    players = [Player(name=name) for name in player_names]
    game = Game(board=board, players=players, rng=rng)

    session = GameSession(
        game_id=uuid.uuid4().hex[:12],
        game=game,
        agents=agents,
        tracker=ActionTracker(),
        human_seats=human_seats,
    )

    # Autoplay bots through placement if needed
    session._autoplay_bots()
    return session


# ---------------------------------------------------------------------------
# Blob serialization (full Game state round-trip for DB persistence)
# ---------------------------------------------------------------------------


def _serialize_counter(c: Counter[Resource]) -> dict[str, int]:
    return {r.value: count for r, count in c.items() if count > 0}


def _deserialize_counter(d: dict[str, int]) -> Counter[Resource]:
    return Counter({Resource(k): v for k, v in d.items()})


def _serialize_player(p: Player) -> dict[str, Any]:
    return {
        "name": p.name,
        "resources": _serialize_counter(p.resources),
        "victory_points": p.victory_points,
        "settlements": p.settlements,
        "cities": p.cities,
        "roads": p.roads,
        "dev_cards": [c.value for c in p.dev_cards],
        "played_knights": p.played_knights,
        "dev_card_played_this_turn": p.dev_card_played_this_turn,
        "new_dev_cards": [c.value for c in p.new_dev_cards],
    }


def _deserialize_player(d: dict[str, Any]) -> Player:
    return Player(
        name=d["name"],
        resources=_deserialize_counter(d["resources"]),
        victory_points=d["victory_points"],
        settlements=d["settlements"],
        cities=d["cities"],
        roads=d["roads"],
        dev_cards=[DevCardType(v) for v in d["dev_cards"]],
        played_knights=d["played_knights"],
        dev_card_played_this_turn=d["dev_card_played_this_turn"],
        new_dev_cards=[DevCardType(v) for v in d["new_dev_cards"]],
    )


def _serialize_game(game: Game) -> dict[str, Any]:
    """Serialize full Game state to a JSON-safe dict (for DB storage)."""
    return {
        "hexes": [
            {"terrain": h.terrain.value, "token": h.token, "coord": list(h.coord)}
            for h in game.board.hexes
        ],
        "players": [_serialize_player(p) for p in game.players],
        "current_player_idx": game.current_player_idx,
        "turn": game.turn,
        "phase": game.phase.value,
        "rng_state": list(game.rng.getstate()[1]),
        "rng_gauss": game.rng.getstate()[2],
        "vertex_owner": {str(k): v for k, v in game.vertex_owner.items()},
        "vertex_building": {str(k): v for k, v in game.vertex_building.items()},
        "edge_owner": {str(k): v for k, v in game.edge_owner.items()},
        "robber_hex": game.robber_hex,
        "dev_card_deck": [c.value for c in game.dev_card_deck],
        "largest_army_player": game.largest_army_player,
        "longest_road_player": game.longest_road_player,
        "placement_round": game.placement_round,
        "placement_step": game.placement_step,
        "last_roll": game.last_roll,
        "players_to_discard": game.players_to_discard,
        "_discard_idx": game._discard_idx,
        "road_building_remaining": game.road_building_remaining,
    }


def _deserialize_game(d: dict[str, Any]) -> Game:
    """Reconstruct a Game from a serialized blob."""
    hexes = [
        Hex(terrain=Terrain(h["terrain"]), token=h["token"], coord=tuple(h["coord"]))
        for h in d["hexes"]
    ]
    board = Board(hexes=hexes)  # __post_init__ rebuilds topology, ports, token_map

    players = [_deserialize_player(p) for p in d["players"]]

    # Restore RNG state
    rng = random.Random()
    internalstate = tuple(d["rng_state"])
    rng.setstate((3, internalstate, d["rng_gauss"]))

    # Build the Game, bypassing __post_init__ defaults for robber_hex and dev_card_deck
    game = Game.__new__(Game)
    game.board = board
    game.players = players
    game.current_player_idx = d["current_player_idx"]
    game.turn = d["turn"]
    game.phase = GamePhase(d["phase"])
    game.rng = rng
    game.vertex_owner = {int(k): v for k, v in d["vertex_owner"].items()}
    game.vertex_building = {int(k): v for k, v in d["vertex_building"].items()}
    game.edge_owner = {int(k): v for k, v in d["edge_owner"].items()}
    game.robber_hex = d["robber_hex"]
    game.dev_card_deck = [DevCardType(v) for v in d["dev_card_deck"]]
    game.largest_army_player = d["largest_army_player"]
    game.longest_road_player = d["longest_road_player"]
    game.placement_round = d["placement_round"]
    game.placement_step = d["placement_step"]
    game.last_roll = d["last_roll"]
    game.players_to_discard = d["players_to_discard"]
    game._discard_idx = d["_discard_idx"]
    game.road_building_remaining = d["road_building_remaining"]

    return game


def _serialize_tracker(tracker: ActionTracker) -> list[dict[str, Any]]:
    """Serialize ActionTracker records to a JSON-safe list."""
    return [
        {
            "turn": r.turn,
            "phase": r.phase,
            "player_idx": r.player_idx,
            "action_type": r.action_type,
            "action_details": r.action_details,
            "vp_snapshot": r.vp_snapshot,
        }
        for r in tracker.records
    ]


def _deserialize_tracker(records: list[dict[str, Any]]) -> ActionTracker:
    """Reconstruct an ActionTracker from serialized records."""
    tracker = ActionTracker()
    tracker.records = [
        ActionRecord(
            turn=r["turn"],
            phase=r["phase"],
            player_idx=r["player_idx"],
            action_type=r["action_type"],
            action_details=r["action_details"],
            vp_snapshot=r["vp_snapshot"],
        )
        for r in records
    ]
    return tracker


def session_to_blob(session: GameSession) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Serialize a GameSession into (state_blob, tracker_blob) for DB storage."""
    state = _serialize_game(session.game)
    state["human_seats"] = sorted(session.human_seats)
    state["finished"] = session.finished
    state["seat_agents"] = [None if a is None else _agent_name(a) for a in session.agents]
    tracker = _serialize_tracker(session.tracker)
    return state, tracker


def session_from_blob(
    game_id: str,
    state_blob: dict[str, Any],
    tracker_blob: list[dict[str, Any]],
) -> GameSession:
    """Reconstruct a GameSession from DB blobs."""
    game = _deserialize_game(state_blob)
    tracker = _deserialize_tracker(tracker_blob)
    human_seats = set(state_blob["human_seats"])

    agents: list[Agent | None] = []
    for agent_name in state_blob["seat_agents"]:
        if agent_name is None:
            agents.append(None)
        else:
            agents.append(AGENT_REGISTRY[agent_name]())

    return GameSession(
        game_id=game_id,
        game=game,
        agents=agents,
        tracker=tracker,
        human_seats=human_seats,
        finished=state_blob.get("finished", False),
    )


def _agent_name(agent: Agent) -> str:
    """Reverse-lookup an agent's registry key from its instance."""
    agent_type = type(agent)
    for key, factory in AGENT_REGISTRY.items():
        if factory is agent_type:
            return key
    return agent.name()
