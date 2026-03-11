"""Game session management: wraps a Game with agents and action tracking."""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass

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
from catan.api.action_tracker import ActionTracker
from catan.board import Board
from catan.game import Game, GamePhase
from catan.player import Player

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
