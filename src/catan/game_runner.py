"""Game runner: play complete games with AI agents."""

from __future__ import annotations

import random
from dataclasses import dataclass

from catan.ai.agent import Agent
from catan.board import Board
from catan.game import Game, GamePhase
from catan.player import Player


@dataclass
class GameResult:
    """Result of a single completed game."""

    winner: int | None  # player index, or None for draw/timeout
    turns: int
    victory_points: list[int]  # final VP per player


@dataclass
class TournamentResult:
    """Aggregated results of multiple games."""

    wins: list[int]  # win count per player slot
    draws: int
    total_games: int
    avg_turns: float
    avg_vps: list[float]


def run_game(
    agents: list[Agent],
    seed: int | None = None,
    shuffle_board: bool = True,
) -> GameResult:
    """Run a complete game with the given agents.

    Parameters
    ----------
    agents : One agent per player (typically 3 or 4).
    seed : Random seed for reproducibility.
    shuffle_board : Whether to randomize the board layout.
    """
    rng = random.Random(seed)

    board = Board.standard(shuffle=shuffle_board, rng=rng)
    players = [Player(name=a.name()) for a in agents]
    game = Game(board=board, players=players, rng=rng)

    while game.phase != GamePhase.FINISHED:
        # Handle phases that don't need agent input
        if game.phase == GamePhase.ROLL:
            game._start_next_turn()
            continue

        # Determine which agent should act
        if game.phase == GamePhase.ROBBER_DISCARD:
            acting_player_idx = game.players_to_discard[game._discard_idx]
        else:
            acting_player_idx = game.current_player_idx

        actions = game.legal_actions()
        if not actions:
            # Safety: if no actions available, end the game
            game.phase = GamePhase.FINISHED
            break

        agent = agents[acting_player_idx]
        action = agent.choose_action(game, actions)
        game.apply_action(action)

        winner = game.check_victory()
        if winner is not None:
            game.phase = GamePhase.FINISHED

    return GameResult(
        winner=game.check_victory(),
        turns=game.turn,
        victory_points=[p.victory_points for p in game.players],
    )


def run_tournament(
    agents: list[Agent],
    n_games: int,
    base_seed: int = 0,
    shuffle_board: bool = True,
) -> TournamentResult:
    """Run multiple games and aggregate results."""
    num_players = len(agents)
    wins = [0] * num_players
    draws = 0
    total_turns = 0
    total_vps = [0.0] * num_players

    for i in range(n_games):
        result = run_game(agents, seed=base_seed + i, shuffle_board=shuffle_board)
        total_turns += result.turns
        for j, vp in enumerate(result.victory_points):
            total_vps[j] += vp
        if result.winner is not None:
            wins[result.winner] += 1
        else:
            draws += 1

    return TournamentResult(
        wins=wins,
        draws=draws,
        total_games=n_games,
        avg_turns=total_turns / max(n_games, 1),
        avg_vps=[v / max(n_games, 1) for v in total_vps],
    )
