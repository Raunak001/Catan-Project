"""Shared test helpers for setting up game states."""

from __future__ import annotations

import random

from catan.board import Board
from catan.dev_cards import DevCardType
from catan.game import Game, GamePhase
from catan.player import Player
from catan.resources import Resource


def make_game(num_players: int = 4, seed: int = 42) -> Game:
    """Create a standard game for testing."""
    rng = random.Random(seed)
    board = Board.standard(shuffle=True, rng=rng)
    players = [Player(name=f"P{i}") for i in range(num_players)]
    return Game(board=board, players=players, rng=rng)


def complete_placement(game: Game) -> None:
    """Run through the entire placement phase using first available actions.

    After this, game.phase will be ROLL or MAIN (first turn started).
    """
    while game.phase == GamePhase.PLACEMENT:
        actions = game.legal_actions()
        assert actions, "No legal actions during placement"
        game.apply_action(actions[0])
    # The game transitions through ROLL automatically via _start_next_turn
    # but game_runner handles that. We may need to handle it here.
    if game.phase == GamePhase.ROLL:
        game._start_next_turn()


def skip_to_main_phase(game: Game) -> None:
    """Complete placement and get to the MAIN phase for the first player."""
    complete_placement(game)
    # If we landed on robber phases, handle them
    while game.phase in (GamePhase.ROBBER_DISCARD, GamePhase.ROBBER_MOVE):
        actions = game.legal_actions()
        game.apply_action(actions[0])


def give_resources(player: Player, resources: dict[Resource, int]) -> None:
    """Give a player specific resources."""
    for res, count in resources.items():
        player.resources[res] += count


def give_dev_card(player: Player, card_type: DevCardType) -> None:
    """Give a player a dev card (as if bought on a previous turn)."""
    player.dev_cards.append(card_type)


def place_settlement_at(game: Game, player_idx: int, vertex_id: int) -> None:
    """Directly place a settlement on the board (bypasses normal rules)."""
    player = game.players[player_idx]
    player.settlements.append(vertex_id)
    game.vertex_owner[vertex_id] = player_idx
    game.vertex_building[vertex_id] = "settlement"
    player.victory_points += 1


def place_city_at(game: Game, player_idx: int, vertex_id: int) -> None:
    """Directly place a city on the board (bypasses normal rules)."""
    player = game.players[player_idx]
    player.cities.append(vertex_id)
    game.vertex_owner[vertex_id] = player_idx
    game.vertex_building[vertex_id] = "city"
    player.victory_points += 2


def place_road_at(game: Game, player_idx: int, edge_id: int) -> None:
    """Directly place a road on the board (bypasses normal rules)."""
    player = game.players[player_idx]
    player.roads.append(edge_id)
    game.edge_owner[edge_id] = player_idx
