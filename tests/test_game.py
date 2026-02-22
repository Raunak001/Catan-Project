"""Tests for the Catan game engine."""

import random

from catan.ai.heuristic import RandomAgent
from catan.board import Board
from catan.constants import MAX_TURNS
from catan.game import Game, GamePhase
from catan.game_runner import run_game
from catan.player import Player


def _make_game(num_players: int = 4, seed: int = 42) -> Game:
    """Create a standard game for testing."""
    rng = random.Random(seed)
    board = Board.standard(shuffle=True, rng=rng)
    players = [Player(name=f"P{i}") for i in range(num_players)]
    return Game(board=board, players=players, rng=rng)


class TestGameInit:
    def test_game_starts_in_placement_phase(self):
        game = _make_game()
        assert game.phase == GamePhase.PLACEMENT

    def test_robber_starts_on_desert(self):
        game = _make_game()
        assert game.board.hexes[game.robber_hex].terrain.value == "desert"

    def test_dev_card_deck_has_25_cards(self):
        game = _make_game()
        assert len(game.dev_card_deck) == 25

    def test_players_start_with_no_resources(self):
        game = _make_game()
        for p in game.players:
            assert p.total_resource_count() == 0

    def test_board_has_9_ports(self):
        game = _make_game()
        assert len(game.board.ports) == 9


class TestPlacement:
    def test_placement_generates_settlement_actions(self):
        game = _make_game()
        actions = game.legal_actions()
        assert len(actions) > 0
        from catan.actions import BuildSettlement

        assert all(isinstance(a, BuildSettlement) for a in actions)

    def test_placement_settlement_then_road(self):
        game = _make_game()
        actions = game.legal_actions()
        # Place first settlement
        game.apply_action(actions[0])
        assert game.placement_step == 1
        # Next actions should be roads
        road_actions = game.legal_actions()
        from catan.actions import BuildRoad

        assert all(isinstance(a, BuildRoad) for a in road_actions)
        assert len(road_actions) > 0


class TestFullGame:
    def test_random_game_completes(self):
        """A game with random agents should terminate."""
        agents = [RandomAgent(random.Random(i)) for i in range(4)]
        result = run_game(agents, seed=42)
        assert result.turns > 0
        # Either someone won or the game hit the turn limit
        assert result.winner is not None or result.turns >= MAX_TURNS

    def test_multiple_random_games_complete(self):
        """Run 10 games and verify all complete without errors."""
        for seed in range(10):
            agents = [RandomAgent(random.Random(seed * 4 + i)) for i in range(4)]
            result = run_game(agents, seed=seed)
            assert result.turns > 0
            assert all(vp >= 0 for vp in result.victory_points)

    def test_three_player_game(self):
        """A 3-player game should also work."""
        agents = [RandomAgent(random.Random(i)) for i in range(3)]
        result = run_game(agents, seed=99)
        assert result.turns > 0
        assert len(result.victory_points) == 3
