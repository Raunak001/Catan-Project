"""Tests for the game runner module."""

from __future__ import annotations

import pytest

from catan.ai.heuristic import RandomAgent
from catan.game_runner import GameResult, TournamentResult, run_game, run_tournament

# ------------------------------------------------------------------ #
#  run_game                                                            #
# ------------------------------------------------------------------ #


class TestRunGame:
    """Test single game execution."""

    def test_returns_game_result(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=42)
        assert isinstance(result, GameResult)

    def test_game_has_winner_or_draw(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=42)
        # Winner is None (draw/timeout) or a valid player index
        assert result.winner is None or 0 <= result.winner < 4

    def test_game_turns_positive(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=42)
        assert result.turns > 0

    def test_victory_points_list(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=42)
        assert len(result.victory_points) == 4
        for vp in result.victory_points:
            assert vp >= 0

    def test_winner_has_highest_or_10_vp(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=42)
        if result.winner is not None:
            assert result.victory_points[result.winner] >= 10

    def test_deterministic_with_same_seed(self):
        import random

        # Create fresh agents with matching RNG state for each run
        agents1 = [RandomAgent(rng=random.Random(i)) for i in range(4)]
        r1 = run_game(agents1, seed=99)
        agents2 = [RandomAgent(rng=random.Random(i)) for i in range(4)]
        r2 = run_game(agents2, seed=99)
        assert r1.winner == r2.winner
        assert r1.turns == r2.turns
        assert r1.victory_points == r2.victory_points

    def test_different_seeds_may_differ(self):
        agents = [RandomAgent() for _ in range(4)]
        results = [run_game(agents, seed=s) for s in range(5)]
        # At least some games should have different outcomes
        winners = [r.winner for r in results]
        assert len(set(winners)) > 1 or len(set(r.turns for r in results)) > 1

    @pytest.mark.parametrize("seed", range(5))
    def test_no_crash_various_seeds(self, seed):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=seed)
        assert isinstance(result, GameResult)

    def test_unshuffled_board(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_game(agents, seed=42, shuffle_board=False)
        assert isinstance(result, GameResult)

    def test_three_player_game(self):
        agents = [RandomAgent() for _ in range(3)]
        result = run_game(agents, seed=42)
        assert isinstance(result, GameResult)
        assert len(result.victory_points) == 3


# ------------------------------------------------------------------ #
#  run_tournament                                                      #
# ------------------------------------------------------------------ #


class TestRunTournament:
    """Test tournament execution."""

    def test_returns_tournament_result(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=3, base_seed=0)
        assert isinstance(result, TournamentResult)

    def test_total_games_matches(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=5, base_seed=0)
        assert result.total_games == 5

    def test_wins_plus_draws_equals_total(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=5, base_seed=0)
        assert sum(result.wins) + result.draws == result.total_games

    def test_wins_list_length(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=3, base_seed=0)
        assert len(result.wins) == 4

    def test_avg_turns_positive(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=3, base_seed=0)
        assert result.avg_turns > 0

    def test_avg_vps_non_negative(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=3, base_seed=0)
        assert len(result.avg_vps) == 4
        for avg in result.avg_vps:
            assert avg >= 0

    def test_single_game_tournament(self):
        agents = [RandomAgent() for _ in range(4)]
        result = run_tournament(agents, n_games=1, base_seed=42)
        assert result.total_games == 1
        assert sum(result.wins) + result.draws == 1


# ------------------------------------------------------------------ #
#  Dataclass structure                                                 #
# ------------------------------------------------------------------ #


class TestDataclasses:
    """Test GameResult and TournamentResult structure."""

    def test_game_result_fields(self):
        r = GameResult(winner=0, turns=50, victory_points=[10, 5, 4, 3])
        assert r.winner == 0
        assert r.turns == 50
        assert r.victory_points == [10, 5, 4, 3]

    def test_tournament_result_fields(self):
        r = TournamentResult(
            wins=[3, 1, 0, 1],
            draws=0,
            total_games=5,
            avg_turns=100.0,
            avg_vps=[7.0, 5.0, 4.0, 4.0],
        )
        assert r.wins == [3, 1, 0, 1]
        assert r.draws == 0
        assert r.total_games == 5
        assert r.avg_turns == 100.0
