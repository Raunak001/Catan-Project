"""Tests for the ActionTracker."""

from catan.actions import BankTrade, EndTurn
from catan.api.action_tracker import ActionTracker
from catan.game import GamePhase
from catan.resources import Resource
from tests.helpers import make_game, skip_to_main_phase


def test_record_captures_pre_apply_phase():
    """EndTurn records MAIN phase (before apply), not ROLL."""
    game = make_game()
    skip_to_main_phase(game)
    assert game.phase == GamePhase.MAIN

    tracker = ActionTracker()
    tracker.record(game, game.current_player_idx, EndTurn())
    assert tracker.records[0].phase == "main"


def test_stats_counts_sum_to_total():
    """Sum of action_type_counts equals total_actions."""
    game = make_game()
    skip_to_main_phase(game)
    tracker = ActionTracker()

    for _ in range(5):
        tracker.record(game, 0, EndTurn())

    stats = tracker.get_stats()
    assert sum(stats.action_type_counts.values()) == stats.total_actions
    assert stats.total_actions == 5


def test_per_player_breakdown_correct():
    """Per-player action counts match expected distribution."""
    game = make_game()
    skip_to_main_phase(game)
    tracker = ActionTracker()

    tracker.record(game, 0, EndTurn())
    tracker.record(game, 0, EndTurn())
    tracker.record(game, 1, EndTurn())

    stats = tracker.get_stats()
    assert stats.per_player[0]["EndTurn"] == 2
    assert stats.per_player[1]["EndTurn"] == 1


def test_vp_timeline_captures_snapshots():
    """VP timeline captures per-turn VP snapshots."""
    game = make_game()
    skip_to_main_phase(game)
    tracker = ActionTracker()

    tracker.record(game, 0, EndTurn())
    stats = tracker.get_stats()
    assert len(stats.vp_timeline) >= 1
    assert "turn" in stats.vp_timeline[0]
    assert "vps" in stats.vp_timeline[0]
    assert len(stats.vp_timeline[0]["vps"]) == 4


def test_trades_counted():
    """BankTrade actions are counted in trades_per_player."""
    game = make_game()
    skip_to_main_phase(game)
    tracker = ActionTracker()

    trade = BankTrade(give=Resource.WOOD, receive=Resource.ORE)
    tracker.record(game, 0, trade)
    tracker.record(game, 0, trade)
    tracker.record(game, 1, trade)

    stats = tracker.get_stats()
    assert stats.trades_per_player[0] == 2
    assert stats.trades_per_player[1] == 1


def test_robber_moves_counted():
    """MoveRobber and PlayKnight both increment robber_moves."""
    from catan.actions import MoveRobber, PlayKnight

    game = make_game()
    skip_to_main_phase(game)
    tracker = ActionTracker()

    tracker.record(game, 0, MoveRobber(target_hex=5, steal_from=None))
    tracker.record(game, 0, PlayKnight(target_hex=3, steal_from=1))

    stats = tracker.get_stats()
    assert stats.robber_moves == 2
