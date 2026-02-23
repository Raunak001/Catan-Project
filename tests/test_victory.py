"""Tests for victory condition detection."""

from catan.constants import VICTORY_POINTS_TO_WIN
from catan.game import GamePhase

from .helpers import make_game, place_city_at, place_settlement_at, skip_to_main_phase


class TestVictoryDetection:
    def test_no_winner_at_start(self):
        game = make_game()
        assert game.check_victory() is None

    def test_no_winner_after_placement(self):
        """After placement each player has 2 VP — no winner."""
        game = make_game()
        skip_to_main_phase(game)
        assert game.check_victory() is None
        for p in game.players:
            assert p.victory_points == 2

    def test_winner_at_10_vp(self):
        game = make_game()
        skip_to_main_phase(game)
        game.players[0].victory_points = VICTORY_POINTS_TO_WIN
        assert game.check_victory() == 0

    def test_winner_above_10_vp(self):
        game = make_game()
        skip_to_main_phase(game)
        game.players[2].victory_points = 12
        assert game.check_victory() == 2

    def test_first_player_checked_wins_on_tie(self):
        """If multiple players have 10+ VP, lowest index wins (checked first)."""
        game = make_game()
        skip_to_main_phase(game)
        game.players[1].victory_points = 10
        game.players[3].victory_points = 10
        assert game.check_victory() == 1

    def test_step_ends_game_on_victory(self):
        """game.step() should set phase to FINISHED when someone wins."""
        from catan.actions import EndTurn

        game = make_game()
        skip_to_main_phase(game)
        game.players[0].victory_points = VICTORY_POINTS_TO_WIN
        game_over, winner = game.step(EndTurn())
        # EndTurn advances to next player, but check_victory runs after
        # The winner check happens after apply_action
        assert game.phase == GamePhase.FINISHED

    def test_game_ends_at_max_turns(self):
        """Game should end when MAX_TURNS is reached."""
        from catan.constants import MAX_TURNS

        game = make_game()
        skip_to_main_phase(game)
        game.turn = MAX_TURNS - 1
        # Trigger next turn
        game._start_next_turn()
        assert game.phase == GamePhase.FINISHED


class TestVictoryPointAccounting:
    def test_settlement_gives_1_vp(self):
        game = make_game()
        p = game.players[0]
        vp_before = p.victory_points
        place_settlement_at(game, 0, 0)
        assert p.victory_points == vp_before + 1

    def test_city_gives_2_vp(self):
        game = make_game()
        p = game.players[0]
        vp_before = p.victory_points
        place_city_at(game, 0, 5)
        assert p.victory_points == vp_before + 2

    def test_city_upgrade_net_1_vp(self):
        """Upgrading settlement to city: settlement was +1, city is +2, net change is +1."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player

        # Player already has settlements from placement
        if player.settlements:
            vid = player.settlements[0]
            vp_before = player.victory_points
            # Manually upgrade
            from catan.resources import Resource

            player.resources.update({Resource.WHEAT: 2, Resource.ORE: 3})
            from catan.actions import BuildCity

            game.apply_action(BuildCity(vid))
            assert player.victory_points == vp_before + 1
