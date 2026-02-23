"""Tests for bank trading and port mechanics."""

from collections import Counter

from catan.actions import BankTrade
from catan.game import GamePhase
from catan.ports import PortType
from catan.resources import Resource

from .helpers import (
    give_resources,
    make_game,
    skip_to_main_phase,
)


class TestBankTrade:
    def test_default_rate_is_4_to_1(self):
        """Without ports, bank trade requires 4 of one resource."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        player.resources = Counter({Resource.WOOD: 4})

        actions = game.legal_actions()
        trade_actions = [a for a in actions if isinstance(a, BankTrade)]
        wood_trades = [a for a in trade_actions if a.give == Resource.WOOD]
        assert len(wood_trades) == 4  # can get any of the other 4 resources

    def test_cannot_trade_with_fewer_than_4(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        player.resources = Counter({Resource.WOOD: 3})

        # Check player has no port access for wood
        pidx = game.current_player_idx
        rate = game._trade_rate(pidx, Resource.WOOD)
        if rate == 4:  # only test if no port
            actions = game.legal_actions()
            trade_actions = [
                a for a in actions if isinstance(a, BankTrade) and a.give == Resource.WOOD
            ]
            assert len(trade_actions) == 0

    def test_trade_deducts_and_grants_resources(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        player.resources = Counter({Resource.WOOD: 4})
        pidx = game.current_player_idx
        rate = game._trade_rate(pidx, Resource.WOOD)

        game.apply_action(BankTrade(Resource.WOOD, Resource.ORE))
        assert player.resources[Resource.WOOD] == 4 - rate
        assert player.resources[Resource.ORE] == 1

    def test_cannot_trade_same_resource(self):
        """Cannot give and receive the same resource."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {r: 10 for r in Resource})
        actions = game.legal_actions()
        trade_actions = [a for a in actions if isinstance(a, BankTrade)]
        for a in trade_actions:
            assert a.give != a.receive


class TestPortTrading:
    def test_generic_port_gives_3_to_1(self):
        """A player on a generic port should trade at 3:1."""
        game = make_game()
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.current_player

        # Find a generic port and place settlement there
        for port in game.board.ports:
            if port.port_type == PortType.GENERIC:
                vid = port.vertices[0]
                if vid not in game.vertex_owner:
                    player.settlements.append(vid)
                    game.vertex_owner[vid] = pidx
                    game.vertex_building[vid] = "settlement"
                    break

        rate = game._trade_rate(pidx, Resource.WOOD)
        assert rate == 3

    def test_specialized_port_gives_2_to_1(self):
        """A player on a wood port should trade wood at 2:1."""
        game = make_game()
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.current_player

        # Find a wood port
        for port in game.board.ports:
            if port.port_type == PortType.WOOD:
                vid = port.vertices[0]
                if vid not in game.vertex_owner:
                    player.settlements.append(vid)
                    game.vertex_owner[vid] = pidx
                    game.vertex_building[vid] = "settlement"
                    break

        rate = game._trade_rate(pidx, Resource.WOOD)
        assert rate == 2

    def test_specialized_port_only_affects_its_resource(self):
        """A wood port should not give 2:1 for brick."""
        from catan.board import Board
        from catan.game import Game
        from catan.player import Player

        # Use a fresh game with no placement to avoid stale port access
        board = Board.standard()
        players = [Player(name=f"P{i}") for i in range(4)]
        game = Game(board=board, players=players, phase=GamePhase.MAIN)
        pidx = 0
        player = players[0]

        # Find a wood port vertex NOT shared with any other port
        wood_port = next(p for p in game.board.ports if p.port_type == PortType.WOOD)
        other_port_verts: set[int] = set()
        for p in game.board.ports:
            if p.port_type != PortType.WOOD:
                other_port_verts.update(p.vertices)

        exclusive_vid = None
        for vid in wood_port.vertices:
            if vid not in other_port_verts:
                exclusive_vid = vid
                break

        if exclusive_vid is not None:
            player.settlements.append(exclusive_vid)
            game.vertex_owner[exclusive_vid] = pidx
            game.vertex_building[exclusive_vid] = "settlement"

            wood_rate = game._trade_rate(pidx, Resource.WOOD)
            brick_rate = game._trade_rate(pidx, Resource.BRICK)
            assert wood_rate == 2
            assert brick_rate == 4  # no brick port, no generic port
        else:
            # Both wood port vertices overlap with other ports — test that
            # wood is still 2:1 and the non-overlapping resource is higher
            vid = wood_port.vertices[0]
            player.settlements.append(vid)
            game.vertex_owner[vid] = pidx
            game.vertex_building[vid] = "settlement"
            wood_rate = game._trade_rate(pidx, Resource.WOOD)
            assert wood_rate == 2
            # Sheep should not be 2:1 if there's no sheep port overlap
            sheep_rate = game._trade_rate(pidx, Resource.SHEEP)
            assert sheep_rate >= 3

    def test_port_access_with_city(self):
        """Cities on port vertices should also grant port access."""
        game = make_game()
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.current_player

        for port in game.board.ports:
            if port.port_type == PortType.GENERIC:
                vid = port.vertices[0]
                if vid not in game.vertex_owner:
                    player.cities.append(vid)
                    game.vertex_owner[vid] = pidx
                    game.vertex_building[vid] = "city"
                    break

        rate = game._trade_rate(pidx, Resource.BRICK)
        assert rate == 3

    def test_board_has_correct_port_distribution(self):
        """9 ports: 4 generic, 5 specialized (1 per resource)."""
        game = make_game()
        port_types = Counter(p.port_type for p in game.board.ports)
        assert port_types[PortType.GENERIC] == 4
        assert port_types[PortType.WOOD] == 1
        assert port_types[PortType.BRICK] == 1
        assert port_types[PortType.SHEEP] == 1
        assert port_types[PortType.WHEAT] == 1
        assert port_types[PortType.ORE] == 1

    def test_each_port_has_two_distinct_vertices(self):
        game = make_game()
        for port in game.board.ports:
            v0, v1 = port.vertices
            assert v0 != v1
            assert 0 <= v0 < game.topology.num_vertices
            assert 0 <= v1 < game.topology.num_vertices
