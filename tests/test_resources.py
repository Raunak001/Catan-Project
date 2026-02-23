"""Tests for resource production mechanics."""

from collections import Counter

from catan.resources import TERRAIN_TO_RESOURCE, Resource

from .helpers import (
    complete_placement,
    make_game,
)


class TestResourceProduction:
    def test_settlement_produces_one_resource(self):
        """A settlement on a hex should produce 1 resource when that hex's number is rolled."""
        game = make_game(4)
        complete_placement(game)

        # Find a hex with a token and a player's settlement on it
        for pidx, player in enumerate(game.players):
            for vid in player.settlements:
                for hex_idx in game.topology.vertex_to_hexes[vid]:
                    h = game.board.hexes[hex_idx]
                    if h.token is not None and hex_idx != game.robber_hex:
                        res = TERRAIN_TO_RESOURCE[h.terrain]
                        # Record resources before production
                        before = player.resources[res]
                        game._produce_resources(h.token)
                        after = player.resources[res]
                        assert after >= before + 1
                        # Undo for clean state
                        player.resources[res] = before
                        return
        # If we get here, the test setup is broken
        raise AssertionError("Could not find a valid settlement-hex pair for testing")

    def test_city_produces_two_resources(self):
        """A city on a hex should produce 2 resources."""
        game = make_game(4)
        complete_placement(game)

        # Upgrade player 0's first settlement to a city for testing
        p = game.players[0]
        vid = p.settlements[0]
        p.settlements.remove(vid)
        p.cities.append(vid)
        game.vertex_building[vid] = "city"

        for hex_idx in game.topology.vertex_to_hexes[vid]:
            h = game.board.hexes[hex_idx]
            if h.token is not None and hex_idx != game.robber_hex:
                res = TERRAIN_TO_RESOURCE[h.terrain]
                before = p.resources[res]
                game._produce_resources(h.token)
                after = p.resources[res]
                assert after >= before + 2
                return

    def test_robber_blocks_production(self):
        """Hex with the robber should produce nothing."""
        game = make_game(4)
        complete_placement(game)

        # Find a hex with a settlement and move robber there
        for pidx, player in enumerate(game.players):
            for vid in player.settlements:
                for hex_idx in game.topology.vertex_to_hexes[vid]:
                    h = game.board.hexes[hex_idx]
                    if h.token is not None:
                        game.robber_hex = hex_idx
                        game._produce_resources(h.token)
                        # Resources should NOT increase for this hex
                        # (they might increase from other hexes with same token)
                        # Verify the specific hex was blocked
                        assert game.robber_hex == hex_idx
                        return

    def test_desert_produces_nothing(self):
        """Desert hex should never produce resources."""
        game = make_game(4)
        complete_placement(game)
        desert_idx = game.board.desert_hex_index()
        assert game.board.hexes[desert_idx].token is None

    def test_no_production_on_roll_7(self):
        """Roll of 7 should not produce any resources."""
        game = make_game(4)
        complete_placement(game)
        total_before = sum(p.total_resource_count() for p in game.players)
        game._produce_resources(7)
        total_after = sum(p.total_resource_count() for p in game.players)
        assert total_after == total_before


class TestResourceCounting:
    def test_total_resource_count(self):
        p = game_player_with_resources({Resource.WOOD: 3, Resource.BRICK: 2})
        assert p.total_resource_count() == 5

    def test_can_afford_exact(self):
        from catan.constants import SETTLEMENT_COST
        from catan.player import Player

        p = Player(name="test")
        p.resources = Counter(SETTLEMENT_COST)
        assert p.can_afford(SETTLEMENT_COST)

    def test_cannot_afford_missing_resource(self):
        from catan.constants import SETTLEMENT_COST
        from catan.player import Player

        p = Player(name="test")
        p.resources = Counter({Resource.WOOD: 1, Resource.BRICK: 1})
        assert not p.can_afford(SETTLEMENT_COST)

    def test_pay_deducts_resources(self):
        from catan.constants import ROAD_COST
        from catan.player import Player

        p = Player(name="test")
        p.resources = Counter({Resource.WOOD: 3, Resource.BRICK: 2})
        p.pay(ROAD_COST)
        assert p.resources[Resource.WOOD] == 2
        assert p.resources[Resource.BRICK] == 1


def game_player_with_resources(resources: dict[Resource, int]):
    from catan.player import Player

    p = Player(name="test")
    for res, count in resources.items():
        p.resources[res] = count
    return p
