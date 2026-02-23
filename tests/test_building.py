"""Tests for building rules: settlements, cities, and roads."""

from collections import Counter

from catan.actions import BuildCity, BuildRoad, BuildSettlement
from catan.resources import Resource

from .helpers import (
    give_resources,
    make_game,
    skip_to_main_phase,
)


class TestSettlementBuilding:
    def test_settlement_requires_resources(self):
        """Cannot build settlement without resources."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        player.resources = Counter()  # no resources
        actions = game.legal_actions()
        assert not any(isinstance(a, BuildSettlement) for a in actions)

    def test_settlement_with_resources_on_road(self):
        """Can build settlement when affordable and on own road with distance rule."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(
            player,
            {
                Resource.WOOD: 5,
                Resource.BRICK: 5,
                Resource.SHEEP: 5,
                Resource.WHEAT: 5,
            },
        )

        actions = game.legal_actions()
        settlement_actions = [a for a in actions if isinstance(a, BuildSettlement)]
        for sa in settlement_actions:
            v = sa.vertex_id
            # Must satisfy distance rule
            for n in game.topology.vertex_neighbors[v]:
                assert n not in game.vertex_owner
            # Must be adjacent to own road
            assert any(e in set(player.roads) for e in game.topology.vertex_to_edges[v])

    def test_settlement_deducts_resources(self):
        """Building a settlement should deduct resources."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(
            player,
            {
                Resource.WOOD: 1,
                Resource.BRICK: 1,
                Resource.SHEEP: 1,
                Resource.WHEAT: 1,
            },
        )

        actions = game.legal_actions()
        settlement_actions = [a for a in actions if isinstance(a, BuildSettlement)]
        if settlement_actions:
            game.apply_action(settlement_actions[0])
            assert player.resources[Resource.WOOD] == 0
            assert player.resources[Resource.BRICK] == 0
            assert player.resources[Resource.SHEEP] == 0
            assert player.resources[Resource.WHEAT] == 0

    def test_settlement_grants_victory_point(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        vp_before = player.victory_points
        give_resources(
            player,
            {
                Resource.WOOD: 1,
                Resource.BRICK: 1,
                Resource.SHEEP: 1,
                Resource.WHEAT: 1,
            },
        )
        actions = [a for a in game.legal_actions() if isinstance(a, BuildSettlement)]
        if actions:
            game.apply_action(actions[0])
            assert player.victory_points == vp_before + 1

    def test_distance_rule_enforced(self):
        """No settlement action should violate the distance rule."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {r: 10 for r in Resource})
        actions = [a for a in game.legal_actions() if isinstance(a, BuildSettlement)]
        for a in actions:
            for n in game.topology.vertex_neighbors[a.vertex_id]:
                assert n not in game.vertex_owner, (
                    f"Settlement at {a.vertex_id} violates distance rule (neighbor {n} occupied)"
                )

    def test_settlement_requires_adjacent_road(self):
        """In main phase, settlements must be on the player's road network."""
        game = make_game()
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.current_player
        give_resources(player, {r: 10 for r in Resource})
        actions = [a for a in game.legal_actions() if isinstance(a, BuildSettlement)]
        road_set = set(player.roads)
        for a in actions:
            adjacent_edges = game.topology.vertex_to_edges[a.vertex_id]
            assert any(e in road_set for e in adjacent_edges), (
                f"Settlement at {a.vertex_id} has no adjacent road for player {pidx}"
            )


class TestCityBuilding:
    def test_city_upgrades_settlement(self):
        """Building a city should remove the settlement and add a city."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.WHEAT: 2, Resource.ORE: 3})

        city_actions = [a for a in game.legal_actions() if isinstance(a, BuildCity)]
        if city_actions:
            vid = city_actions[0].vertex_id
            assert vid in player.settlements
            game.apply_action(city_actions[0])
            assert vid not in player.settlements
            assert vid in player.cities
            assert game.vertex_building[vid] == "city"

    def test_city_grants_one_additional_vp(self):
        """City is 2 VP total, settlement was 1, so net gain is 1."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        vp_before = player.victory_points
        give_resources(player, {Resource.WHEAT: 2, Resource.ORE: 3})
        city_actions = [a for a in game.legal_actions() if isinstance(a, BuildCity)]
        if city_actions:
            game.apply_action(city_actions[0])
            assert player.victory_points == vp_before + 1

    def test_city_requires_existing_settlement(self):
        """Can only build city on own settlement."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.WHEAT: 10, Resource.ORE: 10})
        city_actions = [a for a in game.legal_actions() if isinstance(a, BuildCity)]
        for a in city_actions:
            assert a.vertex_id in player.settlements


class TestRoadBuilding:
    def test_road_requires_connection(self):
        """Roads must connect to existing roads or buildings."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        pidx = game.current_player_idx
        give_resources(player, {Resource.WOOD: 10, Resource.BRICK: 10})

        road_actions = [a for a in game.legal_actions() if isinstance(a, BuildRoad)]
        player_vertices = set(player.settlements) | set(player.cities)
        player_roads = set(player.roads)

        for a in road_actions:
            va, vb = game.topology.edge_vertices[a.edge_id]
            # At least one endpoint must connect to own building or road
            connected = False
            for v in (va, vb):
                if v in player_vertices:
                    connected = True
                    break
                owner = game.vertex_owner.get(v)
                if owner is not None and owner != pidx:
                    continue
                for adj_e in game.topology.vertex_to_edges[v]:
                    if adj_e in player_roads:
                        connected = True
                        break
                if connected:
                    break
            assert connected, f"Road {a.edge_id} not connected to player's network"

    def test_cannot_build_on_occupied_edge(self):
        """No road action should target an already-occupied edge."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.WOOD: 10, Resource.BRICK: 10})
        road_actions = [a for a in game.legal_actions() if isinstance(a, BuildRoad)]
        for a in road_actions:
            assert a.edge_id not in game.edge_owner

    def test_opponent_building_blocks_road_connection(self):
        """An opponent's settlement blocks road connectivity through that vertex."""
        game = make_game()
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.current_player

        # Find a vertex at the end of player's road network
        road_set = set(player.roads)
        for eid in player.roads:
            va, vb = game.topology.edge_vertices[eid]
            for v in (va, vb):
                if v not in game.vertex_owner:
                    # Place opponent settlement here to block
                    opp_idx = (pidx + 1) % game.num_players
                    game.vertex_owner[v] = opp_idx
                    game.vertex_building[v] = "settlement"
                    game.players[opp_idx].settlements.append(v)

                    # Now roads through this vertex should not be connectable
                    # (unless they connect to own building on the other side)
                    give_resources(player, {Resource.WOOD: 10, Resource.BRICK: 10})
                    road_actions = [a for a in game.legal_actions() if isinstance(a, BuildRoad)]
                    for ra in road_actions:
                        ea, eb = game.topology.edge_vertices[ra.edge_id]
                        # Verify the road doesn't connect through the blocked vertex
                        # unless it connects via a different path
                        if ea == v or eb == v:
                            other = eb if ea == v else ea
                            # This road's other endpoint must connect to player's
                            # own road/building without going through the blocked vertex
                            assert other in set(player.settlements) | set(player.cities) or any(
                                adj_e in road_set for adj_e in game.topology.vertex_to_edges[other]
                            ), "Road connected through blocked vertex"
                    # Cleanup
                    del game.vertex_owner[v]
                    del game.vertex_building[v]
                    game.players[opp_idx].settlements.remove(v)
                    return
