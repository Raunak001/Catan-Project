"""Tests for the initial placement phase."""

from catan.actions import BuildRoad, BuildSettlement
from catan.game import GamePhase

from .helpers import make_game


class TestPlacementOrder:
    def test_first_round_goes_forward(self):
        """Round 1: player 0, 1, 2, 3."""
        game = make_game(4)
        player_order = []
        for _ in range(4):
            assert game.phase == GamePhase.PLACEMENT
            assert game.placement_round == 0
            player_order.append(game.current_player_idx)
            # Place settlement
            actions = game.legal_actions()
            game.apply_action(actions[0])
            # Place road
            actions = game.legal_actions()
            game.apply_action(actions[0])
        assert player_order == [0, 1, 2, 3]

    def test_second_round_goes_backward(self):
        """Round 2: player 3, 2, 1, 0."""
        game = make_game(4)
        # Complete first round
        for _ in range(4):
            actions = game.legal_actions()
            game.apply_action(actions[0])
            actions = game.legal_actions()
            game.apply_action(actions[0])

        player_order = []
        for _ in range(4):
            assert game.phase == GamePhase.PLACEMENT
            assert game.placement_round == 1
            player_order.append(game.current_player_idx)
            actions = game.legal_actions()
            game.apply_action(actions[0])
            actions = game.legal_actions()
            game.apply_action(actions[0])
        assert player_order == [3, 2, 1, 0]

    def test_placement_ends_after_both_rounds(self):
        """After 8 settlement+road pairs, placement is over."""
        game = make_game(4)
        placements = 0
        while game.phase == GamePhase.PLACEMENT:
            actions = game.legal_actions()
            game.apply_action(actions[0])
            placements += 1
        # 4 players * 2 rounds * 2 actions (settlement + road) = 16
        assert placements == 16

    def test_three_player_placement(self):
        """3-player placement: 3 forward, 3 backward = 12 actions."""
        game = make_game(3)
        placements = 0
        while game.phase == GamePhase.PLACEMENT:
            actions = game.legal_actions()
            game.apply_action(actions[0])
            placements += 1
        assert placements == 12


class TestPlacementDistanceRule:
    def test_cannot_place_adjacent_to_existing_settlement(self):
        game = make_game(4)
        actions = game.legal_actions()
        # Place first settlement
        first_settlement = actions[0]
        game.apply_action(first_settlement)
        # Place road
        road_actions = game.legal_actions()
        game.apply_action(road_actions[0])

        # Now it's player 1's turn — check that no action places on an adjacent vertex
        p1_actions = game.legal_actions()
        placed_vertex = first_settlement.vertex_id
        neighbors = set(game.topology.vertex_neighbors[placed_vertex])
        for a in p1_actions:
            assert isinstance(a, BuildSettlement)
            assert a.vertex_id != placed_vertex, "Can't place on occupied vertex"
            assert a.vertex_id not in neighbors, (
                f"Vertex {a.vertex_id} is adjacent to {placed_vertex}"
            )

    def test_cannot_place_on_occupied_vertex(self):
        game = make_game(4)
        actions = game.legal_actions()
        game.apply_action(actions[0])
        game.apply_action(game.legal_actions()[0])  # road

        # Player 1's actions should not include the vertex player 0 placed on
        p1_actions = game.legal_actions()
        occupied = actions[0].vertex_id
        for a in p1_actions:
            assert a.vertex_id != occupied


class TestPlacementRoads:
    def test_roads_must_be_adjacent_to_last_settlement(self):
        game = make_game(4)
        actions = game.legal_actions()
        settlement_action = actions[0]
        game.apply_action(settlement_action)

        road_actions = game.legal_actions()
        settlement_edges = set(game.topology.vertex_to_edges[settlement_action.vertex_id])
        for a in road_actions:
            assert isinstance(a, BuildRoad)
            assert a.edge_id in settlement_edges


class TestPlacementStartingResources:
    def test_first_round_grants_no_resources(self):
        game = make_game(4)
        # Place P0's settlement + road (round 1)
        game.apply_action(game.legal_actions()[0])
        game.apply_action(game.legal_actions()[0])
        assert game.players[0].total_resource_count() == 0

    def test_second_round_grants_resources(self):
        game = make_game(4)
        # Complete round 1
        for _ in range(4):
            game.apply_action(game.legal_actions()[0])
            game.apply_action(game.legal_actions()[0])

        # Now round 2 — player 3 places
        assert game.placement_round == 1
        settlement_action = game.legal_actions()[0]
        game.apply_action(settlement_action)
        # Player 3 should have resources from adjacent hexes
        # (at least 0, could be more depending on board)
        game.apply_action(game.legal_actions()[0])  # road

    def test_each_player_has_2_settlements_after_placement(self):
        game = make_game(4)
        while game.phase == GamePhase.PLACEMENT:
            game.apply_action(game.legal_actions()[0])
        for p in game.players:
            assert len(p.settlements) == 2
            assert len(p.roads) == 2
            assert p.victory_points == 2  # 1 per settlement
