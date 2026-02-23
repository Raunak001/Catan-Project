"""Tests for longest road computation."""

from catan.board import Board
from catan.game import GamePhase
from catan.longest_road import compute_longest_road
from catan.topology import Topology

from .helpers import make_game, place_road_at, place_settlement_at


def _topo() -> Topology:
    board = Board.standard()
    return board.topology


class TestLongestRoadBasic:
    def test_no_roads_returns_zero(self):
        topo = _topo()
        assert compute_longest_road([], 0, topo, {}) == 0

    def test_single_road_returns_one(self):
        topo = _topo()
        assert compute_longest_road([0], 0, topo, {}) == 1

    def test_two_connected_roads(self):
        """Two roads sharing a vertex should give length 2."""
        topo = _topo()
        # Edge 0 connects vertices (a, b). Find another edge from vertex b.
        a, b = topo.edge_vertices[0]
        for eid in topo.vertex_to_edges[b]:
            if eid != 0:
                result = compute_longest_road([0, eid], 0, topo, {})
                assert result == 2
                return

    def test_disconnected_roads(self):
        """Two disconnected roads should give length 1 each (max = 1)."""
        topo = _topo()
        # Pick two edges that share no vertex
        e0_verts = set(topo.edge_vertices[0])
        for eid in range(1, topo.num_edges):
            e_verts = set(topo.edge_vertices[eid])
            if not e0_verts & e_verts:
                result = compute_longest_road([0, eid], 0, topo, {})
                assert result == 1
                return

    def test_linear_chain(self):
        """A chain of N connected roads should give length N."""
        topo = _topo()
        # Build a chain starting from edge 0
        chain = [0]
        a, b = topo.edge_vertices[0]
        current_vertex = b
        used_edges = {0}

        for _ in range(4):  # try to build a chain of 5
            found = False
            for eid in topo.vertex_to_edges[current_vertex]:
                if eid not in used_edges:
                    chain.append(eid)
                    used_edges.add(eid)
                    ea, eb = topo.edge_vertices[eid]
                    current_vertex = eb if ea == current_vertex else ea
                    found = True
                    break
            if not found:
                break

        expected = len(chain)
        result = compute_longest_road(chain, 0, topo, {})
        assert result == expected


class TestLongestRoadOpponentBlock:
    def test_opponent_settlement_breaks_path(self):
        """An opponent's settlement should break the road at that vertex."""
        topo = _topo()
        # Build a chain of 3 roads
        chain = [0]
        a, b = topo.edge_vertices[0]
        mid_vertex = b
        used_edges = {0}

        for _ in range(2):
            for eid in topo.vertex_to_edges[mid_vertex]:
                if eid not in used_edges:
                    chain.append(eid)
                    used_edges.add(eid)
                    ea, eb = topo.edge_vertices[eid]
                    next_v = eb if ea == mid_vertex else ea
                    if len(chain) == 2:
                        block_vertex = mid_vertex
                    mid_vertex = next_v
                    break

        # Without blocking, should be length 3
        result_unblocked = compute_longest_road(chain, 0, topo, {})
        assert result_unblocked == 3

        # Block the middle vertex with opponent settlement
        vertex_owner = {block_vertex: 1}  # opponent is player 1
        result_blocked = compute_longest_road(chain, 0, topo, vertex_owner)
        # The path is broken at block_vertex, so max is 2
        # (edge before block + edge to block, or edge after block + 1)
        assert result_blocked < result_unblocked

    def test_own_settlement_does_not_break_path(self):
        """Player's own settlement should not break their road."""
        topo = _topo()
        chain = [0]
        a, b = topo.edge_vertices[0]
        current = b
        used = {0}
        for _ in range(2):
            for eid in topo.vertex_to_edges[current]:
                if eid not in used:
                    chain.append(eid)
                    used.add(eid)
                    ea, eb = topo.edge_vertices[eid]
                    current = eb if ea == current else ea
                    break

        # Own settlement on the middle vertex
        vertex_owner = {b: 0}
        result = compute_longest_road(chain, 0, topo, vertex_owner)
        assert result == len(chain)


class TestLongestRoadInGame:
    def test_longest_road_awards_2_vp(self):
        """Player with 5+ road segments gets 2 VP bonus."""
        game = make_game()
        # Skip placement and set up manually
        game.phase = game.phase  # keep in placement for now

        # Build 5 connected roads for player 0
        topo = game.topology
        chain = [0]
        a, b = topo.edge_vertices[0]
        # Also place a settlement so the roads have a root
        place_settlement_at(game, 0, a)
        place_road_at(game, 0, 0)
        current = b
        used = {0}

        for _ in range(4):
            for eid in topo.vertex_to_edges[current]:
                if eid not in used and eid not in game.edge_owner:
                    place_road_at(game, 0, eid)
                    chain.append(eid)
                    used.add(eid)
                    ea, eb = topo.edge_vertices[eid]
                    current = eb if ea == current else ea
                    break

        game.phase = GamePhase.MAIN
        vp_before = game.players[0].victory_points
        game._update_longest_road()
        if len(chain) >= 5:
            assert game.longest_road_player == 0
            assert game.players[0].victory_points == vp_before + 2

    def test_longest_road_not_awarded_under_5(self):
        """4 roads should not award longest road."""
        game = make_game()
        topo = game.topology
        a, b = topo.edge_vertices[0]
        place_settlement_at(game, 0, a)
        place_road_at(game, 0, 0)
        current = b
        used = {0}

        for _ in range(3):  # total 4 roads
            for eid in topo.vertex_to_edges[current]:
                if eid not in used and eid not in game.edge_owner:
                    place_road_at(game, 0, eid)
                    used.add(eid)
                    ea, eb = topo.edge_vertices[eid]
                    current = eb if ea == current else ea
                    break

        game._update_longest_road()
        assert game.longest_road_player is None
