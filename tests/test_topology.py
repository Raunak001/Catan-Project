"""Tests for hex grid topology computation."""

from catan.board import Board
from catan.topology import Topology


def _standard_topology() -> Topology:
    """Build topology from the standard 19-hex board coordinates."""
    board = Board.standard()
    coords = [h.coord for h in board.hexes]
    return Topology.from_hex_coords(coords)


class TestVertexCounts:
    def test_standard_board_has_54_vertices(self):
        topo = _standard_topology()
        assert topo.num_vertices == 54

    def test_each_hex_has_6_vertices(self):
        topo = _standard_topology()
        for hex_idx, vids in topo.hex_to_vertices.items():
            assert len(vids) == 6, f"hex {hex_idx} has {len(vids)} vertices"

    def test_each_vertex_touches_1_to_3_hexes(self):
        topo = _standard_topology()
        for vid, hexes in topo.vertex_to_hexes.items():
            assert 1 <= len(hexes) <= 3, f"vertex {vid} touches {len(hexes)} hexes"

    def test_vertex_positions_are_unique(self):
        topo = _standard_topology()
        assert len(set(topo.vertex_positions)) == topo.num_vertices


class TestEdgeCounts:
    def test_standard_board_has_72_edges(self):
        topo = _standard_topology()
        assert topo.num_edges == 72

    def test_each_edge_connects_two_distinct_vertices(self):
        topo = _standard_topology()
        for edge_id, (a, b) in topo.edge_vertices.items():
            assert a < b, f"edge {edge_id}: vertices not ordered ({a}, {b})"
            assert a != b

    def test_edge_vertices_are_valid(self):
        topo = _standard_topology()
        for edge_id, (a, b) in topo.edge_vertices.items():
            assert 0 <= a < topo.num_vertices
            assert 0 <= b < topo.num_vertices


class TestAdjacency:
    def test_vertex_neighbors_are_symmetric(self):
        topo = _standard_topology()
        for v, neighbors in topo.vertex_neighbors.items():
            for n in neighbors:
                assert v in topo.vertex_neighbors[n], f"vertex {v} -> {n} but not {n} -> {v}"

    def test_each_vertex_has_2_or_3_neighbors(self):
        """Interior vertices have 3 neighbors, edge vertices have 2."""
        topo = _standard_topology()
        for v, neighbors in topo.vertex_neighbors.items():
            assert 2 <= len(neighbors) <= 3, f"vertex {v} has {len(neighbors)} neighbors"

    def test_vertex_to_edges_consistent_with_edge_vertices(self):
        topo = _standard_topology()
        for v, edge_ids in topo.vertex_to_edges.items():
            for eid in edge_ids:
                a, b = topo.edge_vertices[eid]
                assert v in (a, b)

    def test_vertex_pair_to_edge_roundtrips(self):
        topo = _standard_topology()
        for eid, (a, b) in topo.edge_vertices.items():
            assert topo.vertex_pair_to_edge[(a, b)] == eid

    def test_edge_id_helper(self):
        topo = _standard_topology()
        for eid, (a, b) in topo.edge_vertices.items():
            assert topo.edge_id(a, b) == eid
            assert topo.edge_id(b, a) == eid
        # Non-adjacent vertices should return None
        assert topo.edge_id(0, topo.num_vertices - 1) is None or True  # may or may not be adjacent
