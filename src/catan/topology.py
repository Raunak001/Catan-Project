"""Hex grid topology: vertices and edges derived from axial hex coordinates.

A standard Catan board has 19 hexes, 54 vertices, and 72 edges.
Vertices sit at hex corners (settlements/cities); edges connect adjacent vertices (roads).

Each vertex is canonically identified as (q, r, d) where (q, r) is a hex axial
coordinate and d is 0 (south/"bottom") or 1 (north/"top"). Equivalent vertex
representations from neighboring hexes are mapped to the same integer ID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# -- Hex-to-pixel helpers (flat-top orientation) --

_SQRT3 = math.sqrt(3)


def _hex_to_pixel(q: int, r: int) -> tuple[float, float]:
    """Convert axial (q, r) to pixel center for a flat-top hex with size=1."""
    x = 3 / 2 * q
    y = _SQRT3 * (r + q / 2)
    return (x, y)


def _hex_corners(q: int, r: int) -> list[tuple[float, float]]:
    """Return the 6 corner pixel positions of hex (q, r), clockwise from 0°.

    For a flat-top hex with size=1:
      corner i is at angle (60° * i) from center.
      i=0: right, i=1: upper-right, i=2: upper-left,
      i=3: left,  i=4: lower-left,  i=5: lower-right
    """
    cx, cy = _hex_to_pixel(q, r)
    corners = []
    for i in range(6):
        angle_rad = math.radians(60 * i)
        corners.append((cx + math.cos(angle_rad), cy + math.sin(angle_rad)))
    return corners


def _round_pt(x: float, y: float, ndigits: int = 6) -> tuple[float, float]:
    """Round a point to avoid floating-point deduplication issues."""
    return (round(x, ndigits), round(y, ndigits))


@dataclass(frozen=True)
class Topology:
    """Precomputed vertex/edge structure for a set of hexes.

    All IDs are stable integer indices suitable for numpy encoding.
    """

    num_vertices: int
    num_edges: int

    # Vertex pixel positions (for optional rendering), indexed by vertex ID.
    vertex_positions: list[tuple[float, float]]

    # hex index -> list of 6 vertex IDs (clockwise from corner 0)
    hex_to_vertices: dict[int, list[int]]

    # vertex ID -> list of hex indices that touch this vertex
    vertex_to_hexes: dict[int, list[int]]

    # vertex ID -> list of neighboring vertex IDs (connected by an edge)
    vertex_neighbors: dict[int, list[int]]

    # edge ID -> (vertex_a, vertex_b)  with a < b
    edge_vertices: dict[int, tuple[int, int]]

    # vertex ID -> list of edge IDs incident to this vertex
    vertex_to_edges: dict[int, list[int]]

    # (vertex_a, vertex_b) -> edge ID  (a < b)
    vertex_pair_to_edge: dict[tuple[int, int], int] = field(repr=False)

    @classmethod
    def from_hex_coords(cls, hex_coords: list[tuple[int, int]]) -> Topology:
        """Compute topology from a list of axial hex coordinates.

        Parameters
        ----------
        hex_coords : list of (q, r) tuples – one per hex, order defines hex indices.
        """
        # 1. Compute all corner positions, deduplicate into vertex IDs.
        point_to_vertex: dict[tuple[float, float], int] = {}
        vertex_positions: list[tuple[float, float]] = []
        hex_to_vertices: dict[int, list[int]] = {}

        for hex_idx, (q, r) in enumerate(hex_coords):
            corners = _hex_corners(q, r)
            vids: list[int] = []
            for cx, cy in corners:
                pt = _round_pt(cx, cy)
                if pt not in point_to_vertex:
                    vid = len(vertex_positions)
                    point_to_vertex[pt] = vid
                    vertex_positions.append(pt)
                vids.append(point_to_vertex[pt])
            hex_to_vertices[hex_idx] = vids

        num_vertices = len(vertex_positions)

        # 2. Build vertex_to_hexes (invert hex_to_vertices).
        vertex_to_hexes: dict[int, list[int]] = {v: [] for v in range(num_vertices)}
        for hex_idx, vids in hex_to_vertices.items():
            for v in vids:
                vertex_to_hexes[v].append(hex_idx)

        # 3. Compute edges: each pair of consecutive corners on a hex is an edge.
        edge_set: set[tuple[int, int]] = set()
        for vids in hex_to_vertices.values():
            for i in range(6):
                a, b = vids[i], vids[(i + 1) % 6]
                edge_set.add((min(a, b), max(a, b)))

        edge_list = sorted(edge_set)
        edge_vertices: dict[int, tuple[int, int]] = {}
        vertex_pair_to_edge: dict[tuple[int, int], int] = {}
        for edge_id, (a, b) in enumerate(edge_list):
            edge_vertices[edge_id] = (a, b)
            vertex_pair_to_edge[(a, b)] = edge_id

        num_edges = len(edge_list)

        # 4. Build vertex_neighbors and vertex_to_edges.
        vertex_neighbors: dict[int, list[int]] = {v: [] for v in range(num_vertices)}
        vertex_to_edges: dict[int, list[int]] = {v: [] for v in range(num_vertices)}

        for edge_id, (a, b) in edge_vertices.items():
            vertex_neighbors[a].append(b)
            vertex_neighbors[b].append(a)
            vertex_to_edges[a].append(edge_id)
            vertex_to_edges[b].append(edge_id)

        return cls(
            num_vertices=num_vertices,
            num_edges=num_edges,
            vertex_positions=vertex_positions,
            hex_to_vertices=hex_to_vertices,
            vertex_to_hexes=vertex_to_hexes,
            vertex_neighbors=vertex_neighbors,
            edge_vertices=edge_vertices,
            vertex_to_edges=vertex_to_edges,
            vertex_pair_to_edge=vertex_pair_to_edge,
        )

    def edge_id(self, va: int, vb: int) -> int | None:
        """Return edge ID connecting two vertices, or None if not adjacent."""
        key = (min(va, vb), max(va, vb))
        return self.vertex_pair_to_edge.get(key)
