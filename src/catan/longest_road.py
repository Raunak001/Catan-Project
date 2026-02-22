"""Longest road computation via DFS.

In Catan, longest road is the longest simple path through a player's
connected road segments. Opponent settlements/cities break the path.
"""

from __future__ import annotations

from catan.topology import Topology


def compute_longest_road(
    player_roads: list[int],
    player_idx: int,
    topology: Topology,
    vertex_owner: dict[int, int | None],
) -> int:
    """Return the length of the longest road for a player.

    Parameters
    ----------
    player_roads : Edge IDs owned by this player.
    player_idx : This player's index (to allow traversal through own buildings).
    topology : Board topology.
    vertex_owner : Mapping of vertex_id -> owning player index (or None).
    """
    if not player_roads:
        return 0

    # Build adjacency graph: vertex -> set of vertices connected by this player's roads.
    road_set = set(player_roads)
    adj: dict[int, set[int]] = {}
    for eid in player_roads:
        a, b = topology.edge_vertices[eid]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    best = 0

    def dfs(v: int, visited_edges: set[int], length: int) -> None:
        nonlocal best
        best = max(best, length)
        for neighbor in adj.get(v, ()):
            eid = topology.edge_id(v, neighbor)
            if eid is None or eid not in road_set or eid in visited_edges:
                continue
            # Opponent building on the vertex blocks traversal
            owner = vertex_owner.get(neighbor)
            if owner is not None and owner != player_idx:
                # We can count this edge but can't continue past it
                best = max(best, length + 1)
                continue
            visited_edges.add(eid)
            dfs(neighbor, visited_edges, length + 1)
            visited_edges.remove(eid)

    # Try starting from every vertex that has at least one road
    for start_vertex in adj:
        dfs(start_vertex, set(), 0)

    return best
