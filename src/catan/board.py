"""Board representation: hexes, vertices, edges, and ports."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from catan.ports import Port, PortType
from catan.resources import Terrain
from catan.topology import Topology


@dataclass
class Hex:
    """A single hex tile on the board."""

    terrain: Terrain
    token: int | None  # dice number (2-12), None for desert
    coord: tuple[int, int] = (0, 0)  # axial coordinates (q, r)


# Standard 3-4-5-4-3 axial coordinate layout.
STANDARD_AXIAL_COORDS: list[tuple[int, int]] = [
    (0, -2),
    (1, -2),
    (2, -2),
    (-1, -1),
    (0, -1),
    (1, -1),
    (2, -1),
    (-2, 0),
    (-1, 0),
    (0, 0),
    (1, 0),
    (2, 0),
    (-2, 1),
    (-1, 1),
    (0, 1),
    (1, 1),
    (-2, 2),
    (-1, 2),
    (0, 2),
]

STANDARD_TOKENS: list[int] = [
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
]

STANDARD_TERRAINS: list[Terrain] = (
    [Terrain.FOREST] * 4
    + [Terrain.HILLS] * 3
    + [Terrain.PASTURE] * 4
    + [Terrain.FIELDS] * 4
    + [Terrain.MOUNTAINS] * 3
    + [Terrain.DESERT] * 1
)

# Standard port layout: (port_type, (hex_edge_vertex_pair)).
# These are defined as pairs of vertex indices that will be resolved
# after topology is computed. The ordering follows the standard Catan
# board perimeter clockwise. We store them as "edge-of-board" corner indices
# from specific hex coordinates (the perimeter hexes).
#
# Standard Catan port positions (9 ports, clockwise from top):
# We define each port by the hex it's adjacent to and which edge (corner pair).
# Format: (port_type, hex_coord, corner_index_a, corner_index_b)
_STANDARD_PORT_DEFS: list[tuple[PortType, tuple[int, int], int, int]] = [
    # Top edge
    (PortType.GENERIC, (0, -2), 1, 2),  # top-left hex, upper-right edge
    (PortType.WHEAT, (2, -2), 0, 1),  # top-right hex, right-upper edge
    # Right edge
    (PortType.ORE, (2, -1), 0, 5),  # right side
    (PortType.GENERIC, (2, 0), 5, 4),  # right-middle
    # Bottom-right
    (PortType.SHEEP, (1, 1), 4, 5),  # lower-right area  (was (2,0))
    (PortType.GENERIC, (-1, 2), 4, 3),  # bottom
    # Bottom-left / left
    (PortType.BRICK, (-2, 2), 3, 4),  # lower-left
    (PortType.WOOD, (-2, 1), 2, 3),  # left side
    (PortType.GENERIC, (-2, 0), 2, 1),  # left-upper (was (-1,-1))
]


@dataclass
class Board:
    """The full Catan board state."""

    hexes: list[Hex] = field(default_factory=list)
    topology: Topology = field(init=False, repr=False)
    ports: list[Port] = field(init=False, default_factory=list)

    # Fast lookup: dice token -> list of hex indices with that token.
    token_to_hexes: dict[int, list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        coords = [h.coord for h in self.hexes]
        self.topology = Topology.from_hex_coords(coords)
        self._build_token_map()
        self._build_ports()

    def _build_token_map(self) -> None:
        self.token_to_hexes = {}
        for idx, h in enumerate(self.hexes):
            if h.token is not None:
                self.token_to_hexes.setdefault(h.token, []).append(idx)

    def _build_ports(self) -> None:
        """Resolve port vertex IDs from hex coordinates and corner indices."""
        coord_to_idx = {h.coord: i for i, h in enumerate(self.hexes)}
        self.ports = []
        for port_type, hex_coord, ca, cb in _STANDARD_PORT_DEFS:
            hex_idx = coord_to_idx.get(hex_coord)
            if hex_idx is None:
                continue
            vids = self.topology.hex_to_vertices[hex_idx]
            self.ports.append(Port(port_type=port_type, vertices=(vids[ca], vids[cb])))

    @classmethod
    def standard(cls, shuffle: bool = False, rng: random.Random | None = None) -> Board:
        """Create a standard 19-hex Catan board.

        Parameters
        ----------
        shuffle : If True, randomize terrain and token placement (for RL training).
        rng : Optional random.Random instance for reproducibility.
        """
        terrains = list(STANDARD_TERRAINS)
        tokens = list(STANDARD_TOKENS)

        if shuffle:
            _rng = rng or random.Random()
            _rng.shuffle(terrains)
            _rng.shuffle(tokens)

        hexes = []
        token_iter = iter(tokens)
        for terrain, coord in zip(terrains, STANDARD_AXIAL_COORDS):
            token = None if terrain == Terrain.DESERT else next(token_iter)
            hexes.append(Hex(terrain=terrain, token=token, coord=coord))

        return cls(hexes=hexes)

    def desert_hex_index(self) -> int:
        """Return the index of the desert hex."""
        for i, h in enumerate(self.hexes):
            if h.terrain == Terrain.DESERT:
                return i
        raise ValueError("No desert hex found")
