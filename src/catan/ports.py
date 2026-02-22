"""Port definitions and standard board port layout.

Each port sits on the board perimeter and is accessible from 2 adjacent vertices.
Players with a settlement or city on a port vertex get improved trade rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from catan.resources import Resource


class PortType(Enum):
    GENERIC = "generic"  # 3:1 any resource
    WOOD = "wood"  # 2:1 wood
    BRICK = "brick"  # 2:1 brick
    SHEEP = "sheep"  # 2:1 sheep
    WHEAT = "wheat"  # 2:1 wheat
    ORE = "ore"  # 2:1 ore


PORT_RESOURCE: dict[PortType, Resource | None] = {
    PortType.GENERIC: None,
    PortType.WOOD: Resource.WOOD,
    PortType.BRICK: Resource.BRICK,
    PortType.SHEEP: Resource.SHEEP,
    PortType.WHEAT: Resource.WHEAT,
    PortType.ORE: Resource.ORE,
}

PORT_TRADE_RATE: dict[PortType, int] = {
    PortType.GENERIC: 3,
    PortType.WOOD: 2,
    PortType.BRICK: 2,
    PortType.SHEEP: 2,
    PortType.WHEAT: 2,
    PortType.ORE: 2,
}


@dataclass(frozen=True)
class Port:
    """A port on the board perimeter."""

    port_type: PortType
    vertices: tuple[int, int]  # the 2 vertex IDs that access this port
