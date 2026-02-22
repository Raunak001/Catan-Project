"""Resource types and trading logic."""

from enum import Enum


class Resource(Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"


class Terrain(Enum):
    FOREST = "forest"  # produces WOOD
    HILLS = "hills"  # produces BRICK
    PASTURE = "pasture"  # produces SHEEP
    FIELDS = "fields"  # produces WHEAT
    MOUNTAINS = "mountains"  # produces ORE
    DESERT = "desert"  # produces nothing


TERRAIN_TO_RESOURCE: dict[Terrain, Resource | None] = {
    Terrain.FOREST: Resource.WOOD,
    Terrain.HILLS: Resource.BRICK,
    Terrain.PASTURE: Resource.SHEEP,
    Terrain.FIELDS: Resource.WHEAT,
    Terrain.MOUNTAINS: Resource.ORE,
    Terrain.DESERT: None,
}
