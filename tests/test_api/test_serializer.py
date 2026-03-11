"""Tests for the serializer module."""

import json
from collections import Counter

from catan.actions import (
    AcceptTrade,
    BankTrade,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevCard,
    DiscardResources,
    EndTurn,
    MoveRobber,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    ProposeTrade,
    RejectTrade,
)
from catan.api.serializer import (
    deserialize_action,
    serialize_action,
    serialize_game,
    serialize_player,
)
from catan.player import Player
from catan.resources import Resource
from tests.helpers import make_game, skip_to_main_phase


def test_serialize_game_json_safe():
    """Serialized game state has no Enum or Counter keys — fully JSON-safe."""
    game = make_game()
    skip_to_main_phase(game)
    data = serialize_game(game, human_seats={0})
    # If this doesn't raise, the output is JSON-serializable
    json.dumps(data)


def test_roundtrip_all_action_types():
    """Each action type survives serialize -> deserialize roundtrip."""
    actions = [
        BuildSettlement(vertex_id=10),
        BuildRoad(edge_id=5),
        BuildCity(vertex_id=10),
        BankTrade(give=Resource.WOOD, receive=Resource.ORE),
        ProposeTrade(
            offering=Counter({Resource.WOOD: 2}),
            requesting=Counter({Resource.ORE: 1}),
            target_player_idx=1,
        ),
        AcceptTrade(),
        RejectTrade(),
        BuyDevCard(),
        PlayKnight(target_hex=5, steal_from=2),
        PlayKnight(target_hex=5, steal_from=None),
        PlayRoadBuilding(edge1=3, edge2=7),
        PlayRoadBuilding(edge1=3, edge2=None),
        PlayYearOfPlenty(resource1=Resource.SHEEP, resource2=Resource.WHEAT),
        PlayMonopoly(resource=Resource.ORE),
        MoveRobber(target_hex=8, steal_from=1),
        MoveRobber(target_hex=8, steal_from=None),
        DiscardResources(resources=Counter({Resource.WOOD: 2, Resource.BRICK: 1})),
        EndTurn(),
    ]
    for action in actions:
        serialized = serialize_action(action)
        # Must be JSON-safe
        json.dumps(serialized)
        deserialized = deserialize_action(serialized)
        assert deserialized == action, f"Roundtrip failed for {type(action).__name__}"


def test_deserialize_unknown_type_raises():
    """Unknown action type raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown action type"):
        deserialize_action({"type": "FlyToMoon"})


def test_discard_resources_counter_roundtrip():
    """DiscardResources with Enum-keyed Counter survives roundtrip."""
    action = DiscardResources(resources=Counter({Resource.WOOD: 3, Resource.ORE: 1}))
    serialized = serialize_action(action)
    assert "wood" in serialized["resources"]
    assert "ore" in serialized["resources"]
    deserialized = deserialize_action(serialized)
    assert deserialized == action


def test_bot_player_hides_dev_cards():
    """Bot player serialization hides dev card details, shows only count."""
    from catan.dev_cards import DevCardType

    player = Player(name="Bot")
    player.dev_cards = [DevCardType.KNIGHT, DevCardType.MONOPOLY]
    player.new_dev_cards = [DevCardType.ROAD_BUILDING]

    data = serialize_player(player, is_human=False)
    assert "dev_cards" not in data
    assert "new_dev_cards" not in data
    assert data["dev_card_count"] == 3


def test_human_player_shows_resources_and_cards():
    """Human player serialization includes resources and dev card details."""
    from catan.dev_cards import DevCardType

    player = Player(name="Human")
    player.resources[Resource.WOOD] = 3
    player.resources[Resource.ORE] = 1
    player.dev_cards = [DevCardType.KNIGHT]

    data = serialize_player(player, is_human=True)
    assert data["resources"] == {"wood": 3, "ore": 1}
    assert data["dev_cards"] == ["knight"]
