"""Serialize/deserialize game state and actions to/from JSON-safe dicts."""

from __future__ import annotations

from collections import Counter

from catan.actions import (
    AcceptTrade,
    Action,
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
from catan.board import Board
from catan.game import Game
from catan.player import Player
from catan.resources import Resource


def serialize_game(game: Game, human_seats: set[int]) -> dict:
    """Serialize full game state to a JSON-safe dict."""
    return {
        "phase": game.phase.value,
        "turn": game.turn,
        "current_player_idx": game.current_player_idx,
        "last_roll": game.last_roll,
        "robber_hex": game.robber_hex,
        "longest_road_player": game.longest_road_player,
        "largest_army_player": game.largest_army_player,
        "board": serialize_board(game.board),
        "players": [
            serialize_player(p, is_human=(i in human_seats)) for i, p in enumerate(game.players)
        ],
        "vertex_owner": {str(k): v for k, v in game.vertex_owner.items()},
        "vertex_building": {str(k): v for k, v in game.vertex_building.items()},
        "edge_owner": {str(k): v for k, v in game.edge_owner.items()},
        "dev_cards_remaining": len(game.dev_card_deck),
    }


def serialize_board(board: Board) -> dict:
    """Serialize board layout (hexes, ports, topology vertex positions)."""
    hexes = []
    for idx, h in enumerate(board.hexes):
        hexes.append(
            {
                "terrain": h.terrain.value,
                "token": h.token,
                "coord": {"q": h.coord[0], "r": h.coord[1]},
                "hex_idx": idx,
            }
        )

    ports = []
    for p in board.ports:
        ports.append(
            {
                "port_type": p.port_type.value,
                "vertices": list(p.vertices),
            }
        )

    edge_vertices = [
        list(board.topology.edge_vertices[eid]) for eid in range(board.topology.num_edges)
    ]

    return {
        "hexes": hexes,
        "ports": ports,
        "topology": {
            "num_vertices": board.topology.num_vertices,
            "num_edges": board.topology.num_edges,
            "edge_vertices": edge_vertices,
            "hex_to_vertices": {
                str(k): list(v) for k, v in board.topology.hex_to_vertices.items()
            },
        },
    }


def serialize_player(player: Player, is_human: bool) -> dict:
    """Serialize player state. Hides dev card details for non-human seats."""
    resources = {r.value: c for r, c in player.resources.items() if c > 0}
    data = {
        "name": player.name,
        "victory_points": player.victory_points,
        "settlements": player.settlements,
        "cities": player.cities,
        "roads": player.roads,
        "played_knights": player.played_knights,
    }
    if is_human:
        data["resources"] = resources
        data["dev_cards"] = [c.value for c in player.dev_cards]
        data["new_dev_cards"] = [c.value for c in player.new_dev_cards]
    else:
        data["resource_count"] = sum(player.resources.values())
        data["dev_card_count"] = len(player.dev_cards) + len(player.new_dev_cards)
    return data


def serialize_action(action: Action) -> dict:
    """Serialize an action to a JSON-safe dict."""
    result: dict = {"type": type(action).__name__}
    match action:
        case BuildSettlement(vertex_id=v):
            result["vertex_id"] = v
        case BuildRoad(edge_id=e):
            result["edge_id"] = e
        case BuildCity(vertex_id=v):
            result["vertex_id"] = v
        case BankTrade(give=give, receive=recv):
            result["give"] = give.value
            result["receive"] = recv.value
        case ProposeTrade(offering=off, requesting=req, target_player_idx=t):
            result["offering"] = {r.value: c for r, c in off.items() if c > 0}
            result["requesting"] = {r.value: c for r, c in req.items() if c > 0}
            result["target_player_idx"] = t
        case AcceptTrade() | RejectTrade() | BuyDevCard() | EndTurn():
            pass
        case PlayKnight(target_hex=h, steal_from=s):
            result["target_hex"] = h
            result["steal_from"] = s
        case PlayRoadBuilding(edge1=e1, edge2=e2):
            result["edge1"] = e1
            result["edge2"] = e2
        case PlayYearOfPlenty(resource1=r1, resource2=r2):
            result["resource1"] = r1.value
            result["resource2"] = r2.value
        case PlayMonopoly(resource=r):
            result["resource"] = r.value
        case MoveRobber(target_hex=h, steal_from=s):
            result["target_hex"] = h
            result["steal_from"] = s
        case DiscardResources(resources=res):
            result["resources"] = {r.value: c for r, c in res.items() if c > 0}
    return result


_ACTION_TYPES: dict[str, type] = {
    "BuildSettlement": BuildSettlement,
    "BuildRoad": BuildRoad,
    "BuildCity": BuildCity,
    "BankTrade": BankTrade,
    "ProposeTrade": ProposeTrade,
    "AcceptTrade": AcceptTrade,
    "RejectTrade": RejectTrade,
    "BuyDevCard": BuyDevCard,
    "PlayKnight": PlayKnight,
    "PlayRoadBuilding": PlayRoadBuilding,
    "PlayYearOfPlenty": PlayYearOfPlenty,
    "PlayMonopoly": PlayMonopoly,
    "MoveRobber": MoveRobber,
    "DiscardResources": DiscardResources,
    "EndTurn": EndTurn,
}


def deserialize_action(data: dict) -> Action:
    """Deserialize a dict into an Action. Raises ValueError on invalid input."""
    action_type = data.get("type")
    if action_type not in _ACTION_TYPES:
        raise ValueError(f"Unknown action type: {action_type}")

    match action_type:
        case "BuildSettlement":
            return BuildSettlement(vertex_id=data["vertex_id"])
        case "BuildRoad":
            return BuildRoad(edge_id=data["edge_id"])
        case "BuildCity":
            return BuildCity(vertex_id=data["vertex_id"])
        case "BankTrade":
            return BankTrade(give=Resource(data["give"]), receive=Resource(data["receive"]))
        case "ProposeTrade":
            return ProposeTrade(
                offering=Counter({Resource(k): v for k, v in data["offering"].items()}),
                requesting=Counter({Resource(k): v for k, v in data["requesting"].items()}),
                target_player_idx=data["target_player_idx"],
            )
        case "AcceptTrade":
            return AcceptTrade()
        case "RejectTrade":
            return RejectTrade()
        case "BuyDevCard":
            return BuyDevCard()
        case "PlayKnight":
            return PlayKnight(target_hex=data["target_hex"], steal_from=data.get("steal_from"))
        case "PlayRoadBuilding":
            return PlayRoadBuilding(edge1=data["edge1"], edge2=data.get("edge2"))
        case "PlayYearOfPlenty":
            return PlayYearOfPlenty(
                resource1=Resource(data["resource1"]),
                resource2=Resource(data["resource2"]),
            )
        case "PlayMonopoly":
            return PlayMonopoly(resource=Resource(data["resource"]))
        case "MoveRobber":
            return MoveRobber(target_hex=data["target_hex"], steal_from=data.get("steal_from"))
        case "DiscardResources":
            return DiscardResources(
                resources=Counter({Resource(k): v for k, v in data["resources"].items()})
            )
        case "EndTurn":
            return EndTurn()
        case _:
            raise ValueError(f"Unknown action type: {action_type}")
