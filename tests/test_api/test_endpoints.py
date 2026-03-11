"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from catan.api.server import _sessions, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Clear session store between tests."""
    _sessions.clear()
    yield
    _sessions.clear()


def test_list_agents():
    resp = client.get("/agents")
    assert resp.status_code == 200
    agents = resp.json()["agents"]
    assert "random" in agents
    assert "hybridbot" in agents
    assert "smartbot" in agents


def test_create_game_success():
    resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "random"},
                {"seat": 2, "agent": "random"},
                {"seat": 3, "agent": "random"},
            ],
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "game_id" in data
    assert "state" in data
    assert "legal_actions" in data
    assert data["state"]["phase"] in (
        "placement",
        "main",
        "robber_discard",
        "robber_move",
    )


def test_create_game_invalid_seat_count():
    resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "random"},
            ],
        },
    )
    assert resp.status_code == 422  # Pydantic validation


def test_create_game_unknown_agent():
    resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "nonexistent_bot"},
                {"seat": 2, "agent": "random"},
                {"seat": 3, "agent": "random"},
            ],
        },
    )
    assert resp.status_code == 400


def test_get_game_not_found():
    resp = client.get("/games/nonexistent")
    assert resp.status_code == 404


def test_get_game_success():
    create_resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "random"},
                {"seat": 2, "agent": "random"},
                {"seat": 3, "agent": "random"},
            ],
            "seed": 42,
        },
    )
    game_id = create_resp.json()["game_id"]

    resp = client.get(f"/games/{game_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["game_id"] == game_id
    assert "stats_summary" in data


def test_submit_valid_action():
    """Create game and submit the first legal action."""
    create_resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "random"},
                {"seat": 2, "agent": "random"},
                {"seat": 3, "agent": "random"},
            ],
            "seed": 42,
        },
    )
    data = create_resp.json()
    game_id = data["game_id"]
    legal = data["legal_actions"]

    if legal:
        resp = client.post(f"/games/{game_id}/action", json={"action": legal[0]})
        assert resp.status_code == 200
        assert "state" in resp.json()


def test_submit_invalid_action_returns_400():
    create_resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "random"},
                {"seat": 2, "agent": "random"},
                {"seat": 3, "agent": "random"},
            ],
            "seed": 42,
        },
    )
    game_id = create_resp.json()["game_id"]

    resp = client.post(
        f"/games/{game_id}/action",
        json={
            "action": {"type": "FlyToMoon"},
        },
    )
    assert resp.status_code == 400


def test_get_stats_structure():
    create_resp = client.post(
        "/games",
        json={
            "seats": [
                {"seat": 0, "agent": "human"},
                {"seat": 1, "agent": "random"},
                {"seat": 2, "agent": "random"},
                {"seat": 3, "agent": "random"},
            ],
            "seed": 42,
        },
    )
    game_id = create_resp.json()["game_id"]

    resp = client.get(f"/games/{game_id}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_actions" in data
    assert "action_type_counts" in data
    assert "per_player" in data
    assert "vp_timeline" in data


def test_tournament_returns_results():
    resp = client.post(
        "/tournaments",
        json={
            "agents": ["random", "random", "random", "random"],
            "n_games": 2,
            "seed": 0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_games"] == 2
    assert len(data["wins"]) == 4
    assert sum(data["wins"]) + data["draws"] == 2


def test_tournament_unknown_agent():
    resp = client.post(
        "/tournaments",
        json={
            "agents": ["random", "nonexistent", "random", "random"],
            "n_games": 1,
        },
    )
    assert resp.status_code == 400
