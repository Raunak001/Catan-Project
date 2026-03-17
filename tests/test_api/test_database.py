"""Tests for SQLite persistence: database CRUD and session blob round-trips."""

from __future__ import annotations

import pytest

from catan.api.database import (
    delete_session,
    get_connection,
    init_db,
    list_sessions,
    list_tournaments,
    load_session_row,
    save_session,
    save_tournament,
)
from catan.api.session import (
    GameSession,
    create_session,
    session_from_blob,
    session_to_blob,
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    """Point the database at a temp directory for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    init_db()


# ---------------------------------------------------------------------------
# Database CRUD
# ---------------------------------------------------------------------------


class TestGameSessionCRUD:
    def test_save_and_load(self):
        with get_connection() as conn:
            save_session(
                conn,
                game_id="test123",
                seat_config=[{"seat": 0, "agent": "human"}],
                seed=42,
                finished=False,
                winner=None,
                state_blob={"hello": "world"},
                tracker_blob=[{"turn": 0}],
            )
            row = load_session_row(conn, "test123")

        assert row is not None
        assert row["game_id"] == "test123"
        assert row["finished"] is False
        assert row["winner"] is None
        assert row["seat_config"] == [{"seat": 0, "agent": "human"}]
        assert row["seed"] == 42
        assert row["state_blob"] == {"hello": "world"}
        assert row["tracker_blob"] == [{"turn": 0}]

    def test_load_missing_returns_none(self):
        with get_connection() as conn:
            assert load_session_row(conn, "nonexistent") is None

    def test_save_replaces_existing(self):
        with get_connection() as conn:
            save_session(
                conn,
                "g1",
                [{"seat": 0, "agent": "human"}],
                None,
                False,
                None,
                {"v": 1},
                [],
            )
            save_session(
                conn,
                "g1",
                [{"seat": 0, "agent": "human"}],
                None,
                True,
                0,
                {"v": 2},
                [{"a": 1}],
            )
            row = load_session_row(conn, "g1")

        assert row["finished"] is True
        assert row["winner"] == 0
        assert row["state_blob"] == {"v": 2}

    def test_list_sessions(self):
        with get_connection() as conn:
            for i in range(3):
                save_session(
                    conn,
                    f"g{i}",
                    [{"seat": 0, "agent": "human"}],
                    None,
                    False,
                    None,
                    {},
                    [],
                )
            sessions = list_sessions(conn)

        assert len(sessions) == 3
        # Should not contain blobs
        assert "state_blob" not in sessions[0]

    def test_delete_session(self):
        with get_connection() as conn:
            save_session(
                conn,
                "del1",
                [{"seat": 0, "agent": "human"}],
                None,
                False,
                None,
                {},
                [],
            )
            assert delete_session(conn, "del1") is True
            assert load_session_row(conn, "del1") is None
            assert delete_session(conn, "del1") is False


class TestTournamentCRUD:
    def test_save_and_list(self):
        with get_connection() as conn:
            row_id = save_tournament(
                conn,
                agents=["random", "greedy", "smartbot"],
                n_games=10,
                wins={"random": 2, "greedy": 3, "smartbot": 5},
                draws=0,
                avg_turns=80.5,
                avg_vps={"random": 4.2, "greedy": 5.1, "smartbot": 7.3},
            )
            results = list_tournaments(conn)

        assert row_id is not None
        assert len(results) == 1
        r = results[0]
        assert r["agents"] == ["random", "greedy", "smartbot"]
        assert r["n_games"] == 10
        assert r["wins"]["smartbot"] == 5
        assert r["avg_vps"]["greedy"] == 5.1

    def test_list_pagination(self):
        with get_connection() as conn:
            for i in range(5):
                save_tournament(
                    conn,
                    agents=["random", "greedy", "smartbot"],
                    n_games=i + 1,
                    wins={},
                    draws=0,
                    avg_turns=0,
                    avg_vps={},
                )
            page1 = list_tournaments(conn, limit=2, offset=0)
            page2 = list_tournaments(conn, limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2


# ---------------------------------------------------------------------------
# Session blob round-trip
# ---------------------------------------------------------------------------


class TestSessionBlobRoundTrip:
    def _make_session(self, seed: int = 42) -> GameSession:
        """Create a session with 1 human + 3 bots."""
        seat_configs = [
            (0, "human"),
            (1, "random"),
            (2, "greedy"),
            (3, "smartbot"),
        ]
        return create_session(seat_configs, seed=seed)

    def test_round_trip_preserves_game_phase(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert restored.game.phase == session.game.phase

    def test_round_trip_preserves_board(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert len(restored.game.board.hexes) == 19
        for orig, rest in zip(session.game.board.hexes, restored.game.board.hexes):
            assert orig.terrain == rest.terrain
            assert orig.token == rest.token
            assert orig.coord == rest.coord

    def test_round_trip_preserves_players(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert len(restored.game.players) == len(session.game.players)
        for orig, rest in zip(session.game.players, restored.game.players):
            assert orig.name == rest.name
            assert orig.resources == rest.resources
            assert orig.victory_points == rest.victory_points
            assert orig.settlements == rest.settlements
            assert orig.cities == rest.cities
            assert orig.roads == rest.roads
            assert orig.dev_cards == rest.dev_cards
            assert orig.new_dev_cards == rest.new_dev_cards
            assert orig.played_knights == rest.played_knights

    def test_round_trip_preserves_ownership(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert restored.game.vertex_owner == session.game.vertex_owner
        assert restored.game.vertex_building == session.game.vertex_building
        assert restored.game.edge_owner == session.game.edge_owner

    def test_round_trip_preserves_human_seats(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert restored.human_seats == session.human_seats

    def test_round_trip_preserves_agents(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        for i, (orig, rest) in enumerate(zip(session.agents, restored.agents)):
            if orig is None:
                assert rest is None
            else:
                assert isinstance(rest, type(orig))

    def test_round_trip_preserves_tracker_records(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert len(restored.tracker.records) == len(session.tracker.records)
        for orig, rest in zip(session.tracker.records, restored.tracker.records):
            assert orig.turn == rest.turn
            assert orig.phase == rest.phase
            assert orig.player_idx == rest.player_idx
            assert orig.action_type == rest.action_type

    def test_round_trip_preserves_robber_and_deck(self):
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)

        assert restored.game.robber_hex == session.game.robber_hex
        assert restored.game.dev_card_deck == session.game.dev_card_deck

    def test_round_trip_game_produces_same_legal_actions(self):
        """The restored game should produce the same legal actions as the original."""
        session = self._make_session()
        orig_legal = session.game.legal_actions()

        state_blob, tracker_blob = session_to_blob(session)
        restored = session_from_blob(session.game_id, state_blob, tracker_blob)
        rest_legal = restored.game.legal_actions()

        assert orig_legal == rest_legal

    def test_full_db_round_trip(self):
        """Save to DB and load back, verify game state matches."""
        session = self._make_session()
        state_blob, tracker_blob = session_to_blob(session)
        seat_config = [
            {"seat": 0, "agent": "human"},
            {"seat": 1, "agent": "random"},
            {"seat": 2, "agent": "greedy"},
            {"seat": 3, "agent": "smartbot"},
        ]

        with get_connection() as conn:
            save_session(
                conn,
                session.game_id,
                seat_config,
                42,
                session.finished,
                None,
                state_blob,
                tracker_blob,
            )
            row = load_session_row(conn, session.game_id)

        restored = session_from_blob(row["game_id"], row["state_blob"], row["tracker_blob"])
        assert restored.game.legal_actions() == session.game.legal_actions()
        assert restored.human_seats == session.human_seats
