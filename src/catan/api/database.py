"""SQLite persistence layer for game sessions and tournament results."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    return Path(os.environ.get("DATA_DIR", "data")) / "catan.db"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with WAL mode and foreign keys enabled."""
    path = _db_path()
    _ensure_dir(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS game_sessions (
    game_id      TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    finished     INTEGER NOT NULL DEFAULT 0,
    winner       INTEGER,
    seat_config  TEXT NOT NULL,
    seed         INTEGER,
    state_blob   TEXT NOT NULL,
    tracker_blob TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tournament_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at     TEXT NOT NULL,
    agents     TEXT NOT NULL,
    n_games    INTEGER NOT NULL,
    wins       TEXT NOT NULL,
    draws      INTEGER NOT NULL,
    avg_turns  REAL NOT NULL,
    avg_vps    TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Game session CRUD
# ---------------------------------------------------------------------------


def save_session(
    conn: sqlite3.Connection,
    game_id: str,
    seat_config: list[dict],
    seed: int | None,
    finished: bool,
    winner: int | None,
    state_blob: dict,
    tracker_blob: list[dict],
) -> None:
    """Insert or replace a game session row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO game_sessions
            (game_id, created_at, finished, winner, seat_config, seed, state_blob, tracker_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            datetime.now(UTC).isoformat(),
            int(finished),
            winner,
            json.dumps(seat_config),
            seed,
            json.dumps(state_blob),
            json.dumps(tracker_blob),
        ),
    )
    conn.commit()


def load_session_row(conn: sqlite3.Connection, game_id: str) -> dict[str, Any] | None:
    """Load a game session row as a dict, or None if not found."""
    row = conn.execute("SELECT * FROM game_sessions WHERE game_id = ?", (game_id,)).fetchone()
    if row is None:
        return None
    return {
        "game_id": row["game_id"],
        "created_at": row["created_at"],
        "finished": bool(row["finished"]),
        "winner": row["winner"],
        "seat_config": json.loads(row["seat_config"]),
        "seed": row["seed"],
        "state_blob": json.loads(row["state_blob"]),
        "tracker_blob": json.loads(row["tracker_blob"]),
    }


def list_sessions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """List all game sessions (summary only, no blobs)."""
    rows = conn.execute(
        "SELECT game_id, created_at, finished, winner, seat_config FROM game_sessions "
        "ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "game_id": r["game_id"],
            "created_at": r["created_at"],
            "finished": bool(r["finished"]),
            "winner": r["winner"],
            "seat_config": json.loads(r["seat_config"]),
        }
        for r in rows
    ]


def delete_session(conn: sqlite3.Connection, game_id: str) -> bool:
    """Delete a game session. Returns True if a row was deleted."""
    cursor = conn.execute("DELETE FROM game_sessions WHERE game_id = ?", (game_id,))
    conn.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Tournament results CRUD
# ---------------------------------------------------------------------------


def save_tournament(
    conn: sqlite3.Connection,
    agents: list[str],
    n_games: int,
    wins: dict[str, int],
    draws: int,
    avg_turns: float,
    avg_vps: dict[str, float],
) -> int:
    """Insert a tournament result. Returns the new row ID."""
    cursor = conn.execute(
        """
        INSERT INTO tournament_results (run_at, agents, n_games, wins, draws, avg_turns, avg_vps)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            json.dumps(agents),
            n_games,
            json.dumps(wins),
            draws,
            avg_turns,
            json.dumps(avg_vps),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def list_tournaments(
    conn: sqlite3.Connection, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """List tournament results, most recent first."""
    rows = conn.execute(
        "SELECT * FROM tournament_results ORDER BY run_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_at": r["run_at"],
            "agents": json.loads(r["agents"]),
            "n_games": r["n_games"],
            "wins": json.loads(r["wins"]),
            "draws": r["draws"],
            "avg_turns": r["avg_turns"],
            "avg_vps": json.loads(r["avg_vps"]),
        }
        for r in rows
    ]
