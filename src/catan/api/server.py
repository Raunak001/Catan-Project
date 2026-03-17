"""FastAPI server exposing the Catan game engine."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from catan.api.database import (
    get_connection,
    init_db,
    list_tournaments,
    save_session,
    save_tournament,
)
from catan.api.models import CreateGameRequest, SubmitActionRequest, TournamentRequest
from catan.api.serializer import deserialize_action, serialize_action, serialize_game
from catan.api.session import (
    AGENT_REGISTRY,
    GameSession,
    create_session,
    session_from_blob,
    session_to_blob,
)
from catan.api.training_router import router as training_router
from catan.game_runner import run_tournament as run_tournament_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="Catan API", lifespan=lifespan)
app.include_router(training_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# In-memory write-through cache (avoids deserializing on every GET)
_sessions: dict[str, GameSession] = {}


def _persist_session(session: GameSession, seed: int | None = None) -> None:
    """Save session state to the database."""
    state_blob, tracker_blob = session_to_blob(session)
    seat_config = [
        {"seat": i, "agent": "human" if i in session.human_seats else state_blob["seat_agents"][i]}
        for i in range(len(session.agents))
    ]
    winner = session.game.check_victory() if session.finished else None
    with get_connection() as conn:
        save_session(
            conn,
            game_id=session.game_id,
            seat_config=seat_config,
            seed=seed,
            finished=session.finished,
            winner=winner,
            state_blob=state_blob,
            tracker_blob=tracker_blob,
        )


def _get_session(game_id: str) -> GameSession:
    """Look up session from cache, falling back to database."""
    session = _sessions.get(game_id)
    if session is not None:
        return session

    # Try loading from DB
    from catan.api.database import load_session_row

    with get_connection() as conn:
        row = load_session_row(conn, game_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    session = session_from_blob(game_id, row["state_blob"], row["tracker_blob"])
    _sessions[game_id] = session
    return session


@app.get("/agents")
def list_agents() -> dict:
    """List available AI agent names."""
    return {"agents": sorted(AGENT_REGISTRY.keys())}


@app.post("/games")
def create_game(req: CreateGameRequest) -> dict:
    """Create a new game session."""
    seat_configs = [(s.seat, s.agent) for s in req.seats]

    # Validate agent names
    for seat_idx, agent_name in seat_configs:
        if agent_name != "human" and agent_name not in AGENT_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent: {agent_name}. Available: {sorted(AGENT_REGISTRY.keys())}",
            )

    # Validate seat indices
    seat_indices = [s.seat for s in req.seats]
    if sorted(seat_indices) != list(range(len(req.seats))):
        raise HTTPException(
            status_code=400,
            detail=f"Seats must be consecutive starting from 0. Got: {seat_indices}",
        )

    session = create_session(seat_configs, seed=req.seed, shuffle_board=req.shuffle_board)
    _sessions[session.game_id] = session
    _persist_session(session, seed=req.seed)

    legal = session.game.legal_actions()
    return {
        "game_id": session.game_id,
        "state": serialize_game(session.game, session.human_seats),
        "legal_actions": [serialize_action(a) for a in legal],
    }


@app.get("/games/{game_id}")
def get_game(game_id: str) -> dict:
    """Get current game state."""
    session = _get_session(game_id)
    legal = session.game.legal_actions()
    stats = session.tracker.get_stats()
    return {
        "game_id": session.game_id,
        "state": serialize_game(session.game, session.human_seats),
        "legal_actions": [serialize_action(a) for a in legal],
        "stats_summary": {
            "total_actions": stats.total_actions,
            "turn": session.game.turn,
        },
    }


@app.post("/games/{game_id}/action")
def submit_action(game_id: str, req: SubmitActionRequest) -> dict:
    """Submit a human action."""
    session = _get_session(game_id)

    if session.finished:
        raise HTTPException(status_code=400, detail="Game is already finished")

    # Check it's a human's turn
    acting_idx = session.acting_player_idx()
    if acting_idx not in session.human_seats:
        raise HTTPException(
            status_code=400,
            detail=f"Seat {acting_idx} is not a human seat",
        )

    try:
        action = deserialize_action(req.action)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")

    # Validate action is legal
    legal = session.game.legal_actions()
    if action not in legal:
        raise HTTPException(
            status_code=400,
            detail=f"Action {req.action} is not legal in current state",
        )

    session.apply_and_advance(action)
    _persist_session(session)

    legal = session.game.legal_actions()
    winner = session.game.check_victory()
    last_actions = []
    if session.last_actions:
        last_actions = [
            {"player_idx": idx, "action": serialize_action(act)}
            for idx, act in session.last_actions
        ]
    return {
        "state": serialize_game(session.game, session.human_seats),
        "legal_actions": [serialize_action(a) for a in legal],
        "finished": session.finished,
        "winner": winner,
        "last_actions": last_actions,
    }


@app.get("/games/{game_id}/stats")
def get_stats(game_id: str) -> dict:
    """Get action analytics for a game."""
    session = _get_session(game_id)
    stats = session.tracker.get_stats()
    return {
        "total_actions": stats.total_actions,
        "action_type_counts": stats.action_type_counts,
        "per_player": stats.per_player,
        "phase_distribution": stats.phase_distribution,
        "trades_per_player": stats.trades_per_player,
        "dev_cards_played": stats.dev_cards_played,
        "robber_moves": stats.robber_moves,
        "vp_timeline": stats.vp_timeline,
    }


@app.post("/tournaments")
def run_tournament_endpoint(req: TournamentRequest) -> dict:
    """Run a tournament between AI agents and persist results."""

    for name in req.agents:
        if name not in AGENT_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent: {name}. Available: {sorted(AGENT_REGISTRY.keys())}",
            )

    agents = [AGENT_REGISTRY[name]() for name in req.agents]
    result = run_tournament_engine(agents, n_games=req.n_games, base_seed=req.seed)

    # Persist to DB (convert list-indexed results to agent-keyed dicts)
    wins_dict = {name: result.wins[i] for i, name in enumerate(req.agents)}
    vps_dict = {name: result.avg_vps[i] for i, name in enumerate(req.agents)}
    with get_connection() as conn:
        save_tournament(
            conn,
            agents=req.agents,
            n_games=req.n_games,
            wins=wins_dict,
            draws=result.draws,
            avg_turns=result.avg_turns,
            avg_vps=vps_dict,
        )

    return {
        "wins": result.wins,
        "draws": result.draws,
        "total_games": result.total_games,
        "avg_turns": result.avg_turns,
        "avg_vps": result.avg_vps,
        "agents": req.agents,
    }


@app.get("/tournaments")
def get_tournaments(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List past tournament results."""
    with get_connection() as conn:
        results = list_tournaments(conn, limit=limit, offset=offset)
    return {"results": results}
