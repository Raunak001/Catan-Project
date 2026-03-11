"""Pydantic request/response models for the Catan API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SeatConfig(BaseModel):
    seat: int = Field(ge=0, le=3)
    agent: str  # "human" or agent name from AGENT_REGISTRY


class CreateGameRequest(BaseModel):
    seats: list[SeatConfig] = Field(min_length=3, max_length=4)
    seed: int | None = None
    shuffle_board: bool = True


class SubmitActionRequest(BaseModel):
    action: dict


class TournamentRequest(BaseModel):
    agents: list[str] = Field(min_length=3, max_length=4)
    n_games: int = Field(default=10, ge=1, le=1000)
    seed: int = 0
