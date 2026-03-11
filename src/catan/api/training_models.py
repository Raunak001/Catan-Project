"""Pydantic response models for training dashboard endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class StageInfo(BaseModel):
    name: str
    start_timestep: int
    timesteps: int | None = None
    lr: float | None = None
    ent_coef: float | None = None
    n_envs: int | None = None


class RunSummary(BaseModel):
    run_id: str
    run_name: str
    start_time: str
    end_time: str | None = None
    status: str
    stages: list[StageInfo]
    total_timesteps: int | None = None


class MetricPoint(BaseModel):
    timestep: int
    timestamp: str
    stage: str
    reward_mean: float | None = None
    reward_std: float | None = None
    ep_len_mean: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    approx_kl: float | None = None
    clip_fraction: float | None = None


class EvalPoint(BaseModel):
    timestep: int
    timestamp: str
    stage: str
    baseline: str
    win_rate: float
    avg_vp_ppo: float
    avg_vp_baseline: float
    avg_turns: float
    n_games: int


class RunMetrics(BaseModel):
    run_id: str
    config: dict
    metrics: list[MetricPoint]


class RunEvaluations(BaseModel):
    run_id: str
    evaluations: list[EvalPoint]
