"""FastAPI router for training dashboard endpoints.

Serves training run metadata, metrics, and evaluation results from
the ``training_runs/`` directory (JSONL + JSON files written by
``MetricsCallback``).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from catan.api.training_models import (
    EvalPoint,
    MetricPoint,
    RunEvaluations,
    RunMetrics,
    RunSummary,
)

router = APIRouter(prefix="/training", tags=["training"])

TRAINING_RUNS_DIR = Path("training_runs")


def _resolve_runs_dir() -> Path:
    """Return the training runs directory, creating it if needed."""
    TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return TRAINING_RUNS_DIR


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, returning a list of dicts."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


def _load_config(run_dir: Path) -> dict | None:
    """Load config.json from a run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


@router.get("/runs")
def list_runs() -> dict:
    """List all training runs with summary info."""
    runs_dir = _resolve_runs_dir()
    summaries: list[RunSummary] = []

    for entry in sorted(runs_dir.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        config = _load_config(entry)
        if config is None:
            continue

        # Compute total timesteps from metrics file
        metrics = _read_jsonl(entry / "metrics.jsonl")
        total_ts = metrics[-1]["timestep"] if metrics else None

        summaries.append(
            RunSummary(
                run_id=config["run_id"],
                run_name=config.get("run_name", config["run_id"]),
                start_time=config["start_time"],
                end_time=config.get("end_time"),
                status=config.get("status", "unknown"),
                stages=config.get("stages", []),
                total_timesteps=total_ts,
            )
        )

    return {"runs": [s.model_dump() for s in summaries]}


def _deduplicate_and_downsample(raw: list[dict], downsample: int | None) -> list[dict]:
    """Deduplicate by timestep (keep last per timestep), then optionally downsample."""
    seen: dict[int, dict] = {}
    for m in raw:
        seen[m["timestep"]] = m
    deduped = [seen[ts] for ts in sorted(seen)]
    if downsample and downsample > 1:
        deduped = deduped[::downsample]
    return deduped


@router.get("/runs/compare")
def compare_runs(
    ids: str = Query(..., description="Comma-separated run IDs"),
    downsample: int | None = Query(None, description="Return every Nth unique timestep"),
) -> dict:
    """Compare metrics and evaluations across multiple runs."""
    runs_dir = _resolve_runs_dir()
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]

    if not run_ids:
        raise HTTPException(status_code=400, detail="No run IDs provided")

    result: dict[str, dict] = {}
    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            continue

        config = _load_config(run_dir)
        if config is None:
            continue

        raw_metrics = _deduplicate_and_downsample(_read_jsonl(run_dir / "metrics.jsonl"), downsample)
        metrics = [MetricPoint(**m) for m in raw_metrics]
        evals = [EvalPoint(**e) for e in _read_jsonl(run_dir / "evaluations.jsonl")]
        result[run_id] = {
            "config": config,
            "metrics": [m.model_dump() for m in metrics],
            "evaluations": [e.model_dump() for e in evals],
        }

    if not result:
        raise HTTPException(status_code=404, detail="No valid runs found for given IDs")

    return {"runs": result}


@router.get("/runs/{run_id}")
def get_run_metrics(
    run_id: str,
    stage: str | None = Query(None, description="Filter by stage name"),
    downsample: int | None = Query(None, description="Return every Nth unique timestep"),
) -> dict:
    """Get full metrics for a specific training run."""
    run_dir = _resolve_runs_dir() / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    config = _load_config(run_dir)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Config not found for run {run_id}")

    raw_metrics = _read_jsonl(run_dir / "metrics.jsonl")
    if stage:
        raw_metrics = [m for m in raw_metrics if m.get("stage") == stage]

    raw_metrics = _deduplicate_and_downsample(raw_metrics, downsample)
    metrics = [MetricPoint(**m) for m in raw_metrics]

    return RunMetrics(
        run_id=run_id,
        config=config,
        metrics=metrics,
    ).model_dump()


@router.get("/runs/{run_id}/evaluations")
def get_run_evaluations(
    run_id: str,
    baseline: str | None = Query(None, description="Filter by baseline name"),
) -> dict:
    """Get evaluation results for a specific training run."""
    run_dir = _resolve_runs_dir() / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    raw_evals = _read_jsonl(run_dir / "evaluations.jsonl")
    if baseline:
        raw_evals = [e for e in raw_evals if e.get("baseline") == baseline]

    evaluations = [EvalPoint(**e) for e in raw_evals]

    return RunEvaluations(
        run_id=run_id,
        evaluations=evaluations,
    ).model_dump()
