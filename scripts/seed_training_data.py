"""Generate realistic seed training data for the dashboard demo.

Creates one complete 4-stage training run under training_runs/ so the
dashboard has something to display without requiring an actual PPO run.

Usage:
    uv run python scripts/seed_training_data.py
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

RNG = random.Random(42)

RUN_ID = "20260101_000000_demo_run"
RUN_NAME = "demo_run"

STAGES = [
    {
        "name": "stage1_vs_random",
        "timesteps": 1_000_000,
        "lr": 0.0003,
        "ent_coef": 0.01,
        "n_envs": 16,
        "baselines": ["random", "greedy", "smartbot"],
    },
    {
        "name": "stage2_vs_mixed",
        "timesteps": 1_000_000,
        "lr": 0.0001,
        "ent_coef": 0.008,
        "n_envs": 16,
        "baselines": ["random", "greedy", "smartbot", "hybridbot"],
    },
    {
        "name": "stage3_vs_hybrid",
        "timesteps": 1_500_000,
        "lr": 0.00005,
        "ent_coef": 0.005,
        "n_envs": 16,
        "baselines": ["random", "greedy", "smartbot", "hybridbot"],
    },
    {
        "name": "stage4_selfplay",
        "timesteps": 1_500_000,
        "lr": 0.00003,
        "ent_coef": 0.003,
        "n_envs": 16,
        "baselines": ["random", "greedy", "smartbot", "hybridbot"],
    },
]

EVAL_FREQ = 50_000
N_GAMES = 50

# Win rate at start and end of *each stage* per baseline.
# Tuple: (start_of_stage_wr, end_of_stage_wr)
# Baselines not active in a stage are absent from that stage's dict.
STAGE_WIN_RATES: list[dict[str, tuple[float, float]]] = [
    # Stage 1: learn the rules vs random; greedy/smartbot look very hard
    {
        "random":   (0.10, 0.92),
        "greedy":   (0.03, 0.18),
        "smartbot": (0.02, 0.06),
    },
    # Stage 2: mixed opponents — greedy climbing, smartbot/hybrid slow
    {
        "random":    (0.92, 0.96),
        "greedy":    (0.18, 0.52),
        "smartbot":  (0.06, 0.13),
        "hybridbot": (0.04, 0.10),
    },
    # Stage 3: dedicated hybrid training — smartbot/hybrid grind up
    {
        "random":    (0.96, 0.97),
        "greedy":    (0.52, 0.68),
        "smartbot":  (0.13, 0.19),
        "hybridbot": (0.10, 0.16),
    },
    # Stage 4: self-play — smartbot/hybrid plateau, random/greedy locked
    {
        "random":    (0.97, 0.96),
        "greedy":    (0.68, 0.70),
        "smartbot":  (0.19, 0.21),
        "hybridbot": (0.16, 0.18),
    },
]

# Sigmoid stretch per baseline — how quickly each baseline's curve rises
SIGMOID_STRETCH: dict[str, float] = {
    "random":    10.0,  # fast learner
    "greedy":    7.0,
    "smartbot":  5.0,   # slower, noisier
    "hybridbot": 5.0,
}

# Extra noise per baseline (harder baselines = noisier evals)
NOISE_SCALE: dict[str, float] = {
    "random":    0.015,
    "greedy":    0.025,
    "smartbot":  0.030,
    "hybridbot": 0.030,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _win_rate(baseline: str, stage_idx: int, stage_progress: float) -> float:
    """Sigmoid interpolation within a stage, anchored on per-stage targets."""
    rates = STAGE_WIN_RATES[stage_idx]
    start_wr, end_wr = rates[baseline]
    stretch = SIGMOID_STRETCH.get(baseline, 6.0)
    # Map stage_progress [0,1] through a sigmoid centred at 0.5
    t = _sigmoid(stretch * (stage_progress - 0.5))
    wr = start_wr + (end_wr - start_wr) * t
    noise = RNG.gauss(0, NOISE_SCALE.get(baseline, 0.02))
    return max(0.0, min(1.0, wr + noise))


def _avg_vp_ppo(win_rate: float) -> float:
    base = 3.5 + win_rate * 6.0
    return round(base + RNG.gauss(0, 0.15), 2)


def _avg_vp_baseline(baseline: str, win_rate: float) -> float:
    strength = {"random": 4.0, "greedy": 6.5, "smartbot": 7.8, "hybridbot": 8.0}
    base = strength.get(baseline, 5.0) * (1.0 - win_rate * 0.35)
    return round(max(2.0, base + RNG.gauss(0, 0.2)), 2)


def _avg_turns(win_rate: float) -> float:
    base = 260 - win_rate * 130
    return round(max(80, base + RNG.gauss(0, 5)), 1)


def _policy_loss(global_progress: float) -> float:
    val = 0.05 * math.exp(-3.5 * global_progress) + 0.004
    return round(val + RNG.gauss(0, 0.001), 5)


def _value_loss(global_progress: float) -> float:
    val = 0.55 * math.exp(-2.8 * global_progress) + 0.04
    return round(val + RNG.gauss(0, 0.005), 5)


def _entropy(stage_progress: float, ent_coef: float) -> float:
    base = -2.8 - stage_progress * 1.2 - (1.0 - ent_coef / 0.01) * 0.5
    return round(base + RNG.gauss(0, 0.04), 4)


def _approx_kl(global_progress: float) -> float:
    base = 0.007 + RNG.gauss(0, 0.002)
    spike = 0.025 if RNG.random() < 0.05 else 0.0
    return round(max(0.001, base + spike), 5)


def _clip_fraction(global_progress: float) -> float:
    base = 0.13 - global_progress * 0.05
    return round(max(0.02, base + RNG.gauss(0, 0.01)), 4)


def _reward_mean(win_rate: float) -> float:
    return round(win_rate * 9.0 + RNG.gauss(0, 0.25), 3)


def generate() -> None:
    out_dir = Path("training_runs") / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    current_dt = start_dt
    seconds_per_1k_steps = 0.5

    total_timesteps = sum(s["timesteps"] for s in STAGES)
    global_timestep = 0

    metrics_lines: list[str] = []
    eval_lines: list[str] = []
    stage_configs: list[dict] = []

    for stage_idx, stage in enumerate(STAGES):
        stage_configs.append(
            {
                "name": stage["name"],
                "start_timestep": global_timestep,
                "timesteps": stage["timesteps"],
                "lr": stage["lr"],
                "ent_coef": stage["ent_coef"],
                "n_envs": stage["n_envs"],
            }
        )

        stage_ts = 0
        while stage_ts < stage["timesteps"]:
            step = min(EVAL_FREQ, stage["timesteps"] - stage_ts)
            stage_ts += step
            global_timestep += step

            global_progress = global_timestep / total_timesteps
            stage_progress = stage_ts / stage["timesteps"]

            current_dt += timedelta(seconds=step / 1000 * seconds_per_1k_steps)
            ts_str = current_dt.isoformat()

            # Use random win rate as a proxy for overall agent strength in metrics
            proxy_wr = _win_rate("random", stage_idx, stage_progress)

            metrics_lines.append(
                json.dumps(
                    {
                        "timestep": global_timestep,
                        "timestamp": ts_str,
                        "stage": stage["name"],
                        "reward_mean": _reward_mean(proxy_wr),
                        "reward_std": round(abs(RNG.gauss(1.1, 0.15)), 3),
                        "ep_len_mean": _avg_turns(proxy_wr),
                        "policy_loss": _policy_loss(global_progress),
                        "value_loss": _value_loss(global_progress),
                        "entropy": _entropy(stage_progress, stage["ent_coef"]),
                        "approx_kl": _approx_kl(global_progress),
                        "clip_fraction": _clip_fraction(global_progress),
                    }
                )
            )

            for baseline in stage["baselines"]:
                wr = _win_rate(baseline, stage_idx, stage_progress)
                eval_lines.append(
                    json.dumps(
                        {
                            "timestep": global_timestep,
                            "timestamp": ts_str,
                            "stage": stage["name"],
                            "baseline": baseline,
                            "win_rate": round(wr, 3),
                            "avg_vp_ppo": _avg_vp_ppo(wr),
                            "avg_vp_baseline": _avg_vp_baseline(baseline, wr),
                            "avg_turns": _avg_turns(wr),
                            "n_games": N_GAMES,
                        }
                    )
                )

    end_dt = current_dt

    config = {
        "run_id": RUN_ID,
        "run_name": RUN_NAME,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "status": "completed",
        "stages": stage_configs,
        "hyperparams": {
            "net_arch": [256, 256],
            "n_steps": 4096,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "target_kl": 0.015,
        },
        "eval_freq": EVAL_FREQ,
        "eval_baselines": ["random", "greedy", "smartbot", "hybridbot"],
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n")
    (out_dir / "evaluations.jsonl").write_text("\n".join(eval_lines) + "\n")

    print(f"Wrote seed data to {out_dir}")
    print(f"  {len(metrics_lines)} metric points")
    print(f"  {len(eval_lines)} evaluation points")
    print(f"  Simulated timesteps: {global_timestep:,}")


if __name__ == "__main__":
    generate()
