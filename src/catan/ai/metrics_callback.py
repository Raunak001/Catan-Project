"""Custom SB3 callback that logs structured training metrics to JSON files.

Captures per-rollout metrics (reward, loss, entropy) and periodically runs
evaluation games against baseline agents, writing results to JSONL files
for the training dashboard.

Storage layout::

    training_runs/{timestamp}_{run_name}/
        config.json         # Run metadata, hyperparams, stages
        metrics.jsonl       # One JSON line per rollout
        evaluations.jsonl   # One JSON line per eval checkpoint per baseline
"""

from __future__ import annotations

import json
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from catan.ai.heuristic import (
    GreedyAgent,
    RandomAgent,
    SmartBot,
)
from catan.ai.ppo_agent import PPOAgent
from catan.game_runner import run_tournament

BASELINE_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "smartbot": SmartBot,
}


class MetricsCallback(BaseCallback):
    """Logs training metrics and periodic evaluations to JSONL files.

    Parameters
    ----------
    run_dir : Path
        Directory to write config.json, metrics.jsonl, evaluations.jsonl.
    stage_name : str
        Current curriculum stage name (e.g. "stage1_vs_random").
    eval_freq : int
        Run evaluation every this many timesteps (0 to disable).
    eval_games : int
        Number of games per baseline per evaluation checkpoint.
    eval_baselines : list[str]
        Baseline agent names to evaluate against.
    stage_config : dict
        Stage-specific config (timesteps, lr, ent_coef) to record.
    """

    def __init__(
        self,
        run_dir: Path,
        stage_name: str,
        eval_freq: int = 50_000,
        eval_games: int = 20,
        eval_baselines: list[str] | None = None,
        stage_config: dict | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.stage_name = stage_name
        self.eval_freq = eval_freq
        self.eval_games = eval_games
        self.eval_baselines = eval_baselines or ["random", "greedy", "smartbot"]
        self.stage_config = stage_config or {}
        self._last_eval_timestep = -1
        self._metrics_file = self.run_dir / "metrics.jsonl"
        self._evals_file = self.run_dir / "evaluations.jsonl"

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.run_dir / "config.json"
        if config_path.exists():
            # Append stage info to existing config
            config = json.loads(config_path.read_text())
            config["stages"].append(
                {
                    "name": self.stage_name,
                    "start_timestep": self.num_timesteps,
                    **self.stage_config,
                }
            )
            config_path.write_text(json.dumps(config, indent=2))
        else:
            # First stage — create config
            model = self.model
            config = {
                "run_id": self.run_dir.name,
                "run_name": self.run_dir.name.split("_", 2)[-1]
                if "_" in self.run_dir.name
                else self.run_dir.name,
                "start_time": datetime.now(UTC).isoformat(),
                "end_time": None,
                "status": "running",
                "stages": [
                    {
                        "name": self.stage_name,
                        "start_timestep": 0,
                        **self.stage_config,
                    }
                ],
                "hyperparams": {
                    "net_arch": [256, 256],
                    "n_steps": getattr(model, "n_steps", None),
                    "batch_size": getattr(model, "batch_size", None),
                    "n_epochs": getattr(model, "n_epochs", None),
                    "gamma": getattr(model, "gamma", None),
                    "gae_lambda": getattr(model, "gae_lambda", None),
                    "clip_range": str(getattr(model, "clip_range", None)),
                    "target_kl": getattr(model, "target_kl", None),
                },
                "eval_freq": self.eval_freq,
                "eval_baselines": self.eval_baselines,
            }
            config_path.write_text(json.dumps(config, indent=2))

    def _on_step(self) -> bool:
        # Log metrics from SB3's logger every rollout (called every step,
        # but logger values only update after each rollout completes)
        logger_values = self.model.logger.name_to_value if self.model.logger else {}

        # Only write when we have meaningful data (SB3 populates after rollout)
        if logger_values:
            metric = {
                "timestep": self.num_timesteps,
                "timestamp": datetime.now(UTC).isoformat(),
                "stage": self.stage_name,
                "reward_mean": logger_values.get("rollout/ep_rew_mean"),
                "reward_std": logger_values.get("rollout/ep_rew_std"),
                "ep_len_mean": logger_values.get("rollout/ep_len_mean"),
                "policy_loss": logger_values.get("train/policy_gradient_loss"),
                "value_loss": logger_values.get("train/value_loss"),
                "entropy": logger_values.get("train/entropy_loss"),
                "approx_kl": logger_values.get("train/approx_kl"),
                "clip_fraction": logger_values.get("train/clip_fraction"),
            }
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps(metric, default=lambda x: float(x)) + "\n")

        # Periodic evaluation
        if (
            self.eval_freq > 0
            and self.num_timesteps >= self.eval_freq
            and self.num_timesteps - self._last_eval_timestep >= self.eval_freq
        ):
            self._last_eval_timestep = self.num_timesteps
            self._run_evaluation()

        return True

    def _run_evaluation(self) -> None:
        """Save current model to temp file, run tournaments against baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "eval_model"
            self.model.save(str(tmp_path))
            ppo = PPOAgent(str(tmp_path), deterministic=True)

            for baseline_name in self.eval_baselines:
                if baseline_name not in BASELINE_REGISTRY:
                    continue

                agent_cls = BASELINE_REGISTRY[baseline_name]
                agents = [ppo] + [agent_cls(rng=random.Random(i)) for i in range(3)]

                result = run_tournament(agents, n_games=self.eval_games, base_seed=42)
                win_rate = result.wins[0] / max(self.eval_games, 1)

                eval_point = {
                    "timestep": self.num_timesteps,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "stage": self.stage_name,
                    "baseline": baseline_name,
                    "win_rate": round(win_rate, 4),
                    "avg_vp_ppo": round(result.avg_vps[0], 2),
                    "avg_vp_baseline": round(sum(result.avg_vps[1:]) / 3, 2),
                    "avg_turns": round(result.avg_turns, 1),
                    "n_games": self.eval_games,
                }
                with open(self._evals_file, "a") as f:
                    f.write(json.dumps(eval_point) + "\n")

                if self.verbose > 0:
                    print(
                        f"  [Eval] step {self.num_timesteps}: "
                        f"vs {baseline_name} = {win_rate:.1%} win rate"
                    )

    def _on_training_end(self) -> None:
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            config["end_time"] = datetime.now(UTC).isoformat()
            config["status"] = "completed"
            config_path.write_text(json.dumps(config, indent=2))
