"""Maskable PPO training script with curriculum learning for the Catan environment.

Three-stage curriculum:
  1. Train vs RandomAgent           (learn the rules)
  2. Train vs mixed heuristic bots  (learn strategy)
  3. Self-play + heuristic mix      (refine)

Usage:
    # Full curriculum with defaults
    uv run python -m catan.ai.train

    # Custom timesteps per stage
    uv run python -m catan.ai.train --stage1 500000 --stage2 1000000 --stage3 500000

    # Single stage only (e.g. continue from a checkpoint)
    uv run python -m catan.ai.train --only-stage 2 --load models/stage1

    # Quick smoke test
    uv run python -m catan.ai.train --stage1 1000 --stage2 1000 --stage3 1000
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from catan.ai.agent import Agent
from catan.ai.gym_env import CatanEnv
from catan.ai.heuristic import (
    DevCardBot,
    GreedyAgent,
    LongestRoadBot,
    RandomAgent,
    ResourceHoarder,
    SmartBot,
)


def _get_action_mask(env: CatanEnv) -> np.ndarray:
    """Callback for ActionMasker wrapper."""
    return env.action_masks()


def make_env(
    seed: int | None = None,
    opponent_agents: list[Agent] | None = None,
) -> ActionMasker:
    """Create a wrapped CatanEnv with action masking."""
    env = CatanEnv(seed=seed, opponent_agents=opponent_agents)
    return ActionMasker(env, _get_action_mask)


def make_vec_env(
    n_envs: int = 4,
    seed: int | None = None,
    opponent_factory: Callable[[], list[Agent]] | None = None,
) -> SubprocVecEnv:
    """Create vectorised environments for parallel training.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.
    seed : int | None
        Base seed — each env gets ``seed + i``.
    opponent_factory : callable | None
        Returns a list of 3 opponent agents. If None, defaults to RandomAgents.
    """

    def _make(i: int) -> Callable[[], ActionMasker]:
        def _init() -> ActionMasker:
            env_seed = (seed + i) if seed is not None else None
            opponents = opponent_factory() if opponent_factory else None
            return make_env(seed=env_seed, opponent_agents=opponents)

        return _init

    return SubprocVecEnv([_make(i) for i in range(n_envs)])


def _create_model(
    env: ActionMasker | SubprocVecEnv,
    seed: int | None = None,
    load_path: str | None = None,
) -> MaskablePPO:
    """Create a new model or load from checkpoint."""
    if load_path:
        print(f"Loading model from {load_path}")
        return MaskablePPO.load(load_path, env=env)

    return MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        seed=seed,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.015,
        tensorboard_log="logs/catan_ppo",
    )


def train_stage(
    stage_name: str,
    timesteps: int,
    save_dir: str,
    seed: int | None = None,
    load_path: str | None = None,
    n_envs: int = 4,
    opponent_factory: Callable[[], list[Agent]] | None = None,
    checkpoint_freq: int = 100_000,
) -> str:
    """Train one curriculum stage. Returns path to the saved model."""
    print(f"\n{'=' * 60}")
    print(f"  Stage: {stage_name}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"{'=' * 60}\n")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if n_envs > 1:
        env = make_vec_env(
            n_envs=n_envs, seed=seed, opponent_factory=opponent_factory
        )
    else:
        env = make_env(
            seed=seed,
            opponent_agents=opponent_factory() if opponent_factory else None,
        )

    model = _create_model(env, seed=seed, load_path=load_path)

    # Checkpoint callback: freq is per-env, so divide by n_envs
    effective_freq = max(checkpoint_freq // n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=effective_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix=stage_name,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_cb,
        tb_log_name=stage_name,
    )

    # Save final model
    final_path = str(save_path / stage_name)
    model.save(final_path)
    print(f"Stage '{stage_name}' complete. Model saved to {final_path}")

    env.close()
    return final_path


def train_curriculum(
    stage1_steps: int = 500_000,
    stage2_steps: int = 1_000_000,
    stage3_steps: int = 500_000,
    seed: int | None = None,
    save_dir: str = "models",
    n_envs: int = 4,
    checkpoint_freq: int = 100_000,
    only_stage: int | None = None,
    load_path: str | None = None,
) -> str:
    """Run the full 3-stage training curriculum.

    Returns the path to the final trained model.
    """
    import random as stdlib_random

    stage1_path = load_path
    stage2_path = load_path

    # All heuristic agent classes for mixed-opponent stages
    _heuristic_factories: list[type] = [
        RandomAgent, GreedyAgent, LongestRoadBot,
        DevCardBot, ResourceHoarder, SmartBot,
    ]

    # --- Stage 1: vs RandomAgent ---
    if only_stage is None or only_stage == 1:

        def random_opponents() -> list[Agent]:
            return [RandomAgent(rng=stdlib_random.Random()) for _ in range(3)]

        stage1_path = train_stage(
            stage_name="stage1_vs_random",
            timesteps=stage1_steps,
            save_dir=save_dir,
            seed=seed,
            load_path=load_path if only_stage == 1 else None,
            n_envs=n_envs,
            opponent_factory=random_opponents,
            checkpoint_freq=checkpoint_freq,
        )

    # --- Stage 2: vs mixed heuristic opponents ---
    if only_stage is None or only_stage == 2:

        def mixed_opponents() -> list[Agent]:
            # Weight toward easier bots — harder ones still appear but less often
            weights = [3, 2, 1, 1, 1, 1]  # Random, Greedy, LR, DevCard, RH, SmartBot
            chosen = stdlib_random.choices(_heuristic_factories, weights=weights, k=3)
            return [cls(rng=stdlib_random.Random()) for cls in chosen]

        stage2_load = stage1_path if only_stage != 2 else load_path
        stage2_path = train_stage(
            stage_name="stage2_vs_mixed",
            timesteps=stage2_steps,
            save_dir=save_dir,
            seed=seed,
            load_path=stage2_load,
            n_envs=n_envs,
            opponent_factory=mixed_opponents,
            checkpoint_freq=checkpoint_freq,
        )

    # --- Stage 3: Self-play (1 PPO clone + 2 random heuristic bots) ---
    if only_stage is None or only_stage == 3:
        from catan.ai.ppo_agent import PPOAgent

        selfplay_checkpoint = stage2_path if only_stage != 3 else load_path

        def selfplay_opponents() -> list[Agent]:
            ppo = PPOAgent(selfplay_checkpoint, deterministic=False)
            heuristic_cls = stdlib_random.choices(_heuristic_factories, k=2)
            heuristics = [cls(rng=stdlib_random.Random()) for cls in heuristic_cls]
            return [ppo] + heuristics

        stage3_load = stage2_path if only_stage != 3 else load_path
        final_path = train_stage(
            stage_name="stage3_selfplay",
            timesteps=stage3_steps,
            save_dir=save_dir,
            seed=seed,
            load_path=stage3_load,
            n_envs=n_envs,
            opponent_factory=selfplay_opponents,
            checkpoint_freq=checkpoint_freq,
        )
        return final_path

    return stage2_path if stage2_path else stage1_path


def main() -> None:
    # Disable PyTorch distribution validation — with 463 actions mostly masked
    # to -1e8, the softmax can accumulate enough float error that probs don't
    # sum to exactly 1.0, triggering a spurious Simplex constraint violation.
    torch.distributions.Distribution.set_default_validate_args(False)

    parser = argparse.ArgumentParser(description="Train Catan PPO agent (curriculum)")
    parser.add_argument("--stage1", type=int, default=1_000_000)
    parser.add_argument("--stage2", type=int, default=5_000_000)
    parser.add_argument("--stage3", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument(
        "--only-stage", type=int, default=None, choices=[1, 2, 3],
    )
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    train_curriculum(
        stage1_steps=args.stage1,
        stage2_steps=args.stage2,
        stage3_steps=args.stage3,
        seed=args.seed,
        save_dir=args.save_dir,
        n_envs=args.n_envs,
        checkpoint_freq=args.checkpoint_freq,
        only_stage=args.only_stage,
        load_path=args.load,
    )


if __name__ == "__main__":
    main()
