"""Evaluate a trained PPO agent against baseline agents.

Usage:
    uv run python scripts/evaluate.py --model models/stage3_selfplay
    uv run python scripts/evaluate.py --model models/stage1_vs_random --games 500
    uv run python scripts/evaluate.py --model models/stage2_vs_greedy --baselines random greedy
"""

from __future__ import annotations

import argparse
import random

from catan.ai.heuristic import (
    DevCardBot,
    GreedyAgent,
    LongestRoadBot,
    RandomAgent,
    ResourceHoarder,
    SmartBot,
)
from catan.ai.ppo_agent import PPOAgent
from catan.game_runner import run_tournament

BASELINE_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "longest_road": LongestRoadBot,
    "dev_card": DevCardBot,
    "resource_hoarder": ResourceHoarder,
    "smart": SmartBot,
}


def evaluate(
    model_path: str,
    n_games: int = 1000,
    baselines: list[str] | None = None,
    seed: int = 0,
) -> dict[str, dict]:
    """Run the PPO agent against each baseline and return results.

    Returns a dict mapping baseline name to a result dict with
    keys: win_rate, avg_vp_ppo, avg_vp_baseline, avg_turns.
    """
    if baselines is None:
        baselines = list(BASELINE_REGISTRY.keys())

    ppo = PPOAgent(model_path, deterministic=True)
    results = {}

    for baseline_name in baselines:
        agent_cls = BASELINE_REGISTRY[baseline_name]
        # PPO at seat 0, three copies of the baseline at seats 1-3
        agents = [ppo] + [
            agent_cls(rng=random.Random(seed + i)) for i in range(3)
        ]

        tournament = run_tournament(agents, n_games=n_games, base_seed=seed)

        ppo_wins = tournament.wins[0]
        win_rate = ppo_wins / n_games
        results[baseline_name] = {
            "win_rate": win_rate,
            "ppo_wins": ppo_wins,
            "avg_vp_ppo": tournament.avg_vps[0],
            "avg_vp_baseline": sum(tournament.avg_vps[1:]) / 3,
            "avg_turns": tournament.avg_turns,
            "draws": tournament.draws,
        }

        print(f"\nvs {baseline_name:20s}  |  "
              f"Win rate: {win_rate:5.1%}  |  "
              f"PPO VP: {tournament.avg_vps[0]:.1f}  |  "
              f"Baseline VP: {results[baseline_name]['avg_vp_baseline']:.1f}  |  "
              f"Avg turns: {tournament.avg_turns:.0f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO agent vs baselines")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--games", type=int, default=1000, help="Games per matchup")
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Which baselines to evaluate against",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print(f"Games per matchup: {args.games}")
    print(f"Baselines: {args.baselines or 'all'}")
    print("=" * 70)

    results = evaluate(
        model_path=args.model,
        n_games=args.games,
        baselines=args.baselines,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, r in results.items():
        print(f"  vs {name:20s}: {r['win_rate']:5.1%} win rate")


if __name__ == "__main__":
    main()
