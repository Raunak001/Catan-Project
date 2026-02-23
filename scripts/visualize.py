"""Visualize training results and agent behaviour.

Usage:
    uv run python scripts/visualize.py --model models/stage3_selfplay
    uv run python scripts/visualize.py --model models/stage3_selfplay --games 200
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from catan.ai.heuristic import (
    DevCardBot,
    GreedyAgent,
    LongestRoadBot,
    RandomAgent,
    ResourceHoarder,
)
from catan.ai.ppo_agent import PPOAgent
from catan.game_runner import run_tournament

BASELINES: dict[str, type] = {
    "Random": RandomAgent,
    "Greedy": GreedyAgent,
    "LongestRoad": LongestRoadBot,
    "DevCard": DevCardBot,
    "ResourceHoarder": ResourceHoarder,
}


def plot_win_rates(
    model_path: str, n_games: int = 200, seed: int = 0, output_dir: str = "figures"
) -> None:
    """Bar chart: PPO win rate vs each baseline."""
    ppo = PPOAgent(model_path, deterministic=True)
    names = []
    win_rates = []

    for name, cls in BASELINES.items():
        agents = [ppo] + [cls(rng=random.Random(seed + i)) for i in range(3)]
        result = run_tournament(agents, n_games=n_games, base_seed=seed)
        names.append(name)
        win_rates.append(result.wins[0] / n_games)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, win_rates, color="steelblue", edgecolor="black")
    ax.set_ylabel("Win Rate")
    ax.set_title("PPO Agent Win Rate vs Baselines")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Random baseline (25%)")
    ax.legend()

    for bar, rate in zip(bars, win_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{rate:.0%}", ha="center", va="bottom", fontweight="bold",
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/win_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/win_rates.png")


def plot_settlement_heatmap(
    model_path: str, n_games: int = 200, seed: int = 0, output_dir: str = "figures"
) -> None:
    """Heatmap of settlement placement frequency across vertices 0-53."""
    ppo = PPOAgent(model_path, deterministic=True)
    vertex_counts: Counter[int] = Counter()

    for i in range(n_games):
        rng = random.Random(seed + i)
        from catan.board import Board
        from catan.game import Game, GamePhase
        from catan.player import Player

        board = Board.standard(shuffle=True, rng=rng)
        players = [Player(name="PPO")] + [Player(name=f"R{j}") for j in range(3)]
        game = Game(board=board, players=players, rng=rng)
        opponents = [RandomAgent(rng=random.Random(seed + i + j)) for j in range(3)]
        all_agents = [ppo, *opponents]

        while game.phase != GamePhase.FINISHED:
            if game.phase == GamePhase.ROLL:
                game._start_next_turn()
                continue
            if game.phase == GamePhase.ROBBER_DISCARD:
                acting = game.players_to_discard[game._discard_idx]
            else:
                acting = game.current_player_idx
            legal = game.legal_actions()
            if not legal:
                break
            action = all_agents[acting].choose_action(game, legal)
            game.apply_action(action)
            if game.check_victory() is not None:
                game.phase = GamePhase.FINISHED

        # Record PPO's settlements + cities (cities were settlements first)
        for v in game.players[0].settlements:
            vertex_counts[v] += 1
        for v in game.players[0].cities:
            vertex_counts[v] += 1

    # Plot as a bar chart (vertices 0-53)
    vertices = list(range(54))
    counts = [vertex_counts.get(v, 0) for v in vertices]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(vertices, counts, color="coral", edgecolor="none", width=1.0)
    ax.set_xlabel("Vertex ID")
    ax.set_ylabel("Settlement Count")
    ax.set_title(f"PPO Settlement Placement Frequency ({n_games} games)")
    ax.set_xlim(-0.5, 53.5)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/settlement_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/settlement_heatmap.png")


def plot_resource_patterns(
    model_path: str, n_games: int = 200, seed: int = 0, output_dir: str = "figures"
) -> None:
    """Bar chart of average resources held by the PPO agent at game end."""
    ppo = PPOAgent(model_path, deterministic=True)
    from catan.resources import Resource

    resource_totals: Counter[Resource] = Counter()

    from catan.board import Board
    from catan.game import Game, GamePhase
    from catan.player import Player

    for i in range(n_games):
        rng = random.Random(seed + i)
        board = Board.standard(shuffle=True, rng=rng)
        players = [Player(name="PPO")] + [Player(name=f"R{j}") for j in range(3)]
        game = Game(board=board, players=players, rng=rng)
        opponents = [RandomAgent(rng=random.Random(seed + i + j)) for j in range(3)]
        all_agents = [ppo, *opponents]

        while game.phase != GamePhase.FINISHED:
            if game.phase == GamePhase.ROLL:
                game._start_next_turn()
                continue
            if game.phase == GamePhase.ROBBER_DISCARD:
                acting = game.players_to_discard[game._discard_idx]
            else:
                acting = game.current_player_idx
            legal = game.legal_actions()
            if not legal:
                break
            action = all_agents[acting].choose_action(game, legal)
            game.apply_action(action)
            if game.check_victory() is not None:
                game.phase = GamePhase.FINISHED

        for res, count in game.players[0].resources.items():
            resource_totals[res] += count

    res_names = [r.value for r in Resource]
    res_avgs = [resource_totals.get(r, 0) / max(n_games, 1) for r in Resource]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#8B4513", "#CD853F", "#228B22", "#FFD700", "#808080"]
    ax.bar(res_names, res_avgs, color=colors, edgecolor="black")
    ax.set_ylabel("Average Count at Game End")
    ax.set_title(f"PPO Resource Holdings at Game End ({n_games} games)")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/resource_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/resource_patterns.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Catan PPO agent performance")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--games", type=int, default=200, help="Games per chart")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    args = parser.parse_args()

    print(f"Generating visualisations for: {args.model}")
    print(f"Games per chart: {args.games}")
    print("=" * 60)

    plot_win_rates(args.model, args.games, args.seed, args.output_dir)
    plot_settlement_heatmap(args.model, args.games, args.seed, args.output_dir)
    plot_resource_patterns(args.model, args.games, args.seed, args.output_dir)

    print("\nAll visualisations saved to:", args.output_dir)


if __name__ == "__main__":
    main()
