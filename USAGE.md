# Usage Guide

All commands use `uv run` to execute within the project's virtual environment.

## Development

```bash
uv sync                    # Install/update all dependencies
uv run pytest              # Run all tests (299 tests)
uv run pytest -m "not slow"  # Skip slow tests
uv run ruff check .        # Lint the codebase
uv run ruff format .       # Auto-format code
uv run ruff check --fix .  # Lint with auto-fix
```

## Training

### Full Curriculum (3 stages)

```bash
uv run python -m catan.ai.train
```

Runs all three training stages in sequence:

| Stage | Opponents | Default Steps | Purpose |
|-------|-----------|---------------|---------|
| 1 | RandomAgent | 500,000 | Learn basic rules and legal play |
| 2 | GreedyAgent | 1,000,000 | Learn strategic building/trading |
| 3 | Self-play (stage 2 checkpoint) | 500,000 | Refine against a competent opponent |

Models are saved to `models/` with checkpoints every 100k steps. TensorBoard logs go to `logs/catan_ppo/`.

### Training Options

```bash
uv run python -m catan.ai.train [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--stage1` | int | 500000 | Timesteps for stage 1 (vs Random) |
| `--stage2` | int | 1000000 | Timesteps for stage 2 (vs Greedy) |
| `--stage3` | int | 500000 | Timesteps for stage 3 (self-play) |
| `--seed` | int | None | Random seed for reproducibility |
| `--save-dir` | str | `models` | Directory for saved models and checkpoints |
| `--n-envs` | int | 4 | Number of parallel environments (higher = faster on multi-core CPUs) |
| `--checkpoint-freq` | int | 100000 | Save a checkpoint every N timesteps |
| `--only-stage` | int | None | Run only stage 1, 2, or 3 (requires `--load` for stages 2/3) |
| `--load` | str | None | Path to a previously saved model to continue training from |

A **timestep** is one action the agent takes in the game. A full Catan game is roughly 100-300 agent actions, so 500k timesteps is approximately 1,700-5,000 complete games.

### Examples

```bash
# Quick smoke test (finishes in ~1 minute)
uv run python -m catan.ai.train --stage1 1000 --stage2 1000 --stage3 1000

# Train with 8 parallel environments for speed
uv run python -m catan.ai.train --n-envs 8

# Only run stage 1, then stop
uv run python -m catan.ai.train --only-stage 1

# Continue from a stage 1 checkpoint into stage 2
uv run python -m catan.ai.train --only-stage 2 --load models/stage1_vs_random

# Resume stage 3 with more steps
uv run python -m catan.ai.train --only-stage 3 --load models/stage2_vs_greedy --stage3 1000000
```

### Monitoring with TensorBoard

```bash
uv run tensorboard --logdir logs/catan_ppo
```

Open http://localhost:6006 to see live training curves (episode reward, episode length, loss, etc.).

## Evaluation

```bash
uv run python scripts/evaluate.py --model <MODEL_PATH> [OPTIONS]
```

Runs the trained agent in tournaments against baseline opponents. The agent plays as seat 0 against 3 copies of each baseline.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | (required) | Path to a trained model file |
| `--games` | int | 1000 | Number of games per matchup |
| `--baselines` | str[] | all | Which baselines to test against |
| `--seed` | int | 0 | Random seed |

Available baselines: `random`, `greedy`, `longest_road`, `dev_card`, `resource_hoarder`

### Output

Prints a table per matchup:

```
vs random                |  Win rate: 45.2%  |  PPO VP: 7.3  |  Baseline VP: 5.1  |  Avg turns: 142
vs greedy                |  Win rate: 28.1%  |  PPO VP: 6.8  |  Baseline VP: 6.4  |  Avg turns: 158
```

- **Win rate**: Fraction of games where the PPO agent (seat 0) won
- **PPO VP**: Average victory points the PPO agent had at game end
- **Baseline VP**: Average VP across the 3 baseline opponents
- **Avg turns**: Average number of game turns before completion

### Examples

```bash
# Evaluate against all baselines
uv run python scripts/evaluate.py --model models/stage3_selfplay

# Quick check against just Random and Greedy
uv run python scripts/evaluate.py --model models/stage1_vs_random --baselines random greedy --games 100
```

## Visualization

```bash
uv run python scripts/visualize.py --model <MODEL_PATH> [OPTIONS]
```

Generates matplotlib charts and saves them as PNG files.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | (required) | Path to a trained model file |
| `--games` | int | 200 | Number of games to sample for each chart |
| `--seed` | int | 0 | Random seed |
| `--output-dir` | str | `figures` | Directory to save chart images |

### Charts Generated

| File | Description |
|------|-------------|
| `win_rates.png` | Bar chart of PPO win rate vs each baseline. Red dashed line at 25% marks the random-chance baseline for a 4-player game. |
| `settlement_heatmap.png` | Frequency of PPO settlement placements across all 54 board vertices. Shows which positions the agent prefers. |
| `resource_patterns.png` | Average resource cards the PPO agent holds at game end. Reveals the agent's resource acquisition strategy. |

### Example

```bash
uv run python scripts/visualize.py --model models/stage3_selfplay --games 500 --output-dir figures
```

## Baseline Agents

Five heuristic agents are available for training opponents and evaluation benchmarks:

| Agent | Strategy | Key Behaviour |
|-------|----------|---------------|
| `RandomAgent` | Uniform random | Picks randomly from legal actions. Weakest baseline. |
| `GreedyAgent` | Build everything | Priority: cities > settlements > roads > dev cards. Picks highest-production vertices. |
| `LongestRoadBot` | Road spam | Priority: play road building > build roads > settlements. Pursues longest road bonus (2 VP). |
| `DevCardBot` | Army builder | Priority: play knights > buy dev cards > build. Pursues largest army bonus (2 VP). |
| `ResourceHoarder` | City rush | Priority: cities > bank trade for ore/wheat > settlements. Targets high-probability hexes. |
