# Plan: Fix Training Instability & Improve PPO Performance

## Context

The full 3-stage curriculum (500K + 2M + 1M = 3.5M steps) completed with Phase 4 changes (256x256 network, dense rewards, mixed opponents, `ent_coef=0.02`, `batch_size=128`). Results are **worse** than the previous run:

| Matchup | Previous | Current |
|---------|----------|---------|
| vs Random | 80% | 57.5% |
| vs LongestRoad | 16% | 8.0% |
| vs ResourceHoarder | 9% | 3.0% |
| vs DevCard | 8% | 1.5% |
| vs Greedy | 2% | 1.5% |
| vs SmartBot | 2% | 1.0% |

### Diagnosis from TensorBoard + Checkpoint Evaluation

1. **KL divergence is 4-5x too high** in Stage 2 (0.07 vs target ~0.015) — policy updates are too aggressive
2. **Clip fraction ~25%** — a quarter of actions are hitting the trust region boundary every update
3. **Stage 3 explained variance only 0.40** — value function can't predict returns in self-play
4. **256x256 network is severely undertrained** at 3.5M total steps

The learning *is* progressing (43% → 55% → 57.5% vs Random across stages) but too slowly because updates are noisy/unstable.

### Training Trajectory (checkpoint evaluations)

| Checkpoint | vs Random | vs Greedy |
|---|---|---|
| Stage 1 @ 200K | 24% | — |
| Stage 1 final (500K) | 43% | 0% |
| Stage 2 @ 500K | 40% | 0% |
| Stage 2 @ 1M | 54% | 0% |
| Stage 2 @ 1.5M | 50% | 2% |
| Stage 2 final (2M) | 55% | 1% |
| Stage 3 @ 500K | 56% | 0% |
| Stage 3 final (1M) | 57.5% | 1.5% |

---

## Changes to Implement

### 1. Stabilize training hyperparameters — `src/catan/ai/train.py`

In `_create_model()` (line 99):

| Parameter | Current | New | Why |
|-----------|---------|-----|-----|
| `learning_rate` | `3e-4` | `1e-4` | Directly reduces KL divergence; smaller steps = more stable |
| `batch_size` | `128` | `256` | Larger minibatches → lower gradient variance |
| `n_epochs` | `10` | `4` | Fewer passes per rollout buffer → less overfitting per update |
| `ent_coef` | `0.02` | `0.01` | Current entropy is already dropping fast; 0.02 may be adding noise |
| `target_kl` | (none) | `0.015` | **New.** Early-stops PPO updates when KL exceeds threshold — prevents the destructive large policy jumps seen in Stage 2 |

### 2. Simplify reward shaping — `src/catan/ai/gym_env.py`

The current reward has ~8 different bonus/penalty terms that may create confusing signal. Simplify to focus on what matters:

**Keep (these are aligned with winning):**
- VP gain: `+1.0` per VP gained
- Win/loss: `+10.0 / -5.0`
- Settlement quality: `+0.1 to +0.3` based on production value

**Remove (noisy or misaligned):**
- Resource gain: `+0.01` per resource — rewards hoarding, not spending
- Road bonus: `+0.2` — rewards building roads even when not strategically useful
- Dev card bonus: `+0.3` — rewards buying dev cards regardless of game state
- City bonus: `+0.5` — already captured by VP gain (+1.0 per VP), double-counting
- Port access bonus: `+0.2` — too sparse and situational
- Trade penalty: `-0.05` — discourages all trading, but 4:1 trades are sometimes correct
- Resource diversity: `+0.05 × unique` — rewards holding resources, not spending them

**Result:** The reward simplifies to just VP gain + settlement quality + terminal win/loss. This gives a clearer learning signal: "gain VP, place good settlements, win the game."

### 3. Increase training duration — `src/catan/ai/train.py`

Update CLI defaults in `main()` (line 269):

| Stage | Current | New |
|-------|---------|-----|
| Stage 1 (vs Random) | 500K | **1M** |
| Stage 2 (vs Mixed) | 2M | **5M** |
| Stage 3 (Self-play) | 1M | **2M** |
| **Total** | **3.5M** | **8M** |

The 256x256 network needs ~2-3x more data. Stage 2 gets the biggest increase since that's where strategic learning happens.

### 4. Graduate Stage 2 difficulty — `src/catan/ai/train.py`

Currently Stage 2 randomly picks from all 6 bots equally. This means the agent faces SmartBot (very hard) right after only learning to beat Random. Instead, weight toward easier bots early:

In `train_curriculum()`, replace the `mixed_opponents()` factory with one that uses weighted sampling:
```python
def mixed_opponents() -> list[Agent]:
    # Weight toward easier bots — harder ones still appear but less often
    weights = [3, 2, 1, 1, 1, 1]  # Random, Greedy, LongestRoad, DevCard, ResourceHoarder, SmartBot
    chosen = stdlib_random.choices(_heuristic_factories, weights=weights, k=3)
    return [cls(rng=stdlib_random.Random()) for cls in chosen]
```

This gives ~33% Random, ~22% Greedy, ~11% each for the rest. The agent builds competence against easier opponents before encountering the harder ones.

---

## Files to Modify

1. **`src/catan/ai/train.py`** — hyperparameters, default timesteps, weighted opponent sampling
2. **`src/catan/ai/gym_env.py`** — simplify reward shaping
3. **`tests/test_train.py`** — update any tests that assert on specific hyperparameter values
4. **`tests/test_gym_env.py`** — update reward-related tests for simplified rewards

## Verification

1. `uv run pytest tests/test_train.py tests/test_gym_env.py` — all tests pass
2. `uv run ruff check .` — clean lint
3. Quick smoke test: `uv run python -m catan.ai.train --stage1 2000 --stage2 2000 --stage3 2000` — completes without errors
4. After full training: `uv run python scripts/evaluate.py --model models/stage3_selfplay --games 200` — compare against current numbers
