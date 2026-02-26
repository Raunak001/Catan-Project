# Improvement Implementation Roadmap

## Quick Start: 3-Phase Plan

### Phase A: Retrain with Enriched Obs (Priority 1 — 1-2 days)
**Status**: Ready to execute
**Command**:
```bash
# Full curriculum with enriched 485-dim obs space
uv run python -m catan.ai.train --n-envs 4

# OR customize timesteps (recommended):
uv run python -m catan.ai.train --stage1 1000000 --stage2 5000000 --stage3 1000000 --n-envs 4
```

**What it does**:
- Trains new model with 485 obs dims (vs current 407)
- Model learns vertex production value instead of ID memorization
- Baseline: 38.75% settlements clustered → expected ~5-10% after training

**Expected result**: Win rate vs SmartBot goes from 5% → 15-25%

---

### Phase B: Improve Reward Shaping (Priority 2 — 1-3 hours)

#### B1: Increase Settlement Quality Reward
**File**: `src/catan/ai/gym_env.py:335-342`

**Current**:
```python
if isinstance(game_action, BuildSettlement):
    prod_value = 0.0
    for hex_idx in game.board.topology.vertex_to_hexes[game_action.vertex_id]:
        token = game.board.hexes[hex_idx].token
        if token is not None:
            prod_value += _REWARD_TOKEN_PROB.get(token, 0.0)
    # Scale: best vertex ~0.42 (6+8+5), normalise to [0.1, 0.3]
    reward += 0.1 + min(prod_value / 0.42, 1.0) * 0.2
```

**Change to**:
```python
if isinstance(game_action, BuildSettlement):
    prod_value = 0.0
    for hex_idx in game.board.topology.vertex_to_hexes[game_action.vertex_id]:
        token = game.board.hexes[hex_idx].token
        if token is not None:
            prod_value += _REWARD_TOKEN_PROB.get(token, 0.0)
    # Increased incentive: [0.5, 1.5] (matches VP strength)
    reward += 0.5 + min(prod_value / 0.42, 1.0) * 1.0
```

**Rationale**:
- Old ratio: settlement quality (+0.3 max) vs VP (+1.0) = 30%
- New ratio: settlement quality (+1.5 max) vs VP (+1.0) = 150% (4× more important)
- Incentivizes smart placement from start

#### B2: Add Port Bonus Reward (Optional)
**File**: `src/catan/ai/gym_env.py:343-360` (after settlement block)

**Add**:
```python
# Port bonus: +0.5 if settlement touches 2:1 or 3:1 port
if isinstance(game_action, BuildSettlement):
    port_bonus = 0.0
    vertex = game_action.vertex_id

    # Check adjacent ports
    for port in game.board.ports:
        if vertex in port.vertices:
            if port.port_type in [PortType.TWO_FOR_ONE_SHEEP, PortType.TWO_FOR_ONE_WHEAT,
                                   PortType.TWO_FOR_ONE_ORE, PortType.TWO_FOR_ONE_BRICK,
                                   PortType.TWO_FOR_ONE_WOOD]:
                port_bonus += 0.3
            elif port.port_type == PortType.THREE_FOR_ONE:
                port_bonus += 0.2

    reward += port_bonus
```

**Rationale**: Encourages strategic port access (long-term value)

---

### Phase C: Harden Curriculum (Priority 3 — 30 mins)

#### C1: Adjust Stage 2 Opponent Weights
**File**: `src/catan/ai/train.py:228-232`

**Current**:
```python
def mixed_opponents() -> list[Agent]:
    # Weight toward easier bots
    weights = [3, 2, 1, 1, 1, 1]  # Random, Greedy, LR, DevCard, RH, SmartBot
    chosen = stdlib_random.choices(_heuristic_factories, weights=weights, k=3)
    return [cls(rng=stdlib_random.Random()) for cls in chosen]
```

**Change to**:
```python
def mixed_opponents() -> list[Agent]:
    # More balanced; emphasize stronger bots for better learning
    weights = [1, 1, 1, 1, 1, 2]  # Random, Greedy, LR, DevCard, RH, SmartBot(2)
    chosen = stdlib_random.choices(_heuristic_factories, weights=weights, k=3)
    return [cls(rng=stdlib_random.Random()) for cls in chosen]
```

**Rationale**:
- Current: Random appears 3× more often than SmartBot (too easy)
- New: SmartBot appears 2× more often than others (harder learning)
- Prevents early plateau against weak opponents

#### C2: Extend Stage 3 Duration (Optional)
**File**: `src/catan/ai/train.py:282-283`

**Current**:
```python
parser.add_argument("--stage3", type=int, default=1_000_000)
```

**Change to**:
```python
parser.add_argument("--stage3", type=int, default=2_000_000)
```

**Rationale**: Self-play is highest ROI training phase; extra time compounds improvements

---

## Implementation Order & Testing

### Step 1: Retrain with Phase 5 (No Code Changes)
```bash
# Takes 1-2 days on 4 parallel envs
uv run python -m catan.ai.train --n-envs 4
```
- Evaluate after: `uv run python scripts/evaluate.py --model models/stage3_selfplay --games 200`
- Expected: 5% → 10-15% vs SmartBot

### Step 2: Apply B1 (Reward Increase) + Retrain
```bash
# Edit gym_env.py (2 mins)
# Retrain (1-2 days)
uv run python -m catan.ai.train --n-envs 4
```
- Expected cumulative: 15% → 20-25% vs SmartBot

### Step 3: Apply C1 (Curriculum Hardening) + Retrain
```bash
# Edit train.py (1 min)
# Retrain (1-2 days)
uv run python -m catan.ai.train --n-envs 4
```
- Expected cumulative: 20-25% → 25-35% vs SmartBot

### Step 4: Apply B2 (Port Bonus) + Retrain
```bash
# Edit gym_env.py (5 mins)
# Retrain (1-2 days)
uv run python -m catan.ai.train --n-envs 4
```
- Expected cumulative: 25-35% → 30-40% vs SmartBot

---

## Verification Checklist

After each retrain:

```bash
# 1. Run evaluation (200+ games per opponent)
uv run python scripts/evaluate.py --model models/stage3_selfplay --games 500

# 2. Check settlement heatmap improvement
uv run python scripts/visualize.py  # Updates figures/settlement_heatmap.png

# 3. Run test suite (ensure no regressions)
uv run pytest tests/test_gym_env.py -v
uv run pytest tests/test_heuristic_agents.py -v

# 4. Inspect specific metrics
# - vs SmartBot: should show clear improvement trend
# - Settlement clustering: should drop 38.75% → ~10% → ~5%
# - VP scores: should increase across all opponents
```

---

## Expected Progression

### Timeline
| Phase | Change | Effort | Runtime | Cumulative Impact |
|-------|--------|--------|---------|------------------|
| A | Phase 5 retrain (485 dims) | - | 1-2d | 5% → 15% vs SmartBot |
| B1 | Reward increase | 2 min | 1-2d | 15% → 20-25% |
| C1 | Curriculum hardening | 1 min | 1-2d | 20-25% → 25-35% |
| B2 | Port bonus (opt) | 5 min | 1-2d | 25-35% → 30-40% |

**Total effort**: 4 lines of code, 4-8 days of training
**Total expected improvement**: 5% → 30-40% vs SmartBot (6-8× better)

---

## Advanced Options (Phase D — Lower Priority)

### D1: Add Diversity Penalty
```python
# Penalize redundant settlement placement
if isinstance(game_action, BuildSettlement):
    # Check if player already owns many settlements on same resource type
    # Subtract reward if so
    pass
```

### D2: Road Extension Bonus
```python
# Reward settlements that extend existing road chains
if isinstance(game_action, BuildSettlement):
    road_bonus = 0.2 * (len(connected_roads) / 8)  # Max 0.2
    reward += road_bonus
```

### D3: Early Game Incentives
```python
# Different reward schedule for placement vs main phases
if game.phase == GamePhase.PLACEMENT:
    settlement_reward *= 1.5  # Emphasize placement quality early
```

---

## Success Metrics

**Declare Phase 5 Complete When**:
- [ ] Vertex clustering: 38.75% → <5% ✓
- [ ] vs SmartBot: 5% → >15% ✓
- [ ] vs Greedy: 9% → >20% ✓
- [ ] Avg VP increase: 5-10% across opponents ✓
- [ ] Resource hoarding: Wheat 2.0 → <1.5 avg ✓
- [ ] All 337 tests passing ✓

---

## Questions to Monitor During Retraining

1. **Does clustering really disappear?** (settlement heatmap should show uniform distribution)
2. **Does win rate scale with opponent difficulty?** (should improve vs SmartBot more than Random)
3. **Do resource holdings normalize?** (wheat/ore should drop significantly)
4. **Does VP plateau improve?** (max VP should increase 10-15%)

---

## File Reference

| File | Change Type | Impact |
|------|-------------|--------|
| `src/catan/ai/gym_env.py` | Reward shaping | Medium (phase B) |
| `src/catan/ai/train.py` | Curriculum weights | Low-medium (phase C) |
| Default retrain | Obs space utilization | High (phase A) |

---

## Command Cheatsheet

```bash
# Phase A: Retrain only (recommended first)
uv run python -m catan.ai.train --n-envs 4

# Phase B+C: Apply both changes then retrain
# [Edit gym_env.py, train.py]
uv run python -m catan.ai.train --n-envs 4

# Quick test
uv run pytest tests/test_gym_env.py tests/test_heuristic_agents.py -v

# Evaluate results
uv run python scripts/evaluate.py --model models/stage3_selfplay --games 200

# Visualize (updates figures/)
uv run python scripts/visualize.py
```
