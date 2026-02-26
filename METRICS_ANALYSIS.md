# Phase 4 Metrics Deep Dive

## 📊 Win Rate Analysis

### Raw Performance vs Each Baseline
```
PPO Win Rate Distribution (200 games each):

Random        [████████████████████████] 91%
              ├─ PPO VP avg: 9.9
              ├─ Baseline VP: 3.2
              └─ Comment: Trivial; inflates confidence

LongestRoad   [████████            ] 32%
              ├─ PPO VP avg: 7.6
              ├─ Baseline VP: 6.0
              └─ Comment: Weak opponent pool

ResourceHoard [██████              ] 22%
              ├─ PPO VP avg: 6.8
              ├─ Baseline VP: 6.9
              └─ Comment: ❌ PPO LOSES by VP!

DevCard       [█████               ] 16%
              ├─ PPO VP avg: 6.1
              ├─ Baseline VP: 7.1
              └─ Comment: ❌ PPO LOSES by VP!

Greedy        [██                  ] 9%
              ├─ PPO VP avg: 5.9
              ├─ Baseline VP: 7.2
              └─ Comment: Heuristic mismatch

SmartBot      [█                   ] 5%
              ├─ PPO VP avg: 5.8
              ├─ Baseline VP: 7.1
              └─ Comment: ❌ WORST PERFORMANCE
```

### Performance Decay Analysis
```
Difficulty Progression:
Random → Greedy → SmartBot

Win Rate Drop:
  91% ────────────────────────
       └──→ 9% (─82 pts, -90%)
            └──→ 5% (−4 pts, −44%)

Inference: Model hits strategy ceiling @ ~6 VP

Opponent Strategy:
  Random: No strategy
  Greedy: Simple (build settlements)
  SmartBot: Complex (adaptive play)

PPO Strategy:
  vs Random: "Build anywhere, spam settlements"
  vs Greedy: "Compete with same greedy strategy"
  vs SmartBot: "Can't adapt to smart play"
```

---

## 🏘️ Settlement Placement Analysis

### Vertex Clustering Severity

```
Expected (uniform):
  Per vertex: 800 settlements ÷ 54 vertices = 14.8 avg

Actual distribution:
  Vertex 1:   150 placements (10.1× expected) ⚠️⚠️⚠️
  Vertex 32:  160 placements (10.8× expected) ⚠️⚠️⚠️
  Vertex 0:   42  placements (2.8× expected)
  Others:     ~2  placements avg (0.14× expected)

Clustering metric:
  Concentration at top 2: 310/800 = 38.75%
  Should be: 2/54 = 3.70%
  Multiplier: 10.5× too concentrated
```

### What This Reveals

1. **Static Pattern**: Model learned memorized IDs, not production logic
2. **Board Ignorance**: Board shuffles every game; model ignores new layout
3. **Generalization Failure**: Model can't evaluate positions, only recognize vertex labels
4. **Obs Space Issue**: 407-dim obs doesn't encode "what makes a good vertex"

### Expected After Phase 5 Retrain

```
With 485-dim obs + vertex production features:

Expected uniform clustering:
  Vertex 0-53: ~3-5 placements each (more even spread)
  No vertex > 10% of total

Target: Clustering metric ≤ 5% (near random)
```

---

## 💾 Resource Management Analysis

### End-Game Holdings Pattern

```
Resource Distribution (avg per player at game end):

Wheat   [████████████████████] 2.0
        └─ Should be ~0.5-1.0 (actively spent on dev cards)

Ore     [████████████████] 1.5
        └─ Should be ~0.3-0.8 (expensive resource)

Brick   [████████] 0.7
        └─ Should be ~0.5-1.0 (used for settlements/roads)

Wood    [████████] 0.8
        └─ Should be ~0.5-1.0 (used for roads/settlements)

Sheep   [██████] 0.6
        └─ Should be ~0.3-0.8 (flexible)
```

### Resource Spending Inefficiency

```
Model Behavior:
  - Farms wheat/ore (builds on good hexes)
  - Buys dev cards (converts wheat/ore to knights)
  - Doesn't deploy knights (leaves them unplayed)
  - Hoards remaining wheat/ore

Baseline Behavior:
  - Builds settlements strategically
  - Times dev card plays (knights → longest road)
  - Actively trades resources

Conclusion: Model doesn't understand dev card timing or trading
```

### Comparison: Expected vs Actual

```
Phase Category:
  Early (Turns 1-50):   Model ✓ Good (placement is OK)
  Mid (Turns 51-150):   Model ✗ Weak (strategy diverges)
  Late (Turns 151+):    Model ✗ Poor (can't catch up)

Resource hoarding suggests:
  1. Poor road placement (doesn't need to trade)
  2. Inefficient settlement strategy (generates excess)
  3. No long-term planning (doesn't cash in resources)
```

---

## 🔍 Observation Space Analysis

### Current Size: 485 dimensions
```
Hex Features (95 dims):
  ├─ Hex resources (95):  19 hex × 5 resources one-hot
  └─ Hex tokens (19):     19 hex token probabilities

Board State (146 dims):
  ├─ Robber (19):         19 hex one-hot
  ├─ Vertex owners (54):   54 vertices: [0=empty, 1=P0, 2=P1, ...]
  ├─ Vertex buildings (54): [0=none, 1=settlement, 2=city]
  ├─ Edge owners (72):     72 edges: [0=empty, 1=P0, ...]
  └─ Vertex production (54): NEW—sum of adjacent hex probabilities

Player State (140 dims):
  ├─ Resources (4×5=20):   4 players × 5 resource counts
  ├─ VP (4):              4 players victory points
  ├─ Dev cards (4×5=20):   4 players × 5 card types
  ├─ Knights (4):         4 players knights played
  ├─ Largest army (4):     4 players largest army flag
  ├─ Longest road (4):     4 players longest road lengths
  ├─ Ports (4×6=24):       4 players × 6 port types boolean
  └─ Income (4×5=20): NEW—per-resource income from buildings

Game State (8 dims):
  ├─ Phase (7):            7 phase one-hot
  ├─ Current player (4):    4 player one-hot
  ├─ Turn count (1):        normalized turn number
  ├─ Last roll (1):         normalized 2-12
  ├─ Deck size (1):         normalized dev cards remaining
  └─ Road builder remaining (1): turns left from road building card

Affordability (4 dims): NEW
  ├─ Can afford settlement
  ├─ Can afford city
  ├─ Can afford road
  └─ Can afford dev card
```

### Phase 5 New Features (78 dims, +19%)

| Feature | Dims | Purpose | Status |
|---------|------|---------|--------|
| Vertex production | 54 | Production value per vertex | ✅ Encoded, ❌ Unused (old model) |
| Player income | 20 | Expected income per resource | ✅ Encoded, ❌ Unused |
| Affordability flags | 4 | Can build actions | ✅ Encoded, ❌ Unused |

**Critical Issue**: These features are computed and encoded but the trained model has 407-dim input layer. It ignores the extra 78 dims!

---

## 🎯 Reward Shaping Analysis

### Current Reward Function

```python
# Terminal rewards
win:     +10.0 (agent wins)
loss:    -5.0  (agent loses)

# Intermediate rewards
per_vp:  +1.0  (per victory point gained)

# Settlement quality reward
quality: +0.1 to +0.3 (based on hex production)
         = 0.1 + (production / 0.42) × 0.2

# Max formula example:
best_settlement = (6+8+5=19 probability) / 0.42 = 1.0
quality_bonus = 0.1 + 1.0 × 0.2 = 0.3
```

### Reward Hierarchy Analysis

```
Magnitude Comparison:
  Win game:         +10.0  (terminal)
  Per VP:           +1.0   (intermediate, repeatable)
  Settlement best:  +0.3   (placement)

Relative Importance:
  VP reward is 3.3× more valuable than best settlement
  → Model learns "get VP" >> "place well"

Example trajectory:
  Good settlement:      +0.3 reward
  Bad settlement later: -0.0 (no penalty!)
  But bad placement costs -1.0 VP in future

Mismatch: Immediate reward says "settle anywhere"
          But future says "bad placement = bad VP"
```

### Settlement Reward Distribution

```
Production value (probability sum from 3 adjacent hexes):

Worst settlement (0.0 prob):    +0.1 reward
  └─ Still incentivized! (no penalty)

Good settlement (0.3 prob):     +0.16 reward

Best settlement (0.42+ prob):   +0.3 reward

Ratio: Best/worst = 0.3/0.1 = 3×
       This is WEAK for such a critical decision

By comparison:
  1 VP gain:  +1.0 reward
  Settlement bonus: +0.3 max

  Ratio: VP/settlement = 1.0/0.3 = 3.3×
         Placement is 3× less important than VP!
```

---

## 📈 Performance Scaling by Opponent Type

### VP Trajectory by Opponent

```
Random (weak):
  Game 1:   PPO=8 vs baseline=2   (PPO +6)
  Avg:      PPO=9.9 vs baseline=3.2

SmartBot (strong):
  Game 1:   PPO=6 vs baseline=7   (PPO -1)
  Avg:      PPO=5.8 vs baseline=7.1

Interpretation:
  - vs Random: PPO dominates through any strategy
  - vs SmartBot: PPO is fundamentally weak
  - Gap: 4.1 VP difference (massive in game with 10 VP target)
```

### Win Probability Analysis

```
Assumptions (all players equally skilled):
  - Each player has ~25% win rate
  - PPO beats Random: 91% >> 25% (HUGE advantage)
  - PPO vs SmartBot: 5% << 25% (HUGE disadvantage)

If we treat each baseline as "player skill level":
  - Random: Very weak (near random choices)
  - SmartBot: Strong (heuristic strategic play)

PPO performance:
  - Relative to Random: 91/25 = 3.6× stronger than expected
  - Relative to SmartBot: 5/25 = 0.2× weaker than expected

Conclusion: PPO learned to beat bad players, not good ones
```

---

## 🧪 Test Coverage Implications

### What's NOT Being Tested in Training

```
✓ Tested (Random agents test these):
  - Rule enforcement
  - Action validity
  - Basic decision making

✗ NOT tested (requires strong opponents):
  - Road blocking strategy
  - Port positioning value
  - Resource trade timing
  - Dev card deployment strategy
  - Longest road disruption
  - Robber placement defense
  - Early game timing

✗ NOT tested (model-specific):
  - Generalization to new boards
  - Decision robustness across game states
  - Strategy adaptation
```

### Why Curriculum Matters

```
Stage 1 (vs Random):
  ✓ Learns rules
  ✓ Basic settlement placement
  ✗ Doesn't learn strategy

Stage 2 (vs mixed weak bots):
  ✓ Learns to beat Greedy
  ✓ Learns to counter basic strategies
  ✗ Doesn't learn vs SmartBot or even basic blocking

Stage 3 (self-play):
  ✓ Should learn from self-generated opponents
  ✗ But if Stage 2 is weak, self-play inherits weakness

Result: Strong bottleneck at Stage 2
```

---

## Summary: Metrics → Root Causes

| Metric | Finding | Root Cause |
|--------|---------|-----------|
| 38.75% vertex clustering | Settlement placement by ID | Obs space doesn't encode production |
| 91% vs Random → 5% vs SmartBot | Extreme performance gap | Weak learning signal; easy curriculum |
| VP plateau at 5.8-6.8 | Can't scale performance | Placement + strategy both weak |
| Wheat 2.0, Ore 1.5 at end | Resource hoarding | No reward for trading/timing |
| Settlement reward +0.3 max | Weak placement incentive | Reward is 3× weaker than VP |
| Phase 5 obs unused | New features ignored | Model trained on old 407-dim space |

---

## Expected Improvement Trajectory

### Phase 5 Retrain (Immediate)
```
Issue Fixed: Obs space mismatch (407 → 485 dims)
Improvement: Vertex clustering 38% → ~10%

Before:  91% | 32% | 22% | 16% | 9% | 5%
After:   88% | 35% | 28% | 20% | 15% | 12%
         ────────────────────────────────
         Random through SmartBot win rates
```

### Reward Increase (Phase B1)
```
Issue Fixed: Settlement bonus too weak (3× weaker than VP)
Improvement: Model prioritizes good placement

Before:  88% | 35% | 28% | 20% | 15% | 12%
After:   85% | 38% | 32% | 24% | 20% | 18%
```

### Curriculum Hardening (Phase C1)
```
Issue Fixed: Stage 2 too easy (Random 3× more than SmartBot)
Improvement: Stronger foundations, better strategy scaling

Before:  85% | 38% | 32% | 24% | 20% | 18%
After:   82% | 40% | 35% | 28% | 25% | 25%
```

### Full Implementation
```
All changes + extended training:

Before:  91% | 32% | 22% | 16% | 9% | 5%
After:   82% | 42% | 38% | 32% | 30% | 30%
         ────────────────────────────────
         6× improvement vs SmartBot!
```
