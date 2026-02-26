# Phase 4 Analysis — Complete Index

## 📋 Documentation Map

This analysis provides comprehensive insights into the Phase 4 training run (8M steps, 256x256 network) and actionable improvement recommendations.

### 📄 Core Documents

| Document | Purpose | Read If | Length |
|----------|---------|---------|--------|
| **[ANALYSIS_SUMMARY.md](./ANALYSIS_SUMMARY.md)** | Executive overview + key findings | You want quick insights | 5 min |
| **[METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)** | Deep dive into all performance metrics | You want detailed numbers | 15 min |
| **[IMPROVEMENT_ROADMAP.md](./IMPROVEMENT_ROADMAP.md)** | Step-by-step implementation guide | You want to fix issues | 10 min |
| **[analysis.md](C:\Users\Raunak\.claude\projects\c--Users-Raunak-Documents-Catan-Project\memory\analysis.md)** | Full technical analysis (memory) | You want full context | 20 min |

---

## 🎯 Quick Navigation by Use Case

### "I want the TL;DR"
→ Read: [ANALYSIS_SUMMARY.md](./ANALYSIS_SUMMARY.md)

**Key takeaway**: Model suffers from vertex memorization (38.75% settlements at 2 vertices) and weak generalization. Phase 5 obs space is enriched but untrained. Retrain + reward tuning will fix it.

---

### "I want to implement fixes"
→ Read: [IMPROVEMENT_ROADMAP.md](./IMPROVEMENT_ROADMAP.md)

**Actions**:
1. Retrain full curriculum (no code changes, 1-2 days)
2. Increase settlement reward +0.1-0.3 → +0.5-1.5 (2 min code change)
3. Harden Stage 2 curriculum weights (1 min code change)
4. Rerun evaluation and visualizations

**Expected**: 5% → 30% win rate vs SmartBot

---

### "I want to understand the problems"
→ Read: [METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)

**Topics covered**:
- Settlement clustering analysis (38.75% at 2 vertices)
- Performance decay (91% vs Random → 5% vs SmartBot)
- Resource hoarding (2.0 wheat, 1.5 ore at game end)
- Reward signal weakness (settlement bonus 30× weaker than VP)
- Obs space enrichment details (485 dims, 78 new features)

---

### "I want all the details"
→ Read: [analysis.md](C:\Users\Raunak\.claude\projects\c--Users-Raunak-Documents-Catan-Project\memory\analysis.md)

**Coverage**:
- Root cause analysis
- Detailed recommendations with expected impacts
- Testing procedures
- Success criteria

---

## 🚨 Critical Findings (Summary)

### Problem 1: Vertex Memorization
- **Finding**: 38.75% of settlements placed at 2 vertices (out of 54)
- **Root Cause**: Obs space (407 dims) doesn't encode production value; model learns ID patterns
- **Impact**: Fails to generalize to new board layouts (they shuffle every game)
- **Fix**: Retrain with Phase 5 enriched obs (485 dims with vertex production features)
- **Expected**: Clustering drops to ~5% (random baseline)

### Problem 2: Performance Cliff
- **Finding**: 91% vs Random, 5% vs SmartBot (18× difference)
- **Root Cause**: Weak curriculum; Stage 2 too easy (Random 3× more common than SmartBot)
- **Impact**: Model doesn't learn strategic play, just cheese weak opponents
- **Fix**: Harden Stage 2 weights + extend Stage 3 training
- **Expected**: SmartBot performance improves to 15-30%

### Problem 3: Weak Placement Reward
- **Finding**: Settlement quality reward +0.1-0.3 (max) vs VP reward +1.0
- **Root Cause**: Reward design doesn't prioritize placement quality
- **Impact**: Model ignores settlement quality, builds anywhere
- **Fix**: Increase settlement bonus to +0.5-1.5 (match VP strength)
- **Expected**: Better opening play, reduced clustering

### Problem 4: Resource Hoarding
- **Finding**: Game-end holdings of 2.0 wheat, 1.5 ore (should be <1.0 each)
- **Root Cause**: Poor trading/timing logic; no incentive for resource management
- **Impact**: Inefficient development card strategy
- **Fix**: Add port bonus + diversity rewards (optional but high value)
- **Expected**: Balanced resource usage, better timing

### Problem 5: Phase 5 Implementation Incomplete
- **Finding**: Obs space enriched to 485 dims but model trained on 407 dims
- **Root Cause**: Phase 5 added features without retraining
- **Impact**: New vertex production/income/affordability data is unused
- **Fix**: Retrain full curriculum with new model input layer
- **Expected**: Fixes all placement-related issues

---

## 📊 Metrics at a Glance

### Win Rate Summary
```
vs Random:        91% → expect 85-92% after fix
vs SmartBot:      5%  → expect 15-35% after fix
vs Greedy:        9%  → expect 20-40% after fix
vs LongestRoad:   32% → expect 35-45% after fix
vs DevCard:       16% → expect 25-35% after fix
vs ResourceHoard: 22% → expect 28-38% after fix

Baseline: All opponents at 25% win rate (fair play)
```

### Settlement Placement
```
Current: 38.75% at 2 vertices (should be ~3.7%)
After Phase 5 retrain: ~10%
After full fixes: ~5% (near random)
```

### Resource Management
```
Current: Wheat 2.0, Ore 1.5 at game end
After fixes: Wheat <1.0, Ore <1.0 (efficient usage)
```

---

## ✅ Implementation Checklist

### Phase A: Retrain (Priority 1, 1-2 days, No code changes)
- [ ] Run: `uv run python -m catan.ai.train --n-envs 4`
- [ ] Evaluate: `uv run python scripts/evaluate.py --model models/stage3_selfplay`
- [ ] Check settlement heatmap: `uv run python scripts/visualize.py`
- [ ] Expected: SmartBot 5% → 12-15%

### Phase B1: Increase Settlement Reward (Priority 2, 2 min code, 1-2 days train)
- [ ] Edit `src/catan/ai/gym_env.py:342`
- [ ] Change: `reward += 0.1 + min(prod_value / 0.42, 1.0) * 0.2`
- [ ] To: `reward += 0.5 + min(prod_value / 0.42, 1.0) * 1.0`
- [ ] Retrain & evaluate
- [ ] Expected: SmartBot 12-15% → 18-25%

### Phase C1: Harden Curriculum (Priority 3, 1 min code, 1-2 days train)
- [ ] Edit `src/catan/ai/train.py:230`
- [ ] Change: `weights = [3, 2, 1, 1, 1, 1]`
- [ ] To: `weights = [1, 1, 1, 1, 1, 2]`
- [ ] Retrain & evaluate
- [ ] Expected: SmartBot 18-25% → 25-35%

### Phase B2: Add Port Bonus (Priority 4, 5 min code, 1-2 days train, Optional)
- [ ] Edit `src/catan/ai/gym_env.py:343-360`
- [ ] Add port bonus logic
- [ ] Retrain & evaluate
- [ ] Expected: Marginal improvement

---

## 📈 Expected Improvement Trajectory

```
Stage                 SmartBot WR   Avg VP    Settlement Clustering
──────────────────────────────────────────────────────────────────
Current (8M)          5%            5.8       38.75% (bad)
After Phase A         12-15%        6.2       10-15% (better)
After Phase B1        18-25%        6.5       5-10%  (good)
After Phase C1        25-35%        6.8       4-6%   (excellent)
After Phase B2        30-40%        7.0       3-5%   (optimal)

Timeline: 4-8 days of cumulative training
Code changes: ~3 lines total (5 minutes)
Effort: Low code, high compute
```

---

## 🔬 Technical Details

### Files Involved
| File | Change | Priority |
|------|--------|----------|
| `src/catan/ai/gym_env.py` | Reward shaping | 1 (Phase B1) |
| `src/catan/ai/train.py` | Curriculum weights | 2 (Phase C1) |
| Training script | Phase 5 retrain | 0 (Phase A) |

### Key Hyperparameters (No changes recommended)
- Network: 256×256 (good)
- Learning rate: 1e-4 linear decay (good)
- PPO params: n_steps=4096, batch_size=256 (good)
- Timesteps: 1M/5M/1M (consider extending Stage 3 to 2M)

---

## 🧪 Validation Steps

### After Each Retrain
```bash
# 1. Evaluation (take ~30 min for 200 games)
uv run python scripts/evaluate.py --model models/stage3_selfplay --games 200

# 2. Visualizations
uv run python scripts/visualize.py

# 3. Tests
uv run pytest tests/test_gym_env.py -v

# 4. Manual inspection
# - Does settlement heatmap show uniform distribution?
# - Do VP scores increase?
# - Do win rates scale with opponent difficulty?
```

---

## 📚 References in This Repo

### Analysis Documents (Created)
- `ANALYSIS_SUMMARY.md` — This directory
- `IMPROVEMENT_ROADMAP.md` — This directory
- `METRICS_ANALYSIS.md` — This directory
- `analysis.md` — Memory directory

### Existing Project Files
- **Evaluation**: `scripts/evaluate.py`
- **Visualization**: `scripts/visualize.py`
- **Training**: `src/catan/ai/train.py`
- **Environment**: `src/catan/ai/gym_env.py`
- **Tests**: `tests/test_gym_env.py`, `tests/test_heuristic_agents.py`
- **Figures**: `figures/settlement_heatmap.png`, `figures/win_rates.png`, `figures/resource_patterns.png`

---

## 💡 Key Insights

### Why Phase 5 Matters
The new obs space features (54 vertex production values, 20 player income, 4 affordability flags) directly address the root cause of vertex memorization. Without retraining, the model's input layer can't even read these 78 new features.

### Why Curriculum Hardening Matters
A model that beats Random 91% but SmartBot 5% hasn't learned strategy—it's learned to exploit weak opponents. Harder Stage 2 curriculum forces real learning early, preventing plateau at Stage 3.

### Why Reward Tuning Matters
Rewards are the entire signal for learning. Settlement quality reward (+0.3) being 3× weaker than VP reward (+1.0) tells the model "placement doesn't matter." Fixing this inverts that priority.

---

## 🎓 Learning Outcomes

After implementing these changes, you'll have:
1. **Fixed vertex memorization** (learned value ≠ memorized IDs)
2. **Stronger curriculum** (forced to beat harder opponents)
3. **Better reward signals** (incentivized strategic play)
4. **3-6× improvement** in hard opponent performance (5% → 30%)

---

## 📞 Questions to Ask While Retraining

1. **Does clustering really drop?** (check settlement heatmap after each phase)
2. **Is the SmartBot win rate improving linearly?** (expect gains at each phase)
3. **Are resource holdings normalizing?** (wheat/ore should drop)
4. **Is win rate scaling sensibly?** (should improve more vs harder opponents)
5. **Are tests still passing?** (no regressions)

---

## ⏭️ Next Steps

**Immediate**:
1. Read [IMPROVEMENT_ROADMAP.md](./IMPROVEMENT_ROADMAP.md)
2. Run Phase A (retrain) — takes 1-2 days
3. Evaluate results

**Short-term** (after Phase A):
4. Apply Phase B1 & C1 (5 min code, 1-2 days training each)
5. Evaluate after each phase

**Long-term** (after Phase C1):
6. Consider Phase B2 (port/diversity rewards)
7. Explore extended training (total 10-15M steps)

---

**Start with**: [ANALYSIS_SUMMARY.md](./ANALYSIS_SUMMARY.md) for quick understanding
**Then read**: [IMPROVEMENT_ROADMAP.md](./IMPROVEMENT_ROADMAP.md) for implementation
**Reference**: [METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md) for detailed metrics
