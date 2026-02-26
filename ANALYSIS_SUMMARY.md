# Phase 4 Training Analysis — Executive Summary

## 🚨 Critical Issues Identified

### 1. Vertex Memorization (38.75% of settlements at 2 vertices)
```
Expected distribution: ~1.85% per vertex (1/54)
Actual:
  - Vertex 1:  150 placements (18.75% ← 10× too high!)
  - Vertex 32: 160 placements (20.00% ← 11× too high!)
  - Others:    490 placements (61.25%)

Total settlements: 800 (200 games × 4 players)
Memorized: 310 (38.75%)
```

**Why it matters**: Board shuffles every game. Model isn't learning "what makes a good vertex," it's just memorizing IDs.

---

## 🏆 Performance Analysis

### Win Rates vs Baseline Agents (200 games each)
```
Random        ████████████████████████ 91%     ← Too easy; inflates confidence
LongestRoad   ████████ 32%
ResourceHoard ██████ 22%
DevCard       █████ 16%
Greedy        ██ 9%
SmartBot      █ 5%                              ← Loses to best opponent!
```

### VP Plateau Effect
```
Opponent Difficulty:  Random → Greedy → SmartBot
Model VP Score:        9.9  → 5.9   → 5.8      ← Only 3.4 VP range
Baseline VP:           3.2  → 7.2   → 7.1      ← Increases with difficulty!
```

**Insight**: Model wins from cheese vs Random, but can't scale strategy. Baseline players adapt better to stronger opponents.

---

## 💾 Resource Management Problem

### End-Game Holdings (Average per player)
```
Wheat:  2.0 avg  ← Excessive (dev cards cost 1 ore+1 sheep+1 wheat)
Ore:    1.5 avg  ← Excessive (costly; should trade for settlements)
Brick:  0.7 avg  ← Underused
Wood:   0.8 avg  ← Underused
Sheep:  0.6 avg  ← Underused
```

**Pattern**: Model farms wheat/ore (dev cards) but doesn't use them. Classic suboptimal resource allocation.

---

## 🎯 Root Causes

| Issue | Cause | Evidence |
|-------|-------|----------|
| Vertex memorization | Obs space doesn't encode production value; model learns ID patterns | Heatmap shows static clustering despite board shuffling |
| VP plateau | Placement + strategy are weak; model hits ceiling | VP barely increases vs harder opponents |
| Resource hoarding | No incentive to trade or time plays | 2.0 wheat at game end indicates unused production |
| Performance cliff | Training curriculum too easy; early success prevents adaptation | Stage 2 opponents too weak relative to SmartBot |

---

## 🔧 Phase 5 Status: Obs Space Enriched, Model Not Retrained

**What was done**:
- Added 78 new features to obs space (19% larger)
- Vertex production values (54 dims) — sum of adjacent hex probabilities
- Player income expectations (20 dims) — per-resource income
- Affordability flags (4 dims) — can afford each building type

**What's missing**:
- Current model checkpoint trained on **407 dims**, not 485
- New features are encoded but ignored by old model
- **Needs full retrain** to utilize enriched obs space

---

## ✅ Recommended Improvements (Prioritized)

### TIER 1: IMMEDIATE (1-2 days)
**1. Retrain Full Curriculum with 485-dim Obs**
```bash
uv run python -m catan.ai.train --n-envs 4
```
- Forces model to learn production value instead of memorized IDs
- Expected: 50-70% reduction in vertex clustering, 2-5× win rate vs SmartBot
- Effort: 1-2 days of compute time

### TIER 2: HIGH IMPACT (1-3 hours)
**2. Increase Settlement Quality Reward 3-5×**
```python
# Current: +0.1 to +0.3 (30× weaker than VP)
reward += 0.1 + min(prod_value / 0.42, 1.0) * 0.2

# Proposed: +0.5 to +1.5 (matches VP strength)
reward += 0.5 + min(prod_value / 0.42, 1.0) * 1.0
```
- Makes placement quality equally important as VP gain
- Expected: Better opening play, less clustering
- Effort: 1 hour change + retrain

**3. Harden Stage 2 Curriculum**
```python
# Current: Random(3), Greedy(2), others(1)
weights = [1, 1, 1, 1, 1, 2]  # Equal with SmartBot(2)
```
- Prevents early plateau against weak opponents
- Expected: Stronger mid-game foundation
- Effort: 30 mins + retrain

### TIER 3: MEDIUM IMPACT (2-5 hours)
**4. Add Strategic Reward Bonuses**
- Port bonus: +0.5 for settlements touching 2:1/3:1 ports
- Diversity bonus: +0.1 for resource mix improvement
- Production diversity: reward varied adjacent hexes
- Expected: Less greedy play; better adaptation
- Effort: 2-3 hours

**5. Extend Self-Play Stage 3 Duration**
```bash
# Current: 1M steps
# Proposed: 2M steps
--stage3 2000000
```
- Self-play is highest value training; model needs time
- Expected: Better endgame strategy
- Effort: 12 hours extra compute

---

## 📊 Expected Improvements After Retrain + Adjustments

| Metric | Current | After P5 Retrain | After All Changes |
|--------|---------|------------------|-------------------|
| vs Random | 91% | 85-90% | 85-92% |
| vs SmartBot | 5% | 15-25% | 20-35% |
| vs Greedy | 9% | 20-30% | 25-40% |
| Settlement clustering | 38.75% | 8-12% | 4-6% |
| Avg VP (vs mixed) | 6.0 | 6.5-7.0 | 7.0-7.5 |

---

## 📝 Testing Checklist

- [ ] Retrain Phase 5 curriculum (1-2 days)
- [ ] Evaluate vs all baselines (200+ games each)
- [ ] Verify obs space encoding (heatmap shows 54 distinct vertices)
- [ ] Confirm resource management improves (wheat/ore < 1.0 avg)
- [ ] Check win rate scaling (should improve vs harder opponents)
- [ ] Run full test suite (337 tests)

---

## 🎯 Success Criteria

✅ **Phase 5 Complete When**:
1. Vertex clustering drops from 38.75% → ~5% (random baseline)
2. Win rate vs SmartBot: 5% → 15%+ (3× improvement)
3. Resource hoarding drops 50% (wheat < 1.5 avg)
4. VP scores increase 5-10% across opponents

---

## 📚 Files to Reference
- **Full analysis**: `C:\Users\Raunak\.claude\projects\...\memory\analysis.md`
- **Evaluation results**: `figures/win_rates.png`, `settlement_heatmap.png`, `resource_patterns.png`
- **Reward shaping code**: `src/catan/ai/gym_env.py:335-342`
- **Training code**: `src/catan/ai/train.py:108-124`, `:228-232`
