# Phase 5 Training - Executive Summary

**Status**: ✅ COMPLETE  
**Date**: February 25, 2026  
**Time**: 4:43 PM  

---

## Quick Facts

| Metric | Value |
|--------|-------|
| **Training Duration** | 14-16 hours (16 parallel envs) |
| **Total Steps** | 8M (1M Stage 1 + 5M Stage 2 + 2M Stage 3) |
| **Model File** | `models/stage3_selfplay.zip` |
| **OBS Space** | 485 dims (19% larger than Phase 4) |
| **Network** | 256×256 (pi/vf layers) |

---

## What Changed (Phase 4 → Phase 5)

### Hyperparameters
- Learning rate: 3e-4 → **1e-4** (stabilize)
- Batch size: 128 → **256** (reduce variance)
- Epochs: 10 → **4** (prevent overfitting)
- **NEW**: target_kl = **0.015** (early-stop large updates)

### Training Duration
- Stage 1: 500K → **1M** (+100%)
- Stage 2: 2M → **5M** (+150%)
- Stage 3: 1M → **2M** (+100%)
- **Total**: 3.5M → **8M** (+128%)

### Observation Space
- Added 78 features (485 total, was 407)
- Vertex production values (54 dims)
- Player income expectations (20 dims)
- Affordability flags (4 dims)

---

## Phase 4 Issues Diagnosis

| Issue | Severity | Phase 5 Fix |
|-------|----------|------------|
| Vertex memorization (38.75% at 2 vertices) | 🔴 Critical | Rich obs space now encodes production |
| KL divergence too high (0.07 vs 0.015 target) | 🔴 Critical | target_kl=0.015 added to stop bad updates |
| Explained variance too low (0.40) | 🟠 High | More training steps (8M vs 3.5M) |
| Strategy plateau vs heuristics | 🟠 High | Better hyperparams + curriculum |
| Resource hoarding | 🟡 Medium | Richer obs should help |

---

## Expected Improvements

### Win Rate Uplift Target
| Opponent | Phase 4 | Phase 5 Expected |
|----------|---------|-----------------|
| Random | 57.5% | 65-75% |
| Greedy | 1.5% | 15-25% |
| SmartBot | **1%** | **10-20%** |

**Success metric**: SmartBot win rate > 10% (10× improvement)

### Settlement Clustering Fix
| Metric | Phase 4 | Phase 5 Target |
|--------|---------|----------------|
| Top 2 vertices | 38.75% | 4-8% |
| Memorization | Very high | Much lower |

---

## What's Next

### 🟢 DO THIS FIRST (30 min)
```bash
# Run quick evaluation (50 games per opponent)
uv run python scripts/evaluate.py --model models/stage3_selfplay --games 50
```

### 🟡 THEN DO THIS (2-3 hours)
```bash
# Full evaluation (1000 games per opponent)
uv run python scripts/evaluate.py --model models/stage3_selfplay --games 1000
```

### 🔵 COMPARE RESULTS
Compare Phase 5 results to Phase 4 baseline:
- If SmartBot improved to >10%: ✅ Success — keep model
- If SmartBot still <5%: ❌ Debug needed — check obs space usage

---

## Files to Review

| File | Purpose |
|------|---------|
| [RECENT_RUN_ANALYSIS.md](RECENT_RUN_ANALYSIS.md) | Detailed Phase 5 analysis |
| [TRAINING_IMPROVEMENT_PLAN.md](TRAINING_IMPROVEMENT_PLAN.md) | Phase 5 design spec |
| [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md) | Phase 4 problems & fixes |
| [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md) | Deep dive on Phase 4 failures |

---

## Key Insights from Phase 4 Analysis

> **Vertex Memorization Problem**: Model learned that vertices 1 and 32 are "good" (18-20% of placements), ignoring the actual board layout. This is because the obs space didn't encode what makes a vertex valuable.

> **KL Divergence Issue**: Updates were too aggressive (0.07 vs target 0.015), causing instability. The clip_range=0.2 wasn't strong enough without a target_kl.

> **Training Plateau**: Model hit a ceiling against harder opponents because it didn't learn robust strategy, just pattern-matched against Random.

**How Phase 5 Addresses These**:
- ✅ Obs enriched with production values → model can learn placement value
- ✅ target_kl=0.015 added → prevents destructive large updates
- ✅ Longer training (8M vs 3.5M) → more time for strategy learning
- ✅ Better curriculum (weighted Stage 2) → harder opponents earlier

---

## Timeline

```
Phase 4 Completed: ~Feb 24 afternoon
  Issue Analysis: Feb 24-25
  
Phase 5 Training Started: ~Feb 25 morning
  (with 16 parallel envs)
  
Phase 5 Training Completed: Feb 25 @ 4:43 PM
  ✅ All 8M steps finished
  ✅ Model saved
  
NEXT: Evaluate model (30 min - 3 hours)
```

---

## Success Criteria

### Minimum (Phase 5 Justified)
- [ ] SmartBot win rate: ≥5% (up from 1%)
- [ ] Random win rate: ≥60% (any improvement)

### Target (Phase 5 Successful)
- [ ] SmartBot win rate: ≥10% (10× improvement)
- [ ] Greedy win rate: ≥10% (6× improvement)
- [ ] Vertex clustering: <10% (39× improvement)

### Stretch (Phase 5 Excellent)
- [ ] SmartBot win rate: ≥15%
- [ ] All heuristics: ≥10%
- [ ] SmartBot VP parity: 7.0+ (vs 7.1 baseline)

---

## Recommended Reading Order

1. **This file** — Start here for overview
2. [`RECENT_RUN_ANALYSIS.md`](RECENT_RUN_ANALYSIS.md) — Full details on Phase 5
3. [`TRAINING_IMPROVEMENT_PLAN.md`](TRAINING_IMPROVEMENT_PLAN.md) — Design decisions
4. [`METRICS_ANALYSIS.md`](METRICS_ANALYSIS.md) — Why Phase 4 failed
