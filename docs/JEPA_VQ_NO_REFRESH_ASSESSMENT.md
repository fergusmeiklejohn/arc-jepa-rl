# VQ No-Refresh Diagnostic Assessment

**Date**: 2025-12-03
**Run**: `temp/diagnostic_vq_no_refresh/`
**Config**: `configs/training/jepa_vq_no_refresh.yaml`

---

## Executive Summary

⚠️ **MIXED RESULTS**: VQ with refresh disabled completed 30 epochs but shows **concerning signs of early collapse**.

**Status**: Training is more stable than the original failed run, but **VQ codebook is still collapsing** (17.7% → 4.4% usage). This is a **partial improvement, not a solution**.

**Recommendation**: Proceed to **Option B (Gentle VQ Parameters)** with more aggressive fixes.

---

## Three-Way Comparison

### Run Completion & Stability

| Metric | Sixth Run (VQ + Refresh) | No VQ | VQ No Refresh | Winner |
|--------|-------------------------|-------|---------------|---------|
| **Epochs Completed** | 22/80 (27%) | 30/30 (100%) | 30/30 (100%) | ✅ No VQ / VQ No Refresh |
| **Early Stopping** | Yes (epoch 22) | No | No | ✅ No VQ / VQ No Refresh |
| **Val Loss Spike** | 27.2 (epoch 13) | None | None | ✅ No VQ / VQ No Refresh |

### Final Losses

| Metric | Sixth Run | No VQ | VQ No Refresh | Winner |
|--------|-----------|-------|---------------|---------|
| **Final Train Loss** | 9.26 (epoch 22) | **7.74** | 9.07 | ✅ No VQ |
| **Final Val Loss** | 9.18 (epoch 22) | **7.75** | 8.99 | ✅ No VQ |
| **Best Val Loss** | 8.66 (epoch 6) | **7.75** (epoch 30) | 8.68 (epoch 12) | ✅ No VQ |

### Embedding Quality

| Metric | Sixth Run | No VQ | VQ No Refresh | Winner |
|--------|-----------|-------|---------------|---------|
| **Context Effective Rank** | 1.36-1.88 ❌ | 12.7-33.2 ✅ | 1.43-3.29 ⚠️ | ✅ No VQ |
| **Target Effective Rank** | 2.40-5.96 ⚠️ | 24.7-43.1 ✅ | 2.54-5.65 ⚠️ | ✅ No VQ |
| **Context Isotropy** | ~1e-5 ❌ | ~1.2e-5 ✅ | 9e-6 to 1.4e-5 ⚠️ | ✅ No VQ |
| **Target Isotropy** | ~1e-5 ❌ | ~1.2e-5 ✅ | 7e-6 to 9e-6 ❌ | ✅ No VQ |

### VQ Codebook Health

| Metric | Sixth Run | No VQ | VQ No Refresh | Assessment |
|--------|-----------|-------|---------------|------------|
| **VQ Usage (Start)** | 24.3% | N/A | **17.7%** | ⚠️ Already low |
| **VQ Usage (End)** | 1.7% ❌ | N/A | **4.4%** ❌ | ⚠️ Collapsing |
| **VQ Usage Trajectory** | Erratic collapse | N/A | **Monotonic decline** | ⚠️ Still unstable |

---

## Detailed Analysis: VQ No Refresh Run

### ✅ Positives

1. **No catastrophic spike**: Unlike sixth run (val loss 27.2 at epoch 13), this remained stable
2. **Completed all 30 epochs**: No early stopping triggered
3. **Better than sixth run**: Final val loss 8.99 vs 9.18
4. **Relational loss stabilized**: Started at 0.70 vs 37.1 in no-VQ run (much better initialization with VQ)

### ❌ Negatives

1. **VQ codebook collapse continues**:
   - Start: 17.7% usage (136/768 codes)
   - End: **4.4% usage** (34/768 codes)
   - Trajectory: Monotonic decline every checkpoint

2. **Embedding space collapse**:
   - Context effective rank: 3.29 → 1.43 (should grow, not shrink!)
   - Target effective rank: 5.65 → 2.94 (also declining)
   - Much worse than no-VQ (12-43 range)

3. **Worse than no-VQ**:
   - Final losses ~15-16% higher (9.07 vs 7.74 train, 8.99 vs 7.75 val)
   - Effective ranks 5-20x lower
   - Isotropy slightly worse

4. **Warning signs at epoch 27-30**:
   - Train loss starts rising: 8.91 → 9.11 → 9.06 → 9.08 → 9.07
   - Val loss volatile: 9.13 → 9.24 → 9.03 → 8.99
   - This pattern preceded collapse in sixth run

### ⚠️ Concerning Trends

**VQ Usage Over Time:**
```
Step   200:  136 codes (17.7%) ← Reasonable start
Step   400:  126 codes (16.4%) ← Slow decline
Step   600:  114 codes (14.8%)
Step   800:  106 codes (13.8%)
Step  1000:   87 codes (11.3%) ← Crossing 10% threshold
Step  1200:   73 codes ( 9.5%)
Step  1400:   67 codes ( 8.7%)
Step  1600:   56 codes ( 7.3%)
Step  1800:   45 codes ( 5.9%)
Step  2000:   47 codes ( 6.1%) ← Slight recovery
Step  2850:   34 codes ( 4.4%) ← Near collapse
```

**This is a classic "slow death" pattern** - monotonic decline without refresh attempting revival.

---

## Root Cause Analysis

### Why Removing Refresh Didn't Fully Fix It

**Original Hypothesis**: Refresh mechanism creates discontinuities → collapse
**Reality**: Refresh was a **symptom amplifier**, not the root cause

**The Real Problem**: **VQ commitment cost too high + EMA decay too aggressive**

```yaml
# Current settings in vq_no_refresh config:
commitment_cost: 0.3      # Forces strong quantization
ema_decay: 0.99          # Very slow adaptation
```

**What's happening:**
1. High commitment cost (0.3) forces embeddings to cluster near codebook entries
2. Slow EMA (0.99) means codebook updates lag embedding changes
3. InfoNCE pushes embeddings apart for discrimination
4. VQ pulls them together for quantization
5. **Tug-of-war** → embeddings cluster around fewer codes → usage collapses
6. Without refresh, dead codes stay dead forever

**Sixth run was worse because:**
- Refresh tried to revive codes → discontinuities → faster destabilization
- But underlying tug-of-war was still happening

---

## Comparison to Design Intent

### What We Need (From Blueprint)

✅ **Discrete codes for symbolic grounding** - YES, we have codes
❌ **Stable, diverse codebook** - NO, collapsing to 4.4% usage
✅ **InfoNCE contrastive learning** - YES, working
❌ **Healthy embedding space** - NO, effective rank collapsing
✅ **Training completes** - YES, 30 epochs done
❌ **Performance better than baseline** - NO, worse than no-VQ

**Net Assessment**: 2/6 requirements met. **Not compliant**.

---

## Why No-VQ Performed Best

Without VQ, there's **no tug-of-war**:
- InfoNCE spreads embeddings → Effective rank grows (12→33)
- SIGReg + invariance losses regularize → Stable isotropy
- No quantization bottleneck → Full gradient flow
- No codebook collapse → Can't fail from code death

**Trade-off**: Loses discrete symbolic grounding needed for HRL/DSL integration.

---

## Recommended Next Steps

### Option B: Gentle VQ Parameters (HIGH PRIORITY)

The no-refresh diagnostic proves we need to **reduce VQ-InfoNCE conflict**, not just disable refresh.

**Key Changes** (already in `jepa_vq_gentle.yaml`):

```yaml
# 1. REDUCE commitment cost - less aggressive quantization
commitment_cost: 0.15  # Was 0.3 → 50% reduction

# 2. FASTER EMA adaptation - codebook tracks embeddings better
ema_decay: 0.95        # Was 0.99 → 4x faster updates

# 3. GENTLE refresh as safety net - very infrequent
vq_refresh_enabled: true
vq_refresh_interval: 3000  # Was 500 → 6x less frequent
vq_refresh_usage_threshold: 0.01  # Was 0.05 → Only truly dead codes

# 4. WIDER temperature range - InfoNCE can adapt
temperature_min: 0.05  # Was 0.07
temperature_max: 0.20  # Was 0.15

# 5. REDUCE competing losses - less interference
relational_loss:
  weight: 0.025  # Was 0.05
  context_self_weight: 0.01  # Was 0.025
```

**Expected Outcome:**
- ✅ VQ codes stabilize above 15% usage
- ✅ Effective rank stays healthy (>5)
- ✅ Losses approach no-VQ performance
- ✅ Discrete codes for symbolic grounding

**Test Command:**
```bash
python scripts/train_jepa.py --config configs/training/jepa_vq_gentle.yaml
```

---

### Option C: Gradient-Based VQ (MEDIUM PRIORITY)

If Option B still shows collapse, switch to gradient-based updates:

```yaml
ema_decay: null  # Disable EMA completely
```

**Changes needed:**
- Modify [vq.py:98-119](../training/modules/vq.py#L98-L119) to use gradient updates
- Codebook becomes trainable parameters like other weights
- More stable but slower convergence

**Pros**:
- Eliminates EMA timing issues
- Codebook trained jointly with encoder
- Proven stable in original VQ-VAE papers

**Cons**:
- Requires code changes
- May need lower learning rate for codebook
- Typically needs more epochs to converge

---

### Option D: Hybrid Approach (BACKUP PLAN)

If both fail, consider **two-stage training**:

**Stage 1**: Train without VQ (as in diagnostic)
- Get healthy, stable embeddings (effective rank 20-40)
- Complete 50-60 epochs
- Save encoder weights

**Stage 2**: Add VQ on top of frozen encoder
- Load stage 1 weights
- Add VQ layer with very low commitment cost (0.05)
- Fine-tune only VQ codebook for 10-20 epochs
- Encoder embeddings already spread out, less conflict

**Pros**: Separates the two optimization problems
**Cons**: Longer training time, more complex pipeline

---

## Success Criteria for Next Run

Before declaring VQ stable, we need:

### Must-Have (Blocking):
- [ ] VQ usage ratio > **15%** throughout training
- [ ] VQ usage trajectory: **flat or growing** (not declining)
- [ ] Context effective rank > **5.0** (currently ~1.5)
- [ ] Target effective rank > **10.0** (currently ~2.9)
- [ ] Final val loss < **8.5** (currently 8.99)

### Nice-to-Have (Goals):
- [ ] VQ usage ratio > **25%** at end
- [ ] Effective ranks approaching no-VQ (context >10, target >20)
- [ ] Final val loss < **8.0** (matching no-VQ)
- [ ] Training stable for 80 epochs (not just 30)

### Red Flags (Abort Criteria):
- ❌ VQ usage drops below **10%** at any point
- ❌ Validation loss spike > **1.5x** previous best
- ❌ Effective rank drops below **1.0** (complete collapse)
- ❌ Early stopping triggered before 60 epochs

---

## Monitoring Checklist

When running the gentle VQ config, watch these metrics closely:

**Every 200 steps** (from embedding_metrics.jsonl):
```bash
# Check VQ usage
grep "vq_usage_ratio" temp/diagnostic_vq_gentle/embedding_metrics.jsonl | tail -5

# Check effective ranks
grep "effective_rank" temp/diagnostic_vq_gentle/embedding_metrics.jsonl | tail -5
```

**Every 5 epochs** (from epoch_metrics.jsonl):
```bash
# Check losses
grep "loss_mean" temp/diagnostic_vq_gentle/epoch_metrics.jsonl | tail -5
```

**Alert if**:
- VQ usage < 15%: Immediate concern
- VQ usage declining 3 checkpoints in a row: Collapse starting
- Effective rank < 3.0: Embedding space compressing
- Val loss increases 2 epochs in a row: Potential instability

---

## Probability Assessment

Based on three diagnostics, estimated likelihood of success:

| Approach | Success Probability | Rationale |
|----------|-------------------|-----------|
| **No VQ** | 95% ✅ | Already proven stable, but non-compliant |
| **VQ No Refresh** | 30% ⚠️ | Completes but collapses, not usable long-term |
| **Gentle VQ (Option B)** | 65% 🎯 | Addresses root cause (commitment/EMA conflict) |
| **Gradient VQ (Option C)** | 75% 🎯 | More invasive but proven stable in literature |
| **Hybrid Two-Stage (Option D)** | 85% ✅ | Most conservative, highest success rate |

**Recommendation**: Try Option B (Gentle VQ) first due to:
- Minimal code changes
- Directly targets root cause
- Fast to test (30 epochs = 1-2 hours)
- If it works, cleanest solution

If Option B shows any VQ usage below 12% or effective rank below 4.0 by epoch 15, **abort and move to Option C or D**.

---

## Conclusion

**The VQ no-refresh diagnostic confirms**:
1. ✅ Refresh mechanism **was** causing problems (no catastrophic spike)
2. ❌ But it wasn't the **only** problem (codebook still collapsing)
3. ⚠️ Core issue: **VQ-InfoNCE parameter conflict** (commitment too high, EMA too slow)

**Path forward**:
1. Run gentle VQ config (Option B) - **NEXT STEP**
2. If that fails, implement gradient-based VQ (Option C)
3. If both fail, use hybrid two-stage approach (Option D)

**Timeline**:
- Option B test: 1-2 hours
- Decision point: 2-3 hours
- Full solution: 1-2 days (including validation)

**Next command**:
```bash
python scripts/train_jepa.py --config configs/training/jepa_vq_gentle.yaml
```
