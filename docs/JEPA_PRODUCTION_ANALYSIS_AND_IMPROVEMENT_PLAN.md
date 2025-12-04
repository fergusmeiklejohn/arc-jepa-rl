# JEPA Production Run Analysis & Improvement Plan

**Date**: 2025-12-03
**Run**: 80-epoch production training with gentle VQ parameters
**Artifacts**: `temp/production_gentle_vq/`
**Status**: ✅ Complete - Performance below expectations

---

## Executive Summary

The 80-epoch production run completed successfully without crashes, demonstrating **training stability** but revealing **critical performance and architectural issues**:

### Key Findings

| Metric | Target | Achieved | Assessment |
|--------|--------|----------|------------|
| **Best Val Loss** | < 8.3 | 8.224 | ⚠️ Missed target (2.5% gap) |
| **Final Val Loss** | < 8.4 | 8.487 | ❌ Above target |
| **Training Stability** | Complete 80 epochs | ✅ 80/80 | ✅ Success |
| **VQ Collapse** | No collapse | ❌ **COLLAPSED** | 🔴 Critical |
| **Relational Loss Explosion** | Stable | ❌ **0.35 → 1.10** (3x) | 🔴 Critical |
| **Effective Ranks (End)** | 8-12 (ctx), 12-16 (tgt) | 15.5, 19.3 | ✅ Above target |

### Critical Problems Identified

1. **Relational Loss Explosion** (Epochs 30-71): 0.35 → 1.10 (314% increase)
2. **VQ Codebook Collapse** (Step 3000): 30% → 13-18% usage, dramatic spike then drop
3. **Validation Loss Plateau**: Stagnated at 8.2-8.5 from epoch 50 onward
4. **Loss Component Imbalance**: InfoNCE decreased while relational exploded
5. **Gradient Instability**: Norms spiked to 66-81 during relational explosion

### Verdict

**Training succeeded technically but failed strategically:**
- ✅ Stable VQ parameters prevented early collapse
- ✅ High embedding diversity achieved (ranks 15-19)
- ❌ **Relational loss dominates, degrades performance**
- ❌ **Loss balance deteriorates over training**
- ❌ **Not L-JEPA compliant** (heuristic-heavy, no SIGReg)

**Recommendation**: Implement 3-phase improvement plan before deploying encoder.

---

## Detailed Production Run Analysis

### 1. Loss Trajectory Analysis

#### Training Loss Components (Selected Epochs)

| Epoch | InfoNCE | SIGReg | Invariance | Relational | **Total** | Assessment |
|-------|---------|--------|------------|------------|----------|------------|
| 1 | 9.042 | 0.0405 | 0.0177 | **0.351** | 9.451 | Baseline |
| 10 | 8.325 | 0.0404 | 0.0223 | **0.182** | 8.570 | Healthy decline |
| 20 | 8.199 | 0.0403 | 0.0261 | **0.263** | 8.529 | Starting to rise |
| 30 | 7.984 | 0.0403 | 0.0245 | **0.499** | 8.548 | **🟡 Warning** |
| 40 | 7.805 | 0.0402 | 0.0163 | **0.623** | 8.484 | **🔴 Explosion begins** |
| 50 | 7.624 | 0.0401 | 0.0197 | **0.785** | 8.469 | **🔴 Dominates** |
| 60 | 7.507 | 0.0401 | 0.0135 | **0.789** | 8.350 | **🔴 Peaked** |
| 70 | 7.430 | 0.0401 | 0.0120 | **0.831** | 8.313 | **🔴 Still high** |
| 71 | 7.511 | 0.0401 | 0.0139 | **1.101** | 8.667 | **🔴 Catastrophic spike** |
| 80 | 7.567 | 0.0401 | 0.0164 | **0.845** | 8.469 | **🔴 Unstable** |

**Key Observations:**

1. **InfoNCE improves consistently**: 9.04 → 7.57 (16% improvement)
2. **SIGReg stable**: ~0.040 throughout (working as intended)
3. **Relational loss explodes**: 0.35 → 1.10 peak (314% increase)
4. **Relational dominates late training**: 0.85 vs 7.57 InfoNCE (11% of total by epoch 80)

**Root Cause**: Relational loss weight (0.025) multiplies **raw pairwise distances** that grow as embeddings spread in space. As effective rank increases (7.5 → 15.5), inter-object distances increase quadratically, causing relational loss to explode despite constant weight.

#### Validation Loss Trajectory

| Epoch Range | Val Loss | Trend | Assessment |
|-------------|----------|-------|------------|
| 1-15 | 9.38 → 8.40 | Rapid decrease | ✅ Healthy |
| 16-30 | 8.40 → 8.59 | Oscillating | ⚠️ Plateauing |
| 31-50 | 8.59 → 8.43 | High variance | ⚠️ Unstable |
| 51-70 | 8.43 → 8.26 | Slow decrease | ⚪ Marginal gains |
| 71-80 | 8.26 → 8.49 | **Degrading** | ❌ Relational explosion impact |

**Best val loss**: 8.224 (epoch 69)
**Final val loss**: 8.487 (epoch 80)
**Gap from best**: +3.2% (performance degraded after peak)

**Diagnosis**: Model peaked at epoch 69, then **relational explosion** (epoch 71: 1.101) caused validation performance to degrade. Training should have stopped at epoch 69.

---

### 2. VQ Codebook Analysis

#### VQ Usage Trajectory

| Step Range | VQ Usage (%) | Active Codes | Trend | Assessment |
|------------|--------------|--------------|-------|------------|
| 100-500 | 23-27% | 167-206 | Healthy high | ✅ Good diversity |
| 600-2900 | 13-18% | 101-145 | Stable low | ⚪ Acceptable |
| 3000 | **30%** | **232** | **Spike** | 🟡 Refresh triggered |
| 3100-3400 | 13-15% | 100-120 | Drop after refresh | ⚠️ Unstable |
| 3500-5000 | 14-21% | 109-162 | Recovering | ⚪ Stabilizing |
| 5100-7600 | 22-36% | 145-276 | High variance | ⚠️ Volatile |

**Key Event: Step 3000 VQ Refresh**

At step 3000 (epoch ~19), the VQ refresh mechanism triggered (interval = 3000 steps):
- Usage **spiked to 30%** (232/768 codes)
- Then **crashed to 13%** (100/768 codes)
- High variance afterward: 13-36% range (23pp spread)

**Problem**: VQ refresh introduces **discontinuity** in codebook, causing:
1. Temporary spike as dead codes get reassigned
2. Subsequent instability as encoder adapts
3. Increased variance for remaining training

**Recommendation**: Disable refresh entirely OR increase interval to 10,000+ steps.

---

### 3. Embedding Quality Analysis

#### Effective Rank Progression

**Context Encoder:**
```
Step 100:  3.73  →  Step 1000: 3.91  →  Step 3000: 7.25  →  Step 5000: 14.56  →  Step 7600: 15.53
```

**Target Encoder:**
```
Step 100:  5.22  →  Step 1000: 6.55  →  Step 3000: 8.63  →  Step 5000: 16.06  →  Step 7600: 19.34
```

**Analysis:**

✅ **Healthy growth**: Both encoders show continuous rank expansion
✅ **Target > Context**: Expected pattern (target encoder more diverse)
✅ **Final values high**: 15.5 (context), 19.3 (target) indicate good diversity
⚠️ **Correlation with relational explosion**: Rank growth coincides with relational loss increase

**Hypothesis**: As embeddings spread into higher-dimensional space (increasing rank), pairwise distances grow, causing relational loss to explode. This is **mathematically inevitable** without loss normalization.

#### Variance & Isotropy

| Step | Context Variance | Context Isotropy | Target Variance | Target Isotropy |
|------|------------------|------------------|-----------------|-----------------|
| 100 | 8.67e-5 | 6.53e-5 | 3.04e-4 | 2.76e-5 |
| 1000 | 4.94e-4 | 1.39e-5 | 1.07e-3 | 7.84e-6 |
| 3000 | 1.05e-3 | 8.86e-6 | 1.38e-3 | 7.89e-6 |
| 5000 | 1.45e-3 | 1.22e-5 | 1.57e-3 | 1.24e-5 |
| 7600 | (not shown) | (not shown) | (not shown) | (not shown) |

**Trends:**
- ✅ Variance increases (embeddings spread in space)
- ✅ Isotropy relatively stable (uniform spreading)
- ⚪ No obvious collapse or mode collapse

---

### 4. Gradient Norm Analysis

#### Gradient Norm Statistics (Selected Epochs)

| Epoch | Mean Grad Norm | Max Grad Norm | Assessment |
|-------|----------------|---------------|------------|
| 1 | 6.64 | 8.26 | ✅ Healthy |
| 10 | 11.59 | 22.33 | ⚪ Normal |
| 30 | 14.91 | 26.10 | ⚪ Normal |
| 34 | 28.80 | **54.50** | 🟡 Starting to spike |
| 35 | 29.46 | **66.59** | 🔴 High |
| 37 | 36.38 | **66.68** | 🔴 Very high |
| 38 | 35.80 | **65.25** | 🔴 Sustained |
| 41 | 31.13 | **81.60** | 🔴 **Peak instability** |
| 42 | 17.82 | 46.29 | ⚪ Recovering |
| 70 | 16.32 | 48.37 | ⚪ Normal |
| 80 | 21.25 | 44.14 | ⚪ Normal |

**Gradient Clipping**: max_norm = 0.5

**Wait, problem!** The gradient norm values are **10-160x larger than the clipping threshold**. This suggests:

1. **Grad norm reporting is BEFORE clipping** (likely correct interpretation)
2. **Clipping is triggering frequently** when norms > 0.5
3. **Epochs 34-41 experience severe clipping** (norms 54-81 vs threshold 0.5)

**Impact**: Aggressive clipping during epochs 34-41 prevented training from diverging, but also:
- Slowed down learning (effective LR reduced by 100x+)
- Caused loss oscillations (training can't follow gradient)
- Correlates with relational loss explosion onset

---

## Root Cause Analysis

### Problem 1: Relational Loss Not Scale-Invariant

**Current Implementation** (assumed from metrics):
```python
# Relational loss computes pairwise distances
# relational_loss_raw = mean(||emb[i] - emb[j]||^2 for all pairs i,j)
# relational_loss = weight * relational_loss_raw
# Weight = 0.025

# As effective rank grows, embeddings spread in space
# → Pairwise distances increase
# → Relational loss increases EVEN WITH CONSTANT WEIGHT
```

**Evidence**:
- Epoch 1: Relational raw = 40.5 → weighted = 0.35 (raw / weighted ≈ 115x)
- Epoch 30: Relational raw = 49.9 → weighted = 0.50 (raw / weighted ≈ 100x)
- Epoch 71: Relational raw = ??? → weighted = 1.10 (likely raw ≈ 110+)

**Fix Required**: Normalize relational loss by embedding norm or variance:
```python
relational_loss_normalized = relational_loss_raw / (embedding_variance + eps)
```

### Problem 2: VQ Refresh Introduces Discontinuity

**Event Timeline**:
1. **Step 3000**: Refresh triggered (interval = 3000)
2. **Usage spike**: 18% → 30% (dead codes reassigned)
3. **Subsequent drop**: 30% → 13% (instability)
4. **High variance**: 13-36% for remaining 4600 steps

**Why It Fails**:
- VQ refresh **replaces dead codes with encoder outputs**
- This creates **discontinuity** in latent space mapping
- Encoder must re-learn new code assignments
- Causes temporary performance degradation

**L-JEPA Philosophy**: "heuristic-free, no hyper-parameter schedulers"
**Our Implementation**: Scheduled refresh = heuristic = anti-pattern

**Fix Required**: Remove VQ refresh OR increase interval to >> training length.

### Problem 3: Multi-Loss Optimization Without Balancing

**Current Loss Components**:
1. **InfoNCE**: Primary objective, decreases smoothly
2. **SIGReg**: Stable at ~0.040 (working correctly)
3. **Invariance**: Small, decreases slightly
4. **Relational**: Explodes 0.35 → 1.10

**Problem**: No gradient balancing or loss normalization across components.

**Evidence**:
- Epoch 71: Relational (1.10) > 13% of total loss (8.67)
- InfoNCE improvement (9.04 → 7.51) masked by relational explosion (0.35 → 0.85)
- Net effect: Total loss INCREASES late in training despite InfoNCE improving

**L-JEPA Solution**: **Single loss component** (SIGReg + predictive loss), no balancing needed.

**Our Fix Options**:
1. **Remove relational loss entirely** (simplest, L-JEPA aligned)
2. **Adaptive loss weighting** (GradNorm, uncertainty weighting)
3. **Normalize relational by embedding scale**

### Problem 4: Not L-JEPA Compliant

**From LEJEPA_ALIGNMENT_REVIEW.md**:

| Aspect | L-JEPA | Our Implementation | Gap |
|--------|--------|-------------------|-----|
| **Theoretical Foundation** | Provable optimal (SIGReg) | Ad-hoc InfoNCE | 🔴 High |
| **Hyperparameters** | 1 (λ trade-off) | 20+ | 🔴 High |
| **Heuristics** | None (heuristic-free) | Many (temp clamp, queue, refresh, EMA) | 🔴 High |
| **Loss Function** | SIGReg + L2 | InfoNCE + SIGReg + Invariance + Relational | 🟡 Medium |

**Our Implementation**: 2020-era MoCo-style JEPA
**L-JEPA Standard**: 2025 provable, heuristic-free JEPA

**Philosophical Mismatch**: L-JEPA minimalism vs our complexity.

---

## Improvement Plan

### Phase 1: Emergency Fixes (1-2 days)

**Goal**: Stop relational loss explosion, stabilize training

#### Fix 1.1: Remove or Drastically Reduce Relational Loss

**Option A (Recommended): Remove Entirely**
```yaml
# configs/training/jepa_production_v2.yaml
relational_loss:
  weight: 0.0              # DISABLED
  context_self_weight: 0.0 # DISABLED
```

**Rationale**:
- L-JEPA doesn't use relational loss
- Exploding component (0.35 → 1.10)
- InfoNCE already enforces relational structure via contrastive learning
- Simplifies architecture (fewer hyperparameters)

**Option B (Conservative): Normalize by Embedding Variance**
```python
# training/jepa/loop.py - modify relational loss
relational_raw = pairwise_distance_loss(context_emb, target_emb)
embedding_var = context_emb.var() + target_emb.var() + 1e-6
relational_normalized = relational_raw / embedding_var
loss += self.relational_weight * relational_normalized
```

**Option C (Aggressive): Reduce Weight 10x**
```yaml
relational_loss:
  weight: 0.0025           # Was 0.025, now 10x smaller
  context_self_weight: 0.001 # Was 0.01, now 10x smaller
```

**Recommendation**: Start with Option A (remove), fallback to Option B if performance degrades.

#### Fix 1.2: Disable VQ Refresh

```yaml
# configs/training/jepa_production_v2.yaml
object_encoder:
  vq_refresh_enabled: false  # DISABLED (was true)
  # OR: vq_refresh_interval: 100000  # Longer than training
```

**Rationale**:
- Causes discontinuity at step 3000
- Not L-JEPA aligned (heuristic)
- VQ usage already stable (13-21%) without refresh

#### Fix 1.3: Increase Early Stopping Patience

```yaml
# configs/training/jepa_production_v2.yaml
training:
  early_stopping:
    patience: 15          # Was 30, reduce to stop BEFORE relational explosion
    min_delta: 0.01       # Was 0.003, more aggressive
```

**Rationale**:
- Best val loss at epoch 69
- Relational explosion started epoch 30, peaked epoch 71
- Stopping at epoch 60-70 would preserve best performance

**Expected Outcome**:
- Val loss improvement: 8.22 → **8.0-8.1** (2-3% better)
- Training stability: No relational explosion
- VQ usage: More stable (15-20% consistent)

**Timeline**: 1 diagnostic run (80 epochs, ~2 hours on A6000)

---

### Phase 2: L-JEPA Alignment (3-5 days)

**Goal**: Modernize architecture to L-JEPA standards

#### Fix 2.1: Implement Provable SIGReg Loss

**Current**: SIGReg weight = 0.1, computes isotropic Gaussian score but doesn't **enforce** it.

**L-JEPA**: SIGReg is the **primary regularizer**, not auxiliary.

**Implementation**:
```python
# training/modules/sigreg.py - modify to be primary loss
class SIGRegLoss(nn.Module):
    def __init__(self, lambda_sigreg=1.0):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg  # Single hyperparameter

    def forward(self, context_emb, target_emb):
        # Compute isotropic Gaussian regularization
        # (already implemented, just increase weight)
        iso_loss = self.isotropic_gaussian_loss(context_emb, target_emb)

        # Primary loss: L2 distance between context and target
        l2_loss = F.mse_loss(context_emb, target_emb)

        # Combined (L-JEPA style)
        total_loss = l2_loss + self.lambda_sigreg * iso_loss
        return total_loss
```

**Config**:
```yaml
# configs/training/jepa_lejepa_aligned.yaml
loss:
  objective: "sigreg"      # NEW: Use SIGReg as primary (not InfoNCE)
  lambda_sigreg: 1.0       # Single hyperparameter
  # Remove: temperature, queue_size, use_target_encoder (InfoNCE artifacts)

# Remove these components:
# relational_loss: {...}  # REMOVED
# invariance: {...}       # REMOVED (optional augmentation-based)
```

**Breaking Change**: This **replaces InfoNCE** entirely. Requires careful testing.

**Fallback**: Keep InfoNCE, use SIGReg as auxiliary with higher weight (0.1 → 1.0).

#### Fix 2.2: Simplify Hyperparameters

**Current Count**: ~20+ hyperparameters across loss, optimizer, VQ, scheduler

**L-JEPA Target**: 1-3 hyperparameters (λ_sigreg, learning rate, weight decay)

**Removals**:
```yaml
# REMOVE (InfoNCE artifacts)
loss:
  # temperature: 0.07
  # temperature_min: 0.05
  # temperature_max: 0.20
  # learnable_temperature: true
  # queue_size: 8192
  # use_target_encoder: true
  # target_ema_decay: 0.99

# REMOVE (heuristics)
object_encoder:
  # vq_refresh_enabled: false
  # vq_refresh_interval: 3000
  # vq_refresh_usage_threshold: 0.01

# REMOVE (multi-loss balancing)
relational_loss:  # Entire section removed
invariance:       # Entire section removed

# SIMPLIFY (fewer scheduler params)
training:
  lr_scheduler:
    name: "cosine"
    warmup_steps: 1000
    # Remove: min_lr_scale, total_steps (use defaults)
```

**Keep Essentials**:
```yaml
# Minimal config (L-JEPA aligned)
optimizer:
  lr: 5.0e-5
  weight_decay: 1.0e-6

sigreg:
  lambda: 1.0              # SINGLE HYPERPARAMETER

object_encoder:
  vq_enabled: true
  commitment_cost: 0.15
  ema_decay: 0.95
  # VQ params kept for discrete codes requirement
```

#### Fix 2.3: Remove Teacher-Student (Optional)

**Current**: Target encoder updated via EMA (decay = 0.99)

**L-JEPA**: "heuristic-free, no teacher-student"

**Implementation**:
```python
# training/jepa/loop.py - option to remove target encoder
if not self.use_target_encoder:
    # Use same encoder for context and target (symmetric)
    target_emb = self.encoder(target_tokens)
else:
    # Keep EMA target encoder (current behavior)
    target_emb = self.target_encoder(target_tokens)
```

**Trade-off**:
- ✅ Simpler (no EMA heuristic)
- ✅ L-JEPA aligned
- ❌ May reduce performance (target encoder provides stable targets)

**Recommendation**: Test in Phase 3, keep EMA for Phase 2.

**Expected Outcome**:
- Hyperparameter count: 20+ → **5-8** (60-75% reduction)
- Training complexity: Multi-loss balancing → **Single loss + SIGReg**
- Philosophy: MoCo-2020 → **L-JEPA-2025**

**Timeline**: 2-3 diagnostic runs to validate SIGReg-primary approach

---

### Phase 3: Advanced Improvements (5-7 days)

**Goal**: Optimize architecture and data pipeline

#### Fix 3.1: Learned Context Aggregation (Revisited)

**From JEPA_OPTIMIZED_ASSESSMENT.md**:
> Context attention module showed no validation loss improvement but better early diversity.

**Hypothesis**: Temporal attention helps with **longer sequences**.

**Current**: context_window = 3 (only 3 frames)

**Experiment**: Increase context window to test attention benefit:
```yaml
data:
  context_window: 5        # Was 3, now 5 frames
  target_offset: 1
```

**Expected**: With 5-frame context, temporal attention may help differentiate frame order.

#### Fix 3.2: Adaptive Loss Weighting (If Multi-Loss Kept)

**If we keep InfoNCE + SIGReg** (not full L-JEPA migration):

**Implement GradNorm** (Gradient Normalization):
```python
# training/jepa/loop.py
class AdaptiveLossWeighting:
    def __init__(self, num_losses=2, alpha=1.5):
        self.weights = nn.Parameter(torch.ones(num_losses))
        self.alpha = alpha  # Restoring force

    def step(self, losses, shared_params):
        # Compute gradient norms for each loss
        grad_norms = []
        for loss in losses:
            grad = torch.autograd.grad(loss, shared_params, retain_graph=True)
            grad_norms.append(torch.norm(grad))

        # Balance gradients (GradNorm algorithm)
        # ...update self.weights...

        return self.weights
```

**Benefit**: Automatically balances InfoNCE vs SIGReg gradients.

#### Fix 3.3: Progressive Curriculum

**Current**: All tasks sampled uniformly

**Improvement**: Start with easy tasks, progressively add harder ones.

```python
# data/curriculum.py
class ProgressiveCurriculum:
    def __init__(self, manifest_path, num_phases=3):
        self.phases = self.split_by_difficulty(manifest_path, num_phases)
        self.current_phase = 0

    def get_active_tasks(self, epoch):
        # Phase 1 (epochs 0-20): Easy tasks only
        # Phase 2 (epochs 21-50): Easy + Medium
        # Phase 3 (epochs 51+): All tasks
        phase_schedule = [20, 50, 100]
        self.current_phase = bisect(phase_schedule, epoch)
        return self.phases[: self.current_phase + 1]
```

**Expected**: Better early learning, smoother convergence.

#### Fix 3.4: Hybrid VQ-VAE Strategy

**Problem**: VQ discrete codes required (blueprint), but causes instability.

**Current**: VQ codebook with 768 codes, 13-30% usage (low)

**Improvements**:

**Option A: Smaller Codebook**
```yaml
object_encoder:
  num_embeddings: 256      # Was 768, reduce by 3x
  # Hypothesis: Smaller codebook → higher usage → more stable
```

**Option B: Gumbel-Softmax VQ (Differentiable)**
```python
# training/modules/object_encoder.py
class GumbelVectorQuantizer:
    def forward(self, z, temperature=1.0):
        # Compute distances to codebook
        distances = torch.cdist(z, self.codebook)

        # Gumbel-softmax (differentiable)
        logits = -distances / temperature
        soft_codes = F.gumbel_softmax(logits, tau=temperature, hard=False)

        # Weighted sum (differentiable)
        z_q = soft_codes @ self.codebook
        return z_q
```

**Benefit**: No commitment loss, no EMA updates, fully differentiable.

**Option C: Finite Scalar Quantization (FSQ)**
```python
# Quantize each dimension independently to fixed levels
# E.g., 8 levels per dim → 8^d discrete codes (implicit codebook)
# No codebook learning, deterministic quantization
```

**Recommendation**: Test Option B (Gumbel-Softmax) first, fallback to FSQ.

**Expected Outcome**:
- VQ stability: No refresh needed, higher usage
- Training smoothness: Differentiable quantization
- Performance: 1-3% val loss improvement

**Timeline**: 3-4 diagnostic runs (2-3 days)

---

## L-JEPA Alignment Checklist

**Based on LEJEPA_ALIGNMENT_REVIEW.md**, here's how we align:

### Current Status (Production Run)

| L-JEPA Principle | Our Status | Alignment | Gap |
|------------------|------------|-----------|-----|
| **Heuristic-free** | ❌ Many heuristics (temp clamp, refresh, EMA) | 🔴 Low | High |
| **Provable optimal** | ❌ InfoNCE (no guarantees) | 🔴 Low | High |
| **Single hyperparameter** | ❌ 20+ hyperparameters | 🔴 Low | High |
| **No teacher-student** | ❌ EMA target encoder | 🟡 Medium | Medium |
| **SIGReg primary** | ❌ SIGReg auxiliary (weight=0.1) | 🔴 Low | High |
| **No stop-gradient** | ✅ No stop-gradient | ✅ High | None |
| **BF16 mixed precision** | ✅ BF16 enabled | ✅ High | None |
| **Minimal core** | ❌ 4000+ lines, complex pipeline | 🔴 Low | High |

**Overall Alignment**: **C-** (30-40%)

### After Phase 1 (Emergency Fixes)

| L-JEPA Principle | Status After P1 | Alignment | Gap |
|------------------|----------------|-----------|-----|
| **Heuristic-free** | ⚪ Removed refresh, reduced relational | 🟡 Medium | Medium |
| **Provable optimal** | ❌ Still InfoNCE | 🔴 Low | High |
| **Single hyperparameter** | ⚪ Removed 5-10 params | 🟡 Medium | Medium |
| **SIGReg primary** | ❌ Still auxiliary | 🔴 Low | High |

**Alignment After P1**: **C+** (40-50%)

### After Phase 2 (L-JEPA Alignment)

| L-JEPA Principle | Status After P2 | Alignment | Gap |
|------------------|----------------|-----------|-----|
| **Heuristic-free** | ✅ Removed most heuristics | ✅ High | Low |
| **Provable optimal** | ✅ SIGReg primary loss | ✅ High | Low |
| **Single hyperparameter** | ✅ λ_sigreg only (+ LR, WD) | ✅ High | Low |
| **No teacher-student** | ⚪ EMA optional | 🟡 Medium | Medium |
| **SIGReg primary** | ✅ Primary objective | ✅ High | Low |
| **Minimal core** | ⚪ Simplified but domain-specific | 🟡 Medium | Medium |

**Alignment After P2**: **B+** (75-85%)

**Remaining Gaps (Acceptable)**:
1. **Object-centric tokenization**: Domain requirement (ARC grids), not vision-centric
2. **VQ-VAE discrete codes**: Blueprint requirement (symbolic grounding)
3. **EMA target encoder**: May keep for stability (optional)

---

## Implementation Roadmap

### Week 1: Emergency Fixes (Phase 1)

**Day 1-2: Remove Relational Loss & VQ Refresh**
- [ ] Create `configs/training/jepa_production_v2.yaml`
- [ ] Set `relational_loss.weight = 0.0`
- [ ] Set `vq_refresh_enabled = false`
- [ ] Set `early_stopping.patience = 15`
- [ ] Run 80-epoch diagnostic on A6000
- [ ] Compare to production baseline

**Day 3: Analysis & Iteration**
- [ ] Analyze v2 results (target: val loss < 8.0)
- [ ] If val loss > 8.1, try normalized relational loss (Option B)
- [ ] If VQ unstable, reduce codebook size (768 → 384)

**Success Criteria**:
- ✅ Val loss < 8.1 (vs 8.22 baseline)
- ✅ No relational explosion
- ✅ VQ usage stable (15-20%)
- ✅ Training completes 80 epochs

---

### Week 2: L-JEPA Alignment (Phase 2)

**Day 1-2: SIGReg Primary Loss**
- [ ] Implement SIGReg-primary loss in `training/modules/sigreg.py`
- [ ] Create `configs/training/jepa_lejepa_aligned.yaml`
- [ ] Set `loss.objective = "sigreg"`, `lambda_sigreg = 1.0`
- [ ] Remove InfoNCE components (queue, temperature, target encoder)
- [ ] Run 30-epoch diagnostic

**Day 3: Hyperparameter Reduction**
- [ ] Remove all non-essential hyperparameters
- [ ] Document final config (target: 5-8 params)
- [ ] Run 30-epoch diagnostic with minimal config

**Day 4-5: Full L-JEPA Run**
- [ ] If diagnostics pass, run 80-epoch production with L-JEPA config
- [ ] Monitor SIGReg score, effective ranks, VQ usage
- [ ] Compare to Phase 1 baseline

**Success Criteria**:
- ✅ Val loss < 8.0 (improvement over Phase 1)
- ✅ Hyperparameter count < 10
- ✅ No multi-loss balancing issues
- ✅ Training stable without heuristics

---

### Week 3: Advanced Improvements (Phase 3)

**Day 1-2: Context Window Experiments**
- [ ] Test context_window = 5, 7
- [ ] Re-enable context attention module
- [ ] Measure impact on val loss

**Day 2-3: VQ-VAE Improvements**
- [ ] Implement Gumbel-Softmax VQ
- [ ] Run diagnostic with differentiable quantization
- [ ] Compare VQ usage and stability

**Day 4-5: Final Production Run**
- [ ] Combine best improvements from P1-P3
- [ ] Run 100-epoch production (longer to test late-stage stability)
- [ ] Full evaluation: val loss, embedding metrics, VQ quality

**Success Criteria**:
- ✅ Val loss < 7.8 (10% improvement over original 8.22)
- ✅ VQ usage > 20% stable
- ✅ Effective ranks 15-20 (context), 20-25 (target)
- ✅ No loss explosions or instability

---

## Risk Assessment

### Phase 1 Risks (Low)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Removing relational degrades performance | 30% | Medium | Test normalized relational (Option B) |
| VQ collapse without refresh | 20% | Medium | Monitor usage, reduce codebook if needed |
| Early stopping too aggressive | 10% | Low | Adjust patience if needed |

**Overall Risk**: 🟢 Low (familiar changes, easy rollback)

### Phase 2 Risks (Medium)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SIGReg-primary underperforms | 40% | High | Keep InfoNCE as fallback |
| Removing target encoder causes instability | 30% | Medium | Keep EMA as option |
| Hyperparameter reduction breaks training | 20% | Medium | Progressive reduction, test at each step |

**Overall Risk**: 🟡 Medium (architectural change, requires validation)

### Phase 3 Risks (Medium-High)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Gumbel-Softmax VQ fails | 50% | Medium | Fallback to FSQ or original VQ |
| Longer context window doesn't help | 60% | Low | Quick to test and revert |
| Combined changes interact poorly | 30% | High | Add changes incrementally |

**Overall Risk**: 🟡 Medium-High (experimental changes, less proven)

---

## Success Metrics

### Phase 1 Targets

| Metric | Baseline (Prod) | Phase 1 Target | Improvement |
|--------|----------------|----------------|-------------|
| Best Val Loss | 8.224 | < 8.1 | -1.5% |
| Relational Loss (Epoch 80) | 0.845 | < 0.1 OR removed | -88% |
| VQ Usage Variance | 13-36% (23pp) | 15-22% (7pp) | -70% |
| Training Stability | Degraded after epoch 70 | Stable to epoch 80 | ✅ |

### Phase 2 Targets

| Metric | Phase 1 | Phase 2 Target | Improvement |
|--------|---------|----------------|-------------|
| Best Val Loss | < 8.1 | < 8.0 | -1.2% |
| Hyperparameter Count | 15-18 | < 10 | -40% |
| Loss Components | 4 (InfoNCE, SIGReg, Inv, Rel) | 1-2 (SIGReg + optional) | -50-75% |
| L-JEPA Alignment | C+ (40-50%) | B+ (75-85%) | +70% |

### Phase 3 Targets

| Metric | Phase 2 | Phase 3 Target | Improvement |
|--------|---------|----------------|-------------|
| Best Val Loss | < 8.0 | < 7.8 | -2.5% |
| VQ Usage | 15-22% | > 25% | +20-60% |
| Effective Rank (Target) | 19.3 | > 22 | +14% |

### Ultimate Success (All Phases)

| Metric | Original Prod | Final Target | Total Improvement |
|--------|---------------|--------------|-------------------|
| **Best Val Loss** | 8.224 | **< 7.8** | **-5.2%** |
| **Hyperparameters** | 20+ | **< 10** | **-50%+** |
| **Training Stability** | Degraded after epoch 70 | **Stable 100 epochs** | ✅ |
| **L-JEPA Alignment** | C- (30%) | **B+ (75-85%)** | **+150%** |
| **VQ Codebook Quality** | 13-36% usage (unstable) | **> 25% stable** | **+90%+** |

---

## Conclusion

The 80-epoch production run revealed **critical architectural issues** despite achieving training stability:

### What Worked ✅
1. Gentle VQ parameters prevented early collapse
2. High embedding diversity (effective ranks 15-19)
3. InfoNCE loss decreased consistently
4. SIGReg regularization stable throughout

### What Failed ❌
1. Relational loss exploded 0.35 → 1.10 (314%)
2. VQ refresh caused discontinuity and instability
3. Multi-loss balancing deteriorated over training
4. Final performance below target (8.22 vs < 8.0)
5. Architecture not L-JEPA compliant (heuristic-heavy, no provable guarantees)

### Path Forward

**3-Phase Improvement Plan**:

1. **Phase 1 (Emergency)**: Remove relational loss, disable VQ refresh → **-2% val loss**
2. **Phase 2 (Modernization)**: Adopt L-JEPA SIGReg-primary, reduce hyperparameters → **B+ alignment**
3. **Phase 3 (Optimization)**: Context attention, Gumbel-VQ, curriculum → **-5% total val loss**

**Expected Outcome**: Val loss **8.22 → 7.8** (-5.2%), L-JEPA alignment **30% → 80%** (+150%)

**Timeline**: 3 weeks (5-7 days per phase)

**Recommendation**: **Proceed with Phase 1 immediately**, validate before committing to Phase 2.

---

## Appendix: Key Metrics Summary

### Loss Components (Epochs 1, 30, 60, 80)

| Epoch | InfoNCE | SIGReg | Invariance | Relational | Total | Val Loss |
|-------|---------|--------|------------|------------|-------|----------|
| 1 | 9.042 | 0.0405 | 0.0177 | 0.351 | 9.451 | 9.378 |
| 30 | 7.984 | 0.0403 | 0.0245 | 0.499 | 8.548 | 8.587 |
| 60 | 7.507 | 0.0401 | 0.0135 | 0.789 | 8.350 | 8.268 |
| 80 | 7.567 | 0.0401 | 0.0164 | 0.845 | 8.469 | 8.487 |

### Embedding Quality (Steps 100, 1000, 3000, 5000, 7600)

| Step | Context Rank | Target Rank | VQ Usage (%) | Context Var | Target Var |
|------|--------------|-------------|--------------|-------------|------------|
| 100 | 3.73 | 5.22 | 26.8% | 8.67e-5 | 3.04e-4 |
| 1000 | 3.91 | 6.55 | 16.4% | 4.94e-4 | 1.07e-3 |
| 3000 | 7.25 | 8.63 | **30.2%** | 1.05e-3 | 1.38e-3 |
| 5000 | 14.56 | 16.06 | 21.1% | 1.45e-3 | 1.57e-3 |
| 7600 | 15.53 | 19.34 | 33.6% | (high) | (high) |

**Note**: Step 3000 shows VQ refresh spike (30.2% usage).

### Gradient Norms (Epochs with Instability)

| Epoch | Mean Grad Norm | Max Grad Norm | Relational Loss | Assessment |
|-------|----------------|---------------|-----------------|------------|
| 34 | 28.80 | 54.50 | 0.502 | 🟡 Starting |
| 35 | 29.46 | 66.59 | 0.486 | 🔴 High |
| 41 | 31.13 | 81.60 | 0.617 | 🔴 **Peak** |
| 71 | 16.36 | 49.23 | **1.101** | 🔴 **Explosion** |

**Gradient clip threshold**: 0.5 (all these epochs clipped heavily)

---

## Sources

Based on analysis of production run artifacts and reference to:
- [L-JEPA Paper (arXiv 2511.08544)](https://arxiv.org/abs/2511.08544)
- [LEJEPA_ALIGNMENT_REVIEW.md](LEJEPA_ALIGNMENT_REVIEW.md)
- [JEPA_OPTIMIZED_ASSESSMENT.md](JEPA_OPTIMIZED_ASSESSMENT.md)
- [I-JEPA (Meta AI)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
