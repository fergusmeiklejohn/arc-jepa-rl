# JEPA Training Failure Diagnostic Plan

## Summary of Findings

After analyzing the failed sixth training run (`temp/pretrain_a6000_sixth_run`), I've identified **VQ codebook instability** as the most likely root cause, with several contributing factors.

### Critical Evidence

1. **VQ Usage Collapse**: Codebook usage drops from 24% → 59% → 6% → **1.7%** over 2090 steps
2. **Embedding Space Collapse**:
   - Effective rank: 6.4 → 1.36-1.88 (should stay high)
   - Isotropy: ~1e-5 (extremely low, embeddings clustering)
3. **Validation Spike**: Loss jumped from 8.9 to 27.2 at epoch 13
4. **Early Stopping**: Triggered at epoch 22 (of 80 planned)

### Root Cause Analysis

**Primary Culprit: VQ Codebook Instability**

The VQ-VAE is caught in a destructive feedback loop:
1. Codebook collapses → fewer active codes
2. Refresh mechanism (every 500 steps) tries to revive dead codes
3. This causes discontinuous jumps in the embedding space
4. InfoNCE loss destabilizes from stale negatives
5. Gradients push embeddings away from newly revived codes
6. Loop repeats, getting worse each cycle

**Contributing Factors:**

1. **Multiple Conflicting Losses**: InfoNCE + SIGReg + Invariance + Relational + VQ commitment all pulling in different directions
2. **Tight Temperature Clamping**: Range [0.07, 0.15] too narrow to adapt when embeddings collapse
3. **Context Aggregation Loss**: Double mean-pooling loses sequential information
4. **Timing Mismatch**: VQ EMA updates 2x per optimizer step (grad accumulation), target encoder EMA 1x

## Diagnostic Experiment: VQ Ablation

**Goal**: Determine if VQ is the root cause by running without it.

**What to Run on Paperspace:**

```bash
cd /path/to/arc-jepa-rl
source .venv/bin/activate

# Run diagnostic with VQ disabled (30 epochs, ~1-2 hours)
python scripts/train_jepa.py \
  --config configs/training/jepa_diagnostic_no_vq.yaml
```

**Config Changes** (already in `jepa_diagnostic_no_vq.yaml`):
- `vq_enabled: false` ← Key change
- `epochs: 30` (shorter run)
- `checkpoint_dir: temp/diagnostic_no_vq`
- `patience: 20` (more lenient early stopping)

**Expected Outcomes:**

### If VQ is the problem:
✅ Training should be **stable and smooth**
✅ Validation loss decreases monotonically
✅ No spikes or collapses
✅ Effective rank stays high (>5)
✅ Isotropy stays reasonable (>1e-3)
✅ No codebook metrics (VQ disabled)

### If VQ is NOT the problem:
❌ Similar collapse pattern appears
❌ Validation loss spikes
❌ Embedding metrics still degrade

## Next Steps Based on Results

### Scenario A: VQ Ablation is Stable
**Conclusion**: VQ is definitely the problem

**Fixes to try (in order):**

1. **Disable VQ Refresh** (quickest test)
   ```yaml
   vq_refresh_enabled: false
   ```

2. **Reduce VQ Commitment Cost** (less aggressive quantization)
   ```yaml
   commitment_cost: 0.15  # was 0.3
   ```

3. **Gentler Refresh Parameters** (if keeping refresh)
   ```yaml
   vq_refresh_interval: 2000      # was 500
   vq_refresh_usage_threshold: 0.01  # was 0.05
   ```

4. **Switch to Straight VQ-VAE** (disable EMA updates)
   ```yaml
   ema_decay: null  # Use gradient-based updates instead
   ```

### Scenario B: VQ Ablation Still Unstable
**Conclusion**: Deeper issue with loss combination or architecture

**Fixes to try:**

1. **Minimal Loss Configuration** (InfoNCE only)
   ```yaml
   sigreg:
     weight: 0.0
   invariance:
     color_weight: 0.0
     translation_weight: 0.0
     symmetry_weight: 0.0
   relational_loss:
     weight: 0.0
   ```

2. **Widen Temperature Range**
   ```yaml
   temperature_min: 0.03  # was 0.07
   temperature_max: 0.30  # was 0.15
   ```

3. **Fix Context Aggregation** (requires code change - see below)

## Code Fixes for Context Aggregation (If Needed)

The current double mean-pooling in `training/jepa/loop.py:505-507` loses sequential information.

**Quick Fix** (just use last context grid):
```python
# Line 507 - replace:
aggregated = reshaped.mean(dim=1)

# With:
aggregated = reshaped[:, -1, :]  # Use only the last context grid
```

**Better Fix** (weighted or learned aggregation):
```python
# Add a learnable context aggregator in __init__
self.context_aggregator = nn.Sequential(
    nn.Linear(context_length, 1),
    nn.Softmax(dim=1)
)

# In forward:
weights = self.context_aggregator(reshaped.transpose(1, 2))  # (B, 1, hidden)
aggregated = (reshaped * weights).sum(dim=1)
```

## Monitoring During Diagnostic Run

Watch these metrics in TensorBoard or logs:

### Good Signs:
- ✅ Loss decreasing smoothly
- ✅ Effective rank > 5.0
- ✅ Isotropy > 1e-3
- ✅ No sudden spikes

### Bad Signs:
- ❌ Loss plateau or increase
- ❌ Effective rank < 2.0
- ❌ Isotropy < 1e-4
- ❌ Validation spikes

## Files to Commit

After pushing this diagnostic config to the remote machine:

```bash
git add configs/training/jepa_diagnostic_no_vq.yaml
git add docs/JEPA_DIAGNOSTIC_PLAN.md
git commit -m "Add VQ ablation diagnostic config and analysis"
git push
```

Then on Paperspace:
```bash
git pull
# Run diagnostic as above
```

## Timeline

- **Diagnostic run**: ~1-2 hours (30 epochs)
- **Analysis**: ~15 minutes (check metrics)
- **Fix implementation**: ~30 minutes - 2 hours (depending on which fix)
- **Validation run**: ~2-4 hours (full run with fix)

## Questions to Answer

1. Does removing VQ stabilize training? → **Most critical**
2. If yes, is it the refresh mechanism specifically?
3. If no, which auxiliary loss is causing issues?
4. Is the context aggregation losing critical information?

---

## Additional Experiments (If Time Permits)

### Experiment 2: Minimal Config
Even simpler than no-VQ - just InfoNCE with minimal everything:
- No VQ
- No auxiliary losses
- No target encoder EMA
- Wider temperature range

### Experiment 3: VQ with No Refresh
Keep VQ but disable the refresh mechanism to see if that's the specific culprit.

### Experiment 4: Single Loss Addition
Starting from minimal stable config, add one auxiliary loss at a time to identify which one breaks stability.

---

**Priority**: Run the VQ ablation first. This single experiment will tell us whether we're on the right track.
