# Training and Evaluation Guide

This guide provides comprehensive, step-by-step instructions for training and evaluating the ARC JEPA × HRL system on single-GPU machines (A6000/A100), with robust checkpoint/resume capabilities to handle auto-shutdown scenarios.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Hardware Setup & Optimization](#hardware-setup--optimization)
3. [Checkpoint Management & Auto-Shutdown Resilience](#checkpoint-management--auto-shutdown-resilience)
4. [JEPA Pretraining Workflows](#jepa-pretraining-workflows)
5. [Evaluation & Metrics Interpretation](#evaluation--metrics-interpretation)
6. [Active Reasoner & RL Training](#active-reasoner--rl-training)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **Environment setup:**
   ```bash
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

2. **Generate training data:**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
     --config configs/data/pilot_curriculum.yaml
   ```

3. **Quick JEPA test run (30 seconds):**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
     --config configs/training/jepa_tiny.yaml \
     --device cuda
   ```

---

## Hardware Setup & Optimization

### Single GPU Configuration (A6000/A100)

The repository includes GPU-optimized configurations designed for 40-80GB VRAM GPUs.

#### Recommended Configurations

**A6000 (48GB VRAM):**
```yaml
# configs/training/jepa_pretrain_gpu.yaml
training:
  batch_size: 1024          # Effective batch with grad accumulation
  grad_accum_steps: 2       # ~512 per micro-batch
  num_workers: 8            # DataLoader processes
  mixed_precision: "fp16"   # ~40% memory savings
  pin_memory: true          # Faster CPU→GPU transfer
```

**A100 (80GB VRAM):**
```yaml
training:
  batch_size: 2048          # Larger batches for better throughput
  grad_accum_steps: 4       # 512 micro-batch maintained
  num_workers: 12
  mixed_precision: "bf16"   # Better numerical stability than fp16
  pin_memory: true
```

#### Memory Optimization Strategies

1. **Mixed Precision Training:**
   - `fp16`: Best memory savings, may need loss scaling
   - `bf16`: Better stability, requires Ampere+ GPUs (A100, RTX 30xx+)
   - `none`: Full precision (debugging only)

2. **Gradient Accumulation:**
   - Effective batch = `batch_size / grad_accum_steps`
   - Increase `grad_accum_steps` if hitting OOM
   - Example: batch=2048, accum=4 → 512 samples per GPU step

3. **DataLoader Workers:**
   - Rule of thumb: `num_workers = min(CPU_cores / 2, 12)`
   - Too many workers → RAM exhaustion
   - Too few → GPU starvation (check `nvidia-smi dmon -s u`)

4. **Pre-tokenization (Recommended for long runs):**
   ```bash
   # Pre-compute object tokens once (saves 30-40% training time)
   PYTHONPATH=. .venv/bin/python scripts/pretokenize_jepa.py \
     --config configs/training/jepa_pretrain_gpu.yaml \
     --output artifacts/tokenized/pilot_curriculum
   ```

   Then update config:
   ```yaml
   pre_tokenized:
     path: artifacts/tokenized/pilot_curriculum
   ```

#### Monitoring GPU Utilization

```bash
# Terminal 1: Training
PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
  --config configs/training/jepa_pretrain_gpu.yaml \
  --device cuda

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

**Target metrics:**
- GPU utilization: >85%
- Memory usage: 70-90% (leave headroom for spikes)
- Power draw: Near TDP (350W for A100, 300W for A6000)

If GPU util <70%, increase `num_workers` or reduce `grad_accum_steps`.

---

## Checkpoint Management & Auto-Shutdown Resilience

### Current Checkpoint Behavior

**Automatic Saving:**
- Checkpoints saved every epoch: `checkpoint_dir/checkpoint_epoch_XXXX.pt`
- Contains: model state, optimizer state, queue state, epoch number, config
- Location: `training.checkpoint_dir` in config (default: `artifacts/jepa/pretrain_gpu/`)

**What's Saved:**
```python
{
    "epoch": 15,
    "model_state": encoder.state_dict(),
    "projection_state": projection_head.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "queue_state": queue.state_dict(),
    "config": original_config,
}
```

### Resuming Training After Auto-Shutdown

**⚠️ Current Limitation:** The `train_jepa.py` script does not include a `--resume` flag yet.

#### Option 1: Manual Resume (Python Script)

Create a resume script `scripts/resume_jepa.py`:

```python
"""Resume JEPA training from checkpoint."""
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.jepa import ObjectCentricJEPAExperiment, load_jepa_config

# Load checkpoint
checkpoint_path = Path("artifacts/jepa/pretrain_gpu/checkpoint_epoch_0015.pt")
checkpoint = torch.load(checkpoint_path)

# Restore config and create experiment
config = checkpoint["config"]
device = checkpoint.get("device", "cuda")
experiment = ObjectCentricJEPAExperiment(config, device=device)

# Load model/optimizer state
experiment.trainer.encoder.load_state_dict(checkpoint["model_state"])
experiment.projection_head.load_state_dict(checkpoint["projection_state"])
experiment.optimizer.load_state_dict(checkpoint["optimizer_state"])
experiment.queue.load_state_dict(checkpoint["queue_state"])

start_epoch = checkpoint["epoch"] + 1
print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")

# Continue training loop...
# (Copy training loop from train_jepa.py, adjust epoch range)
```

#### Option 2: Planned Enhancement (Recommended Path)

Add resume capability to `train_jepa.py`:

```python
# Add to parse_args():
parser.add_argument(
    "--resume",
    type=Path,
    default=None,
    help="Path to checkpoint to resume from"
)

# In main(), after creating experiment:
if args.resume:
    checkpoint = torch.load(args.resume)
    experiment.trainer.encoder.load_state_dict(checkpoint["model_state"])
    experiment.projection_head.load_state_dict(checkpoint["projection_state"])
    experiment.optimizer.load_state_dict(checkpoint["optimizer_state"])
    experiment.queue.load_state_dict(checkpoint["queue_state"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 1

# Modify training loop:
for epoch in range(start_epoch, epochs + 1):
    # ... existing training code
```

**Usage after enhancement:**
```bash
PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
  --config configs/training/jepa_pretrain_gpu.yaml \
  --device cuda \
  --resume artifacts/jepa/pretrain_gpu/checkpoint_epoch_0015.pt
```

### Auto-Shutdown Best Practices

1. **Use tmux/screen for persistent sessions:**
   ```bash
   # Start persistent session
   tmux new -s jepa_training

   # Run training
   PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
     --config configs/training/jepa_pretrain_gpu.yaml \
     --device cuda

   # Detach: Ctrl+B, then D
   # Reattach after reconnect: tmux attach -t jepa_training
   ```

2. **Enable TensorBoard logging for progress tracking:**
   ```yaml
   logging:
     enabled: true
     log_dir: artifacts/jepa/pretrain_gpu/tensorboard
     flush_secs: 10  # Flush frequently for shutdown safety
   ```

3. **Monitor training remotely:**
   ```bash
   # Terminal 1: Training in tmux
   # Terminal 2: TensorBoard
   .venv/bin/tensorboard --logdir artifacts/jepa/pretrain_gpu/tensorboard --port 6006
   ```

4. **Checkpoint inspection:**
   ```bash
   # Find latest checkpoint
   ls -lt artifacts/jepa/pretrain_gpu/checkpoint_epoch_*.pt | head -1

   # Check epoch number
   PYTHONPATH=. .venv/bin/python -c "
   import torch
   ckpt = torch.load('artifacts/jepa/pretrain_gpu/checkpoint_epoch_0015.pt')
   print(f'Epoch: {ckpt[\"epoch\"]}, Device: {ckpt[\"device\"]}')
   "
   ```

5. **Backup critical checkpoints:**
   ```bash
   # Periodic backup (add to cron or systemd timer)
   rsync -av artifacts/jepa/pretrain_gpu/ /backup/jepa_$(date +%Y%m%d)/
   ```

---

## JEPA Pretraining Workflows

### Workflow 1: Basic JEPA Pretraining (Single GPU)

**Goal:** Train object-centric JEPA encoder on synthetic ARC tasks.

**Estimated time:** 6-12 hours (30 epochs, 10K tasks, A6000)

#### Step 1: Generate Dataset

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
  --config configs/data/pilot_curriculum.yaml
```

**Output:**
- `data/pilot_curriculum/manifest.jsonl` (10,000 task pairs)
- `data/pilot_curriculum/summary.json` (dataset statistics)

**Verify dataset:**
```bash
# Check task count
wc -l data/pilot_curriculum/manifest.jsonl

# Inspect summary
cat data/pilot_curriculum/summary.json | jq '{total_tasks, phases, program_length_histogram}'
```

#### Step 2: (Optional) Pre-tokenize for Speed


Generate the pilot curriculum dataset first:
```bash
python scripts/generate_dataset.py --config configs/data/pilot_curriculum.yaml
```
This writes data/pilot_curriculum/manifest.jsonl and summary.json.
Then rerun pretokenization:
```bash
python scripts/pretokenize_jepa.py --config configs/training/jepa_pretrain_gpu.yaml --output artifacts/tokenized/pilot_curriculum
```


**Benefits:**
- 30-40% faster training (especially for large grids)
- Lower CPU usage during training
- Consistent tokenization across runs

**Update config to use pre-tokenized data:**
```yaml
# configs/training/jepa_pretrain_gpu.yaml
pre_tokenized:
  path: artifacts/tokenized/pilot_curriculum
```

#### Step 3: Launch Training

```bash
# Start tmux session for auto-shutdown resilience
tmux new -s jepa_pretrain

# Run training
python scripts/train_jepa.py --config configs/training/jepa_pretrain_gpu.yaml --device cuda --mixed-precision fp16
```

**Monitor progress:**
```bash
# Watch console output
# Expected output:
# Epoch 1/30: loss=2.456789
#   Validation loss: 2.123456
# Epoch 2/30: loss=2.012345
#   Validation loss: 1.987654
# ...
```

**TensorBoard (separate terminal):**
```bash
.venv/bin/tensorboard --logdir artifacts/jepa/pretrain_gpu/tensorboard
# Open http://localhost:6006
```

#### Step 4: Validate Training

**Check metrics.json:**
```bash
cat artifacts/jepa/pretrain_gpu/metrics.json | jq '{
  completed_epochs,
  batch_size,
  final_loss: .losses[-1],
  best_val_loss,
  early_stopping_epoch
}'
```

**Inspect checkpoint:**
```bash
ls -lh artifacts/jepa/pretrain_gpu/checkpoint_epoch_*.pt
# Should see files ~50-200MB depending on model size
```

**Visualize embedding quality:**
```bash
cat artifacts/jepa/pretrain_gpu/embedding_metrics.jsonl | tail -5 | jq
```

### Workflow 2: JEPA with LeJEPA Regularization (SIGReg)

**Goal:** Add isotropic Gaussian regularization to reduce embedding collapse.

**When to use:** If standard JEPA shows low embedding variance or codebook collapse.

#### Configuration

```yaml
# configs/training/jepa_sigreg.yaml (create from jepa_pretrain_gpu.yaml)
sigreg:
  weight: 0.1        # Start conservative (0.05-0.2 range)
  num_slices: 1024   # Higher = more accurate, slower
  num_points: 17     # Gauss-Legendre quadrature points

loss:
  objective: "infonce"
  temperature: 0.07
  queue_size: 4096
```

#### Training

```bash
PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
  --config configs/training/jepa_sigreg.yaml \
  --device cuda
```

**Monitoring SIGReg:**
```bash
# TensorBoard will show:
# - loss/sigreg: Raw penalty (before weighting)
# - loss/total: InfoNCE + weighted SIGReg
# - diagnostics/context/isotropy: Embedding isotropy score
```

**Tuning `sigreg.weight`:**
- Too low (0.01): Minimal effect, potential collapse
- Sweet spot (0.05-0.2): Balanced regularization
- Too high (0.5+): Dominates InfoNCE, poor representations

**Validation:** Check `diagnostics/context/isotropy` in TensorBoard. Target: >0.8 (scale 0-1).

### Workflow 3: JEPA Ablation Study

**Goal:** Compare JEPA variants (InfoNCE, +VQ, +relational, +invariance, +SIGReg).

```bash
PYTHONPATH=. .venv/bin/python scripts/run_jepa_ablation.py \
  --tasks data/pilot_curriculum/manifest.jsonl \
  --output artifacts/ablation/jepa_variants.json \
  --device cuda
```

**Output:**
```json
{
  "variants": {
    "infonce_only": {"success_rate": 0.72, "avg_loss": 1.23},
    "infonce_vq": {"success_rate": 0.75, "avg_loss": 1.18},
    "infonce_vq_relational": {"success_rate": 0.78, "avg_loss": 1.12},
    "infonce_vq_relational_sigreg": {"success_rate": 0.81, "avg_loss": 1.08}
  }
}
```

### Workflow 4: Hyperparameter Tuning

**Key hyperparameters for JEPA:**

| Hyperparameter | Default | A6000 Recommended | Tuning Notes |
|----------------|---------|-------------------|--------------|
| `batch_size` | 1024 | 1024-2048 | Larger = more stable but slower per-epoch |
| `grad_accum_steps` | 2 | 2-4 | Increase if OOM, maintain effective batch |
| `lr` | 5e-5 | 3e-5 to 1e-4 | Too high → collapse, too low → slow |
| `warmup_steps` | 1000 | 500-2000 | ~10% of total steps |
| `queue_size` | 4096 | 4096-8192 | Larger = better negatives, more memory |
| `temperature` | 0.07 | 0.05-0.1 | Lower = harder negatives |
| `mixed_precision` | fp16 | fp16/bf16 | bf16 if A100 (better stability) |

**Tuning procedure:**

1. **Baseline run (2-3 epochs):**
   ```bash
   # Quick test with default config
   PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
     --config configs/training/jepa_pretrain_gpu.yaml \
     --device cuda
   ```
   Monitor: Loss should decrease steadily, GPU util >85%

2. **Learning rate sweep:**
   ```yaml
   # Test: 1e-5, 3e-5, 5e-5, 1e-4, 3e-4
   optimizer:
     lr: 5.0e-5  # Start here
   ```
   Rule: If loss plateaus early → increase LR. If loss spikes → decrease LR.

3. **Batch size scaling (if memory allows):**
   ```yaml
   training:
     batch_size: 2048  # Test doubling
     grad_accum_steps: 2  # Maintain effective batch
   ```
   Larger batches → better gradients but longer epochs.

4. **Temperature tuning:**
   ```yaml
   loss:
     temperature: 0.07
     learnable_temperature: true  # Let model adapt
   ```
   Check TensorBoard `loss/temperature` track - should stabilize 0.05-0.15.

---

## Evaluation & Metrics Interpretation

### Evaluation Workflows

#### Workflow 1: Evaluate on Synthetic Tasks

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
  --tasks data/pilot_curriculum/manifest.jsonl \
  --output artifacts/eval/pilot_curriculum.json \
  --latent-config configs/training/jepa_pretrain_gpu.yaml \
  --latent-device cuda \
  --max-nodes 3
```

**Key flags:**
- `--tasks`: JSONL manifest (generated tasks)
- `--latent-config`: JEPA config to enable embedding tracking
- `--latent-device`: GPU for JEPA inference
- `--max-nodes`: Program complexity limit (3 = ~10K programs)

**Output:** `artifacts/eval/pilot_curriculum.json`

```json
{
  "total_tasks": 10000,
  "variants": {
    "dsl_only": {
      "success_rate": 0.82,
      "avg_programs_tested": 127.5,
      "novelty_rate": 0.15,
      "solved_count": 8200
    },
    "meta_guided": {
      "success_rate": 0.88,
      "avg_programs_tested": 68.3,
      "novelty_rate": 0.22,
      "solved_count": 8800
    }
  },
  "latent_distances": {
    "mean_solved": 0.12,
    "mean_failed": 0.87,
    "correlation": -0.68
  }
}
```

#### Workflow 2: Evaluate on ARC-1 Official Dev Set

**Setup (one-time):**
```bash
git clone --depth 1 https://github.com/fchollet/ARC external/ARC
```

**Run evaluation:**
```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
  --arc-dev-root external/ARC/data/training \
  --output artifacts/eval/arc_dev.json \
  --latent-config configs/training/jepa_pretrain_gpu.yaml \
  --latent-device cuda \
  --max-nodes 3
```

**Expected baseline:** 2-5% success rate (ARC-1 training split is hard for symbolic methods)

#### Workflow 3: JEPA Correlation Validation

**Goal:** Quantify how well JEPA embeddings predict solver success.

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_jepa_correlation.py \
  --jepa-config configs/training/jepa_pretrain_gpu.yaml \
  --checkpoints artifacts/jepa/pretrain_gpu/checkpoint_epoch_0030.pt \
  --tasks data/pilot_curriculum/manifest.jsonl \
  --device cuda \
  --output artifacts/eval/jepa_correlation.json
```

**Output metrics:**
```json
{
  "pearson_correlation": -0.72,  // Negative = lower distance → higher solve rate
  "p_value": 0.0001,
  "mean_distance_solved": 0.08,
  "mean_distance_failed": 0.91,
  "threshold_95_recall": 0.15    // Distance threshold for 95% recall
}
```

**Interpreting correlation:**
- **< -0.6:** Strong negative correlation (good!)
- **-0.4 to -0.6:** Moderate correlation (acceptable)
- **> -0.4:** Weak correlation (JEPA needs improvement)

### Metrics Reference

#### Evaluation Metrics

| Metric | Definition | Good Value | Bad Value | Notes |
|--------|------------|------------|-----------|-------|
| **success_rate** | Fraction of tasks solved | >0.8 (synthetic), >0.05 (ARC-1) | <0.5 (synthetic) | Main quality metric |
| **avg_programs_tested** | Mean programs enumerated | <100 (meta-guided) | >500 | Efficiency metric |
| **novelty_rate** | New rules discovered / total tasks | 0.1-0.3 | <0.05 or >0.5 | Too low = limited discovery, too high = overfitting |
| **latent_distances.correlation** | Pearson ρ(distance, solve) | < -0.6 | > -0.3 | JEPA quality indicator |

#### Training Metrics (TensorBoard)

| Metric | Path | Target | Troubleshooting |
|--------|------|--------|-----------------|
| **InfoNCE loss** | `loss/info_nce` | Decrease to 1.0-1.5 | Stuck >3.0 → LR too low or collapse |
| **Validation loss** | `val/loss` | Track train/val gap | Gap >0.5 → overfitting |
| **Temperature** | `loss/temperature` | 0.05-0.15 (stable) | >0.2 → easy negatives (increase queue) |
| **Isotropy** | `diagnostics/context/isotropy` | >0.8 | <0.5 → collapse (add SIGReg) |
| **VQ usage** | `diagnostics/vq/usage_ratio` | >0.6 | <0.3 → dead codes (enable refresh) |

#### Embedding Diagnostics

**Isotropy score (0-1 scale):**
- **1.0:** Perfect spherical distribution
- **0.8-1.0:** Healthy embeddings
- **0.5-0.8:** Mild anisotropy (acceptable)
- **<0.5:** Severe collapse (add regularization)

**VQ codebook usage:**
- **>0.8:** Excellent (most codes active)
- **0.5-0.8:** Good
- **<0.5:** Collapse risk (enable `vq_refresh`)

**Check diagnostics:**
```bash
# Last 10 diagnostic events
tail -10 artifacts/jepa/pretrain_gpu/embedding_metrics.jsonl | jq '.context.isotropy'
```

### Common Evaluation Patterns

#### Pattern 1: High Loss but High Solve Rate

**Symptoms:**
- Final JEPA loss: 2.5+
- Solve rate: 85%+

**Interpretation:** Model learns task-relevant features despite high InfoNCE loss. Acceptable.

**Action:** Continue training, monitor correlation instead of absolute loss.

#### Pattern 2: Low Loss but Low Solve Rate

**Symptoms:**
- Final JEPA loss: 0.8
- Solve rate: 40%

**Interpretation:** Embedding collapse - model memorizes without generalizing.

**Action:**
1. Check isotropy (<0.5 confirms collapse)
2. Add SIGReg regularization (weight 0.1)
3. Increase queue size (8192+)
4. Reduce batch size (more diverse negatives)

#### Pattern 3: Training-Validation Gap

**Symptoms:**
- Train loss: 1.2
- Val loss: 2.5

**Interpretation:** Overfitting (especially if using small datasets <5K tasks)

**Action:**
1. Enable early stopping:
   ```yaml
   training:
     early_stopping:
       enabled: true
       patience: 4
       min_delta: 0.01
   ```
2. Increase dataset size
3. Add augmentation:
   ```yaml
   augmentations:
     mask_ratio: 0.2
     palette_permutation: true
   ```

---

## Active Reasoner & RL Training

### Workflow 1: Program-Conditioned JEPA

**Goal:** Train JEPA to predict latent outcomes of program application (counterfactual reasoning).

#### Step 1: Generate Program Triples

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_program_triples.py \
  --manifest data/pilot_curriculum/manifest.jsonl \
  --output data/program_triples/pilot_curriculum.jsonl
```

**Output format:**
```json
{
  "input": [[0, 1], [1, 0]],
  "program": {"type": "flip_horizontal", "args": {}},
  "output": [[1, 0], [0, 1]],
  "metadata": {"rule_family": "symmetry"}
}
```

#### Step 2: Train Program-Conditioned JEPA

```bash
PYTHONPATH=. .venv/bin/python scripts/train_program_conditioned_jepa.py \
  --config configs/training/jepa_program_conditioned.yaml \
  --device cuda
```

**Key config differences:**
```yaml
# configs/training/jepa_program_conditioned.yaml
dataset_triples: data/program_triples/pilot_curriculum.jsonl  # Not manifest

program_encoder:
  embedding_dim: 64
  hidden_dim: 128
  num_layers: 2

loss:
  objective: "infonce"
  # Same as standard JEPA
```

**Validation:**
```bash
# Check prediction accuracy
cat artifacts/program_conditioned/metrics.json | jq '{
  final_loss: .losses[-1],
  prediction_accuracy: .prediction_metrics.accuracy
}'
```

### Workflow 2: Active Reasoner Policy Training

**Goal:** Learn RL policy to navigate hypothesis search using program-conditioned JEPA.

#### Step 1: (Optional) Behavioral Cloning Bootstrap

```bash
PYTHONPATH=. .venv/bin/python scripts/pretrain_bc.py \
  --config configs/training/rl/bc_active_reasoner.yaml \
  --device cuda
```

**When to use BC:**
- Cold-start RL (sparse rewards)
- Have expert traces from symbolic solver

**Skip if:** Starting from scratch with dense reward shaping.

#### Step 2: Train Active Reasoner

```bash
PYTHONPATH=. .venv/bin/python scripts/train_active_reasoner.py \
  --config configs/training/rl/active_reasoner.yaml
```

**Config structure:**
```yaml
env:
  dataset_path: data/program_triples/pilot_curriculum.jsonl
  jepa_config_path: configs/training/jepa_program_conditioned.yaml
  program_checkpoint: artifacts/program_conditioned/checkpoint_epoch_0030.pt
  max_actions: 64
  reward:
    success_distance: 0.1    # Distance threshold for success
    success_bonus: 1.0       # Reward for solving
    step_penalty: 0.02       # Per-step cost (encourage efficiency)
    simplicity_weight: 0.01  # Prefer shorter programs

trainer:
  iterations: 60             # ~2000 episodes total
  episodes_per_iter: 32
  lr: 3.0e-4
  gamma: 0.95                # Discount factor
```

**Monitoring RL Training:**
```bash
# Watch episode rewards
tail -f artifacts/active_reasoner/train.log | grep "episode_reward_mean"
```

**Convergence indicators:**
- Episode reward: -0.5 → 0.8+ (over 60 iterations)
- Episode length: 50+ → 10-20 (more efficient search)
- Success rate: 10% → 70%+

### Workflow 3: Hierarchical RL (Option Discovery)

**⚠️ Status:** Scaffolding exists but not fully implemented (see `ENGINEERING_REVIEW.md`).

**Planned workflow:**

1. **Collect rollouts:**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/rollout_latent_env.py \
     --env-config configs/training/rl/latent_rollout_env.yaml \
     --episodes 100 \
     --output data/rollouts/pilot_curriculum.jsonl
   ```

2. **Mine options:**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/discover_options.py \
     --traces data/rollouts/pilot_curriculum.jsonl \
     --min-support 3 \
     --output data/options/discovered.json
   ```

3. **Train PPO with options:**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/train_hierarchical.py \
     --config configs/training/rl/ppo_latent_env.yaml
   ```

---

## Advanced Topics

### Multi-Phase Curriculum Training

**Goal:** Gradually increase task complexity (atomic → sequential → compositional).

#### Configuration

```yaml
# configs/data/multi_phase_curriculum.yaml
task_schedule:
  phase_i:   {count: 2000, primitives: ["flip_horizontal", "flip_vertical"]}
  phase_ii:  {count: 3000, primitives: ["flip_horizontal", "rotate_90", "recolor"]}
  phase_iii: {count: 5000}  # All primitives

program:
  length_schedule:
    phase_i:   {1: 1.0}           # Atomic only
    phase_ii:  {1: 0.5, 2: 0.5}   # Mix
    phase_iii: {2: 0.4, 3: 0.4, 4: 0.2}  # Sequential
```

#### Training Strategy

1. **Generate curriculum dataset:**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
     --config configs/data/multi_phase_curriculum.yaml
   ```

2. **Train JEPA on full curriculum:**
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
     --config configs/training/jepa_pretrain_gpu.yaml \
     --device cuda
   ```
   The mixed curriculum acts as implicit curriculum learning.

3. **Evaluate per-phase:**
   ```bash
   # Filter manifest by phase
   jq 'select(.metadata.phase == "phase_i")' \
     data/multi_phase_curriculum/manifest.jsonl > /tmp/phase_i.jsonl

   PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
     --tasks /tmp/phase_i.jsonl \
     --output artifacts/eval/phase_i.json
   ```

**Expected progression:**
- Phase I (atomic): 95%+ solve rate
- Phase II (2-step): 80-90%
- Phase III (3-4 step): 60-80%

### Out-of-Distribution Evaluation

**Goal:** Test generalization to unseen primitives, grid sizes, complexity.

#### Generate OOD Dataset

```yaml
# configs/data/pilot_ood.yaml
grid:
  min_size: 20      # Larger than training (was 8)
  max_size: 40

generator:
  allowed_primitives: ["flood_fill", "connected_components"]  # Held-out primitives

program:
  length_schedule:
    sequential: {4: 0.5, 5: 0.3, 6: 0.2}  # Longer programs
```

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
  --config configs/data/pilot_ood.yaml
```

#### Evaluation

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
  --tasks data/pilot_ood/manifest.jsonl \
  --output artifacts/eval/ood.json \
  --latent-config configs/training/jepa_pretrain_gpu.yaml \
  --latent-device cuda
```

**Analyzing OOD performance:**
```bash
# Compare in-distribution vs OOD
jq '{
  id: .variants.dsl_only.success_rate,
  ood: <ood_result>.variants.dsl_only.success_rate,
  gap: (.variants.dsl_only.success_rate - <ood_result>.variants.dsl_only.success_rate)
}' artifacts/eval/pilot_curriculum.json
```

**Acceptable gap:** <20% (e.g., 82% ID, 65% OOD)

---

## Troubleshooting

### Training Issues

#### Issue 1: CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB (GPU 0; 47.54 GiB total capacity)
```

**Solutions (in order of preference):**

1. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 512  # Was 1024
   ```

2. **Increase gradient accumulation:**
   ```yaml
   training:
     grad_accum_steps: 4  # Was 2, maintains effective batch
   ```

3. **Enable mixed precision:**
   ```yaml
   training:
     mixed_precision: "fp16"  # 40% memory savings
   ```

4. **Reduce model size (last resort):**
   ```yaml
   object_encoder:
     hidden_dim: 128  # Was 256
     num_embeddings: 256  # Was 512
   ```

5. **Clear GPU cache between runs:**
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
     PYTHONPATH=. .venv/bin/python scripts/train_jepa.py ...
   ```

#### Issue 2: Loss Not Decreasing

**Symptoms:**
- Loss stuck at 3.5+ for 10+ epochs
- Validation loss = training loss (no learning)

**Diagnosis:**

1. **Check learning rate:**
   ```bash
   # TensorBoard: optimizer/lr should start ~1e-4, decrease with warmup
   ```
   If LR too low (<1e-6), increase:
   ```yaml
   optimizer:
     lr: 1.0e-4  # Was 1.0e-5
   ```

2. **Check gradient flow:**
   ```python
   # Add to train_jepa.py after backward():
   total_norm = 0
   for p in experiment.trainer.encoder.parameters():
       if p.grad is not None:
           total_norm += p.grad.data.norm(2).item() ** 2
   print(f"Gradient norm: {total_norm ** 0.5}")
   ```
   If norm ~0: Gradients vanishing, reduce depth or add skip connections.
   If norm >100: Gradients exploding, reduce LR or increase grad clipping.

3. **Check data loading:**
   ```bash
   # Verify dataset isn't corrupted
   head -5 data/pilot_curriculum/manifest.jsonl
   ```

#### Issue 3: DataLoader Hangs

**Symptoms:**
- Training freezes after "Epoch 1/30"
- No GPU activity (`nvidia-smi` shows 0% util)

**Solutions:**

1. **Reduce num_workers:**
   ```yaml
   training:
     num_workers: 4  # Was 8
   ```

2. **Disable pin_memory:**
   ```yaml
   training:
     pin_memory: false  # May help with RAM issues
   ```

3. **Check shared memory (Linux):**
   ```bash
   df -h /dev/shm  # Should have >10GB free
   # If low, mount larger tmpfs:
   sudo mount -o remount,size=20G /dev/shm
   ```

#### Issue 4: Embedding Collapse

**Symptoms:**
- Isotropy <0.3
- VQ usage <20%
- Loss decreases but solve rate doesn't improve

**Solutions:**

1. **Add SIGReg regularization:**
   ```yaml
   sigreg:
     weight: 0.1
   ```

2. **Increase queue size:**
   ```yaml
   loss:
     queue_size: 8192  # Was 4096, harder negatives
   ```

3. **Reduce temperature:**
   ```yaml
   loss:
     temperature: 0.05  # Was 0.07, harder negatives
   ```

4. **Enable VQ refresh:**
   ```yaml
   object_encoder:
     vq_refresh_enabled: true
     vq_refresh_interval: 50
     vq_refresh_usage_threshold: 0.01
   ```

### Evaluation Issues

#### Issue 1: Low Solve Rate on Synthetic Tasks

**Symptoms:**
- Training solve rate <50% (expected 80%+)

**Diagnosis:**

1. **Check dataset difficulty:**
   ```bash
   cat data/pilot_curriculum/summary.json | jq .program_length_histogram
   ```
   If dominated by length 4+, reduce:
   ```yaml
   program:
     length_schedule: {1: 0.3, 2: 0.5, 3: 0.2}  # Easier
   ```

2. **Check enumeration depth:**
   ```bash
   # --max-nodes controls search depth
   # If too low, increase:
   PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
     --tasks data/pilot_curriculum/manifest.jsonl \
     --max-nodes 4  # Was 3
   ```

3. **Check primitive coverage:**
   ```bash
   # Ensure dataset uses primitives in DSL registry
   jq '.metadata.rule_trace[].name' data/pilot_curriculum/manifest.jsonl | sort | uniq
   ```

#### Issue 2: Latent Distance Correlation Poor (<-0.4)

**Symptoms:**
- JEPA correlation: -0.25 (weak)
- Embeddings don't predict solve success

**Solutions:**

1. **Train longer:**
   ```yaml
   training:
     epochs: 50  # Was 30
   ```

2. **Improve JEPA objective:**
   - Add relational loss (weight 0.05)
   - Add invariance losses (color, translation)
   - Try SIGReg

3. **Use program-conditioned JEPA:**
   - Explicitly train on program transformations
   - See "Active Reasoner & RL Training" section

### System Issues

#### Issue 1: Auto-Shutdown During Training

**Symptoms:**
- Training interrupted mid-epoch
- No checkpoint saved

**Prevention:**

1. **Use tmux/screen:**
   ```bash
   tmux new -s jepa_training
   # Run training
   # Detach: Ctrl+B, D
   ```

2. **Checkpoint every epoch (default):**
   - Already enabled, checkpoints in `artifacts/jepa/pretrain_gpu/`

3. **Add manual checkpoint hook (if needed):**
   ```python
   # In train_jepa.py training loop:
   import signal

   def save_emergency_checkpoint(signum, frame):
       checkpoint_path = checkpoint_dir / f"emergency_epoch_{epoch}.pt"
       torch.save({...}, checkpoint_path)
       sys.exit(0)

   signal.signal(signal.SIGTERM, save_emergency_checkpoint)
   ```

#### Issue 2: Disk Space Exhausted

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**

1. **Clean old checkpoints:**
   ```bash
   # Keep only last 5 epochs
   cd artifacts/jepa/pretrain_gpu
   ls -t checkpoint_epoch_*.pt | tail -n +6 | xargs rm
   ```

2. **Disable TensorBoard (if needed):**
   ```yaml
   logging:
     enabled: false
   ```

3. **Use compression for tokenized datasets:**
   ```bash
   # Compress shards (50-70% savings)
   gzip artifacts/tokenized/pilot_curriculum/*.pt
   ```

---

## Appendix: Quick Reference

### Essential Commands

```bash
# Generate dataset
PYTHONPATH=. .venv/bin/python scripts/generate_dataset.py \
  --config configs/data/pilot_curriculum.yaml

# Train JEPA (single GPU)
PYTHONPATH=. .venv/bin/python scripts/train_jepa.py \
  --config configs/training/jepa_pretrain_gpu.yaml \
  --device cuda

# Evaluate synthetic
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
  --tasks data/pilot_curriculum/manifest.jsonl \
  --output artifacts/eval/result.json

# Evaluate ARC-1 dev
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
  --arc-dev-root external/ARC/data/training \
  --output artifacts/eval/arc_dev.json

# TensorBoard
.venv/bin/tensorboard --logdir artifacts/jepa/pretrain_gpu/tensorboard

# GPU monitoring
watch -n 1 nvidia-smi
```

### File Locations

| Component | Path |
|-----------|------|
| JEPA checkpoints | `artifacts/jepa/pretrain_gpu/checkpoint_epoch_XXXX.pt` |
| Training metrics | `artifacts/jepa/pretrain_gpu/metrics.json` |
| TensorBoard logs | `artifacts/jepa/pretrain_gpu/tensorboard/` |
| Embedding diagnostics | `artifacts/jepa/pretrain_gpu/embedding_metrics.jsonl` |
| Evaluation results | `artifacts/eval/*.json` |
| Generated datasets | `data/*/manifest.jsonl` |
| Tokenized datasets | `artifacts/tokenized/*/` |

### Config Templates

| Use Case | Config File |
|----------|-------------|
| Quick test (CPU) | `configs/training/jepa_tiny.yaml` |
| Full training (A6000) | `configs/training/jepa_pretrain_gpu.yaml` |
| LeJEPA regularization | `configs/training/active_reasoner_sigreg.yaml` |
| Program-conditioned | `configs/training/jepa_program_conditioned.yaml` |
| Active Reasoner | `configs/training/rl/active_reasoner.yaml` |
| Curriculum data | `configs/data/pilot_curriculum.yaml` |
| OOD data | `configs/data/pilot_ood.yaml` |

### Performance Benchmarks (A6000)

| Task | Dataset Size | Epochs | Time | Peak VRAM |
|------|--------------|--------|------|-----------|
| JEPA pretrain | 10K tasks | 30 | 8 hours | 28GB |
| JEPA pretrain | 50K tasks | 30 | 36 hours | 35GB |
| Program-conditioned | 10K triples | 20 | 6 hours | 22GB |
| Active Reasoner | 5K tasks | 60 iters | 12 hours | 18GB |
| Evaluation (synthetic) | 10K tasks | - | 2 hours | 8GB |
| Evaluation (ARC-1 dev) | 400 tasks | - | 30 min | 6GB |

### Troubleshooting Decision Tree

```
Loss not decreasing?
├─ LR too low? → Increase to 1e-4
├─ Gradients vanishing? → Check grad norm, reduce depth
└─ Data issue? → Verify manifest integrity

GPU OOM?
├─ Reduce batch_size → 512, 256, ...
├─ Increase grad_accum_steps → 4, 8, ...
└─ Enable mixed_precision → fp16/bf16

Low solve rate?
├─ Dataset too hard? → Check program_length_histogram
├─ Search depth too low? → Increase --max-nodes
└─ JEPA correlation poor? → Train longer, add regularization

DataLoader hangs?
├─ Reduce num_workers → 4, 2, 0
└─ Check /dev/shm space → Remount with larger size
```

---

## Next Steps

After completing this guide:

1. **Production Training Run:**
   - Generate 50K task curriculum
   - Train JEPA for 30 epochs (~36 hours on A6000)
   - Validate correlation >-0.6

2. **Implement Resume Functionality:**
   - Add `--resume` flag to `train_jepa.py`
   - Test checkpoint recovery after interruption

3. **Hierarchical RL (Phase 2):**
   - Complete option discovery pipeline
   - Train PPO on latent environment
   - See `ACTIVE_REASONER_IMPLEMENTATION_GUIDE.md`

4. **ARC-2 Evaluation:**
   - Download ARC-2 test set (when released)
   - Run full evaluation suite
   - Compare with ARC-1 baseline

---

## Additional Resources

- **Project Blueprint:** `Project_Blueprint.md` (strategic roadmap)
- **Active Reasoner Guide:** `ACTIVE_REASONER_IMPLEMENTATION_GUIDE.md` (RL details)
- **Engineering Review:** `ENGINEERING_REVIEW.md` (architecture decisions)
- **Dependencies:** `docs/DEPENDENCIES.md` (package documentation)
- **Contributing:** `CONTRIBUTING.md` (development workflow)

For questions or issues, refer to the troubleshooting section or consult the main `README.md`.
