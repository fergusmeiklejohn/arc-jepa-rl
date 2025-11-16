# Blueprint: Training a JEPA × Hierarchical RL System for OOD ARC Reasoning

## Overview
This document outlines how to design, train, and grow a hybrid **JEPA + Hierarchical RL** system capable of abstract, compositional reasoning and eventual *out-of-distribution (OOD)* thinking on ARC-style tasks.  

The goal is not only to solve known ARC tasks but to *imagine new solutions* for unseen ones by learning transferable primitives, abstract representations, and meta-rules.

---

## 1. Conceptual Foundation

| Component | Role | Analogy |
|------------|------|----------|
| **JEPA** | Learns *what changes* and *what stays invariant* between input-output pairs; compresses temporal relations. | “Perception and memory” — an understanding of structure. |
| **Hierarchical RL (HRL)** | Learns *how to act* via reusable, parameterized skills (options). | “Action and planning” — structured exploration of rule space. |
| **Meta-JEPA** | Learns *relations between rules*, predicting transformations-of-transformations. | “Imagination” — reasoning about rules themselves. |

Together, they enable a pipeline from concrete perception → abstract structure → meta-level reasoning.

---

## 2. Training Stages Overview

| Stage | Focus | Data Source | Main Objective |
|--------|--------|--------------|----------------|
| **A. Pretraining (JEPA)** | Learn invariant latent representations | Synthetic ARC-like data | Encode spatial/temporal invariants |
| **B. Skill Learning (HRL)** | Ground transformations as options | ARC + synthetic data | Learn reusable primitive operators |
| **C. Meta-Generalization (Meta-JEPA)** | Model relationships among rules | Generated rule compositions | Learn to infer rules from few examples |
| **D. OOD Evaluation** | Test zero-shot and compositional reasoning | ARC Eval + human-crafted tasks | Measure imagination and abstraction |

### 2.1 Staged Implementation Tracker (Beads)
The following Beads issues track concrete, staged milestones that operationalize this plan:

- JEPA InfoNCE objective + projection head + memory queue — arc-jepa-rl-4fd0 (P1)
- JEPA dataset loader + augmentations + JSONL manifest reader — arc-jepa-rl-1b0e (P1)
- Relational attention/GNN over object tokens using adjacency — arc-jepa-rl-b062 (P2)
- Synthetic ARC generator + JSONL exporter (Atomic→Sequential curriculum) — arc-jepa-rl-78eb (P1)
- Define ARC DSL and symbolic enumerator — arc-jepa-rl-fb5b.3 (P1, in_progress)
- HRL options + envs with JEPA latent reward — arc-jepa-rl-c8da (P1)
- Meta-JEPA rule-family trainer — arc-jepa-rl-217a (P2)

Notes:
- JEPA representation work (above) feeds HRL options and Meta-JEPA; generator and DSL enable curriculum and symbolic evaluation.
- Refer to `.beads/issues.jsonl` for live status; new work should be created in Beads and linked via discovered-from.

---

## 3. Stage A: JEPA Pretraining

### Objective
Train an **object-centric Context-Target JEPA** to predict latent changes between grid pairs without reconstructing pixels, while preserving discrete relational structure.

### Input
- Synthetic input-output pairs (100k+)
- Object-centric view: each grid is decomposed into connected components with geometry/color features and adjacency graphs (see `arcgen/objects.py`)
- Data augmentations: masking, cropping, noise, color shuffling

### Architecture & Loss
- **Object tokenizer**: converts grids into padded sets of object tokens with relational adjacency (`training/modules/object_tokenizer.py`)
- **Vector-quantized encoder**: projects object tokens through an MLP + VQ-VAE bottleneck to encourage crisp, reusable codes (`training/modules/vq.py`)
- **Relational attention heads**: multi-layer adjacency-aware self-attention stack over object tokens (`training/modules/relational.py`) for symmetry and counting.
- **Joint embedding predictive loss** (contrastive/multi-step InfoNCE) computed on mean-pooled object embeddings; future work adds relational consistency losses.
- **Invariance penalties** now configurable via `training.invariance` weights (color permutations, translations, horizontal/vertical symmetries) so JEPA learns coordinate- and palette-stable encodings before downstream use.
- Invariance constraints for symmetry, color permutation, translation.

### Expected Outcome
A **latent codebook** capturing transformation invariants (e.g., “reflection”, “flood-fill”, “recolor”) with discrete tokens that can be composed downstream by HRL and program-synthesis components.

### Training Infrastructure
- `training/jepa/object_pipeline.py` and `training/jepa/trainer.py` build the tokenizer/encoder from YAML configs and expose encoding helpers.
- `training/jepa/loop.py` provides `ObjectCentricJEPAExperiment` with optimizer wiring and epoch helpers; `scripts/train_jepa.py --dry-run` exercises the stack.
- `scripts/train_jepa.py` now drives a `torch.utils.data.DataLoader` around JEPA datasets: manifest samples are tokenized inside `ManifestTokenizedPairDataset` workers, pin-memory transfers use `non_blocking=True`, and `training.num_workers`/`training.pin_memory` knobs tune host-device overlap.
- Object tokenization runs through a vectorized NumPy path (connected-components, feature extraction, adjacency) with a legacy fallback plus `scripts/benchmark_tokenizer.py` to measure the >2× speedup and guard numerical fidelity.
- When memory is tight, `training.grad_accum_steps` accumulates micro-batches before stepping the optimizer/InfoNCE queue so we can emulate large effective batch sizes without OOMs.
- Mixed precision is gated behind `training.amp`: enabling it on CUDA boxes wraps encoder/projection/InfoNCE computations in `torch.cuda.amp.autocast` with `GradScaler`, while CPU runs automatically fall back to standard FP32 so configs remain portable.
- Status update (2025-11-05): projection heads + InfoNCE queue + manifest loader now implemented; synthetic generator + latent option env wired for HRL integration; typed DSL primitives + enumerator + interpreter landed; neural guidance scaffolding (dataset builder, scorer, beam search, training CLI) in place; relational graph attention now backed by configurable multi-head layers in the object encoder.

---

## 4. Stage B: Skill Learning with Hierarchical RL

### Architecture
- Low-level skills (“options”): learned policies mapping grid → grid over k steps.
- High-level planner: selects options in latent space to minimize JEPA “distance-to-goal.”

### Training Procedure
1. Use heuristic or symbolic solvers to generate demonstration trajectories.
2. Pretrain options with behavioral cloning.
3. Fine-tune with RL (e.g., PPO, A2C) using **latent JEPA rewards**:
   - Reward = −‖z_pred − z_goal‖ + novelty_bonus

### Skills Library (Examples)
| Category | Description |
|-----------|-------------|
| Geometry | Mirror, rotate, translate, scale |
| Topology | Flood-fill, connect components |
| Color | Remap palette, threshold |
| Logic | If-condition-based replace, repeat-until-stable |
| **Discovered Options** | Auto-promoted latent skills identified via VQ code clusters / curiosity (see Section 9) |

---

## 5. Stage C: Meta-JEPA (Rule-of-Rules Learning)

### Purpose
Train a **meta-model** that predicts transformations between *tasks*, not just *grids*.

### Input
- Each training example = multiple (input, output) pairs from one rule family
- Target: latent vector representing that rule family

### Training Signal
- Contrastive learning between rule embeddings (close within families, distant otherwise)
- Incorporate **neural program traces**: map JEPA latent deltas + primitive sequences into a DSL (Section 6b) to provide supervision for meta-rule clustering.

### Effect
The model learns to reason in **rule space** — forming clusters of related transformations, while emitting priors that guide few-shot program induction (Section 6c).

- Implementation status: `training/meta_jepa/` provides dataset builders, a
  contrastive encoder, and a `MetaJEPATrainer` with CLI support
  (`scripts/train_meta_jepa.py`) for learning rule-family embeddings that feed
  few-shot priors.

---

## 6. Symbolic-Neural Bridge & Few-Shot Inference

### 6a. Object-Aware JEPA Outputs
- JEPA yields discrete code indices (VQ) per object along with relational adjacency.
- HRL consumes codes as option arguments; Meta-JEPA consumes aggregated summaries.

### 6b. Program Synthesis Integration
- Define a strongly-typed grid DSL (map/filter/group/paint/flood-fill, etc.) with cost heuristics.
- Use JEPA object embeddings to produce **skeleton proposals**; perform neural-guided beam search over DSL instantiations.
- Maintain a version space of programs consistent with provided examples; apply constraint solving for pruning (symmetry, color counts, adjacency checks).

### 6c. Few-Shot Solver Pipeline
1. Encode context/target examples with the object-centric JEPA encoder.
2. Meta-JEPA provides priors over rule families and likely skeletons (see
   `training/meta_jepa/prior.py` — now wired into `GuidedBeamSearch` and the
   `FewShotSolver` to bias program ordering).
3. Neural-guided enumerator scores candidate programs; symbolic executor validates against examples.
4. Resulting program is executed on ARC test grids; fallback strategies include caching partial programs and using HRL options to refine mismatched outputs.

### 6d. Evaluation Hooks
- Track success rate vs. beam width, solver compute, and JEPA code diversity.
- Ablations: JEPA-only embeddings, DSL-only search, hybrid pipeline.
- `training/eval/` now provides an `EvaluationSuite` with CLI support
  (`scripts/evaluate_arc.py`) to run DSL-only vs meta-guided ablations and
  collect success/program-count metrics over JSONL task manifests.

---

## 7. Stage D: Out-of-Distribution Evaluation

### Datasets
1. **ARC-1 Eval** (official)
2. **ARC-2 Eval**
3. **Human-crafted surprise tasks** intentionally outside training priors

### Evaluation Metrics
- Success rate (exact match)
- Latent “distance-to-goal”
- Number of options executed
- Program search depth / enumerated candidates
- Novel rule discovery rate (emergent skills)

---

## 7. Data Generation Strategy

### 7.1 Core Sources
| Source | Size | Role |
|---------|------|------|
| ARC-1 Train | ~400 | Fine-tuning & grounding |
| ARC-2 Train | ~400 | Meta-validation |
| Synthetic ARC-like | 50k–500k | JEPA pretraining & invariance learning |
| Recomposed rule sequences | 10k–100k | Compositional generalization |
| Human-crafted “trick” tasks | 100–500 | OOD benchmarking |

---

### 7.2 Synthetic Task Generators

| Family | Description | Goal |
|---------|--------------|------|
| **Color Permutations** | Recolor objects systematically | Build color invariance |
| **Geometric Transforms** | Rotate, mirror, scale, crop | Learn spatial symmetry |
| **Rule Composition** | Sequentially combine primitives | Temporal abstraction |
| **Noise Injection** | Add irrelevant patterns | Force relational focus |
| **Meta-Rule Shifts** | Swap input/output roles | Teach causal inference |
| **Interpolated Programs** | Blend two known rules | Encourage latent smoothness |
| **Discoverable Primitives** | Generate patterns requiring novel subroutines | Test option discovery |

All generators output (input grid, output grid, rule_trace).

---

## 8. Curriculum Design

### Phase 1 — Structural Pretraining
- Train the object-centric JEPA encoder on synthetic data for 1–2 epochs (using `ObjectCentricJEPAExperiment`).
- Validate on small held-out synthetic set; inspect VQ code usage and object-level reconstruction proxies.

### Phase 2 — Grounded Skill Learning
- Initialize option policies via supervised imitation of known transformations.
- Fine-tune with latent JEPA reward to discover efficient temporal compositions.
- Auto-promote frequently-used VQ codes into the primitive registry.

### Phase 3 — Meta-Reasoning
- Train meta-JEPA on embeddings of rulesets (task families).
- Encourage clustering and relational prediction.
- Align meta-JEPA outputs with DSL skeleton categories for better few-shot priors.

### Phase 4 — Symbolic Search & Few-Shot Evaluation
- Integrate neural-guided program search; run few-shot evaluation loops.
- Use discrepancies to trigger option discovery / data augmentation.
- Automated rollout mining (`training/options`) now discovers recurring option
  sequences and promotes them into typed primitives with regression coverage
  (`tests/test_option_discovery.py`).
- The few-shot solver (`training/solver.FewShotSolver`) enumerates DSL programs
  with optional neural guidance, consuming the promoted primitives to solve
  demos under tight node budgets (`tests/test_few_shot_solver.py`).

### Phase 5 — OOD Reinforcement
- Evaluate and fine-tune using exploration bonuses for **latent novelty** (e.g., high JEPA prediction error).
- Curiosity drives discovery of new compositional primitives.

---

## 9. Growing Toward OOD “Imagination”

| Growth Mechanism | Implementation | Effect |
|------------------|----------------|--------|
| **Latent Option Discovery** | Cluster unexplained transitions / VQ residues to spawn new skills | Expands primitive vocabulary |
| **Curiosity Reward** | Reward transitions with high JEPA error | Encourages novelty search |
| **Meta-JEPA Bootstrapping** | Predict rule embeddings for unseen tasks | Enables analogical reasoning |
| **Symbolic Distillation** | Extract and name new skill clusters | Stabilizes compositional grammar |
| **Program Cache Mining** | Promote frequently re-used program fragments into DSL macros | Accelerates few-shot solving |

---

## 10. Practical Stack

| Component | Library/Framework |
|------------|------------------|
| Representation Learning | PyTorch / Lightning + object-centric modules (`training/jepa/*`) |
| RL Engine | PufferLib / RLlib |
| Program Search | Custom typed DSL + constraint solving + neural heuristics |
| Data Generation | Python procedural grid DSL |
| Evaluation | ARC runtime + JSON task specs |
| Visualization | Streamlit / TensorBoard |

---

## 11. Expected Evolution

| Stage | Capability |
|--------|-------------|
| Early | Recombines known skills for novel ARC tasks |
| Mid | Learns meta-patterns (“count, then fill”, “mirror conditional on color”) |
| Late | Discovers new transformation families; self-bootstraps rule space |
| Mature | Imaginative recombination, true OOD rule inference |

---

## 12. Summary

> The system’s imagination is grown, not pre-coded.  
> JEPA provides structural compression, HRL provides compositional action, and a meta-JEPA turns structure into concept-space.  
> By iteratively expanding the training universe with procedurally generated and recombined rules, we approach a system that **thinks beyond its training distribution**.

---

## Next Steps
1. Implement a **synthetic ARC generator** with controllable rule grammar.  
2. Pretrain the **object-centric JEPA backbone** (tokenizer + VQ encoder) on 50k+ synthetic pairs, validating codebook usage.  
3. Design and implement the **typed DSL enumerator** with neural guidance + constraint pruning.  
4. Train **option policies** with heuristic traces + RL fine-tuning, seeding the primitive registry with discovered VQ skills.  
5. Build the **few-shot solver loop** combining JEPA embeddings, Meta-JEPA priors, and DSL search; evaluate on ARC dev tasks.  
6. Run ablations (JEPA-only, HRL-only, DSL-only, hybrid) to quantify contributions against the critique concerns.  
4. Build **meta-JEPA** for task-level embedding and reasoning.  
5. Benchmark systematically on ARC-1, ARC-2, and human-designed surprise tasks.
