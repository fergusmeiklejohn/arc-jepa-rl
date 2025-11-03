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

---

## 3. Stage A: JEPA Pretraining

### Objective
Train a **Context-Target JEPA** to predict latent changes between grid pairs without reconstructing pixels.

### Input
- Synthetic input-output pairs (100k+)
- Data augmentations: masking, cropping, noise, color shuffling

### Loss
- **Joint embedding predictive loss** (contrastive or cosine)
- Invariance constraints for symmetry, color permutation, translation

### Expected Outcome
A **latent space** capturing transformation invariants (e.g., “reflection”, “flood-fill”, “recolor”).

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

---

## 5. Stage C: Meta-JEPA (Rule-of-Rules Learning)

### Purpose
Train a **meta-model** that predicts transformations between *tasks*, not just *grids*.

### Input
- Each training example = multiple (input, output) pairs from one rule family
- Target: latent vector representing that rule family

### Training Signal
Contrastive learning between rule embeddings:
- Close for tasks from same transformation family
- Distant for unrelated rule types

### Effect
The model learns to reason in **rule space** — forming clusters of related transformations.

---

## 6. Stage D: Out-of-Distribution Evaluation

### Datasets
1. **ARC-1 Eval** (official)
2. **ARC-2 Eval**
3. **Human-crafted surprise tasks** intentionally outside training priors

### Evaluation Metrics
- Success rate (exact match)
- Latent “distance-to-goal”
- Number of options executed
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

All generators output (input grid, output grid, rule_trace).

---

## 8. Curriculum Design

### Phase 1 — Structural Pretraining
- Train JEPA on synthetic data for 1–2 epochs.
- Validate on small held-out synthetic set.

### Phase 2 — Grounded Skill Learning
- Initialize option policies via supervised imitation of known transformations.
- Fine-tune with latent JEPA reward to discover efficient temporal compositions.

### Phase 3 — Meta-Reasoning
- Train meta-JEPA on embeddings of rulesets (task families).
- Encourage clustering and relational prediction.

### Phase 4 — OOD Reinforcement
- Evaluate and fine-tune using exploration bonuses for **latent novelty** (e.g., high JEPA prediction error).
- Curiosity drives discovery of new compositional primitives.

---

## 9. Growing Toward OOD “Imagination”

| Growth Mechanism | Implementation | Effect |
|------------------|----------------|--------|
| **Latent Option Discovery** | Cluster unexplained transitions to spawn new skills | Expands primitive vocabulary |
| **Curiosity Reward** | Reward transitions with high JEPA error | Encourages novelty search |
| **Meta-JEPA Bootstrapping** | Predict rule embeddings for unseen tasks | Enables analogical reasoning |
| **Symbolic Distillation** | Extract and name new skill clusters | Stabilizes compositional grammar |

---

## 10. Practical Stack

| Component | Library/Framework |
|------------|------------------|
| Representation Learning | PyTorch / Lightning |
| RL Engine | PufferLib / RLlib |
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
2. Pretrain a **JEPA backbone** on 50k+ synthetic pairs.  
3. Train **option policies** with heuristic traces + RL fine-tuning.  
4. Build **meta-JEPA** for task-level embedding and reasoning.  
5. Benchmark systematically on ARC-1, ARC-2, and human-designed surprise tasks.

