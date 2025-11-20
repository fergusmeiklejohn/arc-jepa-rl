# Synthetic ARC Task Generator Specification

## 1. Purpose
To create a scalable, controllable data generator that produces **ARC-like tasks** for training and evaluating a JEPA × HRL system.  
The generator must emulate the **compositional, abstract, and relational structure** of ARC tasks while being configurable enough to generate OOD (out-of-distribution) variants.

---

## 2. Design Goals

| Goal | Description |
|------|--------------|
| **Structural richness** | Cover key ARC transformation types (geometric, color, logic, counting, repetition). |
| **Composable rules** | Allow combining primitives into multi-step programs. |
| **Controllable diversity** | Parameterized generation for spatial size, noise, rule complexity, color palette, etc. |
| **Label-free supervision** | Self-supervised: generate (input, output, transformation_trace) triplets automatically. |
| **Curriculum scalability** | Support progressive complexity — from single-rule to multi-rule meta-tasks. |

---

## 3. System Architecture

### 3.1 Modules
1. **Grid Engine**
   - Represents N×N discrete color grids.
   - API: `Grid.random(pattern, colors, size)` and transformation methods (`rotate`, `mirror`, `fill`, `replace`, etc.).
2. **Primitive Library**
   - Core transformation primitives (see §4).
   - Composable as sequential or conditional rules.
3. **Program Synthesizer**
   - Generates random or curriculum-driven compositions of primitives.
   - Example: `[mirror_x, flood_fill(color=2), recolor(3→1)]`
4. **Task Generator**
   - Executes synthesized program to produce (input, output, metadata).
5. **Evaluator**
   - Computes complexity metrics (rule depth, branching factor, object count).
6. **Exporter**
   - Saves as JSON or NumPy arrays with metadata for JEPA training.

---

## 4. Primitive Transformation Grammar

### 4.1 Base Categories

| Category | Example Primitives | Description |
|-----------|-------------------|--------------|
| **Geometry** | `mirror_x`, `mirror_y`, `rotate90`, `translate(dx, dy)`, `scale(k)` | Spatial transformations |
| **Color/Value** | `recolor(a→b)`, `invert_palette`, `threshold_gt(v)` | Pixel/value remapping |
| **Topology/Object** | `flood_fill(region, color)`, `extract_shape(mask)`, `merge_touching()` | Object-level reasoning |
| **Logical/Conditional** | `if(color=a, replace_with=b)`, `repeat_until_stable(rule)` | Conditional logic |
| **Counting/Pattern** | `repeat(pattern, n, direction)`, `tile(shape, spacing)` | Quantitative reasoning |
| **Meta-rules** | `apply(rule_A) then (rule_B)`, `apply_random(rule_set)` | Compositional reasoning |

### 4.2 Compositional Grammar

RULE ::= PRIMITIVE | SEQ(RULE, RULE) | IF(CONDITION, RULE) | REPEAT(RULE, k)

Example program:

IF(has_symmetry_x)
THEN mirror_x
ELSE recolor(1→2)
REPEAT(flood_fill(random_region), 2)

---

## 5. Parameter Ranges

| Parameter | Range / Options | Effect |
|------------|-----------------|--------|
| Grid size | 5×5 – 30×30 | Spatial complexity |
| Colors | 3–10 | Combinatorial diversity |
| Object count | 1–10 | Visual clutter |
| Rule depth | 1–5 | Temporal abstraction |
| Conditional probability | 0.0–0.5 | Logical reasoning density |
| Noise level | 0–0.2 | Robustness pressure |
| Novelty bias | 0–1 | Encourages OOD rule synthesis |

---

## 6. Curriculum Phases

| Phase | Composition Type | Example | Purpose |
|--------|------------------|----------|----------|
| **I. Atomic** | Single primitive | `mirror_x`, `recolor` | Learn basic invariants |
| **II. Sequential** | 2–3 chained rules | `rotate → recolor` | Teach composition |
| **III. Conditional** | If/Else logic | `if(color=2) → fill(3)` | Contextual reasoning |
| **IV. Recursive** | Repeat until stable | `repeat(remove_outer_layer)` | Temporal recursion |
| **V. Meta-Rule** | Program-space ops | `apply(rule_A) then inverse(rule_B)` | Rule-of-rules abstraction |

Each curriculum stage doubles as a **JEPA time horizon** — enabling temporal abstraction during training.

---

## 7. Output Format

Each generated task stored as JSON:

```json
{
  "id": "task_000123",
  "input": [[0,1,1],[2,0,1],[0,0,2]],
  "output": [[1,1,1],[2,0,1],[1,1,2]],
  "rule_trace": [
    {"primitive": "mirror_x"},
    {"primitive": "recolor", "params": {"from":2,"to":1}}
  ],
  "metadata": {
    "phase": "II",
    "grid_size": 9,
    "color_count": 4,
    "complexity": 2.3
  }
}

### Program triple export (for counterfactual JEPA + Active Reasoner)

Emit an auxiliary JSONL where each record captures (input, program, output) triples with tokenized program metadata:

```json
{
  "input": [[0,1,1],[2,0,1],[0,0,2]],
  "output": [[1,1,1],[2,0,1],[1,1,2]],
  "program": {
    "primitives": ["mirror_x", "recolor"],
    "params": [{}, {"from": 2, "to": 1}],
    "ids": [5, 17],            // integer IDs aligned to the primitive vocab
    "param_vectors": [[0,0,..],[2,1,..]], // fixed-width parameter embeddings
    "mask": [1,1]              // 0/1 padding mask for variable length
  },
  "context_window": 3          // ensure at least k=3 frames per ADR-0002/JEPA loader
}
```

The JEPA `ManifestTokenizedPairDataset` expects `context_window>=3`; generator traces should provide at least that many sequential frames when available so multi-step contexts can be sliced without re-tokenizing. Program IDs/parameter vectors are consumed by `ProgramTripleDataset` and the Active Reasoner's `HypothesisSearchEnv`.


⸻

8. Dataset Assembly Plan

Dataset	Size	Source	Usage
Pretrain-A (Atomic)	50k	Random single rules	JEPA invariance learning
Pretrain-B (Sequential)	100k	Composed rules	JEPA temporal compression
Fine-Tune (Mixed ARC-like)	5k	Weighted mix of ARC-1 tasks + synthetic	HRL skill training
Meta-Ruleset (Rule families)	10k	Generated rule compositions	Meta-JEPA training
Evaluation (OOD)	500	Held-out unseen rules + human tasks	Zero-shot test


⸻

9. Integration with Training Pipeline

Component	Consumes	Produces
Task Generator	rule grammar + parameters	JSON tasks
JEPA Trainer	input/output pairs	latent encoder
HRL Trainer	environment interface	skill policies
Meta-JEPA	rule embeddings	meta-latent space
Evaluator	ARC Eval sets	OOD metrics


⸻

10. Extending for OOD Research

10.1 Out-of-Distribution Axes
	1.	Structural OOD — new rule compositions unseen in training.
	2.	Semantic OOD — same structures, new color/object meanings.
	3.	Program OOD — entirely new primitives derived from discovered latent clusters.
	4.	Causal OOD — input/output relationships reversed or altered.

10.2 Controlled OOD Experiments
	•	Withhold entire primitive families (e.g., no rotations during training).
	•	Train on 5–6 colors, test on 9–10.
	•	Train on 10×10 grids, test on 20×20.
	•	Mix deterministic and stochastic rule generators.

⸻

11. Implementation Notes
	•	Language: Python (NumPy + PIL + networkx for object graphs)
	•	Output: JSON + PNG pairs
	•	Reproducibility: seeded PRNG and metadata logging
	•	Performance: ~10–50 tasks/sec per CPU core
	•	Extensibility: add new primitives via class registry pattern

⸻

12. Example Usage

from arcgen import TaskGenerator

gen = TaskGenerator(grid_size=(10,10), colors=6)
task = gen.sample(rule_depth=3, conditional_prob=0.2)
task.save("data/train/task_00123.json")


⸻

13. Future Extensions

Direction	Description
Latent Skill Discovery	Cluster JEPA latent deltas to auto-spawn new primitives.
Curriculum Auto-tuning	Adaptive sampling based on JEPA loss or HRL success rate.
ARC-2 Analog Tasks	Blend human and synthetic rules to simulate unseen reasoning types.
Meta-JEPA World Model	Predict next task embeddings; generate imagined ARC tasks.
