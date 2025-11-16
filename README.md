# ARC JEPA × HRL Research Sandbox

This repository implements the plan outlined in `Project_Blueprint.md` and
`Synthetic_ARC_Task_Generator_Spec.md`. The goal is to build a synthetic ARC
task generator, train JEPA representations, learn hierarchical option policies,
and evaluate out-of-distribution reasoning.

## Repository Layout

- `arcgen/` — Core grid utilities, primitive transformations, and program
  synthesizers for generating ARC-like tasks. Includes object extraction and
  relational helpers in `arcgen/objects.py`.
- `training/` — Training loops and experiment orchestration for JEPA, HRL, and
  Meta-JEPA components. Object tokenization and discrete latent modules live in
  `training/modules/`, object-centric JEPA helpers under `training/jepa/`, the
  typed DSL / enumerator scaffolding under `training/dsl/`, rollout mining +
  promotion utilities in `training/options/`, the few-shot solver pipeline in
  `training/solver/`, meta-rule clustering/training in `training/meta_jepa/`,
  and evaluation helpers in `training/eval/`.
- `envs/` — Environment wrappers exposing ARC-style tasks to RL agents.
- `configs/` — YAML configuration files for data generation, training, and
  evaluation runs.
- `scripts/` — Command-line entry points for dataset generation and training.
  - `generate_dataset.py` — Build synthetic manifests; supports curriculum schedules and `allowed_primitives` constraints.
  - `train_jepa.py` — Run manifest-backed JEPA pretraining (supports `--dry-run` and `--device`).
  - `train_meta_jepa.py` — Train the rule-family encoder on JSONL tasks with contrastive loss.
  - `train_hierarchical.py` — Launch RLlib PPO over the latent option environment using configs under `configs/training/rl/`.
  - `evaluate_arc.py` — Run the evaluation/ablation suite and emit JSON metrics.
- `tests/` — Unit and integration tests covering generators, models, and envs.

### DSL Primitive Coverage

The default DSL registry (`training/dsl/primitives.py`) now spans topology and
control-flow operators in addition to basic geometry:

- Topology: `flood_fill`, `connected_components`, `shape_bbox`, `shape_centroid`,
  and `shape_area` expose component-level reasoning hooks.
- Collections: `components_filter_by_*`, `components_map_to_subgrids`, and
  `components_fold_overlay` implement map/filter/fold combinators over the new
  `ShapeList`/`GridList` types.
- Logic: integer/shape list predicates plus `if_then_else` allow conditional
  program branches.

10+ regression tests live in `tests/test_dsl_primitives.py`, ensuring the few-shot
solver enumerator can rely on the extended registry without regressions.

## Getting Started

1. Create a Python environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install core dependencies (placeholder): `pip install -r requirements.txt`.
3. Generate synthetic datasets (pick a recipe below) and start pretraining.

### Dataset recipes

- **Baseline sequential mix** — `python scripts/generate_dataset.py --config configs/data/pilot.yaml`
- **Curriculum (atomic + sequential)** — `python scripts/generate_dataset.py --config configs/data/pilot_curriculum.yaml`
- **OOD slice (large grids, constrained primitives)** — `python scripts/generate_dataset.py --config configs/data/pilot_ood.yaml`

Each config writes a manifest and `summary.json` to its `output_root`. The generator now accepts `task_schedule` at the config root (or under `generator`) and optional `generator.allowed_primitives` for structural constraints/outlier pockets.

#### Program length curriculum

Control program complexity directly inside the `program` block:

```yaml
program:
  max_depth: 4  # optional safety clamp for any phase
  length_schedule:
    sequential:   # phase-specific histogram (atomic/sequential or roman numerals)
      2: 0.5
      3: 0.3
      4: 0.2
    atomic:
      1: 1.0
```

Supplying a single mapping (e.g., `{1: 0.4, 2: 0.4, 3: 0.2}`) applies to every phase. Weights are normalised automatically and entries beyond `max_depth` are dropped. The generator stores the sampled `program_length` per task, and `summary.json` now reports both descriptive stats and a literal `program_length_histogram` to verify the curriculum.

Evaluate a manifest to sanity-check solve rates and program counts:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py --tasks data/pilot_curriculum/manifest.jsonl --output artifacts/eval/pilot_curriculum.json
```

To benchmark against the official ARC dev (training) set, point the harness at the
directory (or individual JSON file) that contains the canonical tasks:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
  --arc-dev-root /path/to/arc-dataset/training \
  --output artifacts/eval/arc_dev.json
```

The loader validates each ARC file, uses all provided train pairs as few-shot
examples, and checks predictions against any available test outputs.

### ARC dev baseline snapshot (ARC-1 training)

The repo tracks a baseline evaluation against the official ARC-1 training split.

1. Download the dataset once (the official JSON bundle is open-source):  
   `git clone --depth 1 https://github.com/fchollet/ARC external/ARC`
2. Run the evaluation harness against the training directory:

   ```bash
   PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py \
     --arc-dev-root external/ARC/data/training \
     --output artifacts/eval/arc_dev_baseline.json
   ```

Current baseline (DSL enumerator, max_nodes=3, no meta priors available for ARC-1):

| Variant     | Success Rate | Avg. Programs Tested |
|-------------|--------------|----------------------|
| `dsl_only`  | 2.25%        | 235.57               |
| `meta_guided`* | 2.25%    | 235.57               |

\*Meta-guided mode falls back to the vanilla registry on ARC-1 because no rule traces
are provided to derive primitive histograms.

The JSON summary for reproducibility lives at `artifacts/eval/arc_dev_baseline.json`.

### Pre-tokenized JEPA manifests

Long JEPA runs avoid Python tokenization overhead by precomputing object tokens
once and streaming them from disk:

```bash
PYTHONPATH=. .venv/bin/python scripts/pretokenize_jepa.py \
  --config configs/training/jepa_pretrain.yaml \
  --output artifacts/tokenized/pilot_curriculum
```

This reads the manifest/config defaults, writes sharded `.pt` tensors plus a
`metadata.json` descriptor, and preserves per-sample metadata in each shard.
Point the trainer at the tokenized directory to enable the zero-tokenization
path (use the same `tokenizer` + `data.context_window` you precomputed with):

```yaml
# configs/training/jepa_pretrain.yaml
pre_tokenized:
  path: artifacts/tokenized/pilot_curriculum
```

`scripts/train_jepa.py` automatically switches to the new
`TokenizedPairDataset` when `pre_tokenized.path` is set; otherwise it falls back
to manifest-time tokenization.

Need to sanity-check tokenizer throughput? Run the micro-benchmark:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_tokenizer.py --samples 512 --height 24 --width 24 --respect-colors --connectivity 8
```

The script times both the legacy path and the vectorized implementation and
validates numerical parity. For heavy ARC grids (120x120, 1024 object slots, 512
color features) the command below reports ~3.1× speedup on a CPU-only run:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_tokenizer.py \
  --samples 16 --height 120 --width 120 \
  --respect-colors --connectivity 8 \
  --max-objects 1024 --max-color-features 512
```

### JEPA pretraining

Run full JEPA training against any manifest:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_jepa.py --config configs/training/jepa_pretrain.yaml --device cpu
```

Pass `--device cuda` on GPU boxes. Checkpoints and `metrics.json` land in `artifacts/jepa/pretrain/` (configurable via `training.checkpoint_dir`). Use `--dry-run` for a single dummy optimisation step.

For full A6000 runs, start from `configs/training/jepa_pretrain_gpu.yaml`; it sets a larger batch, longer schedule, and defaults to TensorBoard logging under `artifacts/jepa/pretrain_gpu/tensorboard/`.

Mixed precision is controlled via `training.amp` in the config. Set it to `true` on CUDA hosts to enable `torch.cuda.amp` autocast + GradScaler; the loop automatically reverts to FP32 if CUDA/AMP is unavailable (or when forcing `--device cpu`).

**BYOL-style target encoder:** set `loss.use_target_encoder=true` and `loss.target_ema_decay=<0-1]` in the JEPA config to enable a stop-gradient EMA copy of the encoder/projection head. The training loop automatically keeps the target network in sync via EMA updates and routes the contrastive loss through the stabilized branch.

### Hierarchical option training (RLlib)

The RL stack uses RLlib. Install the optional dependency first:

```bash
pip install "ray[rllib]"
```

Then launch PPO on the latent option environment:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_hierarchical.py \
  --config configs/training/rl/ppo_latent_env.yaml \
  --output-dir artifacts/rl/ppo_run --stop-iters 2
```

The sample config wires RLlib into `ArcLatentOptionEnv`, loads JEPA encoder settings via `jepa_config_path`, and exposes knobs for reward shaping, curriculum schedules, and PPO hyper-parameters. Metrics plus the final checkpoint path are stored under the chosen output directory.

#### Mining RL traces for new options

1. Collect rollouts (random or trained policies) with `scripts/rollout_latent_env.py --env-config configs/training/rl/latent_rollout_env.yaml --jepa-config <jepa.yaml> --episodes 100 --output data/latent_option_traces.jsonl`. Each step now records the exact `grid_before`/`grid_after` pairs required for replay.
2. Extract candidate macros via `python scripts/discover_options.py --env-config configs/training/rl/latent_rollout_env.yaml --traces data/latent_option_traces.jsonl --output artifacts/options/discovered.json --min-support 3 --max-length 3`.
3. (Optional) Pass `--promote` to validate registration in the DSL registry; the JSON summary lists each auto-named primitive and its underlying option sequence so you can wire them into future solver runs.

`training/options/traces.py` exposes `load_option_episodes_from_traces(...)` if you want to plug the mined episodes directly into the few-shot solver or custom discovery logic.

> This project is under active development. Consult the blueprint documents for
> the full roadmap and open Beads issues for next steps.

## Dependencies

- Python deps are tracked via `requirements*.txt` files in the repo root. Use `uv venv --python 3.11 .venv` followed by `uv pip install --python .venv/bin/python -r requirements.txt`.
- Install optional extras as needed:
  - `requirements-dev.txt` for test tooling (`pytest`).
  - `requirements-rl.txt` for RLlib-based PPO/A2C training.
- `scipy>=1.11` is included in `requirements.txt` to power the vectorized object
  tokenization path; when it's missing the code falls back to the legacy Python
  implementation.
- See `docs/DEPENDENCIES.md` for the full process, package descriptions, and contribution guidelines.

## Decision Records

Architecture choices are tracked in `docs/adr/`.
- `0001-rl-engine.md` — RLlib selected for hierarchical option training.
- `0002-jepa-objective.md` — JEPA uses multi-step InfoNCE with memory queue and
  specified augmentation policy.
