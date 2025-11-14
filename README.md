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

## Getting Started

1. Create a Python environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install core dependencies (placeholder): `pip install -r requirements.txt`.
3. Generate synthetic datasets (pick a recipe below) and start pretraining.

### Dataset recipes

- **Baseline sequential mix** — `python scripts/generate_dataset.py --config configs/data/pilot.yaml`
- **Curriculum (atomic + sequential)** — `python scripts/generate_dataset.py --config configs/data/pilot_curriculum.yaml`
- **OOD slice (large grids, constrained primitives)** — `python scripts/generate_dataset.py --config configs/data/pilot_ood.yaml`

Each config writes a manifest and `summary.json` to its `output_root`. The generator now accepts `task_schedule` at the config root (or under `generator`) and optional `generator.allowed_primitives` for structural constraints/outlier pockets.

Evaluate a manifest to sanity-check solve rates and program counts:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_arc.py --tasks data/pilot_curriculum/manifest.jsonl --output artifacts/eval/pilot_curriculum.json
```

### JEPA pretraining

Run full JEPA training against any manifest:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_jepa.py --config configs/training/jepa_pretrain.yaml --device cpu
```

Pass `--device cuda` on GPU boxes. Checkpoints and `metrics.json` land in `artifacts/jepa/pretrain/` (configurable via `training.checkpoint_dir`). Use `--dry-run` for a single dummy optimisation step.

For full A6000 runs, start from `configs/training/jepa_pretrain_gpu.yaml`; it sets a larger batch, longer schedule, and defaults to TensorBoard logging under `artifacts/jepa/pretrain_gpu/tensorboard/`.

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

> This project is under active development. Consult the blueprint documents for
> the full roadmap and open Beads issues for next steps.

## Dependencies

- Python deps are tracked via `requirements*.txt` files in the repo root. Use `uv venv --python 3.11 .venv` followed by `uv pip install --python .venv/bin/python -r requirements.txt`.
- Install optional extras as needed:
  - `requirements-dev.txt` for test tooling (`pytest`).
  - `requirements-rl.txt` for RLlib-based PPO/A2C training.
- See `docs/DEPENDENCIES.md` for the full process, package descriptions, and contribution guidelines.

## Decision Records

Architecture choices are tracked in `docs/adr/`.
- `0001-rl-engine.md` — RLlib selected for hierarchical option training.
- `0002-jepa-objective.md` — JEPA uses multi-step InfoNCE with memory queue and
  specified augmentation policy.
