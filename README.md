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
  `training/modules/`, with object-centric JEPA helpers under `training/jepa/`.
- `envs/` — Environment wrappers exposing ARC-style tasks to RL agents.
- `configs/` — YAML configuration files for data generation, training, and
  evaluation runs.
- `scripts/` — Command-line entry points for dataset generation and training.
  - `train_jepa.py` — CLI stub that exercises the object-centric JEPA encoder (supports `--dry-run`).
- `tests/` — Unit and integration tests covering generators, models, and envs.

## Getting Started

1. Create a Python environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install core dependencies (placeholder): `pip install -r requirements.txt`.
3. Generate a sanity dataset once the generator is implemented: `python scripts/generate_dataset.py --config configs/data/pilot.yaml`.

> This project is under active development. Consult the blueprint documents for
> the full roadmap and open Beads issues for next steps.

## Decision Records

Architecture choices are tracked in `docs/adr/`.
- `0001-rl-engine.md` — RLlib selected for hierarchical option training.
- `0002-jepa-objective.md` — JEPA uses multi-step InfoNCE with memory queue and
  specified augmentation policy.
