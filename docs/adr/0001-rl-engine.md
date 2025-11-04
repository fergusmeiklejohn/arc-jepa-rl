# ADR 0001: Adopt RLlib for Hierarchical Option Training

## Status
Accepted

## Context
We need an RL framework that can:

- support hierarchical policies where a high-level controller selects parameterised options/primitives;
- integrate custom latent-space rewards derived from the JEPA encoder;
- scale from local prototyping to distributed training as experiments grow;
- provide solid logging, checkpointing, and evaluation tooling out of the box;
- interoperate cleanly with Gymnasium-style environments and PyTorch models.

Candidate stacks considered: **RLlib**, **PufferLib**, and a custom lightweight PPO/A2C implementation.

## Decision
We will standardise on **RLlib** (via Ray) for option learning, imitation fine-tuning, and RL fine-tuning.

Key commitments:

- Use Gymnasium-compatible environments so RLlib can manage vectorised rollouts.
- Model each primitive option as a parameterised action selectable by the high-level policy using RLlib's hierarchical policy APIs (options or custom action distributions).
- Integrate latent rewards by extending RLlib callbacks to fetch JEPA embeddings and compute novelty bonuses.
- Leverage Ray for multi-process sampling once workloads exceed a single machine.

## Consequences

### Positive

- Rich hierarchical tooling and multi-agent abstractions reduce bespoke code for options.
- Built-in experiment management simplifies checkpointing and evaluation.
- Easy path to distributed rollouts and GPU acceleration.

### Negative / Mitigations

- Ray introduces runtime overhead and operational complexity. We will begin with local Ray runtimes and document minimal configurations.
- RLlib's API surface is large; contributors must follow internal usage guides. We'll add wrappers/adapters in `training/rllib_utils/` to hide boilerplate.
- Dependency footprint increases. Requirements and CI images must include compatible Ray/RLlib versions.

## Alternatives Considered

- **PufferLib**: lighter-weight but lacks first-class hierarchical/option abstractions; would require custom rollout glue code.
- **Custom PPO/A2C**: maximum control but significant engineering cost for vectorised envs, logging, and distributed scale.

## Follow-up Tasks

1. Add RLlib + Ray to the base requirements file with pinned versions.
2. Implement an environment adapter exposing ARC tasks with option metadata.
3. Build helper utilities for latent reward computation inside RLlib callbacks.
