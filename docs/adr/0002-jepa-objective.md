# ADR 0002: Configure JEPA Objective with Multi-Step InfoNCE

## Status
Accepted

## Context

Our JEPA module must learn transformation-aware embeddings that transfer to
hierarchical option policies and meta-reasoning. We need to decide on the
predictive loss, context-target structure, augmentation policy, projection
dimensions, temperature handling, and negative sampling strategy.

## Decision

We will train the JEPA encoder using a **multi-step InfoNCE objective** with the
following configuration:

1. **Context–target tuples**: context length `k=3` (two preceding grid states +
   the current one) predicting the latent representation of the output grid one
   step ahead. This captures short transformation chains without excessive
   memory overhead.
2. **Loss**: InfoNCE with a shared encoder for context and target projections;
   target branch uses stop-gradients for stability.
3. **Projection head**: encoder outputs 512-d embeddings, projected to 256-d via
   a two-layer MLP with LayerNorm.
4. **Temperature**: learnable scalar initialised at `τ=0.07`, clamped to
   `[0.03, 0.3]`.
5. **Negative sampling**: in-batch negatives augmented with a FIFO queue (memory
   bank) of size 4096 to widen the contrastive set.
6. **Augmentations**: random 20% masking, ±1 cell random crops (with padding),
   palette permutation, and Gaussian noise (σ=0.05) applied to context grids.

## Consequences

### Positive
- InfoNCE strengthens separation between distinct transformation families,
  improving latent reward shaping and meta-clustering.
- Multi-step context aligns with sequential option execution.
- Learnable temperature allows automatic calibration as training progresses.
- Memory queue increases negative diversity without requiring huge batches.

### Negative / Mitigations
- InfoNCE adds computational overhead; we will prototype with queue size 1024 in
  smoke tests before scaling to 4096.
- Multi-step tuples need sequential data; generators must emit intermediate
  states. The synthetic task generator roadmap already includes rule traces we
  can leverage.
- Augmentations risk over-randomising small grids. We'll gate aggressive crops
  and masking for grids smaller than 8×8 in the implementation.

## Alternatives Considered

- Cosine regression loss: simpler but weaker discrimination; rejected due to
  limited separation between rule families.
- Pure in-batch negatives: easier but insufficient diversity for 256-d
  projections.
- BYOL-style predictor: interesting but riskier collapse without strong
  augmentation; deferred until baseline established.

## Follow-up Tasks

1. Update training configs to reflect the InfoNCE setup (context window,
   augmentations, queue size).
2. Modify dataset pipelines to output multi-step sequences + intermediate grids.
3. Implement the encoder, projection head, and memory queue in the JEPA module.
