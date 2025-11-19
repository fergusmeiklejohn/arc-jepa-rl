# Active Reasoner Implementation Guide

**Goal:** Transform arc-jepa-rl from a passive pattern predictor into an **active reasoner** that generates and tests hypotheses through reinforcement learning.

**Status:** Blueprint describes this vision (Section 4: Hierarchical RL), but RL training loop is not implemented.

**Effort:** 8-12 weeks for full implementation across 4 phases.

---

## Executive Summary

### What We're Building

An RL-powered hypothesis search system that:
1. **Generates** candidate program chains in latent space (fast)
2. **Scores** hypotheses using JEPA similarity to target (provable via SIGReg)
3. **Learns** which primitives to compose via policy gradients (active exploration)
4. **Selects** simplest programs that explain the data (Occam's razor)

### Current State vs Target

| Component | Current Status | Target State |
|-----------|---------------|--------------|
| **JEPA Encoder** | ✅ Learns grid representations | ✅ Same + predict counterfactual latents |
| **Hypothesis Generator** | ⚠️ Brute-force enumeration | ✅ RL policy learns to navigate space |
| **Evaluator** | ⚠️ Grid execution (slow) | ✅ Latent similarity (fast) |
| **Search Strategy** | ❌ Heuristic only | ✅ Learned via PPO/A2C |
| **Simplicity Prior** | ⚠️ MDL exists, not integrated | ✅ Built into RL reward |

### Key Insight

**Current approach:** Enumerate programs → Execute on grids → Check match (expensive)

**Active reasoner:** RL policy → Predict in latent space → Score similarity → Execute only top-K (efficient)

**Speedup:** ~100-1000× for hypothesis pruning

---

## Phase 1: Counterfactual Latent Prediction (2-3 weeks)

### Objective

Enable JEPA to predict: **"If I apply program P to state S, what latent will result?"**

This is the foundation for fast hypothesis evaluation without grid execution.

### Current Gap

```python
# Current: training/jepa/dataset.py
# JEPA trained on (grid_context, grid_target) pairs
# NO program information

class ManifestTokenizedPairDataset:
    def __getitem__(self, idx):
        return context_grid, target_grid  # Missing: what transformation?
```

### Target Architecture

```python
# Proposed: training/jepa/program_conditioned.py
class ProgramConditionedJEPA(nn.Module):
    """
    JEPA that predicts latent outcomes of program application.

    Input: (current_latent, program_embedding)
    Output: predicted_next_latent
    """
    def __init__(self, latent_dim=512, program_embed_dim=64):
        super().__init__()
        self.object_encoder = ObjectEncoder(...)  # Existing

        # NEW: Program encoder
        self.program_encoder = nn.Sequential(
            nn.Embedding(num_primitives, program_embed_dim),
            nn.TransformerEncoder(...)  # For program sequences
        )

        # NEW: Transition predictor
        self.transition_model = nn.Sequential(
            nn.Linear(latent_dim + program_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, current_latent, program):
        """
        Predict latent after applying program.

        Args:
            current_latent: (B, latent_dim) current state embedding
            program: (B, seq_len) primitive indices

        Returns:
            predicted_latent: (B, latent_dim) predicted next state
        """
        program_emb = self.program_encoder(program)  # (B, program_embed_dim)
        combined = torch.cat([current_latent, program_emb], dim=1)
        predicted_latent = self.transition_model(combined)
        return predicted_latent

    def predict_counterfactual(self, input_grid, candidate_programs):
        """
        Batch predict outcomes for multiple hypotheses.

        Args:
            input_grid: (H, W) starting grid
            candidate_programs: List[Program] hypotheses to test

        Returns:
            predicted_latents: (N, latent_dim) predicted outcomes
        """
        current_latent = self.object_encoder(input_grid)

        # Encode all programs
        program_tensors = [prog.to_tensor() for prog in candidate_programs]
        program_batch = torch.stack(program_tensors)

        # Batch prediction (NO GRID EXECUTION)
        predicted_latents = self.forward(
            current_latent.unsqueeze(0).expand(len(candidate_programs), -1),
            program_batch
        )

        return predicted_latents
```

### Training Data Preparation

**New dataset format:**
```python
# scripts/prepare_program_triples.py
def create_program_triple_dataset(manifest_path, output_path):
    """
    Convert (input, output) pairs to (input, program, output) triples.

    Reads manifest JSONL with program traces from synthetic generator.
    """
    triples = []

    for task in load_manifest(manifest_path):
        input_grid = task['train'][0]['input']
        output_grid = task['train'][0]['output']
        program = task['metadata']['program']  # From generator

        triples.append({
            'input': input_grid,
            'program': program,  # NEW: ground truth transformation
            'output': output_grid
        })

    save_jsonl(triples, output_path)
```

**Dataset loader:**
```python
# training/jepa/program_dataset.py
class ProgramTripleDataset(Dataset):
    """Dataset of (grid_before, program, grid_after) triples."""

    def __getitem__(self, idx):
        triple = self.triples[idx]

        # Tokenize grids
        input_tokens = tokenize_grid(triple['input'])
        output_tokens = tokenize_grid(triple['output'])

        # Encode program as sequence of primitive IDs
        program_tensor = encode_program(triple['program'])

        return input_tokens, program_tensor, output_tokens
```

### Training Loop

```python
# training/jepa/train_program_conditioned.py
def train_program_conditioned_jepa(config):
    model = ProgramConditionedJEPA(config)
    dataset = ProgramTripleDataset(config.data.manifest)

    for epoch in range(config.training.epochs):
        for batch in dataloader:
            input_tokens, programs, output_tokens = batch

            # Encode current and target states
            current_latent = model.object_encoder(input_tokens)
            target_latent = model.object_encoder(output_tokens)

            # Predict next latent given program
            predicted_latent = model(current_latent, programs)

            # Loss: InfoNCE + SIGReg
            info_nce_loss = contrastive_loss(predicted_latent, target_latent)
            sigreg_loss = sigreg(predicted_latent)

            total_loss = info_nce_loss + λ_sigreg * sigreg_loss

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### Validation

```python
# tests/test_program_conditioned_jepa.py
def test_counterfactual_accuracy():
    """Validate predicted latents match actual execution."""
    model = load_trained_model()

    for task in test_tasks:
        input_grid = task['input']
        program = task['program']
        target_grid = task['output']

        # Predict in latent space
        predicted_latent = model.predict_counterfactual(input_grid, [program])[0]

        # Execute on actual grid
        actual_output = program.execute(input_grid)
        actual_latent = model.object_encoder(actual_output)

        # Should be close
        similarity = cosine_similarity(predicted_latent, actual_latent)
        assert similarity > 0.9, f"Prediction accuracy too low: {similarity}"
```

### Deliverables

- [ ] `training/jepa/program_conditioned.py` - ProgramConditionedJEPA model
- [ ] `training/jepa/program_dataset.py` - Triple dataset loader
- [ ] `scripts/prepare_program_triples.py` - Data preparation script
- [ ] `scripts/train_program_conditioned_jepa.py` - Training CLI
- [ ] `tests/test_program_conditioned_jepa.py` - Validation tests
- [ ] Updated configs: `configs/training/jepa_program_conditioned.yaml`

### Success Metrics

- **Prediction accuracy:** Cosine similarity > 0.85 between predicted and actual latents
- **Speedup:** 100-1000× faster than grid execution for hypothesis pruning
- **Generalization:** Works on held-out program combinations

---

## Phase 2: Active Hypothesis Search with RL (2-3 weeks)

### Objective

Train a policy network that learns **which primitives to compose** to reach a target state, using JEPA similarity as reward signal.

### Current Gap

```python
# Current: training/solver/few_shot.py
# Brute-force enumeration, no learning

for program in enumerate_all_programs(max_nodes=3):
    output = program.execute(input_grid)  # Expensive!
    if matches(output, target_grid):
        return program
```

### Target Architecture

```python
# Proposed: training/reasoner/active_search.py
class ActiveReasonerPolicy(nn.Module):
    """
    RL policy for hypothesis navigation.

    State: (current_latent, partial_program_embedding, target_latent)
    Action: Which primitive to apply next (discrete)
    """
    def __init__(self, latent_dim=512, num_primitives=50, max_chain_length=4):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2 + 64, 512),  # current + target + program
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Actor: state -> action distribution
        self.actor = nn.Linear(256, num_primitives)

        # Critic: state -> value estimate
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        """
        Args:
            state: dict with keys:
                - 'current_latent': (B, latent_dim)
                - 'target_latent': (B, latent_dim)
                - 'partial_program': (B, program_embed_dim)

        Returns:
            action_logits: (B, num_primitives) action distribution
            value: (B, 1) state value estimate
        """
        # Concatenate state components
        state_vec = torch.cat([
            state['current_latent'],
            state['target_latent'],
            state['partial_program']
        ], dim=1)

        # Encode
        features = self.state_encoder(state_vec)

        # Actor-critic outputs
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value
```

### Training Environment

```python
# training/reasoner/hypothesis_env.py
class HypothesisSearchEnv(gym.Env):
    """
    RL environment for hypothesis navigation.

    Observation: (current_latent, target_latent, partial_program)
    Action: Primitive index to append
    Reward: JEPA similarity improvement + simplicity penalty
    """
    def __init__(self, jepa_model, primitive_registry, max_steps=4):
        self.jepa = jepa_model
        self.primitives = primitive_registry.primitives
        self.max_steps = max_steps

        self.action_space = gym.spaces.Discrete(len(self.primitives))
        self.observation_space = gym.spaces.Dict({
            'current_latent': gym.spaces.Box(-np.inf, np.inf, shape=(512,)),
            'target_latent': gym.spaces.Box(-np.inf, np.inf, shape=(512,)),
            'partial_program': gym.spaces.Box(-np.inf, np.inf, shape=(64,))
        })

    def reset(self, input_grid, target_grid):
        """Start new episode with a task."""
        self.input_grid = input_grid
        self.target_grid = target_grid

        self.current_latent = self.jepa.encode(input_grid)
        self.target_latent = self.jepa.encode(target_grid)

        self.partial_program = []
        self.step_count = 0

        return self._get_obs()

    def step(self, action):
        """Apply primitive, compute reward."""
        primitive = self.primitives[action]

        # Predict next latent (FAST - no grid execution)
        prev_latent = self.current_latent
        next_latent = self.jepa.predict_after_program(
            prev_latent,
            [primitive]
        )

        # Reward = similarity improvement + simplicity penalty
        prev_similarity = F.cosine_similarity(
            prev_latent, self.target_latent, dim=0
        )
        new_similarity = F.cosine_similarity(
            next_latent, self.target_latent, dim=0
        )

        similarity_gain = new_similarity - prev_similarity
        simplicity_penalty = -0.01 * len(self.partial_program)

        reward = similarity_gain + simplicity_penalty

        # Update state
        self.current_latent = next_latent
        self.partial_program.append(primitive)
        self.step_count += 1

        # Done if close enough or max steps
        done = (new_similarity > 0.95) or (self.step_count >= self.max_steps)

        # Bonus for success
        if done and new_similarity > 0.95:
            reward += 1.0

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        program_emb = self._embed_program(self.partial_program)
        return {
            'current_latent': self.current_latent.cpu().numpy(),
            'target_latent': self.target_latent.cpu().numpy(),
            'partial_program': program_emb.cpu().numpy()
        }
```

### RLlib Integration

```python
# training/reasoner/train_active_search.py
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune

def train_active_reasoner(config):
    """Train RL policy for hypothesis navigation."""

    # Configure PPO
    ppo_config = (
        PPOConfig()
        .environment(
            env=HypothesisSearchEnv,
            env_config={
                'jepa_model_path': config.jepa.checkpoint,
                'primitive_registry': config.dsl.primitives,
                'max_steps': config.search.max_chain_length
            }
        )
        .framework('torch')
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,  # Encourage exploration
            train_batch_size=4096
        )
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=200
        )
    )

    # Run training
    tuner = tune.Tuner(
        'PPO',
        param_space=ppo_config.to_dict(),
        run_config=tune.RunConfig(
            stop={'training_iteration': 100},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10
            )
        )
    )

    results = tuner.fit()

    # Return best checkpoint
    best_result = results.get_best_result(metric='episode_reward_mean', mode='max')
    return best_result.checkpoint
```

### Behavioral Cloning Pretraining

**Bootstrap with heuristic solver:**

```python
# training/reasoner/pretrain_bc.py
def pretrain_policy_with_bc(policy, heuristic_traces):
    """
    Pretrain policy via behavioral cloning.

    Args:
        policy: ActiveReasonerPolicy
        heuristic_traces: List of (state, action) pairs from DSL enumerator
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for epoch in range(config.bc.epochs):
        for state, action in heuristic_traces:
            # Policy output
            action_logits, _ = policy(state)

            # Cross-entropy loss
            loss = F.cross_entropy(action_logits, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Deliverables

- [ ] `training/reasoner/active_search.py` - ActiveReasonerPolicy model
- [ ] `training/reasoner/hypothesis_env.py` - RL environment
- [ ] `training/reasoner/train_active_search.py` - PPO training script
- [ ] `training/reasoner/pretrain_bc.py` - Behavioral cloning bootstrap
- [ ] `scripts/collect_heuristic_traces.py` - Generate BC data from DSL solver
- [ ] Updated configs: `configs/training/rl/active_reasoner.yaml`
- [ ] Tests: `tests/test_hypothesis_env.py`

### Success Metrics

- **Policy performance:** Achieves >0.9 similarity on 80%+ of training tasks
- **Search efficiency:** Finds solutions in <10 episodes vs >1000 enumerations
- **Generalization:** Works on held-out task families
- **Chain parsimony:** Learns shorter programs than brute-force baseline

---

## Phase 3: LeJEPA Integration (1-2 weeks)

### Objective

Add SIGReg regularization to ensure JEPA embeddings have provable quality, making similarity scoring reliable for RL rewards.

### Motivation

**Current problem:** No guarantee that JEPA similarity is a good reward signal.

**LeJEPA solution:** Embeddings following isotropic Gaussian distribution provably minimize downstream prediction error.

**Impact on RL:** More stable training, better policy convergence.

### Implementation

```python
# training/jepa/sigreg.py (from LEJEPA_ALIGNMENT_REVIEW.md Appendix)
class SIGRegLoss:
    """Sketched Isotropic Gaussian Regularization."""
    def __init__(self, num_slices=1024, num_points=17):
        self.num_slices = num_slices
        self.num_points = num_points

    def forward(self, embeddings):
        """
        Compute penalty for deviation from isotropic Gaussian.

        Returns:
            Scalar penalty (lower = closer to optimal distribution)
        """
        # Implementation details in LEJEPA_ALIGNMENT_REVIEW.md
        # ...
        return penalty
```

**Integration into training:**

```python
# Modify training/jepa/train_program_conditioned.py
def train_step(batch):
    # ... existing forward pass ...

    # InfoNCE loss (existing)
    info_nce = contrastive_loss(predicted_latent, target_latent)

    # SIGReg loss (NEW)
    sigreg = sigreg_loss(predicted_latent)

    # Combined
    total_loss = info_nce + config.sigreg.weight * sigreg

    return total_loss
```

### Embedding Quality Diagnostics

```python
# training/jepa/diagnostics.py (from review)
def compute_embedding_quality(embeddings, vq_codes=None):
    """Track representation health."""
    return {
        'variance/mean': ...,
        'variance/std': ...,
        'isotropy': ...,  # Should be high
        'rank/effective': ...,  # Should approach embedding_dim
        'gaussian_ness': ...,  # Should be ~1.0
        'codebook/usage': ...  # If using VQ
    }
```

**Log during training:**

```python
if step % 100 == 0:
    metrics = compute_embedding_quality(predicted_latents)
    tensorboard.log_metrics(metrics, step)
```

### Correlation Study

**Validate that JEPA loss predicts solver performance:**

```python
# scripts/validate_jepa_correlation.py
def jepa_correlation_study(jepa_checkpoints, arc_tasks):
    """
    Test LeJEPA's claim: "Training loss correlates with downstream performance."
    """
    results = []

    for checkpoint in jepa_checkpoints:
        model = load_checkpoint(checkpoint)

        # JEPA loss on validation set
        jepa_loss = evaluate_jepa_loss(model, val_manifest)

        # Downstream solver success rate
        solve_rate = evaluate_solver(model, arc_tasks)

        results.append({
            'checkpoint': checkpoint,
            'jepa_loss': jepa_loss,
            'solve_rate': solve_rate
        })

    # Compute correlation
    losses = [r['jepa_loss'] for r in results]
    rates = [r['solve_rate'] for r in results]
    correlation = pearsonr(losses, rates)

    print(f"JEPA loss ↔ solve rate correlation: {correlation:.3f}")

    # Should be strong negative (lower loss = higher solve rate)
    assert correlation < -0.7, "Weak correlation - JEPA not predictive"
```

### Deliverables

- [ ] `training/jepa/sigreg.py` - SIGReg loss implementation
- [ ] `training/jepa/diagnostics.py` - Embedding quality metrics
- [ ] `scripts/validate_jepa_correlation.py` - Correlation study
- [ ] Updated training loop with SIGReg
- [ ] TensorBoard logging for embedding metrics
- [ ] Documentation: How to tune λ_sigreg

### Success Metrics

- **Embedding isotropy:** >0.8 (high = well-distributed)
- **Effective rank ratio:** >0.7 (high = using full dimensionality)
- **Gaussian-ness:** >0.9 (close to 1.0 = ideal distribution)
- **Correlation:** JEPA loss ↔ solve rate < -0.7 (strong negative)
- **RL stability:** Policy training converges faster with SIGReg

---

## Phase 4: Curriculum & Scale (3-4 weeks)

### Objective

Implement operational 5-phase curriculum to grow from simple to complex reasoning, tracking metrics to validate progress.

### Current Gap

Blueprint Section 8 describes curriculum, but it's not implemented:
- No progression logic
- No metrics tracking (codebook usage, option diversity, solve rates)
- No automatic phase transitions

### Curriculum Design

**Phase 1: Atomic Primitives (1-2 primitives, no composition)**
- Tasks: Single transformations (mirror, rotate, translate)
- Goal: Learn basic primitive → latent mappings
- Metrics:
  - JEPA prediction accuracy >0.9
  - Policy success rate >80%
  - Average chain length ≈1.0

**Phase 2: Sequential Composition (2-3 primitives)**
- Tasks: Chain 2-3 primitives
- Goal: Learn composition rules
- Metrics:
  - Successful 2-step chains >70%
  - Codebook usage >60% of codes active

**Phase 3: Relational Reasoning (3-4 primitives with spatial dependencies)**
- Tasks: Require adjacency/connectivity awareness
- Goal: Use relational attention effectively
- Metrics:
  - Relational tasks >60% solve rate
  - Embedding isotropy maintained >0.75

**Phase 4: Conditional & Meta-reasoning (4+ primitives, conditionals)**
- Tasks: If-then rules, count-based transformations
- Goal: Meta-JEPA provides useful priors
- Metrics:
  - Meta-guided solve rate >50%
  - Novel primitive discovery rate >0

**Phase 5: OOD Generalization (unseen combinations)**
- Tasks: Held-out rule families, larger grids, new colors
- Goal: Validate transfer learning
- Metrics:
  - OOD solve rate >30%
  - Chain length stays bounded

### Implementation

```python
# training/curriculum/manager.py
class CurriculumManager:
    """Manages progressive task difficulty."""

    def __init__(self, config):
        self.phases = config.curriculum.phases
        self.current_phase = 0
        self.metrics_history = []

    def should_advance_phase(self, current_metrics):
        """
        Check if ready to move to next phase.

        Criteria:
        - Solve rate above threshold for N consecutive epochs
        - Embedding quality metrics healthy
        - Policy not overfitting (train/val gap <10%)
        """
        phase_config = self.phases[self.current_phase]

        # Check thresholds
        if current_metrics['solve_rate'] < phase_config.min_solve_rate:
            return False

        if current_metrics['isotropy'] < 0.7:
            return False  # Embeddings degrading

        if current_metrics['train_val_gap'] > 0.1:
            return False  # Overfitting

        # Check consistency (N consecutive epochs above threshold)
        recent = self.metrics_history[-phase_config.consistency_window:]
        all_above = all(m['solve_rate'] > phase_config.min_solve_rate
                       for m in recent)

        return all_above

    def advance_phase(self):
        """Move to next curriculum phase."""
        self.current_phase += 1
        logger.info(f"Advancing to Phase {self.current_phase}")

        # Update task distribution
        self.update_task_sampler()

        # Optionally adjust hyperparameters
        self.adjust_training_params()

    def get_current_tasks(self):
        """Sample tasks appropriate for current phase."""
        phase = self.phases[self.current_phase]
        return sample_tasks(
            max_primitives=phase.max_primitives,
            max_chain_length=phase.max_chain_length,
            allow_conditionals=phase.allow_conditionals,
            grid_size_range=phase.grid_size_range
        )
```

### Metrics Tracking

```python
# training/curriculum/metrics.py
class CurriculumMetrics:
    """Track comprehensive metrics across curriculum phases."""

    def compute(self, model, tasks, phase):
        """Compute all relevant metrics for current phase."""

        # Solver metrics
        solve_rate = evaluate_solve_rate(model, tasks)
        avg_chain_length = compute_avg_chain_length(model, tasks)

        # JEPA quality
        embedding_metrics = compute_embedding_quality(model)
        jepa_loss = compute_jepa_loss(model, tasks)

        # RL policy metrics
        policy_entropy = compute_policy_entropy(model)  # High = exploring
        value_accuracy = compute_value_error(model, tasks)

        # Primitive usage
        primitive_counts = track_primitive_usage(model, tasks)
        primitive_diversity = compute_diversity(primitive_counts)

        # Codebook (if VQ)
        if model.use_vq:
            codebook_usage = compute_active_codes(model) / model.num_codes

        # Option discovery
        novel_options = count_discovered_options(model, phase)

        return {
            # Core metrics
            'solve_rate': solve_rate,
            'avg_chain_length': avg_chain_length,

            # JEPA health
            'jepa_loss': jepa_loss,
            'isotropy': embedding_metrics['isotropy'],
            'effective_rank': embedding_metrics['rank/effective'],
            'gaussian_ness': embedding_metrics['gaussian_ness'],

            # RL health
            'policy_entropy': policy_entropy,
            'value_error': value_accuracy,

            # Diversity
            'primitive_diversity': primitive_diversity,
            'codebook_usage': codebook_usage,
            'novel_options': novel_options,

            # Phase info
            'phase': phase,
            'timestamp': datetime.now()
        }
```

### Training Loop with Curriculum

```python
# scripts/train_with_curriculum.py
def train_curriculum(config):
    """Full curriculum training pipeline."""

    # Initialize
    jepa_model = load_jepa_model(config.jepa.checkpoint)
    policy = ActiveReasonerPolicy(config)
    curriculum = CurriculumManager(config.curriculum)

    for epoch in range(config.training.max_epochs):
        # Get tasks for current phase
        tasks = curriculum.get_current_tasks()

        # Train RL policy on tasks
        policy_metrics = train_policy_epoch(policy, jepa_model, tasks)

        # Evaluate
        eval_metrics = curriculum.metrics.compute(
            model={'jepa': jepa_model, 'policy': policy},
            tasks=tasks,
            phase=curriculum.current_phase
        )

        # Log
        log_metrics(eval_metrics, epoch)
        curriculum.metrics_history.append(eval_metrics)

        # Check for phase advancement
        if curriculum.should_advance_phase(eval_metrics):
            curriculum.advance_phase()

            # Save checkpoint at phase transition
            save_checkpoint({
                'jepa': jepa_model.state_dict(),
                'policy': policy.state_dict(),
                'phase': curriculum.current_phase,
                'metrics': eval_metrics
            }, f'checkpoints/phase_{curriculum.current_phase}.pt')
```

### Deliverables

- [ ] `training/curriculum/manager.py` - Curriculum progression logic
- [ ] `training/curriculum/metrics.py` - Comprehensive metric tracking
- [ ] `scripts/train_with_curriculum.py` - Full curriculum training
- [ ] Curriculum configs: `configs/curriculum/5_phase.yaml`
- [ ] Visualization: `scripts/plot_curriculum_progress.py`
- [ ] Documentation: Curriculum design rationale

### Success Metrics

**Phase 1 completion:**
- Solve rate >80% on atomic tasks
- JEPA prediction accuracy >0.9

**Phase 2 completion:**
- Solve rate >70% on 2-3 step compositions
- Codebook usage >60%

**Phase 3 completion:**
- Relational task solve rate >60%
- Isotropy maintained >0.75

**Phase 4 completion:**
- Meta-guided solve rate >50%
- Novel primitive discovery >0

**Phase 5 (OOD):**
- OOD solve rate >30%
- Demonstrates transfer learning

---

## Integration Points with Existing Code

### What We Keep (Strong Foundations)

✅ **Object tokenizer** - `training/modules/object_tokenizer.py`
- Already extracts features, adjacency
- Used as input to JEPA encoder

✅ **VQ-VAE bottleneck** - `training/modules/vq.py`
- Discrete codes useful for symbolic grounding
- Keep alongside continuous embeddings for RL

✅ **Relational attention** - `training/modules/relational.py`
- Graph-aware reasoning over objects
- Critical for spatial dependencies

✅ **DSL primitives** - `training/dsl/primitives.py`
- Expand from 13 to 50+ (separate task)
- Registry provides action space for RL

✅ **Meta-JEPA** - `training/meta_jepa/`
- Provides priors for few-shot tasks
- Integrate into Phase 4 curriculum

✅ **Evaluation harness** - `training/eval/suite.py`
- Use for curriculum metrics
- Track solve rates across phases

### What We Modify

⚠️ **JEPA training** - `training/jepa/loop.py`
- Add program conditioning
- Add SIGReg loss
- Add embedding diagnostics

⚠️ **Dataset** - `training/jepa/dataset.py`
- Change from pairs to (input, program, output) triples
- Add program encoding

⚠️ **Solver** - `training/solver/few_shot.py`
- Replace brute-force with RL policy
- Use latent-space search before grid execution

### What We Add (New Components)

➕ **Program-conditioned JEPA** - `training/jepa/program_conditioned.py`
➕ **Active reasoner policy** - `training/reasoner/active_search.py`
➕ **Hypothesis search env** - `training/reasoner/hypothesis_env.py`
➕ **SIGReg loss** - `training/jepa/sigreg.py`
➕ **Curriculum manager** - `training/curriculum/manager.py`
➕ **Comprehensive metrics** - `training/curriculum/metrics.py`

---

## Dependencies & Prerequisites

### Required Beads Issues to Close First

**Critical blockers:**
- `arc-jepa-rl-au0` ✅ (closed): RLlib integration scaffold
- `arc-jepa-rl-qjp` ✅ (closed): Actor-critic policy networks
- `arc-jepa-rl-tkr` ✅ (closed): PPO/A2C training scripts
- `arc-jepa-rl-btf` ✅ (closed): Behavioral cloning pretraining

**Helpful but not blocking:**
- `arc-jepa-rl-9df` ✅ (closed): Expand DSL primitives to 50+
- `arc-jepa-rl-4dp` ✅ (closed): Structure-aware program encoder
- `arc-jepa-rl-156` ✅ (closed): Multi-step context (k=3)

### External Dependencies

**Python packages:**
```txt
# Already in requirements.txt:
torch>=2.0
ray[rllib]>=2.7
gymnasium

# May need to add:
scipy  # For statistical tests in SIGReg
```

**Compute resources:**
- Phase 1-2: Single GPU (A6000 sufficient)
- Phase 3-4: Multi-GPU helpful (DDP)
- Curriculum training: ~48 GPU-hours estimated

---

## Testing Strategy

### Unit Tests

```python
# tests/test_program_conditioned_jepa.py
def test_counterfactual_prediction():
    """JEPA predicts program effects accurately."""
    assert prediction_accuracy > 0.85

def test_sigreg_convergence():
    """SIGReg drives embeddings toward isotropic Gaussian."""
    assert isotropy > 0.8
    assert gaussian_ness > 0.9

# tests/test_hypothesis_env.py
def test_env_reward_shaping():
    """Reward increases when approaching target."""
    assert reward_correlation > 0.9

def test_env_episode_termination():
    """Episodes terminate correctly."""
    assert all_episodes_terminate

# tests/test_active_search.py
def test_policy_learns_simple_tasks():
    """Policy solves atomic tasks after BC pretraining."""
    assert solve_rate_phase1 > 0.8

def test_policy_exploration():
    """Policy entropy decreases as it learns."""
    assert entropy_final < entropy_initial
```

### Integration Tests

```python
# tests/integration/test_end_to_end_active_reasoner.py
def test_full_pipeline():
    """
    End-to-end test: JEPA → RL → Solve
    """
    # 1. Train program-conditioned JEPA on tiny dataset
    jepa = train_jepa_quick(num_tasks=100, epochs=5)

    # 2. Pretrain policy with BC
    policy = pretrain_bc(jepa, heuristic_traces)

    # 3. Train RL on simple tasks
    policy = train_rl(policy, jepa, phase1_tasks, iterations=10)

    # 4. Evaluate on held-out tasks
    solve_rate = evaluate(policy, jepa, test_tasks)

    assert solve_rate > 0.5, "End-to-end pipeline fails"
```

### Ablation Studies

**Critical experiments to validate approach:**

1. **Counterfactual vs Grid Execution**
   - Compare: RL with latent prediction vs RL with grid execution
   - Metric: Wall-clock time to solution
   - Hypothesis: Latent is 100-1000× faster

2. **SIGReg Impact**
   - Compare: JEPA with vs without SIGReg
   - Metric: RL policy convergence rate, solve rate
   - Hypothesis: SIGReg improves stability

3. **BC Pretraining**
   - Compare: Random init vs BC pretrained policy
   - Metric: Episodes to 80% solve rate
   - Hypothesis: BC reduces sample complexity 10×

4. **Curriculum vs Random**
   - Compare: Curriculum progression vs uniform task sampling
   - Metric: Final OOD solve rate
   - Hypothesis: Curriculum achieves >2× OOD performance

---

## Risk Mitigation

### Risk 1: JEPA predictions inaccurate in latent space

**Symptom:** Predicted latents don't match actual outcomes.

**Mitigation:**
- Start with simple primitives (mirror, rotate) where prediction is easier
- Validate prediction accuracy >0.85 before moving to RL
- Add auxiliary loss: predict actual grid changes as regularizer
- Fallback: Hybrid approach (predict top-K in latent, execute on grid)

### Risk 2: RL policy doesn't converge

**Symptom:** Reward doesn't improve over training.

**Mitigation:**
- BC pretraining provides warm start
- Curriculum ensures early tasks are solvable (shaped reward signal)
- SIGReg ensures reward signal is smooth
- Intrinsic curiosity bonus if stuck (explore novel states)

### Risk 3: Sample efficiency too low

**Symptom:** Needs millions of episodes to learn simple tasks.

**Mitigation:**
- BC pretraining from heuristic solver (skip random exploration)
- JEPA counterfactual search is cheap (no grid execution overhead)
- Small action space (13-50 primitives, not pixel-level)
- Replay buffer with prioritization (focus on informative episodes)

### Risk 4: Doesn't generalize to OOD tasks

**Symptom:** Phase 5 solve rate <10%.

**Mitigation:**
- Meta-JEPA provides task family priors
- Curriculum explicitly trains for composition
- Regularization (entropy bonus, simplicity prior) prevents overfitting
- Data augmentation (color permutations, grid transformations)

---

## Timeline & Milestones

### Week 1-3: Phase 1 (Counterfactual Prediction)
- [ ] Week 1: Implement ProgramConditionedJEPA
- [ ] Week 2: Prepare triple dataset, train model
- [ ] Week 3: Validate prediction accuracy, integrate SIGReg

**Checkpoint:** Prediction accuracy >0.85 on validation set

### Week 4-6: Phase 2 (Active Search RL)
- [ ] Week 4: Implement HypothesisSearchEnv, policy network
- [ ] Week 5: BC pretraining, basic RL training
- [ ] Week 6: Tune hyperparameters, validate on simple tasks

**Checkpoint:** Policy solves 80%+ of atomic tasks

### Week 7-8: Phase 3 (LeJEPA Integration)
- [ ] Week 7: Implement SIGReg, embedding diagnostics
- [ ] Week 8: Correlation study, tune λ_sigreg

**Checkpoint:** Embedding metrics healthy, correlation >0.7

### Week 9-12: Phase 4 (Curriculum)
- [ ] Week 9: Implement curriculum manager, metrics tracking
- [ ] Week 10-11: Train through Phase 1-3 of curriculum
- [ ] Week 12: Phase 4-5, ablation studies, documentation

**Checkpoint:** OOD solve rate >30%

---

## Success Criteria

### Phase 1: Counterfactual Prediction
- ✅ Prediction accuracy: Cosine similarity >0.85
- ✅ Speedup: 100-1000× vs grid execution
- ✅ Generalization: Works on held-out program combos

### Phase 2: Active Search
- ✅ Policy performance: >80% solve rate on Phase 1 tasks
- ✅ Search efficiency: <10 episodes vs >1000 enumerations
- ✅ Chain parsimony: Shorter programs than brute-force

### Phase 3: LeJEPA Integration
- ✅ Embedding isotropy: >0.8
- ✅ Correlation: JEPA loss ↔ solve rate < -0.7
- ✅ RL stability: Faster convergence with SIGReg

### Phase 4: Curriculum
- ✅ Phase 1-4 completion criteria met
- ✅ OOD solve rate: >30%
- ✅ Novel primitive discovery: >0

### Overall System
- ✅ **Active reasoning demonstrated:** Policy learns to navigate hypothesis space
- ✅ **Efficiency gain:** 100× speedup over brute-force
- ✅ **Generalization:** Solves tasks outside training distribution
- ✅ **Parsimony:** Prefers simple explanations (Occam's razor)

---

## Next Steps for Team

### Immediate Actions

1. **Review this guide** with team leads (1-2 hours)
   - Clarify any questions
   - Assign phase ownership

2. **Break down into Beads tickets** (half-day)
   - Each phase becomes an epic
   - Deliverables become individual tasks
   - Link dependencies

3. **Read LeJEPA resources** (2-4 hours for implementers)
   - arXiv paper Sections 1-2.1 (theory)
   - rbalestr-lab/lejepa repo (reference implementation)
   - `LEJEPA_ALIGNMENT_REVIEW.md` Appendix (code examples)

4. **Validate Phase 1 feasibility** (1 week spike)
   - Implement minimal ProgramConditionedJEPA
   - Test on 10 simple tasks
   - Measure prediction accuracy
   - Go/no-go decision

### Resources

- **Blueprint:** `Project_Blueprint.md` Section 4 (HRL vision)
- **LeJEPA Review:** `LEJEPA_ALIGNMENT_REVIEW.md` (theory + code)
- **Existing Code:** Review beads issues that are closed (RLlib scaffold done)
- **Papers:** arXiv 2511.08544 (LeJEPA)

### Communication Channels

- **Weekly sync:** Review curriculum metrics, adjust priorities
- **Phase transitions:** Demo + retrospective when advancing phases
- **Blockers:** Flag early if prediction accuracy or RL convergence fails

---

## Appendix: Code Snippets

### A. Minimal Working Example

```python
# Minimal end-to-end example (simplified)

# 1. Train program-conditioned JEPA
jepa = ProgramConditionedJEPA()
for input_grid, program, output_grid in dataset:
    predicted_latent = jepa(input_grid, program)
    target_latent = jepa.encode(output_grid)
    loss = F.mse_loss(predicted_latent, target_latent)
    loss.backward()

# 2. Create RL environment
env = HypothesisSearchEnv(jepa)
state = env.reset(input_grid, target_grid)

# 3. Train policy
policy = ActiveReasonerPolicy()
for episode in range(1000):
    action = policy(state)
    next_state, reward, done = env.step(action)
    policy.update(state, action, reward)
    if done: break

# 4. Solve new task
solution_program = policy.search(new_input, new_target)
output = solution_program.execute(new_input)
assert output == new_target
```

### B. Config Example

```yaml
# configs/training/active_reasoner.yaml

jepa:
  checkpoint: artifacts/jepa/program_conditioned.pt
  latent_dim: 512
  sigreg:
    enabled: true
    weight: 0.1
    num_slices: 1024

rl:
  algorithm: ppo
  learning_rate: 3e-4
  gamma: 0.99
  entropy_coeff: 0.01
  max_episodes: 10000

  env:
    max_steps: 4
    similarity_threshold: 0.95
    simplicity_penalty: 0.01

curriculum:
  phases:
    - name: atomic
      max_primitives: 1
      max_chain_length: 1
      min_solve_rate: 0.8

    - name: sequential
      max_primitives: 3
      max_chain_length: 3
      min_solve_rate: 0.7

    # ... phases 3-5
```

---

**End of Implementation Guide**
