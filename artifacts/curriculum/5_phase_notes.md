5-phase curriculum wiring
-------------------------

- New utilities under `training/curriculum/`:
  - `CurriculumManager` manages ordered phases, thresholds, and advancement.
  - `PhaseMetrics`/`compute_phase_metrics` aggregate solve rate, codebook usage, and option diversity.
  - `run_curriculum` helper calls a user-supplied per-phase training/eval function and advances when thresholds are met.
- Example config: `configs/curriculum/5_phase.yaml` covers atomic → sequential → relational → conditional/meta → OOD, with per-phase task schedules and thresholds.
- Usage sketch:
  ```python
  from training.curriculum import CurriculumConfig, CurriculumManager, run_curriculum
  import yaml

  cfg = CurriculumConfig.from_mapping(yaml.safe_load(open("configs/curriculum/5_phase.yaml")))
  manager = CurriculumManager(cfg)

  def train_phase(phase):
      # wire phase.task_schedule into env_config["task_schedule"]
      # run RL/solver training, then aggregate metrics
      return compute_phase_metrics(
          episodes=episodes,
          successes=successes,
          codebook_usage=codebook_usage,
          option_counts=option_counts,
      )

  history = run_curriculum(manager, train_phase)
  ```
- The manager only advances when `solve_rate` and, when provided, `codebook_usage`/`option_diversity` exceed thresholds and `min_episodes` is met. History is stored in `manager.metrics_history` for later analysis/visualization.
