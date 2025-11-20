Active Reasoner SIGReg notes
----------------------------

- HypothesisSearchEnv now supports SIGReg monitoring via `sigreg_monitor` (enabled/num_slices/num_points). It computes a SIGReg penalty on current+target latents each step and surfaces an average per episode in `info["sigreg_penalty"]`, logged by `train_active_reasoner.py` as `sigreg_penalty_mean`.
- Example config: `configs/training/active_reasoner_sigreg.yaml` (uses program-conditioned JEPA config with SIGReg weight already >0. Operators can swap in a larger JEPA checkpoint and dataset for real runs).
- Example runs to compare FP32/SIGReg monitoring (needs CUDA box):
  - Baseline (no monitor): `python scripts/train_active_reasoner.py --config configs/training/active_reasoner_sigreg.yaml --output-dir artifacts/active_reasoner/baseline`
  - With SIGReg monitor on (default in config): same as above; check `metrics.jsonl` for `sigreg_penalty_mean`.
- Reward shaping unchanged; SIGReg is used purely for monitoring embedding quality so the latent-distance reward remains stable.
- This machine lacks GPU; RL convergence comparison not run here. Run short iterations (e.g., `iterations=10`, `episodes_per_iter=16`) on a CUDA node to compare reward/success curves and SIGReg penalties with/without monitor.
