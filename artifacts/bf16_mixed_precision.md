BF16 mixed-precision support summary (JEPA, Meta-JEPA, guidance)
================================================================

- Added `mixed_precision` config knob (none|fp16|bf16) across JEPA, Meta-JEPA, and guidance training loops. Defaults remain FP32/disabled for CPU configs; legacy `training.amp` in JEPA maps to `fp16`.
- Each loop now gates autocast + (when fp16) GradScaler on CUDA availability and BF16 support, warning and falling back to FP32 when unsupported.
- Updated example configs:
  - `configs/training/jepa_pretrain_gpu.yaml` uses `mixed_precision: "fp16"` (match prior AMP behavior).
  - CPU configs set `mixed_precision: "none"` to keep behavior unchanged.
  - Guidance config exposes the knob under `train.mixed_precision`.
- Quick GPU sanity commands (run on a CUDA node):
  - FP32 baseline: `python scripts/train_jepa.py --config configs/training/jepa_tiny.yaml --device cuda --mixed-precision none`
  - FP16: `python scripts/train_jepa.py --config configs/training/jepa_tiny.yaml --device cuda --mixed-precision fp16`
  - BF16: `python scripts/train_jepa.py --config configs/training/jepa_tiny.yaml --device cuda --mixed-precision bf16`
  - Guidance: `python scripts/train_guidance.py --jepa-config configs/training/jepa_tiny.yaml --dsl-config configs/dsl/guidance.yaml --device cuda --mixed-precision bf16`
  - Meta-JEPA: `python scripts/train_meta_jepa.py --tasks <jsonl> --device cuda --mixed-precision bf16 --epochs 2 --batch-size 16`
- This machine reports `torch.cuda.is_available() == False`, so GPU throughput/stability comparisons for FP32 vs BF16 were **not run here**. Run the commands above on a CUDA box to gather numbers; expect BF16 throughput >= FP16 with comparable losses on Ampere+ GPUs.
