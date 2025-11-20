# LeJEPA Ablation Summary

| Variant | Description | Loss | Alignment Rate | Codebook Usage |
| --- | --- | --- | --- | --- |
| baseline_infonce | InfoNCE-only encoder (no VQ, no relational, invariance off, SIGReg off). | 0.6824444532394409 | 1.0 | n/a |
| vq_only | +VQ bottleneck, otherwise baseline. | 0.6931 | 1.000 | 0.016 |
| vq_relational | +VQ + relational attention stack. | 0.6931 | 1.000 | 0.008 |
| vq_relational_invariance | +VQ + relational + invariance penalties (color/translation/symmetry). | 0.8030 | 1.000 | 0.016 |
| vq_relational_invariance_sigreg | +VQ + relational + invariance + SIGReg regularisation. | 0.8192 | 1.000 | 0.016 |