# Historical frozen/FT experimental protocol

For the final LoRA configuration and current positive-only Micro Dice contract, see [methodology](methodology.md). This document preserves the original frozen/FT protocol and historical macro metrics.

## Data and sampling

The published ordered manifest is `runtime/reports/manifest.jsonl`, bound to its SHA256 in `audit-summary.json`. It contains 1,066 training rows and 257 internal validation rows from C1–C4. The frozen baseline excludes 96 `normal_expansion_candidate` training frames; 970 rows are eligible. This exclusion is enforced by `source_sampler.draws(..., "old_C1")`, not by deleting manifest rows. The sampler consumes the same auxiliary random draws as the earlier source comparison, including metadata for the three unused training videos. Do not reorder/filter the manifest before replay.

Each update: two images per center (eight total) plus one original C1 normal frame, avoiding a duplicate within that batch. The latter comes from only four original C1 normals. This is repeated exposure, not additional independent supervision. CKA uses 128 training-only samples from the 970-frame eligible pool, 32 per center, stratified by normal/positive status. CKA is not computed on validation images.

## Fixed training settings

| Item | Value |
|---|---|
| Seed | 20261002 |
| Epochs / updates | 20 / 3040 (152 per epoch) |
| Batch | 9 |
| Optimizer | AdamW, head LR 0.001, weight decay 0.0001 |
| Objective | Per-image valid-region BCE + soft Dice; unweighted image mean |
| Precision | FP32, TF32 off, no AMP |
| Input | RGB-only conservative left crop; 448-square aspect-preserving letterbox |
| Augmentation | Independent horizontal/vertical flips; brightness and contrast factors in [0.9,1.1] |
| Checkpoints evaluated | 0, 5, 10, 15, 20; fixed epoch 20 reported |
| Encoder | Frozen/eval/no_grad for selected baseline |

The replay hash is `457640139e657d7307b3f2124bf2dd78f25a1bc3eebf212d8a44773561e696e8`. The portable training wrapper keeps the original assertion. `configs/frozen.json` records the fixed protocol; the loop enforces these values and is not a general hyperparameter-search interface.

## Evaluation

Restore logits to native image resolution before applying strict logit > 0. The padded area is removed; a cropped-out region receives logit −30. Per-positive-image Dice = 2 intersection / (prediction area + GT area), IoU = intersection / union; then take the image mean without smoothing. Severe failure means Dice < 0.5. Empty prediction means zero foreground pixels. Small/medium/large are native GT fractions <5%, [5%,20%), and ≥20% of the full image; these are project strata, not clinical size classes.

For normal frames, separately report any foreground, foreground area >0.1%, and mean foreground fraction. Internal denominator: 38 normal frames; positive denominator: 219. Do not treat normal images as perfect-Dice cases.

## Comparability

Phase 1/2 reference runs used Kaggle; Phase 3 two-layer candidates ran on a different environment and reused reference scores. Phase 4 made a new same-environment baseline on RTX 5090. Use 82.7368% for Phase 4/5 comparison and Phase 6 checkpoint identity, not 82.4341%. One seed was used; no confidence interval or statistical superiority claim is supported. CKA is a descriptive representation diagnostic, not a guarantee that a fusion will improve segmentation.
