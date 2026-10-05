# Reproduction scope and portability changes

## Retained computational behavior

`geometry.py`, `source_sampler.py` and the model-variant modules preserve the experimental implementation. `models.py` keeps DINO initialization/forward behavior while removing the unused U-Net class and accepting the local pretrained path via `DINO_WEIGHTS`. `data_io.py` accepts `POLYP_DATA_ROOT`; the ordered split, per-file hashes, crops and augmentation are retained. The original `summarize` function is extracted into `metrics.py` to avoid obsolete training imports. RGB crop helpers keep their original functions and remove unrelated contact-sheet/report entry points.

The Phase 4 `pilot_entry.py` becomes `runtime/train_entry.py`: the actual training/evaluation loop, sampling hash, optimizer, checkpoint schedule, freeze audit and loss are retained. Machine-specific package/GPU-name assertions and private cloud authorization plumbing are replaced by `scripts/train.py`. It provides a fresh-output context, a user-set time limit and local full-state saves. It does not silently resume, allocate GPUs, access remote services or delete checkpoints. Hardware/package identity now appears in the run record rather than being assumed. `configs/frozen.json` documents the fixed settings; it is not a supported arbitrary tuning interface.

`eval.py` is a new portable orchestration wrapper around the original model, crop, geometry and metric functions. It supports CPU/CUDA, hash-verified external manifests and internal validation. Outputs are scalar/row summaries, without image dumps. `--require-final-lock` verifies the published checkpoint identity; omit only when intentionally evaluating a different run.

The CKA wrapper adapts paths around the actual training-only extraction code and unchanged CKA calculation. Intermediate features are quantized to float16 then calculated in float32, matching the recorded analysis. `--sanity` runs only one image; omit for all 128. No selected data are sampled again.

## Historical environment and runnable scope

Phase 4/6: Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0, timm 1.0.26, NumPy 2.1.3, Pillow 11.3.0; external RGB screening used SciPy 1.16.3. RTX 5090. The dependency list describes the numerical core, not a full cloud OS lock. Local publication smoke checks may use a different CPU PyTorch build; see `verification.json`.

Public commands cover frozen/DetailSkip training, frozen checkpoint evaluation and CKA. Other ablation model definitions and numerical evidence are retained for inspection, but private phase-specific deployment runners are intentionally not presented as portable reproductions. No fresh GPU training, full external inference or clean-download data reconstruction was performed for this release. A new run is not a replay of the already-observed validation selection process.

The original selected checkpoint is not distributed. Metric CSVs remain inspectable, and a compatible trusted local checkpoint can be supplied to the CLI. Full-state recovery files use Python pickle and are for one's own trusted checkpoints only; hash binding checks identity, not trust in an unknown file.

## Provenance

`source-provenance.json` records original relative source locations and source/export SHA256. Changed files are flagged. Original absolute paths, accounts and remote execution settings are omitted. `configs/final-model-lock.json` is an explicitly sanitized metadata export; its original code hashes refer to the historic pre-publication files, not the portable wrappers. `runtime/public-freeze.json` binds this release's computational files and metadata instead.

Publication formatting also normalizes CSV line endings and trailing whitespace without changing numerical cells.
