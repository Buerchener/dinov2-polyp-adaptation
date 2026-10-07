# Reproduction scope and portability changes

## Retained computational behavior

`geometry.py`, `source_sampler.py` and the model-variant modules preserve the experimental implementation. `models.py` keeps DINO initialization/forward behavior while removing the unused U-Net class and accepting the local pretrained path via `DINO_WEIGHTS`. `data_io.py` accepts `POLYP_DATA_ROOT`; the ordered split, per-file hashes, crops and augmentation are retained. The original `summarize` function is extracted into `metrics.py` to avoid obsolete training imports. RGB crop helpers keep their original functions and remove unrelated contact-sheet/report entry points.

The Phase 4 `pilot_entry.py` becomes `runtime/train_entry.py`: the actual training/evaluation loop, sampling hash, optimizer, checkpoint schedule, freeze audit and loss are retained. Machine-specific package/GPU-name assertions and private cloud authorization plumbing are replaced by `scripts/train.py`. It provides a fresh-output context, a user-set time limit and local full-state saves. It does not silently resume, allocate GPUs, access remote services or delete checkpoints. Hardware/package identity now appears in the run record rather than being assumed. `configs/frozen.json` documents the fixed settings; it is not a supported arbitrary tuning interface.

`eval.py` is a new portable orchestration wrapper around the original model, crop, geometry and metric functions. It supports CPU/CUDA, hash-verified external manifests and internal validation. Outputs are scalar/row summaries, without image dumps. `--require-final-lock` verifies the selected model-specific checkpoint identity; omit only when intentionally evaluating a different run.

The CKA wrapper adapts paths around the actual training-only extraction code and unchanged CKA calculation. Intermediate features are quantized to float16 then calculated in float32, matching the recorded analysis. `--sanity` runs only one image; omit for all 128. No selected data are sampled again.

## Historical environment and runnable scope

Phase 4/6: Python 3.12, PyTorch 2.8.0+cu128, torchvision 0.23.0, timm 1.0.26, NumPy 2.1.3, Pillow 11.3.0; external RGB screening used SciPy 1.16.3. RTX 5090. The dependency list describes the numerical core, not a full cloud OS lock. Local publication smoke checks may use a different CPU PyTorch build; see `verification.json`.

Public commands cover frozen/DetailSkip and final LoRA training, frozen/LastBlock/LoRA checkpoint evaluation, micro aggregation and CKA. Other ablation model definitions and numerical evidence are retained for inspection, but private phase-specific deployment runners are intentionally not presented as portable reproductions. No fresh GPU training, full external inference or clean-download data reconstruction was performed for this release. A new run is not a replay of the already-observed validation selection process.

The original selected checkpoint is not distributed. Metric CSVs remain inspectable, and a compatible trusted local checkpoint can be supplied to the CLI. Full-state recovery files use Python pickle and are for one's own trusted checkpoints only; hash binding checks identity, not trust in an unknown file.

## Provenance

`source-provenance.json` records original relative source locations and source/export SHA256. Changed files are flagged. Original absolute paths, accounts and remote execution settings are omitted. `configs/final-model-lock.json` is an explicitly sanitized metadata export; its original code hashes refer to the historic pre-publication files, not the portable wrappers. `runtime/public-freeze.json` binds this release's computational files and metadata instead.

Publication formatting also normalizes CSV line endings and trailing whitespace without changing numerical cells.

## October 7 finalization

`lora_model.py` is copied unchanged from the completed LR5e-5 run. `lora_train_entry.py` preserves its sampler, optimizer, gradient path, checkpoint cadence and encoder digest check. Machine-specific GPU/package equality and private execution plumbing are replaced by a portable context. The initial encoder and decoder hashes remain checked; CPU/CUDA RNG identity relative to the historical environment is recorded, because cross-platform RNG equivalence is not guaranteed. A mismatch is not a reproduction of the original environment. No full GPU training was run for this release.

`configs/lora_final.json` documents fixed protocol settings; it is not an arbitrary tuning interface. `configs/frozen.json` retains the original frozen protocol. `runtime/public-freeze.json` is regenerated for the current portable source. Original source/export provenance remains in `source-provenance.json`; finalization additions and modifications are recorded separately.

The original final LoRA checkpoint SHA256 is in `configs/lora-final-model-lock.json`; the legacy `final-model-lock.json` still identifies the Frozen checkpoint used externally. Supplying the wrong architecture/checkpoint is rejected through strict state loading and optional model-specific hash validation. Neither weights nor data are distributed.

`eval_micro_dice.py --records` verifies arithmetic using exported integer counts; `--predictions ... --data-root ...` independently recomputes from native binary masks against hash-verified original GT. The latter was used for five final rows (1,285 predictions), with an exact match to saved counts and common validation IDs. Normal masks are never included in Dice. The public script accepts extracted dataset files; the inference loader also supports the original dataset ZIP.

`smoke_lora.py` verifies zero-initialized LoRA equivalence to Frozen, parameter counts, gradient connectivity, unchanged original encoder tensors after an optimizer step, strict checkpoint round-trip, and micro-vs-macro/normal separation. It uses synthetic CPU input, not patient data. The existing broader smoke verifies the ordered manifest and full draw-sequence hash. CPU checks do not establish CUDA training equivalence.

LOCO and random-split summaries remain historical audited exports, not fresh fold retraining. Only the primary internal Frozen/LoRA training path has a portable training wrapper; the LOCO protocol and manifest are provided for inspection, but this release does not claim a one-command exact LOCO rerun.
