# Repository curation and publication boundary

This is an update to the existing `Buerchener/dinov2-polyp-adaptation` repository on `main`; no repository is created and no visibility setting is changed. The original experiment workspace remains separate from this curated checkout. Its new local index links to this checkout; its previous README is preserved under `archive/legacy_documentation/`. No important original experiment files were deleted or moved out of their evidence trees.

The existing `runtime/` architecture is retained rather than performing a cosmetic relocation that would break audited imports. Only new LoRA computational modules, portable entry points, compact counts and selected audit summaries were added. Remote orchestration, duplicated historical runners and unrelated Medical-SAM3 experiments remain in the original local workspace and are not included in this Git repository.

Excluded: datasets and images/masks; pretrained and trained weights (`.pt`, `.pth`, `.ckpt`, `.safetensors`); predictions/logits, feature caches and large archives (`.zip`, `.tar`, `.zst`, `.npy`, `.npz`); environments, caches, logs, runs and wandb; `.env`, SSH keys, credentials, cloud authorization/configuration and machine-specific paths. No Git LFS upload is used. Published JSONL files contain integer counts and data manifests, not pixel arrays. Credentials supplied during this task were not written to the repository.

The safety report scans all tracked/nonignored candidate files and reachable historical Git blobs for common secret/private-path/endpoint patterns, excluded binaries, file-size issues and broken local Markdown links. No candidate exceeds 10 MB. A pattern scan is not an exhaustive secret-detection guarantee. Original experiment files outside this curated Git checkout are deliberately not publication candidates.

CPU verification covers the original sampler, geometry, model parameter counts, LoRA gradients and unchanged original tensors, real checkpoint strict loading, two-image historical count equivalence, and all public train/eval CLI help entry points. Native saved-mask recomputation covers five rows and 1,285 masks. No fresh GPU training, remote login or new cloud job was performed.

The remaining manual decision is the project license. The existing pending-license statement is retained; model/data/third-party licensing stays separate. No license is chosen on the user's behalf.
