# Research lineage and current scope

Reviewed on 2026-10-21. This document separates the finalized DINOv2 study from the broader local research history. Historical experiments are not a single ascending performance curve, and their datasets, model identities and metrics cannot be merged into one leaderboard.

## Four research tracks

| Track | What was investigated | Evidence status | Boundary |
|---|---|---|---|
| Earlier Medical-SAM3 analysis | Instance versus semantic output; head adaptation; last-two-block LoRA; prompt learning; predicted-box and box-perturbation studies | Completed development experiments with mixed/negative outcomes | Distinct model and protocol from DINOv2; upstream and earlier team ROI ownership remain separate |
| U-Net pseudo-label students | Stronger pretrained baseline; 194 selected Kvasir pseudo-labels; identical added RGB with human-label control; repeated continuation and validation redesign | Completed; ETIS improvements coexist with insufficient ClinicDB gain and severe-failure regressions | Observed development sets; teacher training sources include Kvasir; no annotation-cost or clinical claim |
| DINOv2 adaptation | Depth probing, fusion, CKA, decoder alternatives, LastBlock FT, all12 QKV LoRA; Frozen cross-center and external evaluation | Completed evidence reported in this repository | Final LoRA is an internal FP trade-off selection; Frozen generalization scores do not transfer to LoRA |
| Teacher audit and selective KD | Existing teacher-mask complementarity, source/overlap audit, minimal positive-only GT-ROI distillation implementation | Audits and local preparation only; no formal KD A/B results | No fabricated teacher probabilities, no SD-SAM performance result, no completed EndoDINO probe |

The public repository remains focused on the DINOv2 study. The earlier experimental archives, raw images, checkpoints, predictions and cloud execution material remain outside this curated Git checkout. The broad local evidence index contains 39 stages; it is not evidence that all stages succeeded.

## Metric and exposure timeline

Earlier ClinicDB comparisons use sequence-equal Dice over metadata groups. Pseudo-label studies report ClinicDB sequence-equal results separately from ETIS image-macro Dice. Early PolypGen C5 development uses positive-image macro and normal-video-equal FP; the current internal DINOv2 table uses positive-only pixel-micro alongside image-macro and separate normal metrics. LOCO averages six center-wise positive-image macro scores. These aggregations are different estimands.

Statements in old reports that C6 was reserved describe their historical stage. Later single-frame LOCO includes the audited C6 subset; the overall project cannot now claim that C6 has never been observed. Likewise, Kvasir was used in earlier U-Net pseudo-label research, but has not been evaluated as a formal benchmark for this DINOv2 study. ETIS exposure and the CVC-300/ColonDB overlap remain disclosed.

## Two Frozen controls and unresolved reproducibility difference

The historical FT control and the LoRA control use the same Frozen DINOv2/four-depth architecture and 467,649 trainable decoder parameters. They are separate trained checkpoints. Their recorded seed, draw sequence and main settings agree, and their epoch-zero per-image statistics agree. Saved training losses first differ at update 5 (1.0847238302230835 versus 1.084723949432373). The cause of the eventual difference has not been isolated; a small initial numerical difference does not prove a particular hardware or optimizer explanation.

LastBlock's paired macro gain is +1.7103 pp over its historical Frozen control. Final LoRA's paired macro gain is +2.9037 pp over the LoRA control. Do not choose whichever control gives a preferred gain. The earlier incomplete LoRA v1 had a documented CUDA-RNG isolation bug; that is a separate incident and is not established as the explanation for the two completed Frozen runs.

## Pending work is not a result

- Minimal positive-only selective KD has a prepared implementation and historical CPU checks. The real training-positive teacher cache and formal GT-only/KD A/B evaluations have not been produced.
- SD-SAM stopped at the provenance gate because the audit did not establish a reliable official trained checkpoint. This is the status of that historical audit, not a current claim about all upstream releases.
- EndoDINO has no completed matched frozen-representation probe in the project.
- Final LoRA has no formal LOCO or external benchmark. The existing Frozen results must not be relabeled.
- A historical normal-loss-weight experiment has recovered predictions but incomplete trained-weight archival. Prediction-level evaluation does not imply a restorable training checkpoint.
- License selection is still pending.

## October 21 organization review

The local original workspace showed no experiment file modifications after October 7 within the documented inventory scope. Modification times alone cannot exclude deleted files or preserved-timestamp changes. The existing public checkout and remote started at `24a2d18b42ee65bba7527829b1103dc9a1aea85a` with a clean working tree.

An authorized read-only remote check compared 36 selected primary epoch20 result/checkpoint files. Thirty had identical local files; six small Phase3 statistics files were recovered separately (323,290 bytes), after which all 36 had local SHA256 matches. This is targeted evidence coverage, not a complete remote-disk backup. No training, remote installation or dataset upload occurred. Local organization preserved original experiment paths and failures; historical entry documents were backed up before navigation updates.

This review updates documentation and verifies available records. It does not replace the prior native-mask/CPU audits with a new full training or inference replication. See [methodology](methodology.md), [experiments](experiments.md), [limitations](limitations.md), and [reproduction scope](reproducibility.md).
