# Experimental record

## Representation and feature analysis (historical macro)

Parameter-matched L12-only: 78.1703%; L9-only: 80.1793%; four-depth fusion: 82.4341%. Fusion gains 4.2638 pp over L12 and 2.2548 pp over L9. L9+L12 yields 81.7294%; none of the three tested pairs beat four depths. Reference runs span environments. Training-only CKA is descriptive: L6–L9 0.9173, L9–L12 0.7678; similarity is not causal proof of complementarity. Original numerical tables are retained in phase1–phase3 result directories.

## Decoder and partial fine-tuning

Historical paired Frozen reference: macro 82.7367516%, any-FP 11/38, mean normal foreground 0.1606318%. DetailSkip: macro 82.8096329% (+0.0728813 pp), small Dice 78.6949% versus 80.0398%, any-FP 18/38; reject. LastBlock FT: macro 84.4470494% (+1.7102978 pp), 2,242,881 trainable parameters, any-FP 13/38, area-FP 11/38, normal mean foreground 0.6242827%. Threshold sweeps on observed validation did not satisfy both tested frozen FP budgets.

Phase 7 is complete: hierarchical macro 83.2941767% (+0.5574251 pp), small Dice 78.8123% (−1.2275 pp), any-FP 24/38, area-FP 20/38, mean normal foreground 0.6839798%, 478,465 trainable parameters. Reject under the declared trade-off. The earlier partial checkpoint is historical execution evidence only; the completed table supersedes the former incomplete status.

## LoRA variants

See [LoRA ablation CSV](../results/lora_ablation.csv) for exact macro and FP values. The A/B Frozen control is 81.9431041% macro, not the historical 82.7367516% control. All12 LR1e-4: 84.8736913% macro, 89.9912800% micro, any-FP 10/38, area-FP 9/38, mean normal foreground 1.1078316%. All12 fixed LR5e-5: 84.8468007% macro, 89.1933486% micro, any-FP 6/38, area-FP 3/38, normal mean foreground 0.5694654%; retained as the selected trade-off. Seven empty positives remain, compared with three for matched Frozen. Original DINO tensors remain unchanged; LoRA changes the effective projections through additive trainable matrices.

L7–L12-only LoRA: approximately 82.64% macro and nine empty positives; not retained. Warmup+cosine: 84.8740746% macro, any-FP 14/38, area-FP 12/38, mean normal foreground 0.8406475%; not retained. These source-reported macro ablations are not newly native-mask recomputed in the finalization; no micro values are inferred for them.

## Generalization experiments use Frozen

Single-frame LOCO: six center-wise image-macro Dice values 80.0735, 82.7803, 83.5280, 59.8776, 67.8580, 80.6386%; center-equal mean 75.7927%, sample SD 9.6615 pp. All 1,518 audited single frames are covered across held-out folds; all sequences excluded. Pooled image-macro 77.9144% is a different secondary aggregation. C5/C6 were previously examined; this is not blind testing.

Random image splits: seeds 20261006/7/8 yield macro 79.1518, 82.3667, 80.2554%; mean 80.5913%, sample SD 1.6336 pp. The descriptive gap versus center-equal LOCO is 4.7986 pp. Neither split establishes patient/video independence. Different training pool sizes and partition designs prevent a pure domain-shift causal interpretation.

Post-hoc main-view preprocessing raises C4 from 59.88% to 63.77%; report it separately from locked LOCO. C5 small/medium/large macro Dice is 58.7837/80.9908/89.0573%. External Frozen macro: ColonDB 70.63%, ETIS 70.71%, CVC-300 89.25%. ETIS was exposed during development; all 60 CVC-300 frames overlap ColonDB. Kvasir was not formally evaluated for this DINOv2 study. No LoRA LOCO or external run has been performed.

## Status and ownership

This release consolidates completed evidence and runnable local wrappers. It does not launch selective KD, further decoder search, new cloud experiments, or GPU training. No superiority, novelty, clinical utility or authorship of upstream/teammate models follows from this engineering study.
