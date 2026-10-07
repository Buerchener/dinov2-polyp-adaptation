# Methodology and metric contract

The final model is a conventional early-fusion segmentation head over normalized DINOv2 ViT-S/14 layers 3, 6, 9 and 12 (zero-based indices 2, 5, 8, 11). Each feature grid is 32×32 for 448×448 inputs. The original pretrained encoder is frozen; low-rank updates are added to all 12 fused QKV projections. Rank 4 maps 384→4→1152, adding 6,144 parameters per block, or 73,728 total. The decoder adds 467,649 trainable parameters, for a total of 541,377.

Training uses seed 20261002, FP32, no TF32, AdamW with weight decay 1e-4, decoder LR 1e-3, fixed LoRA LR 5e-5, and no scheduler. The fixed old_C1 draw sequence has 3,040 batches of 9, over 20 epochs. Augmentation preserves the original horizontal/vertical flips and brightness/contrast jitter; validation has no augmentation. Input preparation, valid-region BCE plus soft Dice, and native logit restoration are unchanged. Epoch 20 is the reported endpoint rather than a best validation checkpoint.

## Metric definition

Threshold restored logits strictly above zero (probability >0.5). For each positive ground-truth image, count intersection TP, prediction foreground P and ground-truth foreground G. FP=P−TP and FN=G−TP. Sum these counts over positive images before division: Micro Dice = 2ΣTP/(2ΣTP+ΣFP+ΣFN). Macro Dice averages 2TP/(P+G) across positive images. No smoothing epsilon is needed when GT is non-empty. Normal images never contribute to either Dice. Report any-pixel FP count, foreground area strictly >0.001 count, and unweighted mean foreground fraction separately. Empty positive predictions score zero. Severe failure means image Dice <0.5.

The October 7 finalization recomputed five prediction archives against hash-verified native GT. All 1,285 predictions matched saved per-image integer counts. Same validation IDs and GT do not turn separate training batches into a paired repeated experiment. The historical FT Frozen baseline and the later LoRA Frozen control are distinct rows.

## Selection and generalization boundary

The primary reporting metric was changed retrospectively from image-macro to positive-only micro; experiments and model selection originally used macro and failure/FP gates. LR5e-5 is a normal-FP trade-off selection, not the highest micro score. Its full continuation gate failed because empty positives increased relative to the matched Frozen model. Frozen LOCO/external results cannot be assigned to final LoRA. No patient/video independence is established.

## Audited internal composition

Eligible old_C1 training: 876 positive single frames + 94 normal single frames = 970. The 96 additional negative sequence candidates remain in the 1,066-row manifest but are excluded by the sampler. Validation: 219 positive single frames + 23 normal single frames + 15 negative sequence frames = 257. Thus all evaluated positive masks are single frames, but the internal normal-FP cohort includes sequences. This is distinct from the sequence-excluded LOCO protocol.
