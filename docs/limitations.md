# Limitations and attribution boundaries

- Final LoRA is single-seed internal development evidence. Repeated selection observed the same validation data. Micro Dice became primary retrospectively; the original macro/FP selection and failed full continuation gate remain disclosed.
- The retained LR5e-5 has lower micro than LR1e-4. It is a normal-FP trade-off, not the global numerical optimum. Normal mean foreground and empty positive predictions remain worse than matched Frozen.
- Micro emphasizes large foreground regions. Report macro, empty predictions, severe failures, small-target strata and all normal FP measures alongside it.
- The two Frozen references are different trained runs. All final rows share evaluation semantics; only the historically matched pairs support their stated experimental comparisons. No repeated-seed LoRA confidence interval or significance claim is available.
- Six-fold LOCO, random splits and external results are Frozen-model evidence. They do not prove final LoRA cross-center generalization. Random-split SD spans three splits/seeds; LOCO SD spans six centers. Neither is a confidence interval for LoRA.
- Patient/video IDs are unreliable or unavailable. All sequences are excluded from primary LOCO, but this does not establish patient independence. Historical internal validation includes sequence frames. Image-level random splits can be optimistic.
- C5/C6 and ETIS were examined during development. All 60 CVC-300 frames overlap this ColonDB release. Upstream pretraining overlap is unknown. Kvasir is not a completed benchmark here.
- Normal-frame coverage is limited. External sets contain positives and do not measure external rejection of normal tissue. C4 over-segmentation and C5 small-target misses remain substantial; post-hoc preprocessing is diagnostic only.
- DetailSkip and completed hierarchical decoder are rejected trade-offs, not omitted negative experiments. The small average gains do not outweigh observed failure/FP regressions.
- This is adaptation/evaluation work using DINOv2, timm and conventional decoder operations, not a new foundation model, SOTA claim or clinically ready system. Earlier Medical-SAM3 team ownership is separate.
- Portable code has CPU verification, not a fresh full training replication. Model checkpoints and images remain local; public integer counts reproduce reported arithmetic, while full inference needs the original compatible checkpoint and independently obtained data. Hardware/version variation can change results.
- License selection remains pending; no new LICENSE is invented. Third-party code, weights and dataset terms remain applicable.
