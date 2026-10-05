# Limitations and attribution boundaries

- This is an experimental adaptation/evaluation project, not a new foundation model. DINOv2, timm and CKA are credited to their authors. The lightweight head uses conventional operations; no first/novel/SOTA claim is made.
- Repeated decisions used the same internal development validation. Scores are not independent held-out estimates after model selection. Only one seed and limited normal-frame coverage are available.
- Frames may be correlated within video/source groups; patient independence is not established. Four original C1 normal images are repeatedly sampled in the selected run.
- ETIS and C5 were exposed earlier in development. CVC-300 overlaps this ColonDB release. Foundation-model pretraining overlap is unknown.
- The selected frozen model favors a particular positive-quality/normal-FP trade-off; this does not prove universally superior calibration or clinical safety. The threshold result is limited to the tested grid, not every possible calibration method.
- The DetailSkip difference in mean Dice is tiny and accompanied by worse small-polyp and normal-frame results. FT improves positive Dice but worsens normal metrics. Report both outcomes.
- Phase 7 stopped before a formal endpoint. Its architecture and partial state are not evidence of either performance improvement or architectural failure.
- Original full experimental evidence is retained locally. The public CLI is an audited portability adaptation, not the original remote orchestration environment. No GPU retraining was performed for publication; data and selected checkpoint are not bundled.
- CV statements should describe personally verified work on adaptation, controlled ablations, CKA and error/generalization analysis. Do not claim ownership of upstream architectures or earlier team Medical-SAM3 work on the strength of this repository alone.
- License selection is pending. No LICENSE file was invented; upstream licenses and dataset terms remain separate.
