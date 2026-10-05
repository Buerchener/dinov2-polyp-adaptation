# Project overview

The project asks how much a frozen DINOv2 representation can support colonoscopic polyp segmentation with a small supervised decoder. Its main deliverable is an evidence trail through representation selection, failure analysis and a locked external evaluation.

The selected baseline uses four transformer depths and 467,649 trainable parameters. Phase 1 establishes the single-last-layer comparison; Phase 2 probes depths; Phase 2.5 describes feature similarity; Phase 3 tests selected pairs. Phase 4 rejects DetailSkip on the overall trade-off. Phase 5/5.5 retain the frozen baseline despite higher positive Dice from partial FT, because normal-frame metrics worsen and the tested threshold sweep does not resolve the trade-off. Phase 6 evaluates that selected checkpoint with explicit exposure/overlap limits. Phase 7 is a parameter-matched hierarchical decoder implementation with interrupted training, not a completed result.

The repository is a curated export. It retains small numerical evidence and real computational code, while excluding source datasets, cloud access material, redundant logs and tensor archives. `source-provenance.json` maps each selected source file to its export and SHA256; it is not a substitute for the original experiment's full-state archives.
