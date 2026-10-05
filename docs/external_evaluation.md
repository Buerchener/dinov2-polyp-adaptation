# External evaluation and independence

The Phase 4 frozen epoch-20 checkpoint was locked before Phase 6 evaluation. The model, RGB crop rule, 448 input geometry, FP32 precision and threshold 0.5 remained fixed. FT was not used for these scores.

`results/phase6_external_generalization/` retains per-image numeric scores, dataset summaries and the overlap audits. CVC-ColonDB: 380 frames, mean Dice 0.7062587517 / IoU 0.6193740171. ETIS: 196, Dice 0.7070955503 / IoU 0.6213791032. CVC-300: 60, Dice 0.8924622892 / IoU 0.8184341018.

All 60 CVC-300 images overlap the evaluated ColonDB release. There are also duplicates within releases; image-level N is not a unique-patient count. Consult `unique-image-counts.json` and `duplicate-annotation-check.json` before interpreting scores. Annotation differences can occur even for duplicated images. Do not pool/average releases as independent tests.

ETIS had already been observed during earlier project development and was included in a prior U-Net continue/stop gate. No direct use in this DINO training was found, but ETIS is an exposed external benchmark, not a blind test. C5 was also previously observed and is omitted from headline generalization claims. No earlier ColonDB use was found in the local source/report audit; that is not proof of absence of contact or foundation pretraining overlap.

External datasets here contain positive frames only. They measure positive segmentation, not external normal-frame rejection. No patient independence or clinical deployment claim follows from these results.
