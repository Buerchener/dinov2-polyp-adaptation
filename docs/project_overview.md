# Project overview

The study starts from pretrained DINOv2 ViT-S/14, freezes the encoder and trains a lightweight decoder, tests intermediate depths, compares final-layer and four-depth fusion, and then studies LastBlock fine-tuning and QKV LoRA. The selected configuration uses all12 LoRA (rank4, alpha8, fixed LR5e-5) plus the original decoder: 541,377 trainable parameters, including 73,728 added encoder parameters. Original DINO tensors remain unchanged.

Final LoRA obtains positive-only Micro Dice 89.193349% on observed internal validation. LR1e-4 has higher micro, but the retained lower learning rate reduces normal FP. Seven empty positive predictions persist. This is a trade-off choice, not a claim that every metric improves or the original full continuation gate passed.

The completed representation, negative decoder results, six-center single-frame LOCO, random-split comparison and external evaluation are retained with distinct metric and model identities. LOCO/random/external results belong to Frozen, not final LoRA. Phase7 is completed and rejected for worse small-target and normal-FP behavior.

The repository is a curated export with portable computational code and compact numerical evidence. Original data, weights, predictions, cloud access material and recovery archives remain local. See the README, methodology, experiments, limitations and provenance documents for evidence and reproduction boundaries.
