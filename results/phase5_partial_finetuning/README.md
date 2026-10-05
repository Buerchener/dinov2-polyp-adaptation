# Phase 5: partial fine-tuning

Frozen baseline is reused from Phase 4. Last-block FT uses 2,242,881 trainable parameters. Positive Dice rises to 84.4470%, but normal any-FP rises 11→13, >0.1%-area FP 7→11, and mean foreground area 0.1606%→0.6243%. FT is not the selected Phase 6 model. `epoch-comparison.csv` retains intermediate evaluations, not independent runs.
