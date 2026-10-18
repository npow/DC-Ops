# E8 validation evidence

The JSON reports in `translator/` are snapshots from the physical-board and
host checks performed during development. `translator/validation_summary.json`
links the model comparisons, memory budgets, timings, and current demo mode.
`translator/live_sign_demo_check.json` verifies that live camera frames and
automatic OCR continue advancing without display pauses or hardware errors.
The live demo uses a 60-entry phrase table; it does not translate arbitrary text.

The frozen neural-translator evaluation finished 8/8 phrases and matched
executable TOSA tokens 8/8, with 7/8 matching FP32 tokens. These parity checks
do not establish translation quality. The pretrained model translates uppercase
`STOP` incorrectly on both host and device. Vocabulary-pruning experiments are
included as failed candidates and are not deployed.

Reports retain their original local artifact paths and firmware hashes. Model
weights and board memory images stay outside Git. Reproduce exports and live
checks using [the live firmware instructions](../translation/live/README.md).
