# R12 Full Autotune Evidence: Tail512

This package records R12 full tail512 autotune results without changing the canonical three-line evidence package.

Current winner: `R12_tail_lr1e-6_beta004_from512`.

Best checkpoint summary:

- beta=0.04, lr=1e-6: step 841 exact 65.625 partial 67.1875
- beta=0.06, lr=1e-6: step 512 exact 62.5 partial 64.0625
- beta=0.04, lr=5e-7: step 512 exact 62.5 partial 64.0625
- canonical R12 full best in clean table: step 384 exact 62.5 partial 65.625

Conclusion: beta=0.06/lr=1e-6 and beta=0.04/lr=5e-7 did not beat beta=0.04/lr=1e-6.

Cleanup note: the non-winner beta=0.04/lr=5e-7 raw run and clean-plot folder
were removed from the top-level local/remote artifact roots after its metrics
were copied into this evidence package. Keep `r12-full-autotune-tail512-001`
as the retained winner artifact.

Key tables:

- `tables/summary.csv`
- `tables/autotune_tail_all_checkpoint_eval.csv`
- `tables/canonical_r12_full_checkpoint_eval.csv`

Clean plots:

- `artifacts/reports/r12-full-autotune-tail512-001-clean`
