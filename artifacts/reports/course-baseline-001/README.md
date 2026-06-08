# `course-baseline-001` report package

This folder is a self-contained evidence package for the GRPO baseline run.

Open `report.html` for the reader-facing report, or `report.md` for markdown editing.

## Headline numbers

- Base accuracy: **51.56%**
- Best saved/fetched LoRA checkpoint: **step 2000**, **28.13%**
- Final LoRA checkpoint: **step 3364**, **3.13%**
- Complete scalar peak correction: `eval_reward_score` peaks at step **448**, `eval_numeric_exact_rate` peaks at step **256**, and `eval_format_accuracy` peaks at step **704**. These are scalar peaks, not checkpoint evals.
- Conclusion: the run completed, but the final checkpoint collapsed and should not be presented as an improvement over base. Step 2000 is only the best among saved/fetched LoRA checkpoints, not the best point in the full scalar timeline.

## Folder map

- `figures/`: report-ready PNG/PDF charts.
- `tables/`: eval summaries, selected TensorBoard scalars, config, chart map.
- `samples/`: rollout examples and failure taxonomy.
- `provenance/`: sanitized manifest, baseline parameter check, source references.
- `raw_refs/`: copied raw evidence files used by the report.
- `full_scalar_analysis/`: no-drop scalar analysis with full long CSV, pivot CSV, peak summary, and early-window plots.

## Suggested citation in the coursework report

Use `figures/02_checkpoint_accuracy_ci.png` for saved-checkpoint comparison, but cite
`full_scalar_analysis/figures/03_eval_scalars_early_0_1200.png` and
`full_scalar_analysis/tables/key_scalar_peak_summary.csv` for the actual early scalar peaks.
Do not claim step 2000 is the overall training-process optimum; it is only the best saved/fetched checkpoint eval in this package.
