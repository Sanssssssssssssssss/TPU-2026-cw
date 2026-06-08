# `course-baseline-001` report package

This folder is a self-contained evidence package for the GRPO baseline run.

Open `report.html` for the reader-facing report, or `report.md` for markdown editing.

## Headline numbers

- Base accuracy: **51.56%**
- Best LoRA checkpoint: **step 2000**, **28.13%**
- Final LoRA checkpoint: **step 3364**, **3.13%**
- Conclusion: the run completed, but the final checkpoint collapsed and should not be presented as an improvement over base.

## Folder map

- `figures/`: report-ready PNG/PDF charts.
- `tables/`: eval summaries, selected TensorBoard scalars, config, chart map.
- `samples/`: rollout examples and failure taxonomy.
- `provenance/`: sanitized manifest, baseline parameter check, source references.
- `raw_refs/`: copied raw evidence files used by the report.

## Suggested citation in the coursework report

Use `figures/02_checkpoint_accuracy_ci.png`, `figures/03_reward_kl_timeline.png`,
and `figures/04_response_health.png` together: they show that training ran to completion,
but checkpoint selection matters because late-stage collapse damaged final LoRA accuracy.
