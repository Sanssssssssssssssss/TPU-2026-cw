# GRPO rollout320 report figures

Clean report-ready static figures and compact web-readable data for the selected rollout-aligned comparison runs.

Lines:
- R0 baseline: K=2, 3,364 steps, 6,728 rollouts.
- R2 baseline K=8: 841 steps, beta=0.04, lr=3e-6, 6,728 rollouts.
- R4 selected format-aware run: K=8, beta=0.04, lr=3e-6, 841 steps, 6,728 rollouts.

All trend plots use `rollouts_seen = step * num_generations` on the x-axis.
Report-facing `train_reward_score` and `eval_reward_score` are rewritten onto the shared baseline 0-10 reward scale.
Original TensorBoard-derived native reward values are retained as `*_reward_score_native` and in each run's `data/<line>/tensorboard_derived/scalar_pivot_raw.csv`.
Checkpoint plots contain exactly the 22 official rollout-aligned eval points.
Compact TensorBoard-derived scalar tables, checkpoint evals, run manifests, run env files, and trace summaries are under `data/`.
Very large local raw sources such as TensorBoard event files, scalar_metrics JSON, flat trace rows, and checkpoint archives are listed in `data_manifest.json` but omitted from Git.

Figures:
- `figures\00_contact_sheet.png` - Contact sheet. Quick visual index of all generated report figures.
- `figures\01_training_reward_and_kl.png` - All selected runs: training reward and KL. Raw and smoothed training reward/KL traces.
- `figures\01b_training_reward_aligned_baseline_0_10.png` - Training reward on the unified report scale. Report-facing reward score on the shared baseline 0-10 reward range.
- `figures\01c_all_runs_health_answer_quality.png` - All selected runs: answer quality and format health. All selected runs shown together for answer-quality diagnostics.
- `figures\per_run\R0_reward_and_kl.png` - R0 reward and KL over GRPO training. Per-run total reward and KL with train/eval traces where available.
- `figures\per_run\R0_health_answer_quality.png` - R0 rollout health and answer quality. Per-run numeric exact, format, empty response, and no-answer rates.
- `figures\per_run\R1_reward_and_kl.png` - R1 reward and KL over GRPO training. Per-run total reward and KL with train/eval traces where available.
- `figures\per_run\R1_health_answer_quality.png` - R1 rollout health and answer quality. Per-run numeric exact, format, empty response, and no-answer rates.
- `figures\per_run\R2_reward_and_kl.png` - R2 reward and KL over GRPO training. Per-run total reward and KL with train/eval traces where available.
- `figures\per_run\R2_health_answer_quality.png` - R2 rollout health and answer quality. Per-run numeric exact, format, empty response, and no-answer rates.
- `figures\per_run\R3_reward_and_kl.png` - R3 reward and KL over GRPO training. Per-run total reward and KL with train/eval traces where available.
- `figures\per_run\R3_health_answer_quality.png` - R3 rollout health and answer quality. Per-run numeric exact, format, empty response, and no-answer rates.
- `figures\per_run\R4_format_lr3e6_reward_and_kl.png` - R4_format_lr3e6 reward and KL over GRPO training. Per-run total reward and KL with train/eval traces where available.
- `figures\per_run\R4_format_lr3e6_health_answer_quality.png` - R4_format_lr3e6 rollout health and answer quality. Per-run numeric exact, format, empty response, and no-answer rates.
- `figures\02_checkpoint_exact_accuracy_ci.png` - Held-out exact accuracy. Exact held-out accuracy with uncertainty and best/final markers.
- `figures\03_checkpoint_eval_three_panel.png` - Checkpoint evaluation panel. Three checkpoint metrics aligned by generated rollouts.
- `figures\04_training_answer_quality_rates.png` - Training batch answer-quality rates. Training exact/partial/format rates as smoothed trajectories.
- `figures\05_reward_components.png` - Reward component trajectories. Reward terms separated by compatible reward mechanism.
- `figures\06_grpo_training_diagnostics.png` - GRPO training diagnostics. Core GRPO variance and rollout health diagnostics.
- `figures\07_optimization_traces.png` - Optimization traces. Optimization scalar trajectories.
- `figures\08_best_final_checkpoint_summary.png` - Best and final checkpoint summary. Compact best-versus-final checkpoint comparison.
