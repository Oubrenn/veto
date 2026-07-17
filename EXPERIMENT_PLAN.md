# VETO Experiment Execution Plan

This plan turns the paper-review checklist into runnable experiments. Use small
mechanism checks first, then scale to the 26-dataset main table.

## 0. Smoke / Command Generation

Use the TFproject Python environment. The helper first tries
`TFproject\Scripts\python.exe`, then `D:\anaconda3\envs\TFproject\python.exe`.

```powershell
.\run_experiment_framework.ps1 smoke cuda
.\run_experiment_framework.ps1 commands cuda
```

The `commands` mode writes full command lists under
`diagnostics\framework_commands` without running long training jobs.

## 1. Synthetic Phase-Path

Fast mechanism diagnostic for PPA, delta G, valid/corrupted AUROC, path accuracy,
phase accuracy, and transition-F1.

```powershell
python experiments\synthetic_phase_path.py --noise 0.1 --corruption swap
python experiments\synthetic_phase_path.py --noise 0.3 --duration_jitter 0.2 --corruption reverse --output diagnostics\synthetic_hard.csv --json_output diagnostics\synthetic_hard.json
python experiments\synthetic_path_control.py --output diagnostics\synthetic_path_control.csv --json_output diagnostics\synthetic_path_control.json
```

`synthetic_path_control.py` is the paper-facing mechanism test. It reports
Occurrence-only, Order-only, and Mixed tasks with Accuracy, Macro-F1,
`G_true - G_wrong`, `DeltaG`, and valid-path versus shuffled-path AUROC.

The manuscript Figure 4 uses the stricter matched-occurrence and marginal-shift
protocol with five seeds:

```powershell
python experiments\synthetic_phase_order_tables.py --seeds 42 43 44 45 46
python experiments\plot_figure4_controlled_synthetic.py
```

## 2. Six-Dataset Core Ablation

Generate the exact commands first:

```powershell
python experiments\run_experiment_suite.py --suite ablations --device cuda --epochs 50
```

Run with `--run` only when the command list looks correct.

Each ablation reports Accuracy, Macro-F1, model parameters, latency, throughput,
memory, diagnostic PPA, delta G, valid/corrupted AUROC, prototype drift, and
memory drift. Run order:

1. `full_veto`
2. `backbone_only`
3. `local_only`
4. `raw_transition`
5. `class_independent_transition`
6. `class_specific_prototypes`
7. `shared_dictionary_only`
8. `uniform_transition`
9. `free_transition_matrix`
10. `neural_transition_generator`
11. `wo_counterfactual`
12. `full_rank_prototypes`
13. `wo_confirmed_memory`
14. `direct_ema_memory`

## 3. Main 26-UEA Table

Use `benchmark_official_splits.py` with the 26 official equal-length UEA dataset
names and 5 seeds. Keep every seed in the raw CSV, then aggregate by dataset.

```powershell
python experiments\run_experiment_suite.py --suite main --python D:\anaconda3\envs\TFproject\python.exe --device cuda --epochs 100 --seeds 42 43 44 45 46 --output_dir diagnostics\main_26uea
```

## 4. Generator Low-Sample Comparison

```powershell
python experiments\run_experiment_suite.py --suite generator --device cuda --epochs 50
```

This emits 10%, 25%, 50%, and 100% training-fraction commands for:

- free transition matrix
- parameter-matched attention head
- neural transition generator

## 5. Confirmed Memory Pollution

```powershell
python experiments\memory_pollution_stress.py --pollutions 0 0.1 0.2 0.3 0.4 --seeds 42 43 44 45 46
```

Report False Commit Rate, normalized drift, commit precision/recall proxy,
confirmation delay, recovery epochs, clean/corrupted/worst-case accuracy,
accepted-sample purity, prototype drift, transition consistency, and
wrong-class write ratio.

The current controlled stress reports held-out nearest-prototype query accuracy,
not UEA task accuracy. The present `PhasePathNet` has no inference-time memory
read path, so do not describe this proxy as model classification accuracy.

## 6. Real Order Corruption

After training a checkpoint:

```powershell
python experiments\order_corruption_diagnostics.py --checkpoint checkpoints\Handwriting_best.pth --dataset Handwriting --levels latent raw --n_permutations 5
```

Report local correctness checks separately from global path metrics. The paper
metrics are occurrence invariance error, IID-reference invariance error,
transition-gain drop, accuracy drop, prediction flip rate, PPA, valid/corrupted
AUROC, confidence drop, and severity-AUC.

The strict Figure 6 source-data protocol trains all six datasets and five seeds,
keeps the signed order gap, and activates the counterfactual objective in the
one-epoch mechanism run:

```powershell
python experiments\run_counterfactual_order_experiment.py --task order --device cuda --epochs 1 --cf_start_epoch 1 --n_shuffles 10
python experiments\run_counterfactual_order_experiment.py --task variants --device cuda --epochs 1 --cf_start_epoch 1 --n_shuffles 5
python experiments\plot_figure6_fix.py
```

The plotter is fail-closed: incomplete five-seed grids, inactive
counterfactual variants, unsigned gaps, or duplicated drift/accuracy values are
rejected instead of replaced by demonstration data.

## 6.1. Canonical Figure Regeneration

Once the strict source files exist, regenerate the manuscript set with:

```powershell
python experiments\redraw_paper_figures.py
```

Legacy CD/overview figure entry points are retired because they can overwrite
the canonical files or generate placeholder panels.

## 7. Deferred Experiments

The final paper plan does not add cross-backbone experiments. VETO is evaluated
with an Inception-style temporal encoder; claims such as "backbone-agnostic" or
"not restricted to one particular backbone" should not be used.

## 8. Statistics

Aggregate seed-level results at dataset level. Do not treat seeds as independent
datasets.

```powershell
python experiments\summarize_results.py --inputs diagnostics\main_veto.csv diagnostics\baseline_inception.csv --reference full_veto --metric selected_test_acc --output_dir diagnostics\summary_acc
python experiments\summarize_results.py --inputs diagnostics\main_veto.csv diagnostics\baseline_inception.csv --reference full_veto --metric selected_test_macro_f1 --output_dir diagnostics\summary_f1
```

Outputs:

- `seed_aggregated_metrics.csv`
- `selected_test_acc_mean_table.csv`
- `selected_test_acc_std_table.csv`
- `average_rank_table.csv`
- `comparison_summary.csv`
- `statistical_summary.json`
- `nemenyi_cd.txt`
- `cd_diagram.png`

## 9. Efficiency and Scaling

```powershell
python experiments\efficiency_scaling.py --device cuda --output diagnostics\efficiency_scaling.csv --json_output diagnostics\efficiency_scaling.json
```

Reports parameters, estimated VETO-head FLOPs, inference ms/sample,
throughput, peak GPU memory, and scaling with class count `Y`, window count
`N`, and phase count `K`.

## 10. Order Sensitivity Correlation

Train one ordered baseline, one orderless baseline, and full VETO with the same
dataset/seed schedule. Then compute whether larger order sensitivity predicts
larger VETO gain:

```powershell
python benchmark_official_splits.py --experiment ordered_backbone --head_mode backbone --no_cf --no_trans --no_memory --datasets Handwriting UWaveGestureLibrary --seeds 42 43 44 --output diagnostics\ordered_backbone.csv --json_output diagnostics\ordered_backbone.json
python benchmark_official_splits.py --experiment orderless --head_mode orderless --no_cf --no_trans --no_memory --datasets Handwriting UWaveGestureLibrary --seeds 42 43 44 --output diagnostics\orderless.csv --json_output diagnostics\orderless.json
python experiments\order_sensitivity_correlation.py --ordered_baseline diagnostics\ordered_backbone.csv --orderless_baseline diagnostics\orderless.csv --veto diagnostics\main_veto.csv
```

## Recommended Run Order

1. Synthetic mechanism check.
2. Six-dataset ablation smoke at low epoch count.
3. Six-dataset ablation full run.
4. 26-dataset main table.
5. Generator low-sample comparison.
6. Memory pollution stress.
7. Cross-backbone.
8. Optional MV-core extension and OSI correlation.
