# Temporal-Density Sparsity Experiment — Notes

Corresponds to proposal Section 5.5 (Controlled variable: temporal
density) and Section 5.7 (Evaluation metrics); this is the central
experiment for RQ1/RQ3.

## What was run

Marlene (frozen architecture, unmodified) was trained and evaluated on the
SERGIO synthetic dataset across three temporal-density tiers, each
repeated over 3 random seeds (9 runs total):

| Tier | Label       | `n_timepoints_keep` |
|------|-------------|----------------------|
| 1    | Dense       | 15 (all available replicates) |
| 2    | Sparse      | 5 |
| 3    | Ultra-sparse| 3 |

Each seed uses genuinely different data, not just a different training
run: `sergio_prepare_data.py --seed` controls which replicate-timepoints
are randomly kept when subsampling to a tier's `n_timepoints_keep`, and
`train_sergio.py --seed` seeds torch/numpy so weight initialization and
batch sampling also differ (and are individually reproducible) across
seeds. This was necessary because prior sweeps on this exact setup showed
Marlene's results are highly seed-dependent — a single run per tier is
not trustworthy evidence, so every tier-level number below is reported as
mean ± std across seeds, never a single run.

## Hyperparameters held fixed across all 9 runs

- Dataset: SERGIO `De-noised_400G_9T_300cPerT_5_DS2` (400 genes, 37 TFs,
  9 genuinely distinct SERGIO-simulated cell types used as `cell_type`;
  `--n_bins 9`)
- `--n_replicates 15` (all pre-simulated replicates loaded, then
  subsampled per tier)
- `--frac_top_edges 0.05` — the best-performing sparsification threshold
  from prior hyperparameter sweeps on this same dataset
- `--n_epochs 500`
- `--lr 1e-4`, `--inner_lr 1e-3` (matches the original, peer-reviewed
  real-data configs — see `train_sergio.py`'s module docstring for why
  these specific values matter: an earlier, 10x larger default caused the
  attention mechanism to collapse to exactly zero)
- Expression data is normalized (`sc.pp.normalize_total` + `log1p`)
  before Marlene ever sees it, matching how every real dataset Marlene
  was validated on (PBMC/HLCA) is preprocessed

## How to reproduce

```
python run_density_experiment.py --device cuda         # ~9 runs, ~18 min each, ~2.7h total
python aggregate_density_results.py
```

`run_density_experiment.py` is resumable — it skips any (tier, seed)
combo whose `Marlene_results/SERGIO/tier<N>_seed<S>/metrics.json` already
exists, so a Kaggle session timeout doesn't require starting over. It
also survives individual run failures (logs and continues to the next
combo). Progress is logged to `density_experiment_log.txt`.

## Outputs (for the paper)

- `density_results_raw.csv` — one row per individual run (tier, seed,
  n_timepoints, mean_auprc, mean_auroc); include as a supplementary/
  appendix table for full transparency
- `density_results_summary.csv` — one row per tier (mean ± std of
  mean_auprc / mean_auroc across the 3 seeds, plus n_seeds); the table to
  cite directly in the Results section
- `density_degradation_curve.png` (300 DPI) / `.pdf` (vector) — the
  central RQ1/RQ3 figure: AUPRC/AUROC mean ± std vs. number of
  timepoints, with random-baseline reference lines at AUPRC=1155/14800≈
  0.078 and AUROC=0.5 for this dataset
- Console output from `aggregate_density_results.py` — an honest,
  non-overclaiming per-tier assessment of whether performance is
  meaningfully above baseline given the seed variance (mirrors the
  framing already used for the `frac_top_edges=0.05` finding: "mean
  AUROC=0.506±0.025 was NOT distinguishable from the 0.5 baseline given
  that variance")

Results are not yet in — the actual training runs on Kaggle GPU. Once
`aggregate_density_results.py` has been run, paste its console summary
and the two CSVs' contents into the Results section, and insert
`density_degradation_curve.pdf` as the RQ1/RQ3 figure.
