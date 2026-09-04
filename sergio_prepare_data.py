"""
Converts a SERGIO-simulated dataset into the AnnData format Marlene's
(frozen, unmodified) scRNATimeSeriesDataset expects.

v3 CHANGE FROM PREVIOUS VERSIONS -- READ THIS FIRST
-----------------------------------------------------
Earlier versions of this script relabeled SERGIO's "bins" as timepoints
and used a synthetic single/replicate-based cell_type label. Diagnosis
showed that was backwards:

  - SERGIO's number_bins are NOT interchangeable snapshots of the same
    population -- each bin gets its own regulator basal-production-rate
    profile (see the dataset's Regs_*.txt file), so different bins are
    GENUINELY, GRN-drivenly different cell states.
  - A replicate (same bin, different SERGIO random seed) differs from
    another replicate ONLY by simulation noise -- there is no systematic,
    GRN-driven difference between replicates at all.

Marlene's entire training signal comes from a cell-type classification
pretext loss. Using replicate-id as that pretext label gave the model a
task it could solve without ever learning correct attention/GRN
structure (confirmed: AUPRC stuck exactly at the random baseline even
after training). Using SERGIO's real bins as cell_type gives the
classification task genuine GRN-driven signal to learn from.

THIS VERSION:
  - cell_type = SERGIO bin id (genuinely distinct, regulator-driven)
  - timepoint = replicate id (arbitrary re-sampling "snapshots" for the
    temporal-density ablation study, proposal Section 5.5). These do NOT
    represent real biological time -- they are repeated draws of the
    SAME underlying population under simulation noise, used purely to
    give Marlene multiple sequence positions to subsample for the
    density-tier experiments (Tier 1/2/3). State this simplification
    explicitly in your methodology, same as the single-cell-type
    simplification in earlier versions.

v4 CHANGE -- expression normalization (see main() for details)
-----------------------------------------------------------------
Raw SERGIO counts are now passed through scanpy's normalize_total +
log1p before being written to the h5ad, matching how every real dataset
Marlene was validated on (PBMC/HLCA) is preprocessed before training
(see train.py's preprocess()). Without this, SERGIO bins' large
basal-production-rate-driven differences in overall expression SCALE
gave the classification pretext task a trivial shortcut, causing the
attention output to collapse to exactly zero (confirmed via a controlled
repro) instead of learning real per-gene structure. This alone was NOT
sufficient to fix the collapse -- see train_sergio.py's lr/inner_lr
defaults for the other half of this fix.

v5 CHANGE -- seeded timepoint subsampling (see --seed / main() for details)
-----------------------------------------------------------------------------
--n_timepoints_keep used to always pick the same evenly-spaced replicate
indices regardless of seed, so repeated "seeds" of the same density tier
would silently train on identical data. It now takes a random (seeded)
subset of size --n_timepoints_keep out of --n_replicates, so seed=0/1/2
of the same tier see genuinely different timepoints -- needed for the
multi-seed robustness experiment (prompt_for_experiment.txt): prior sweeps
showed Marlene's results are highly seed-dependent, so a single run per
tier is not trustworthy evidence.

OUTPUT
------
- data/SERGIO-Marlene.h5ad
- data/sergio_gt_edges.csv
"""
import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def load_sergio_dataset(
    dataset_dir: Path,
    n_bins: int,
    replicate_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Returns X, cell_type (=bin id), timepoint (=replicate id), gene_ids, gt_edges."""
    gt_path = dataset_dir / "gt_GRN.csv"
    gt_df = pd.read_csv(gt_path, header=None, names=["reg", "target"])
    gt_edges = list(zip(gt_df["reg"].astype(int), gt_df["target"].astype(int)))

    X_list, CT_list, T_list = [], [], []
    gene_ids = None

    for t, rep in enumerate(replicate_ids):
        csv_path = dataset_dir / f"simulated_noNoise_{rep}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path.name} not found")
            continue

        raw = pd.read_csv(csv_path, header=None)
        gene_ids_this = raw.iloc[1:, 0].to_numpy().astype(int)
        expr = raw.iloc[1:, 1:].to_numpy().astype(np.float32)  # (n_genes, n_cells)

        if gene_ids is None:
            gene_ids = gene_ids_this
        else:
            assert np.array_equal(gene_ids, gene_ids_this), \
                "gene order mismatch between replicates"

        n_genes, n_cells_total = expr.shape
        assert n_cells_total % n_bins == 0, \
            f"{n_cells_total} cells not divisible by {n_bins} bins"
        cells_per_bin = n_cells_total // n_bins

        # columns are grouped by bin: first cells_per_bin cols -> bin 0, etc.
        cell_type = np.repeat(np.arange(n_bins), cells_per_bin)
        timepoint = np.full(n_cells_total, t, dtype=int)

        X_list.append(expr.T)  # -> (n_cells, n_genes)
        CT_list.append(cell_type)
        T_list.append(timepoint)
        print(f"  loaded {csv_path.name} as timepoint={t}: "
              f"{expr.shape[1]} cells x {n_genes} genes, {n_bins} cell types")

    X = np.concatenate(X_list, axis=0)
    cell_type = np.concatenate(CT_list, axis=0)
    timepoint = np.concatenate(T_list, axis=0)
    return X, cell_type, timepoint, gene_ids, gt_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2",
        help="Path to a SERGIO bundled dataset folder. Must have enough "
             "TFs to avoid dead attention -- use the 400-gene (37 TF) or "
             "larger dataset, NOT the 100-gene (10 TF) one.",
    )
    ap.add_argument(
        "--n_bins", type=int, default=9,
        help="Number of SERGIO bins/cell-types in this dataset (9 for DS2)",
    )
    ap.add_argument(
        "--n_replicates", type=int, default=6,
        help="How many of the 15 pre-simulated replicate runs to use as "
             "timepoints. More = more density tiers available to test "
             "(proposal Tier 1 wants 6-21; use up to 15 here).",
    )
    ap.add_argument(
        "--n_timepoints_keep", type=int, default=None,
        help="If set, keep only this many replicate-timepoints out of "
             "n_replicates (density-tier ablation). Which ones are kept is "
             "controlled by --seed (a random subset, not fixed evenly-"
             "spaced positions) so that different seeds of the same tier "
             "see genuinely different data.",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Random seed controlling which replicate-timepoints "
             "--n_timepoints_keep subsamples. Vary this across repeated "
             "runs of the same tier for the multi-seed robustness "
             "experiment (prior sweeps showed Marlene is highly "
             "seed-dependent) -- otherwise every 'seed' of a tier would "
             "silently reuse identical data.",
    )
    ap.add_argument(
        "--out_dir", type=str, default="/kaggle/working/Marlene/data",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SERGIO dataset from {dataset_dir}")
    X, cell_type, timepoint, gene_ids, gt_edges = load_sergio_dataset(
        dataset_dir, n_bins=args.n_bins,
        replicate_ids=list(range(args.n_replicates)),
    )
    print(f"Total: {X.shape[0]} cells x {X.shape[1]} genes, "
          f"{len(gt_edges)} ground-truth edges, "
          f"{len(np.unique(cell_type))} cell types, "
          f"{len(np.unique(timepoint))} timepoints")

    gene_names = np.array([f"G{gid}" for gid in gene_ids])
    id_to_name = {gid: f"G{gid}" for gid in gene_ids}

    if args.n_timepoints_keep is not None:
        k = min(args.n_timepoints_keep, args.n_replicates)
        rng = np.random.RandomState(args.seed)
        keep_t = np.sort(rng.choice(args.n_replicates, size=k, replace=False))
        mask = np.isin(timepoint, keep_t)
        X, cell_type, timepoint = X[mask], cell_type[mask], timepoint[mask]
        remap = {old: new for new, old in enumerate(sorted(keep_t))}
        timepoint = np.array([remap[t] for t in timepoint])
        print(f"Subsampled to {len(keep_t)} timepoints (seed={args.seed}, "
              f"replicates {sorted(keep_t)})")

    regulators = set(r for r, _ in gt_edges)
    is_tf = np.array([gid in regulators for gid in gene_ids])
    print(f"{is_tf.sum()} / {len(gene_ids)} genes are regulators (TFs)")
    if is_tf.sum() < 30:
        print("WARNING: fewer than 30 TFs -- risk of dead attention "
              "(see module docstring history). Use a larger SERGIO dataset.")

    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "timepoint": timepoint.astype(str),
            "cell_type": np.array([f"CT{c}" for c in cell_type]),
        }),
        var=pd.DataFrame({"is_TF": is_tf}, index=gene_names),
    )

    # v4 FIX -- see module docstring: SERGIO bins carry very different
    # overall expression SCALE (different basal production rates per bin,
    # by design). Marlene's classification loss only reaches the model
    # through the attention-weighted reconstruction (x @ A.T), so feeding
    # it raw, unnormalized counts lets the network solve the pretext task
    # with a trivial "overall expression level" shortcut and never learn
    # real per-gene attention -- confirmed: this collapsed attention to
    # exactly zero everywhere (dead ReLU in the sparsification step) even
    # after hundreds of epochs. Every real dataset Marlene was validated
    # on (PBMC/HLCA) is normalized+log-transformed before Marlene sees it
    # (see train.py's preprocess()); this restores that same parity here.
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    h5ad_path = out_dir / "SERGIO-Marlene.h5ad"
    adata.write(h5ad_path)
    print(f"Wrote {h5ad_path}  ({adata.shape[0]} cells x {adata.shape[1]} genes)")

    gt_edges_named = [(id_to_name[r], id_to_name[t]) for r, t in gt_edges
                       if r in id_to_name and t in id_to_name]
    gt_path = out_dir / "sergio_gt_edges.csv"
    pd.DataFrame(gt_edges_named, columns=["regulator", "target"]).to_csv(
        gt_path, index=False
    )
    print(f"Wrote {gt_path}  ({len(gt_edges_named)} edges)")


if __name__ == "__main__":
    main()
