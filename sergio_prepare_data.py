"""
Converts a SERGIO-simulated dataset into the AnnData format Marlene's
(frozen, unmodified) scRNATimeSeriesDataset expects.

WHAT THIS DOES
---------------
- Loads one or more of SERGIO's pre-simulated expression CSVs
  (data_sets/De-noised_100G_9T_300cPerT_4_DS1/simulated_noNoise_*.csv).
- SERGIO's "bins" (originally meant as distinct cell TYPES) are relabeled
  as discrete TIMEPOINTS. This is a deliberate simplification, not an
  oversight: the proposal's primary axis under test is temporal density
  (RQ1/RQ3), not cell-type diversity, and SERGIO's steady-state mode does
  not natively produce multiple genuinely distinct cell types sharing one
  GRN.
- Cells are labeled by REPLICATE (>=2 required, see --n_replicates) as the
  "cell_type" fed to Marlene, NOT a single constant label. This is a
  compatibility requirement, not a stylistic choice: Marlene's MAML loop
  is trained entirely through a cell-type classification pretext loss
  (cross_entropy over `n_classes` logits), and its attention weights --
  which ARE the predicted GRN -- only receive gradient because the model
  must learn to tell classes apart. With a single constant label
  (n_classes=1), softmax over one logit is identically 1.0 for any input,
  so the loss and its gradient are exactly zero for the entire run --
  the model never leaves its random initialization, silently. Labeling by
  replicate (independent random seeds, IDENTICAL ground-truth GRN) avoids
  reintroducing a cell-type-diversity confound into the true edges used
  for evaluation, while still giving MAML >=2 classes to learn from.
  Caveat: what separates replicates is simulation noise, not biology, so
  this pretext signal is weaker than a real cell-type distinction would
  be -- more replicates (and/or more cells per replicate) strengthen it.
- The ground-truth GRN (data_sets/.../gt_GRN.csv) is used directly to
  build the TF mask and the true-edge list for evaluation, replacing
  Marlene's original TRRUST/RegNetwork lookups (those are irrelevant here
  since SERGIO's gene IDs are arbitrary integers, not real gene symbols).

OUTPUT
------
- data/SERGIO-Marlene.h5ad          <- adata for training
- data/sergio_gt_edges.csv          <- ground-truth (regulator, target) edges
"""
import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


def load_sergio_dataset(
    dataset_dir: Path,
    n_bins: int,
    replicate_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Load and concatenate SERGIO replicate simulations.

    Returns
    -------
    X: (n_cells, n_genes) expression matrix
    timepoint: (n_cells,) integer array, 0..n_bins-1
    replicate: (n_cells,) integer array identifying which replicate a cell
        came from. Every replicate shares the exact same ground-truth GRN
        (same simulator config, different random seed) -- this is used as
        a pseudo cell-type label so Marlene's classification pretext task
        has >=2 classes to learn from (see main()'s docstring note).
    gene_ids: (n_genes,) original SERGIO gene indices, in column order of X
    gt_edges: list of (regulator_gene_id, target_gene_id)
    """
    gt_path = dataset_dir / "gt_GRN.csv"
    gt_df = pd.read_csv(gt_path, header=None, names=["reg", "target"])
    gt_edges = list(zip(gt_df["reg"].astype(int), gt_df["target"].astype(int)))

    X_list, T_list, R_list = [], [], []
    gene_ids = None

    for rep in replicate_ids:
        csv_path = dataset_dir / f"simulated_noNoise_{rep}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path.name} not found")
            continue

        raw = pd.read_csv(csv_path, header=None)
        # raw layout: row0 = cell index header (ignored), col0 = gene index
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
        timepoint = np.repeat(np.arange(n_bins), cells_per_bin)

        X_list.append(expr.T)  # -> (n_cells, n_genes)
        T_list.append(timepoint)
        R_list.append(np.full(n_cells_total, rep, dtype=int))
        print(f"  loaded {csv_path.name}: {expr.shape[1]} cells x {n_genes} genes")

    X = np.concatenate(X_list, axis=0)
    timepoint = np.concatenate(T_list, axis=0)
    replicate = np.concatenate(R_list, axis=0)
    return X, timepoint, replicate, gene_ids, gt_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1",
        help="Path to a SERGIO bundled dataset folder (must contain "
             "gt_GRN.csv and simulated_noNoise_*.csv)",
    )
    ap.add_argument(
        "--n_bins", type=int, default=9,
        help="Number of SERGIO bins in this dataset (9 for the DS1 dataset)",
    )
    ap.add_argument(
        "--n_replicates", type=int, default=2,
        help="How many of the 15 pre-simulated replicate runs to use "
             "(more replicates = more cells, slower). Must be >=2: each "
             "replicate is used as a pseudo cell-type label so Marlene's "
             "classification pretext loss has >=2 classes to learn from "
             "(n_classes=1 gives an identically-zero loss/gradient -- see "
             "module docstring).",
    )
    ap.add_argument(
        "--n_timepoints_keep", type=int, default=None,
        help="If set, keep only this many EVENLY-SPACED timepoints out of "
             "n_bins (e.g. 3 for an ultra-sparse Tier-3 run). Default: keep all.",
    )
    ap.add_argument(
        "--out_dir", type=str, default="/kaggle/working/Marlene/data",
    )
    args = ap.parse_args()

    if args.n_replicates < 2:
        raise ValueError(
            f"--n_replicates={args.n_replicates} but must be >=2: Marlene's "
            "MAML loop needs >=2 cell-type classes to get a nonzero "
            "classification loss/gradient (see module docstring)."
        )

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SERGIO dataset from {dataset_dir}")
    X, timepoint, replicate, gene_ids, gt_edges = load_sergio_dataset(
        dataset_dir, n_bins=args.n_bins,
        replicate_ids=list(range(args.n_replicates)),
    )
    print(f"Total: {X.shape[0]} cells x {X.shape[1]} genes, "
          f"{len(gt_edges)} ground-truth edges, "
          f"{len(np.unique(replicate))} replicate pseudo-classes")

    # gene symbols = "G{sergio_gene_id}", zero-padded for readable sorting
    gene_names = np.array([f"G{gid}" for gid in gene_ids])
    id_to_name = {gid: f"G{gid}" for gid in gene_ids}

    # optionally sparsify timepoints (temporal-density tiers, proposal 5.5)
    if args.n_timepoints_keep is not None:
        keep_bins = np.unique(np.linspace(
            0, args.n_bins - 1, args.n_timepoints_keep
        ).round().astype(int))
        mask = np.isin(timepoint, keep_bins)
        X, timepoint, replicate = X[mask], timepoint[mask], replicate[mask]
        # re-map kept bin ids to consecutive 0..k-1 for cleanliness
        remap = {old: new for new, old in enumerate(sorted(keep_bins))}
        timepoint = np.array([remap[t] for t in timepoint])
        print(f"Subsampled to {len(keep_bins)} timepoints (bins {sorted(keep_bins)})")

    regulators = set(r for r, _ in gt_edges)
    is_tf = np.array([gid in regulators for gid in gene_ids])
    print(f"{is_tf.sum()} / {len(gene_ids)} genes are regulators (TFs)")

    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({
            "timepoint": timepoint.astype(str),
            # replicate id used as pseudo cell-type label, see module docstring
            "cell_type": np.array([f"rep{r}" for r in replicate]),
        }),
        var=pd.DataFrame({"is_TF": is_tf}, index=gene_names),
    )

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
