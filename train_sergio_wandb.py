"""
Marlene training on SERGIO synthetic data.

FROZEN (imported as-is from the original repo, never modified):
    - marlene.models.marlene.Marlene   (architecture)
    - marlene.maml_ct.Meta             (MAML meta-learning training loop)
    - marlene.datasets.scRNATimeSeriesDataset (episodic batching; this class
      is dataset-agnostic -- it only reads adata.obs[timepoint/celltype
      keys] and adata.X, so it needs no changes for SERGIO data)
    - marlene.utils.evaluation.score   (hypergeometric overlap scoring;
      generic over any true_links set, reused as-is)

REWRITTEN for this project (not present in the original repo):
    - load_data(): reads our SERGIO-derived h5ad + gt edge list instead of
      TRRUST/RegNetwork
    - AUPRC/AUROC evaluation: the proposal (Section 5.7) requires AUPRC as
      the primary metric; the original repo only reported hypergeometric
      p-values, so this is added on top, not a replacement
    - Optional Weights & Biases logging: mirrors train.py's use of
      wandb.init/wandb.watch/wandb.log for the training curve, and on top
      of that also logs the final AUPRC/AUROC-per-timepoint eval curve
      (train.py had no quantitative metric worth logging at that point).
      Unlike train.py, wandb is imported lazily (only if --use_wandb is
      passed) so this script has no hard dependency on it.

USAGE
-----
python train_sergio.py \
    --data_dir /kaggle/working/Marlene/data \
    --runid marlene_sergio_dense \
    --n_epochs 500 --device cuda \
    --use_wandb --wandb_project Marlene-SERGIO
"""
import argparse
import copy
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import Adam
from tqdm.auto import tqdm

# numpy>=2.0 removed np.in1d. marlene/datasets.py itself only calls
# np.isin (already numpy>=2.0-safe), but this shim guards against older
# transitive dependencies (pandas/scikit-learn/anndata versions) that may
# still call the removed alias internally. np.isin is a drop-in,
# functionally identical replacement, so this is a compatibility fix, not
# a behavior change.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# --- FROZEN architecture imports, do not modify these three files ---------
from marlene.datasets import scRNATimeSeriesDataset
from marlene.maml_ct import Meta
from marlene.models.marlene import Marlene
from marlene.utils.evaluation import score
# ---------------------------------------------------------------------------


def load_data(data_dir: Path):
    """Load our SERGIO-converted h5ad + its ground-truth edge list.

    Replaces the original load_data(), which called load_trrust() /
    load_regnetwork(). Those are irrelevant for synthetic data with
    arbitrary integer gene IDs -- we already have the true GRN directly
    from the simulator.
    """
    adata = anndata.read_h5ad(data_dir / "SERGIO-Marlene.h5ad")
    gt_df = pd.read_csv(data_dir / "sergio_gt_edges.csv")
    gt_edges = set(zip(gt_df["regulator"], gt_df["target"]))

    n_total_links = adata.shape[1] * adata.var["is_TF"].sum()
    print(f"(cells, genes) = {adata.shape}")
    print(f"n_total_links = {n_total_links}")
    print(f"n ground-truth edges = {len(gt_edges)}")

    return {"adata": adata, "gt_edges": gt_edges, "n_total_links": n_total_links}


def loss_function(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)


def predict_celltype(adata, celltype, dataset, predictor, n_draws=50, quantile=0.98):
    """Unchanged in logic from the original train.py -- extracts predicted
    edges by averaging attention matrices over random cell batches."""
    predictor.eval()
    tfs = adata.var_names[adata.var["is_TF"]].to_numpy()
    targets = adata.var_names.to_numpy()

    A_seq = np.zeros((dataset.n_timepoints, len(targets), len(tfs)))
    predictor.to(dataset.device)

    with torch.no_grad():
        for _ in range(n_draws):
            x = dataset.sample_celltype_batch(celltype)
            attn = predictor(x, attn_only=True).detach().cpu().numpy()
            A_seq += attn
    A_seq /= n_draws

    q_seq = [np.quantile(A, quantile) for A in A_seq]
    preds = []
    for A, q in zip(A_seq, q_seq):
        a, b = np.argwhere(A > q).T
        links = list(zip(tfs[b], targets[a]))
        preds.append({"links": links, "attention": A[a, b], "A_full": A})
    return preds


def compute_auprc_auroc(A_full, tfs, targets, gt_edges):
    """New: primary quantitative metric per proposal Section 5.7.
    Scores the CONTINUOUS attention matrix against the true edge set,
    rather than only a thresholded overlap p-value.
    """
    tf_idx = {g: i for i, g in enumerate(tfs)}
    y_true, y_score = [], []
    for ti, target in enumerate(targets):
        for tf, tf_i in tf_idx.items():
            y_true.append(1 if (tf, target) in gt_edges else 0)
            y_score.append(A_full[ti, tf_i])
    y_true, y_score = np.array(y_true), np.array(y_score)
    if y_true.sum() == 0:
        return float("nan"), float("nan")
    return average_precision_score(y_true, y_score), roc_auc_score(y_true, y_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/kaggle/working/Marlene/data")
    ap.add_argument("--runid", type=str, default="marlene_sergio_run")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--inner_lr", type=float, default=1e-2)
    ap.add_argument("--update_step", type=int, default=5)
    ap.add_argument("--gradient_clip", type=float, default=0.1)
    ap.add_argument("--n_seeds", type=int, default=16)
    ap.add_argument("--frac_top_edges", type=float, default=0.02)
    ap.add_argument("--n_draws", type=int, default=50)
    ap.add_argument("--use_wandb", action="store_true",
                     help="Log training curve + eval metrics to Weights & Biases")
    ap.add_argument("--wandb_project", type=str, default="Marlene-SERGIO")
    ap.add_argument("--wandb_entity", type=str, default=None)
    args = ap.parse_args()

    if args.use_wandb:
        import wandb

    data_dir = Path(args.data_dir)
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"CUDA not available, falling back to {device}")

    print(f"Initializing run '{args.runid}'")
    data_dict = load_data(data_dir)
    adata, gt_edges = data_dict["adata"], data_dict["gt_edges"]

    dataset = scRNATimeSeriesDataset(
        adata,
        timepoint_key="timepoint",
        celltype_key="cell_type",
        timepoint_order=sorted(adata.obs["timepoint"].unique(), key=int),
        batch_size=args.batch_size,
        device=device,
    )
    print(f"n_timepoints={dataset.n_timepoints}, n_classes={dataset.n_classes} "
          f"(pseudo cell-type = SERGIO replicate id, see sergio_prepare_data.py)")
    if dataset.n_classes < 2:
        raise ValueError(
            f"n_classes={dataset.n_classes}, but Marlene's MAML loop is "
            "trained entirely through a cell-type classification loss: "
            "with a single class, cross_entropy is identically 0 everywhere "
            "(softmax over one logit is always 1.0), so gradients are "
            "exactly zero and no training happens. Regenerate the h5ad "
            "with `sergio_prepare_data.py --n_replicates >=2`."
        )

    model = Marlene(
        n_genes=dataset.n_genes,
        n_classes=dataset.n_classes,
        TF_mask=adata.var["is_TF"].to_numpy(),
        n_seeds=args.n_seeds,
        sparse_q=1 - args.frac_top_edges,
    ).train()
    if device == "cuda":
        model.cuda()

    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.runid, config=vars(args))
        wandb.watch(model, log="all", log_freq=100)

    meta_optimizer = Adam(model.parameters(), lr=args.lr)
    meta = Meta(
        model, meta_optimizer,
        update_lr=args.inner_lr,
        update_step=args.update_step,
        loss=loss_function,
        gradient_clip=args.gradient_clip,
    )

    best_model, best_loss = None, 1e10
    bar = tqdm(range(args.n_epochs))
    try:
        for epoch in bar:
            batch = dataset.sample_meta_batch()
            epoch_loss, epoch_acc = meta(
                batch["support"], batch["label"], batch["query"], batch["label"],
            )
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.cpu())
                if device == "cuda":
                    model.cuda()
            bar.set_description(f"epoch={epoch}, loss={epoch_loss:.4g}, acc={epoch_acc:.3g}")
            if args.use_wandb:
                wandb.log({
                    "loss": epoch_loss,
                    "accuracy": epoch_acc,
                    "lr": meta_optimizer.param_groups[0]["lr"],
                })
    except KeyboardInterrupt:
        pass

    # --- Evaluation ---------------------------------------------------
    tfs = adata.var_names[adata.var["is_TF"]].to_numpy()
    targets = adata.var_names.to_numpy()

    y_pred_seq, auprc_per_t, auroc_per_t = {}, [], []
    eval_rows = []
    for celltype in dataset.unq_celltype:
        preds = predict_celltype(
            adata, celltype, dataset, best_model,
            n_draws=args.n_draws, quantile=1 - args.frac_top_edges,
        )
        y_pred_seq[celltype] = [
            {"links": p["links"], "attention": p["attention"]} for p in preds
        ]
        for t, p in enumerate(preds):
            auprc, auroc = compute_auprc_auroc(p["A_full"], tfs, targets, gt_edges)
            auprc_per_t.append(auprc)
            auroc_per_t.append(auroc)
            eval_rows.append([celltype, t, auprc, auroc])
            print(f"  celltype={celltype} t={t}: AUPRC={auprc:.4f} AUROC={auroc:.4f}")

    mean_auprc, mean_auroc = np.nanmean(auprc_per_t), np.nanmean(auroc_per_t)
    print(f"\nMean AUPRC across timepoints: {mean_auprc:.4f}")
    print(f"Mean AUROC across timepoints: {mean_auroc:.4f}")

    # original hypergeometric-overlap metric, kept for continuity with the
    # authors' own reported statistic
    n_total_links = data_dict["n_total_links"]
    scores_df = score(y_pred_seq, gt_edges, n_total_links=n_total_links)
    mean_significant = (scores_df["p-val"] < 0.05).mean()
    mean_overlap = scores_df["N. overlap"].mean()
    print("Mean significant (p<0.05):", mean_significant)
    print("Mean overlap:", mean_overlap)

    if args.use_wandb:
        wandb.log({
            "eval/auprc_auroc_per_timepoint": wandb.Table(
                columns=["cell_type", "t", "auprc", "auroc"], data=eval_rows,
            ),
        })
        wandb.summary["mean_auprc"] = mean_auprc
        wandb.summary["mean_auroc"] = mean_auroc
        wandb.summary["mean_significant"] = mean_significant
        wandb.summary["mean_overlap"] = mean_overlap

    out_dir = Path("Marlene_results") / "SERGIO" / args.runid
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), out_dir / "best_model.ckpt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_auprc": float(np.nanmean(auprc_per_t)),
            "mean_auroc": float(np.nanmean(auroc_per_t)),
            "auprc_per_t": [float(x) for x in auprc_per_t],
            "auroc_per_t": [float(x) for x in auroc_per_t],
            "n_timepoints": dataset.n_timepoints,
        }, f, indent=2)
    print(f"\nSaved checkpoint + metrics to {out_dir}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
