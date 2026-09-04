"""
Aggregates the 3-tier x N_SEEDS results produced by run_density_experiment.py
into a summary table, a raw per-run table, and a publication-ready
degradation-curve figure for the proposal's RQ1/RQ3 (Sections 5.5/5.7).

INPUTS
------
Walks <results_root>/tier<N>_seed<S>/metrics.json (default results_root:
Marlene_results/SERGIO, matching train_sergio.py's own output layout).

OUTPUTS (written under --out_prefix, default ".")
--------------------------------------------------
- density_results_raw.csv:      one row per individual run
                                 (tier, seed, n_timepoints, mean_auprc, mean_auroc)
- density_results_summary.csv:  one row per tier
                                 (tier, n_timepoints, mean_auprc_mean,
                                  mean_auprc_std, mean_auroc_mean,
                                  mean_auroc_std, n_seeds)
- density_degradation_curve.png (300 DPI) and .pdf (vector): the central
  RQ1/RQ3 figure -- AUPRC/AUROC mean +/- std (across seeds) vs. number of
  timepoints, with random-baseline reference lines.
- A plain-text console summary assessing, per tier, whether mean_auprc /
  mean_auroc is meaningfully above the random baseline given the
  across-seed std (never overclaiming when the baseline falls within the
  std band).

BASELINE CONSTANTS
-------------------
AUPRC baseline = true-edge prevalence = n_ground_truth_edges / n_total_links
               = 1155 / 14800 = 0.078 for this 400-gene/37-TF SERGIO dataset
               (confirmed against actual run logs: n_total_links=14800,
               n ground-truth edges=1155).
AUROC baseline = 0.5 (chance ranking), always.
metrics.json does not itself store n_total_links/n_ground_truth_edges, so
these are hardcoded here rather than recomputed per run -- valid as long as
every run in the sweep uses the same SERGIO dataset (it does, by design:
run_density_experiment.py holds --dataset_dir fixed across all tiers/seeds).

STD CONVENTION
--------------
Sample standard deviation (ddof=1, pandas' .std() default) is used, matching
the convention already used in prior reporting for this project (e.g. the
frac_top_edges=0.05 finding: "mean AUROC=0.506+/-0.025").
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIER_SHORT = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
TIER_FULL = {
    1: "Tier 1 (dense)",
    2: "Tier 2 (sparse)",
    3: "Tier 3 (ultra-sparse)",
}

# See module docstring "BASELINE CONSTANTS".
N_GROUND_TRUTH_EDGES = 1155
N_TOTAL_LINKS = 14800
BASELINE_AUPRC = N_GROUND_TRUTH_EDGES / N_TOTAL_LINKS
BASELINE_AUROC = 0.5

RUNID_RE = re.compile(r"^tier(\d+)_seed(\d+)$")


def find_runs(results_root: Path):
    """Yields (tier, seed, metrics_dict) for every tier*_seed*/metrics.json found."""
    for metrics_path in sorted(results_root.glob("tier*_seed*/metrics.json")):
        m = RUNID_RE.match(metrics_path.parent.name)
        if not m:
            continue
        tier, seed = int(m.group(1)), int(m.group(2))
        with open(metrics_path) as f:
            metrics = json.load(f)
        yield tier, seed, metrics


def assess_vs_baseline(mean: float, std: float, baseline: float, metric_name: str) -> str:
    """Honest above/below/indistinguishable-from-baseline assessment.

    Mirrors the framing already used for the frac_top_edges=0.05 finding:
    "mean AUROC=0.506+/-0.025 was NOT distinguishable from the 0.5 baseline
    given that variance" -- never overclaim just because mean > baseline.
    """
    diff = mean - baseline
    if std == 0 or np.isnan(std):
        return (f"{metric_name}: mean={mean:.4f} (baseline={baseline:.4f}) -- "
                f"only one seed available, cannot assess significance")
    if abs(diff) <= std:
        return (f"{metric_name}: mean={mean:.4f} +/- {std:.4f} was NOT "
                f"distinguishable from the {baseline:.4f} baseline given "
                f"that variance")
    direction = "above" if diff > 0 else "below"
    return (f"{metric_name}: mean={mean:.4f} +/- {std:.4f} is meaningfully "
            f"{direction} the {baseline:.4f} baseline "
            f"(delta={diff:+.4f}, {abs(diff) / std:.1f}x std)")


def make_figure(summary_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            plt.style.use("ggplot")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    # Okabe-Ito colorblind-safe palette.
    color_auprc = "#0072B2"  # blue
    color_auroc = "#D55E00"  # vermillion

    df = summary_df.sort_values("n_timepoints").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        df["n_timepoints"], df["mean_auprc_mean"], yerr=df["mean_auprc_std"],
        marker="o", markersize=7, color=color_auprc, linewidth=2,
        capsize=5, capthick=1.5, label="AUPRC (mean ± std across seeds)",
    )
    ax.errorbar(
        df["n_timepoints"], df["mean_auroc_mean"], yerr=df["mean_auroc_std"],
        marker="s", markersize=7, color=color_auroc, linewidth=2,
        capsize=5, capthick=1.5, label="AUROC (mean ± std across seeds)",
    )

    ax.axhline(BASELINE_AUPRC, color="gray", linestyle="--", linewidth=1.3,
               label="Random baseline (AUPRC)")
    ax.axhline(BASELINE_AUROC, color="gray", linestyle=":", linewidth=1.3,
               label="Random baseline (AUROC)")

    ax.set_xlabel("Number of Timepoints", fontsize=11)
    ax.set_ylabel("Score (AUPRC / AUROC)", fontsize=11)
    ax.set_title(
        "Marlene Performance Degradation Under Temporal Sparsity\n"
        "(SERGIO Synthetic Data)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)

    xtick_labels = [f"{n} ({t})" for n, t in zip(df["n_timepoints"], df["tier"])]
    ax.set_xticks(df["n_timepoints"].tolist())
    ax.set_xticklabels(xtick_labels)

    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()

    png_path = out_dir / "density_degradation_curve.png"
    pdf_path = out_dir / "density_degradation_curve.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results_root", type=str, default="Marlene_results/SERGIO")
    ap.add_argument("--out_prefix", type=str, default=".",
                     help="Directory to write summary/raw CSVs and the figure into.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "tier": tier,
            "seed": seed,
            "n_timepoints": metrics.get("n_timepoints"),
            "mean_auprc": metrics["mean_auprc"],
            "mean_auroc": metrics["mean_auroc"],
        }
        for tier, seed, metrics in find_runs(results_root)
    ]

    if not rows:
        print(f"No metrics.json files found under {results_root}/tier*_seed*/. "
              f"Nothing to aggregate -- run run_density_experiment.py first.")
        sys.exit(1)

    raw_df = pd.DataFrame(rows).sort_values(["tier", "seed"]).reset_index(drop=True)
    raw_path = out_dir / "density_results_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path} ({len(raw_df)} rows)")

    expected_tiers = set(TIER_SHORT)
    found_tiers = set(raw_df["tier"].unique())
    missing_tiers = expected_tiers - found_tiers
    if missing_tiers:
        print(f"WARNING: no runs found at all for tier(s): {sorted(missing_tiers)}")

    summary_rows = []
    for tier in sorted(found_tiers):
        sub = raw_df[raw_df["tier"] == tier]
        n_seeds_found = len(sub)
        n_tp_values = sub["n_timepoints"].unique()
        if len(n_tp_values) > 1:
            print(f"WARNING: tier {tier} has inconsistent n_timepoints across "
                  f"seeds: {n_tp_values} -- using the first value")
        std_auprc = sub["mean_auprc"].std() if n_seeds_found > 1 else 0.0  # ddof=1
        std_auroc = sub["mean_auroc"].std() if n_seeds_found > 1 else 0.0
        summary_rows.append({
            "tier": TIER_SHORT[tier],
            "n_timepoints": int(n_tp_values[0]),
            "mean_auprc_mean": sub["mean_auprc"].mean(),
            "mean_auprc_std": std_auprc,
            "mean_auroc_mean": sub["mean_auroc"].mean(),
            "mean_auroc_std": std_auroc,
            "n_seeds": n_seeds_found,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("n_timepoints").reset_index(drop=True)
    summary_path = out_dir / "density_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print("\n=== Density-tier summary vs. random baseline ===")
    for _, row in summary_df.iterrows():
        print(f"\n{row['tier']} (n_timepoints={row['n_timepoints']}, "
              f"n_seeds={row['n_seeds']}):")
        print("  " + assess_vs_baseline(
            row["mean_auprc_mean"], row["mean_auprc_std"], BASELINE_AUPRC, "AUPRC"))
        print("  " + assess_vs_baseline(
            row["mean_auroc_mean"], row["mean_auroc_std"], BASELINE_AUROC, "AUROC"))
        if row["n_seeds"] < 3:
            print(f"  NOTE: only {row['n_seeds']}/3 seeds completed for this "
                  f"tier -- treat this std as provisional")

    make_figure(summary_df, out_dir)


if __name__ == "__main__":
    main()
