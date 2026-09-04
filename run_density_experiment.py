"""
Orchestrates the temporal-density sparsity experiment (proposal Section 5.5:
Controlled variable: temporal density) across three density tiers x N_SEEDS
random seeds, by calling sergio_prepare_data.py + train_sergio.py as
subprocesses for each (tier, seed) combination.

Neither sergio_prepare_data.py nor train_sergio.py is modified by this
script -- it only drives them with the right flags and collects results.

WHY MULTIPLE SEEDS
------------------
Prior experiments on this exact setup showed Marlene's results are highly
seed-dependent -- a single training run is not trustworthy evidence for a
tier's performance. Every (tier, seed) combination below produces its own
independent metrics.json under Marlene_results/SERGIO/tier<N>_seed<S>/;
aggregate_density_results.py then reports mean +/- std across seeds per
tier, never a single run.

Each seed gets genuinely different data (not just a different training
run): sergio_prepare_data.py's --seed controls which replicate-timepoints
are randomly kept for --n_timepoints_keep < --n_replicates (see its v5
docstring note), and train_sergio.py's --seed seeds torch/numpy so weight
init and batch sampling differ (and are reproducible) across seeds too.

RESUMABILITY
------------
Before running a (tier, seed) combo, this script checks whether
Marlene_results/SERGIO/tier<N>_seed<S>/metrics.json already exists and
skips it if so. This means an interrupted sweep (e.g. a Kaggle session
timeout) can simply be re-run from the top -- already-completed combos
are skipped, incomplete/not-yet-run ones proceed.

USAGE
-----
Run from the Marlene/ repo root (relative paths for data dirs and results
are resolved against the current working directory, matching where
train_sergio.py itself writes Marlene_results/SERGIO/<runid>/):

    python run_density_experiment.py --device cuda

Sanity-check the sweep plan (prints every command that would run, touches
nothing) before committing GPU time:

    python run_density_experiment.py --dry_run

Progress (which tier/seed is running, elapsed time, failures) is both
printed and appended to --log_file (default density_experiment_log.txt),
so progress survives a Kaggle session getting cut off mid-sweep.
"""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_SCRIPT = SCRIPT_DIR / "sergio_prepare_data.py"
TRAIN_SCRIPT = SCRIPT_DIR / "train_sergio.py"

# Proposal Section 5.5 temporal-density tiers -- n_timepoints_keep values
# out of the 15 pre-simulated SERGIO replicates loaded as --n_replicates.
TIERS = {
    1: 15,  # Tier 1 -- dense (all available replicates)
    2: 5,   # Tier 2 -- sparse
    3: 3,   # Tier 3 -- ultra-sparse
}

N_SEEDS = 3  # prior sweeps showed high seed-dependence -- see module docstring


def log(msg: str, log_file: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def run_prepare(*, seed, n_timepoints_keep, out_dir, dataset_dir, n_bins,
                 n_replicates, dry_run, log_file):
    cmd = [
        sys.executable, str(PREPARE_SCRIPT),
        "--dataset_dir", dataset_dir,
        "--n_bins", str(n_bins),
        "--n_replicates", str(n_replicates),
        "--n_timepoints_keep", str(n_timepoints_keep),
        "--seed", str(seed),
        "--out_dir", str(out_dir),
    ]
    log(f"  [prepare] {' '.join(cmd)}", log_file)
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_train(*, data_dir, runid, frac_top_edges, n_epochs, device, seed,
              dry_run, log_file):
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data_dir", str(data_dir),
        "--runid", runid,
        "--frac_top_edges", str(frac_top_edges),
        "--n_epochs", str(n_epochs),
        "--device", device,
        "--seed", str(seed),
    ]
    log(f"  [train]   {' '.join(cmd)}", log_file)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset_dir", type=str,
        default="/kaggle/working/SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2",
        help="SERGIO bundled dataset folder, forwarded to sergio_prepare_data.py "
             "(400-gene/37-TF dataset, per the proposal's Kaggle-feasibility "
             "gene-set-size guidance).",
    )
    ap.add_argument("--n_bins", type=int, default=9)
    ap.add_argument("--n_replicates", type=int, default=15,
                     help="All 15 pre-simulated replicates are loaded; tiers "
                          "then subsample down to n_timepoints_keep of them.")
    ap.add_argument("--frac_top_edges", type=float, default=0.05,
                     help="Best-performing sparsification threshold found in "
                          "prior hyperparameter sweeps on this dataset.")
    ap.add_argument("--n_epochs", type=int, default=500)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--data_root", type=str, default=".",
                     help="Where per-(tier,seed) data_tier<N>_seed<S>/ dirs "
                          "are created.")
    ap.add_argument(
        "--results_root", type=str, default="Marlene_results/SERGIO",
        help="Where train_sergio.py writes <runid>/metrics.json. Must match "
             "train_sergio.py's own hardcoded relative "
             "Marlene_results/SERGIO/<runid> layout -- so run this script "
             "from the same working directory every time (the repo root).",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print every command that would run for every (tier, seed) "
             "combo, without executing them or touching the filesystem. "
             "Use this to sanity-check the sweep plan before committing "
             "~2.7 hours of GPU time.",
    )
    ap.add_argument("--log_file", type=str, default="density_experiment_log.txt")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    log_file = Path(args.log_file)

    combos = [
        (tier, n_tp, seed)
        for tier, n_tp in TIERS.items()
        for seed in range(args.n_seeds)
    ]

    log(f"Starting density sweep: {len(TIERS)} tiers x {args.n_seeds} seeds "
        f"= {len(combos)} runs. dry_run={args.dry_run}", log_file)

    n_done = n_skipped = n_failed = 0
    failed_combos = []
    t_sweep_start = time.time()

    for tier, n_tp, seed in combos:
        runid = f"tier{tier}_seed{seed}"
        data_dir = data_root / f"data_tier{tier}_seed{seed}"
        metrics_path = results_root / runid / "metrics.json"

        log(f"=== {runid} (n_timepoints_keep={n_tp}) ===", log_file)

        if metrics_path.exists() and not args.dry_run:
            log(f"  SKIP -- {metrics_path} already exists (resuming)", log_file)
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            run_prepare(
                seed=seed, n_timepoints_keep=n_tp, out_dir=data_dir,
                dataset_dir=args.dataset_dir, n_bins=args.n_bins,
                n_replicates=args.n_replicates,
                dry_run=args.dry_run, log_file=log_file,
            )
            run_train(
                data_dir=data_dir, runid=runid,
                frac_top_edges=args.frac_top_edges, n_epochs=args.n_epochs,
                device=args.device, seed=seed,
                dry_run=args.dry_run, log_file=log_file,
            )
        except Exception:
            elapsed = time.time() - t0
            log(f"  FAILED after {elapsed / 60:.1f} min:\n{traceback.format_exc()}",
                log_file)
            n_failed += 1
            failed_combos.append(runid)
            continue

        elapsed = time.time() - t0
        log(f"  done in {elapsed / 60:.1f} min", log_file)
        n_done += 1

    total_elapsed = time.time() - t_sweep_start
    log(f"\nSweep finished in {total_elapsed / 60:.1f} min. "
        f"done={n_done} skipped={n_skipped} failed={n_failed} "
        f"(of {len(combos)} total)", log_file)
    if failed_combos:
        log(f"WARNING: failed combo(s): {failed_combos} -- see {log_file} "
            f"for tracebacks. Re-running this script will retry them "
            f"(only combos with an existing metrics.json are skipped).",
            log_file)


if __name__ == "__main__":
    main()
