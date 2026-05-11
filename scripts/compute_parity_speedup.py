"""
compute_parity_speedup.py
--------------------------
Post-hoc helper that fetches the single-GPU and DDP-4 University-1652 runs
from wandb and writes `speedup_vs_single_gpu` plus a side-by-side parity
table back to each run's summary.

Usage:
    python scripts/compute_parity_speedup.py \
        --single-run w2c-lab/hcvgloc-distributed/cj4ntmp6 \
        --ddp-run    w2c-lab/hcvgloc-distributed/<run_id>

Both runs must already have `total_wall_clock_sec` in their summary
(written by tools/train_stage1.py at end-of-training).
"""

import argparse
import json
from pathlib import Path

import wandb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single-run", required=True,
                   help="entity/project/run_id for the single-GPU run")
    p.add_argument("--ddp-run", required=True,
                   help="entity/project/run_id for the DDP-4 run")
    p.add_argument("--output", type=str,
                   default="results/tables/convergence_parity.json",
                   help="local table written alongside the wandb summary updates")
    return p.parse_args()


def _fetch(api, path):
    run = api.run(path)
    s = dict(run.summary)
    keys = [
        "total_wall_clock_sec", "final_R@1", "final_R@5", "final_R@10",
        "best_R@1", "epoch_to_R1_gt_50", "peak_gpu_mem_MB",
    ]
    return run, {k: s.get(k) for k in keys}


def main():
    args = parse_args()
    api = wandb.Api()

    single_run, single = _fetch(api, args.single_run)
    ddp_run,    ddp    = _fetch(api, args.ddp_run)

    sg_sec  = single.get("total_wall_clock_sec")
    ddp_sec = ddp.get("total_wall_clock_sec")
    speedup = (sg_sec / ddp_sec) if (sg_sec and ddp_sec) else None

    # Update both runs' summaries with the cross-run number so it's
    # discoverable from either dashboard.
    for r in (single_run, ddp_run):
        if speedup is not None:
            r.summary["speedup_vs_single_gpu"] = round(speedup, 3)
        r.summary.update()

    table = {
        "single_gpu": {"path": args.single_run, **single},
        "ddp4":       {"path": args.ddp_run,    **ddp},
        "speedup_vs_single_gpu": speedup,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(table, f, indent=2)

    print("Convergence-parity comparison")
    print("-" * 50)
    print(f"  single-GPU wall-clock : {sg_sec:.1f}s ({sg_sec/3600:.2f} h)")
    print(f"  DDP-4    wall-clock   : {ddp_sec:.1f}s ({ddp_sec/3600:.2f} h)")
    print(f"  speedup_vs_single_gpu : {speedup:.2f}×" if speedup else "  speedup: n/a")
    print()
    print(f"  single-GPU final R@1  : {single.get('final_R@1')}")
    print(f"  DDP-4    final R@1    : {ddp.get('final_R@1')}")
    print(f"  single-GPU best R@1   : {single.get('best_R@1')}")
    print(f"  DDP-4    best R@1     : {ddp.get('best_R@1')}")
    print()
    print(f"  Table written: {out}")


if __name__ == "__main__":
    main()
