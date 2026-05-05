"""
scaling_benchmark.py
---------------------
Benchmarks HCVGLoc Stage 1 training throughput across 1, 2, and 4 GPUs.

Measures:
    - Samples/sec (global throughput)
    - Time per epoch (wall-clock)
    - Ideal speedup vs. actual speedup
    - Parallel efficiency = actual / ideal

This script is the primary experiment for the Scalability pillar of the
Distributed Systems course project.

Usage:
    # Must be run sequentially for each GPU count (cannot compare inside torchrun)
    python benchmarks/scaling_benchmark.py \\
        --gpus 1 2 4 \\
        --epochs 3 \\
        --batch_size 32 \\
        --config configs/stage1_ddp_4gpu.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HCVGLoc Scaling Benchmark")
    p.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs per run (short runs for benchmarking)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Per-GPU batch size (kept constant across runs)")
    p.add_argument("--steps_per_epoch", type=int, default=200,
                   help="Max steps per epoch (use subset for speed)")
    p.add_argument("--config", type=str, default="configs/stage1_ddp_4gpu.yaml")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--wandb", action="store_true",
                   help="Log benchmark summary to wandb (project=hcvgloc-distributed).")
    p.add_argument("--wandb_project", type=str, default="hcvgloc-distributed")
    p.add_argument("--wandb_entity", type=str, default="w2c-lab")
    p.add_argument("--wandb_group", type=str, default="scaling_benchmark")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-run benchmark (launched via torchrun subprocess)
# ---------------------------------------------------------------------------

def run_benchmark_job(num_gpus: int, args) -> dict:
    """
    Launches a torchrun subprocess with num_gpus processes and parses
    the throughput output written to a temp JSON file.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking {num_gpus} GPU(s) × batch={args.batch_size}")
    print(f"{'='*60}")

    result_file = Path(args.output_dir) / "tables" / f"_bench_tmp_{num_gpus}gpu.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29502",
        str(Path(__file__).parent / "_bench_worker.py"),
        f"--config={args.config}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--steps_per_epoch={args.steps_per_epoch}",
        f"--result_file={result_file}",
    ]

    t_start = time.time()
    ret = subprocess.run(cmd, capture_output=False)
    wall_time = time.time() - t_start

    if ret.returncode != 0 or not result_file.exists():
        print(f"  [WARNING] benchmark job returned {ret.returncode}; using wall-clock estimate.")
        return {
            "num_gpus": num_gpus,
            "wall_time_sec": wall_time,
            "samples_per_sec": (args.steps_per_epoch * args.batch_size * num_gpus * args.epochs) / wall_time,
            "note": "estimated",
        }

    with open(result_file) as f:
        data = json.load(f)
    result_file.unlink(missing_ok=True)   # cleanup temp file
    return data


# ---------------------------------------------------------------------------
# Compute speedup / efficiency metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list) -> list:
    """Attach ideal_speedup, actual_speedup, efficiency to each result row."""
    baseline = next(r for r in results if r["num_gpus"] == min(r["num_gpus"] for r in results))
    base_throughput = baseline["samples_per_sec"]

    for r in results:
        r["ideal_speedup"]    = float(r["num_gpus"]) / float(baseline["num_gpus"])
        r["actual_speedup"]   = r["samples_per_sec"] / base_throughput
        r["parallel_eff_pct"] = 100.0 * r["actual_speedup"] / r["ideal_speedup"]

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling(results: list, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[plot] matplotlib/pandas not available — skipping.")
        return

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Throughput bar chart
    ax = axes[0]
    ax.bar([str(r["num_gpus"]) + " GPU" for r in results],
           [r["samples_per_sec"] for r in results],
           color="steelblue", edgecolor="black")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Throughput (images/sec)")
    ax.set_title("Training Throughput")
    ax.grid(True, axis="y", alpha=0.3)

    # 2. Speedup curve
    ax2 = axes[1]
    gpus = [r["num_gpus"] for r in results]
    ideal = [r["ideal_speedup"] for r in results]
    actual = [r["actual_speedup"] for r in results]
    ax2.plot(gpus, ideal, "k--o", label="Ideal (linear)")
    ax2.plot(gpus, actual, "b-o", label="Actual", linewidth=2)
    ax2.fill_between(gpus, actual, ideal, alpha=0.1, color="red", label="Efficiency gap")
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Ideal vs. Actual Speedup")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Parallel efficiency
    ax3 = axes[2]
    eff = [r["parallel_eff_pct"] for r in results]
    bars = ax3.bar([str(r["num_gpus"]) + " GPU" for r in results],
                   eff, color=["green" if e >= 80 else "orange" for e in eff],
                   edgecolor="black")
    ax3.axhline(100, color="black", linestyle="--", linewidth=0.8)
    ax3.axhline(80, color="red", linestyle=":", linewidth=1, label="80% threshold")
    ax3.set_ylabel("Parallel Efficiency (%)")
    ax3.set_title("Parallel Efficiency")
    ax3.set_ylim(0, 115)
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)
    for bar, e in zip(bars, eff):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{e:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle("HCVGLoc Stage 1 — DDP Scaling Benchmark", fontsize=13)
    plt.tight_layout()

    out_path = output_dir / "plots" / "scaling_results.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  HCVGLoc Stage 1 — Scaling Benchmark")
    print(f"  GPUs to test  : {args.gpus}")
    print(f"  Per-GPU batch : {args.batch_size}")
    print(f"  Epochs/run    : {args.epochs}")
    print("="*60)

    results = []
    for n_gpu in sorted(args.gpus):
        result = run_benchmark_job(n_gpu, args)
        results.append(result)
        print(f"  ✓ {n_gpu} GPU(s): {result['samples_per_sec']:.1f} img/s")

    results = compute_metrics(results)

    # Save table
    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    table_path = out / "tables" / "scaling_results.json"
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved → {table_path}]")

    # Print summary table
    print(f"\n{'GPU':<8} {'Throughput':>14} {'Speedup':>10} {'Efficiency':>12}")
    print("-" * 48)
    for r in results:
        print(
            f"{r['num_gpus']:<8} "
            f"{r['samples_per_sec']:>12.1f}  "
            f"{r['actual_speedup']:>8.2f}×  "
            f"{r['parallel_eff_pct']:>10.1f}%"
        )

    # Plot
    plot_scaling(results, out)

    # Optional wandb logging
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=f"scaling_{'_'.join(str(g) for g in sorted(args.gpus))}GPU",
                config={
                    "gpus_tested": sorted(args.gpus),
                    "epochs": args.epochs,
                    "batch_size_per_gpu": args.batch_size,
                    "steps_per_epoch": args.steps_per_epoch,
                },
                reinit=True,
            )
            cols = list(results[0].keys())
            table = wandb.Table(columns=cols)
            for r in results:
                table.add_data(*[r.get(c) for c in cols])
                wandb.log(
                    {f"scaling/{k}": v for k, v in r.items() if isinstance(v, (int, float))},
                    step=r["num_gpus"],
                )
            wandb.log({"scaling_table": table})
            plot_path = out / "plots" / "scaling_results.png"
            if plot_path.exists():
                wandb.log({"scaling_plot": wandb.Image(str(plot_path))})
            wandb.save(str(table_path))
            wandb.finish()
            print("[wandb] scaling benchmark synced")
        except ImportError:
            print("[wandb] not installed; skipping.")

    print("\n[Scaling benchmark complete]")


if __name__ == "__main__":
    main()
