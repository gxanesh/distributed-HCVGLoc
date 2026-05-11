"""
scaling_2d_sweep.py
--------------------
Two-dimensional scalability sweep: world_size × per-GPU batch_size.

For each (W, B), launches torchrun with `nproc_per_node=W` and runs
benchmarks/_bench_worker.py on the synthetic-data proxy model. The
actual_speedup baseline for each B is the W=1 throughput at the SAME B
(weak scaling — ideal = W at fixed per-GPU batch).

Outputs:
    results/tables/scaling_2d.json        — full grid as JSON
    results/tables/scaling_2d.csv         — flat CSV
    results/plots/scaling_2d_throughput.png
    results/plots/scaling_2d_speedup.png
    results/plots/scaling_2d_efficiency.png

Usage:
    python benchmarks/scaling_2d_sweep.py \
        --gpus 1 2 4 --batches 16 32 64 128 \
        --epochs 3 --steps_per_epoch 200

Optional:  --wandb   (also push grid + plots to wandb run "scaling_2d_sweep")
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="2D scalability sweep (W × B)")
    p.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--config", type=str, default="configs/stage1_ddp_4gpu.yaml")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="hcvgloc-distributed")
    p.add_argument("--wandb_entity", type=str, default="w2c-lab")
    return p.parse_args()


def run_one(world: int, batch: int, args, master_port: int) -> dict:
    """Launch one torchrun job and parse its JSON result."""
    out = Path(args.output_dir)
    tmp = out / "tables" / f"_2d_tmp_W{world}_B{batch}.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    cmd = [
        "torchrun",
        f"--nproc_per_node={world}",
        f"--master_port={master_port}",
        str(Path(__file__).parent / "_bench_worker.py"),
        f"--config={args.config}",
        f"--epochs={args.epochs}",
        f"--batch_size={batch}",
        f"--steps_per_epoch={args.steps_per_epoch}",
        f"--result_file={tmp}",
    ]
    print(f"\n[W={world} B={batch}] {' '.join(cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd, capture_output=False)
    wall = time.time() - t0

    if ret.returncode != 0 or not tmp.exists():
        print(f"  [WARN] W={world} B={batch} returncode={ret.returncode}; estimating throughput")
        return {
            "num_gpus": world,
            "batch_size_per_gpu": batch,
            "global_batch_size": batch * world,
            "samples_per_sec": (args.steps_per_epoch * batch * world * args.epochs) / max(wall, 1e-6),
            "wall_time_sec": round(wall, 3),
            "note": "estimated",
        }

    with open(tmp) as f:
        result = json.load(f)
    tmp.unlink(missing_ok=True)
    return result


def attach_speedup(results: list) -> list:
    """For each per-GPU batch, baseline = world=1 throughput at the same batch."""
    by_batch_baseline = {}
    for r in results:
        if r["num_gpus"] == 1:
            by_batch_baseline[r["batch_size_per_gpu"]] = r["samples_per_sec"]

    for r in results:
        base = by_batch_baseline.get(r["batch_size_per_gpu"])
        if base and base > 0:
            r["ideal_speedup"] = float(r["num_gpus"])
            r["actual_speedup"] = r["samples_per_sec"] / base
            r["parallel_eff_pct"] = 100.0 * r["actual_speedup"] / r["ideal_speedup"]
        else:
            r["ideal_speedup"] = float(r["num_gpus"])
            r["actual_speedup"] = float("nan")
            r["parallel_eff_pct"] = float("nan")
    return results


def write_csv(results, path: Path):
    cols = ["num_gpus", "batch_size_per_gpu", "global_batch_size",
            "samples_per_sec", "actual_speedup", "ideal_speedup",
            "parallel_eff_pct", "wall_time_sec"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            row = [
                str(r.get(c, "")) if not isinstance(r.get(c), float)
                else f"{r[c]:.4f}"
                for c in cols
            ]
            f.write(",".join(row) + "\n")


def plot_grids(results, plot_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[plot] matplotlib/pandas missing; skipping plots")
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    batches = sorted(df["batch_size_per_gpu"].unique())
    gpus = sorted(df["num_gpus"].unique())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # ── 1. throughput vs world for each batch ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, B in zip(colors, batches):
        sub = df[df["batch_size_per_gpu"] == B].sort_values("num_gpus")
        ax.plot(sub["num_gpus"], sub["samples_per_sec"], "o-",
                color=c, linewidth=2, label=f"per-GPU batch={B}")
        for _, row in sub.iterrows():
            ax.annotate(f"{row['samples_per_sec']:.0f}",
                        (row["num_gpus"], row["samples_per_sec"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xticks(gpus)
    ax.set_xlabel("Number of GPUs (world size)")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Scaling — Throughput vs. World Size by Per-GPU Batch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = plot_dir / "scaling_2d_throughput.png"
    plt.tight_layout(); plt.savefig(p1, dpi=150); plt.close()
    print(f"[Saved] {p1}")

    # ── 2. speedup vs world ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ideal_x = list(range(1, max(gpus) + 1))
    ax.plot(ideal_x, ideal_x, "k--", linewidth=1.5, label="Ideal (linear)")
    for c, B in zip(colors, batches):
        sub = df[df["batch_size_per_gpu"] == B].sort_values("num_gpus")
        ax.plot(sub["num_gpus"], sub["actual_speedup"], "o-",
                color=c, linewidth=2, label=f"batch={B}")
    ax.set_xticks(gpus)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Speedup (×) vs single-GPU baseline (same per-GPU batch)")
    ax.set_title("Scaling — Actual vs. Ideal Speedup")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p2 = plot_dir / "scaling_2d_speedup.png"
    plt.tight_layout(); plt.savefig(p2, dpi=150); plt.close()
    print(f"[Saved] {p2}")

    # ── 3. parallel efficiency heat-grid ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(batches), 1)
    for i, B in enumerate(batches):
        sub = df[df["batch_size_per_gpu"] == B].sort_values("num_gpus")
        x = [g + (i - (len(batches) - 1) / 2) * width for g in sub["num_gpus"]]
        bars = ax.bar(x, sub["parallel_eff_pct"], width=width,
                      color=colors[i], label=f"batch={B}", edgecolor="black")
        for bar, v in zip(bars, sub["parallel_eff_pct"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(100, color="black", linestyle=":", linewidth=0.8)
    ax.axhline(80, color="red", linestyle=":", linewidth=1, label="80% threshold")
    ax.set_xticks(gpus)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_ylim(0, 130)
    ax.set_title("Scaling — Parallel Efficiency by Per-GPU Batch")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    p3 = plot_dir / "scaling_2d_efficiency.png"
    plt.tight_layout(); plt.savefig(p3, dpi=150); plt.close()
    print(f"[Saved] {p3}")

    return [p1, p2, p3]


def maybe_log_to_wandb(args, results, plot_paths, json_path: Path):
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping")
        return
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group="scaling_2d_sweep", name="scaling_2d",
        config={"gpus": args.gpus, "batches": args.batches,
                "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch},
        reinit=True,
    )
    cols = ["num_gpus", "batch_size_per_gpu", "global_batch_size",
            "samples_per_sec", "actual_speedup", "ideal_speedup",
            "parallel_eff_pct", "wall_time_sec"]
    table = wandb.Table(columns=cols)
    for r in results:
        table.add_data(*[r.get(c) for c in cols])
    wandb.log({"scaling_2d_table": table})
    for p in plot_paths or []:
        wandb.log({p.stem: wandb.Image(str(p))})
    wandb.save(str(json_path))
    wandb.finish()
    print("[wandb] sweep synced")


def main():
    args = parse_args()
    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"  Scaling 2D Sweep: GPUs={args.gpus} × per-GPU batches={args.batches}")
    print(f"  Epochs/run={args.epochs}, steps/epoch={args.steps_per_epoch}")
    print("=" * 64)

    port_base = 29550
    results = []
    port = port_base
    for B in sorted(args.batches):
        for W in sorted(args.gpus):
            r = run_one(W, B, args, master_port=port)
            r["batch_size_per_gpu"] = B
            r["num_gpus"] = W
            r["global_batch_size"] = B * W
            results.append(r)
            port += 1

    results = attach_speedup(results)

    json_path = out / "tables" / "scaling_2d.json"
    csv_path = out / "tables" / "scaling_2d.csv"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    write_csv(results, csv_path)
    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")

    # Console table
    header = (f"{'GPUs':>4} {'B/GPU':>6} {'GlobalB':>8} "
              f"{'Throughput':>12} {'Speedup':>9} {'Ideal':>7} {'Eff%':>7}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['num_gpus']:>4} {r['batch_size_per_gpu']:>6} "
              f"{r['global_batch_size']:>8} "
              f"{r['samples_per_sec']:>12.1f} "
              f"{r['actual_speedup']:>8.2f}× "
              f"{r['ideal_speedup']:>6.1f}× "
              f"{r['parallel_eff_pct']:>6.1f}%")

    plot_paths = plot_grids(results, out / "plots")

    if args.wandb:
        maybe_log_to_wandb(args, results, plot_paths, json_path)

    print("\n[scaling 2D sweep complete]")


if __name__ == "__main__":
    main()
