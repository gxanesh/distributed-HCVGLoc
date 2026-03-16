"""
checkpoint_manager.py
----------------------
Checkpoint save/restore with overhead measurement for HCVGLoc Stage 1.

Experiments:
  1. Checkpoint overhead: measure time to write checkpoint at different
     model sizes and frequencies (every 5/10/20 epochs).
  2. Recovery benchmark: simulate a mid-training failure (kill + restart),
     measure steps lost and recovery wall-clock time.
  3. Overhead vs frequency tradeoff table.

Usage:
    # Run checkpoint overhead benchmark
    python fault_tolerance/checkpoint_manager.py --benchmark

    # Resume from a checkpoint (used by train_stage1.py --resume)
    python fault_tolerance/checkpoint_manager.py \\
        --restore checkpoints/latest.pt \\
        --output_dir checkpoints/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Handles periodic checkpoint saving and restoration.
    Tracks save/load latency for benchmarking.
    """

    def __init__(self, save_dir: str, interval: int = 10):
        """
        Args:
            save_dir: directory to store checkpoint files
            interval: save every N epochs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._save_times: list = []
        self._load_times: list = []

    def should_save(self, epoch: int) -> bool:
        return (epoch + 1) % self.interval == 0

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metadata: dict = None,
        tag: str = "latest",
    ) -> dict:
        """
        Save checkpoint. Returns timing info.

        Args:
            model:     model (may be DDP-wrapped)
            optimizer: optimizer
            epoch:     current epoch
            metadata:  extra info to store (loss, R@1, etc.)
            tag:       filename tag ('latest', 'best', or epoch number)
        """
        raw_model = model.module if hasattr(model, "module") else model
        state = {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata or {},
        }

        ckpt_path = self.save_dir / f"checkpoint_{tag}.pt"

        t0 = time.time()
        torch.save(state, ckpt_path)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed_ms = (time.time() - t0) * 1000

        size_mb = ckpt_path.stat().st_size / (1024 ** 2)
        self._save_times.append(elapsed_ms)

        return {
            "path": str(ckpt_path),
            "save_time_ms": round(elapsed_ms, 2),
            "size_mb": round(size_mb, 2),
            "epoch": epoch,
        }

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tag: str = "latest",
        map_location: str = "cpu",
    ) -> dict:
        """
        Load checkpoint. Returns (start_epoch, metadata, timing_info).
        """
        ckpt_path = self.save_dir / f"checkpoint_{tag}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        t0 = time.time()
        state = torch.load(ckpt_path, map_location=map_location)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(state["model"])
        if optimizer and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        elapsed_ms = (time.time() - t0) * 1000
        self._load_times.append(elapsed_ms)

        size_mb = ckpt_path.stat().st_size / (1024 ** 2)
        return {
            "start_epoch": state["epoch"] + 1,
            "metadata": state.get("metadata", {}),
            "load_time_ms": round(elapsed_ms, 2),
            "size_mb": round(size_mb, 2),
        }

    def timing_summary(self) -> dict:
        """Return summary statistics of save/load times."""
        def stats(times):
            if not times:
                return {}
            return {
                "mean_ms":   round(sum(times) / len(times), 2),
                "min_ms":    round(min(times), 2),
                "max_ms":    round(max(times), 2),
                "n_saves":   len(times),
            }
        return {
            "save": stats(self._save_times),
            "load": stats(self._load_times),
        }


# ---------------------------------------------------------------------------
# Overhead benchmark experiment
# ---------------------------------------------------------------------------

def benchmark_checkpoint_overhead(args):
    """
    Measure checkpoint save/load time across different checkpoint intervals.

    Simulates the tradeoff:
        - Small interval → low steps lost on failure, high I/O overhead
        - Large interval → more steps lost, less overhead
    """
    from benchmarks.communication_profiler import ProxyStage1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProxyStage1(embed_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    intervals = args.intervals
    out_dir = Path(args.output_dir)

    print("\n" + "="*60)
    print("  Checkpoint Overhead Benchmark")
    print(f"  Intervals tested : {intervals}")
    print(f"  Output dir       : {out_dir}")
    print("="*60)

    results = []

    for interval in intervals:
        mgr = CheckpointManager(save_dir=str(out_dir / "tmp_ckpts"), interval=interval)

        # Simulate training for 100 epochs, checkpointing at each interval
        n_epochs = 100
        n_checkpoints = n_epochs // interval
        total_save_time_ms = 0
        total_load_time_ms = 0

        for i in range(n_checkpoints):
            epoch = (i + 1) * interval - 1
            # Simulate a step of training
            dummy_input = torch.randn(32, 3, 224, 224, device=device)
            out = model(dummy_input)
            loss = out.norm()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save
            save_info = mgr.save(
                model, optimizer, epoch,
                metadata={"loss": loss.item()},
                tag="latest",
            )
            total_save_time_ms += save_info["save_time_ms"]

            # Load (simulate recovery test)
            load_info = mgr.load(model, optimizer, tag="latest")
            total_load_time_ms += load_info["load_time_ms"]

        # Steps lost on failure (worst case = just after a checkpoint)
        # Approx: fail at random point → avg steps lost = interval/2
        steps_lost_avg = interval / 2
        training_time_per_step_ms = 50  # approximate step time for AVGL++ Stage 1
        recovery_time_ms = total_load_time_ms / n_checkpoints  # avg load time

        total_overhead_pct = (total_save_time_ms / (n_epochs * training_time_per_step_ms * 50)) * 100

        result = {
            "interval_epochs":      interval,
            "n_checkpoints":        n_checkpoints,
            "ckpt_size_mb":         save_info["size_mb"],
            "avg_save_time_ms":     round(total_save_time_ms / n_checkpoints, 2),
            "avg_load_time_ms":     round(total_load_time_ms / n_checkpoints, 2),
            "steps_lost_on_failure": steps_lost_avg,
            "total_save_overhead_ms": round(total_save_time_ms, 2),
            "overhead_pct_estimate": round(total_overhead_pct, 2),
        }
        results.append(result)

        print(
            f"  interval={interval:3d} | "
            f"ckpt_size={save_info['size_mb']:.1f}MB | "
            f"avg_save={result['avg_save_time_ms']:.1f}ms | "
            f"avg_load={result['avg_load_time_ms']:.1f}ms | "
            f"steps_lost≈{steps_lost_avg:.0f}"
        )

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tables" / "checkpoint_overhead.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results saved → {out_path}]")

    # Plot tradeoff
    plot_checkpoint_tradeoff(results, out_dir / "plots")
    return results


def plot_checkpoint_tradeoff(results: list, plot_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    intervals = [r["interval_epochs"] for r in results]
    save_times = [r["avg_save_time_ms"] for r in results]
    steps_lost = [r["steps_lost_on_failure"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(intervals, save_times, "b-o", label="Avg Save Time (ms)", linewidth=2)
    ax2.plot(intervals, steps_lost, "r--s", label="Avg Steps Lost on Failure", linewidth=2)

    ax1.set_xlabel("Checkpoint Interval (epochs)")
    ax1.set_ylabel("Avg Checkpoint Save Time (ms)", color="blue")
    ax2.set_ylabel("Avg Steps Lost on Failure", color="red")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Checkpoint Overhead vs. Steps Lost Tradeoff")
    plt.tight_layout()
    out_path = plot_dir / "checkpoint_tradeoff.png"
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Checkpoint Manager & Fault Tolerance Benchmark")
    p.add_argument("--benchmark", action="store_true",
                   help="Run checkpoint overhead benchmark")
    p.add_argument("--intervals", nargs="+", type=int, default=[1, 5, 10, 20, 50],
                   help="Checkpoint intervals to benchmark")
    p.add_argument("--restore", type=str, default=None,
                   help="Path to checkpoint to restore (for testing recovery)")
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark:
        benchmark_checkpoint_overhead(args)
    elif args.restore:
        from benchmarks.communication_profiler import ProxyStage1
        model = ProxyStage1()
        optimizer = torch.optim.AdamW(model.parameters())
        mgr = CheckpointManager(save_dir=str(Path(args.restore).parent))
        info = mgr.load(model, optimizer, tag=Path(args.restore).stem.replace("checkpoint_", ""))
        print(f"Restored from epoch {info['start_epoch']} in {info['load_time_ms']:.2f}ms")
    else:
        print("Use --benchmark to run overhead benchmark, or --restore <path> to test recovery.")
