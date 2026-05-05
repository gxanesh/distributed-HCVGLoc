"""
make_progress_summary.py
-------------------------
Produce a single combined summary figure covering pillars 1-3 for the
progress report. Reads the raw JSON tables under results/tables/ and writes
results/plots/progress_summary.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
PLOT_DIR = ROOT / "results" / "plots"


def _load(name: str):
    with open(TABLE_DIR / name) as f:
        return json.load(f)


def main():
    comm_w4 = _load("comm_profile_W4.json")
    comm_w2 = _load("comm_profile_W2.json")
    scaling = _load("scaling_results.json")
    ckpt = _load("checkpoint_overhead.json")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # --- Panel A: comm overhead % vs batch size for W=2 and W=4 -------------
    axA = axes[0]
    axA.plot([r["batch_size"] for r in comm_w2],
             [r["comm_overhead_pct"] for r in comm_w2],
             "o-", color="tab:blue", linewidth=2, label="W=2 (NCCL ring)")
    axA.plot([r["batch_size"] for r in comm_w4],
             [r["comm_overhead_pct"] for r in comm_w4],
             "s-", color="tab:red", linewidth=2, label="W=4 (NCCL ring)")
    axA.axhline(10, color="green", linestyle=":", linewidth=1, label="10% threshold")
    axA.set_xscale("log", base=2)
    axA.set_xticks([16, 32, 64, 128])
    axA.set_xticklabels(["16", "32", "64", "128"])
    axA.set_xlabel("Per-GPU batch size")
    axA.set_ylabel("AllReduce overhead (% of step time)")
    axA.set_title("Pillar 2 — Communication overhead")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="upper right")

    # --- Panel B: scaling speedup vs GPU count -----------------------------
    axB = axes[1]
    gpus = [r["num_gpus"] for r in scaling]
    actual = [r["actual_speedup"] for r in scaling]
    ideal = [r["ideal_speedup"] for r in scaling]
    axB.plot(gpus, ideal, "k--o", label="Ideal (linear)", linewidth=1.5)
    axB.plot(gpus, actual, "b-o", label="Actual (proxy model)", linewidth=2)
    axB.fill_between(gpus, actual, ideal, alpha=0.12, color="red",
                     label="Efficiency gap")
    for g, a in zip(gpus, actual):
        axB.annotate(f"{a:.2f}×", (g, a),
                     textcoords="offset points", xytext=(8, -4), fontsize=9)
    axB.set_xticks(gpus)
    axB.set_xlabel("Number of GPUs")
    axB.set_ylabel("Speedup (×)")
    axB.set_title("Pillar 1 — DDP scaling speedup")
    axB.legend(loc="upper left")
    axB.grid(True, alpha=0.3)

    # --- Panel C: checkpoint tradeoff --------------------------------------
    axC = axes[2]
    intervals = [r["interval_epochs"] for r in ckpt]
    overhead_pct = [r["overhead_pct_estimate"] for r in ckpt]
    steps_lost = [r["steps_lost_on_failure"] for r in ckpt]

    twin = axC.twinx()
    l1 = axC.plot(intervals, overhead_pct, "o-", color="tab:blue",
                  linewidth=2, label="Overhead % estimate")
    l2 = twin.plot(intervals, steps_lost, "s--", color="tab:red",
                   linewidth=2, label="Avg steps lost on failure")
    axC.set_xlabel("Checkpoint interval (epochs)")
    axC.set_ylabel("Overhead (% of training time)", color="tab:blue")
    twin.set_ylabel("Avg steps lost on failure", color="tab:red")
    axC.set_title("Pillar 3 — Checkpoint tradeoff")
    axC.tick_params(axis="y", labelcolor="tab:blue")
    twin.tick_params(axis="y", labelcolor="tab:red")
    axC.grid(True, alpha=0.3)
    lines = l1 + l2
    axC.legend(lines, [l.get_label() for l in lines], loc="upper right")

    plt.suptitle(
        "distributed-HCVGLoc — Progress Report Summary (2026-04-21)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = PLOT_DIR / "progress_summary.png"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
