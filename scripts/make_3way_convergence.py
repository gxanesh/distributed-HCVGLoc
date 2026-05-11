"""
make_3way_convergence.py
-------------------------
Plot per-epoch convergence + wall-clock for the three University-1652 runs:
  Run A — 1 GPU @ 32/GPU
  Run B — DDP-4 @ 64/GPU
  Run C — DDP-2 @ 64/GPU

Output:
  results/plots/convergence_3way.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "results" / "plots" / "convergence_3way.png"

# Hard-coded per-epoch data harvested from the training logs (3 runs × 20 epochs).
RUN_A = {  # 1 GPU, B=32/GPU, total wall-clock = 16505 s
    "name": "Run A — 1 GPU (B=32)", "color": "tab:blue",
    "epochs":     list(range(20)),
    "nce":        [6.042,5.045,5.534,5.690,5.661,5.696,5.705,5.541,5.578,5.538,
                   5.727,5.813,5.688,5.735,5.816,5.816,5.829,5.906,5.936,6.102],
    "R@1":        [0.11,0.10,0.10,0.10,0.12,0.10,0.12,0.15,0.10,0.10,
                   0.13,0.10,0.08,0.12,0.12,0.16,0.13,0.09,0.12,0.11],
    "R@5":        [0.57,0.53,0.56,0.51,0.61,0.53,0.54,0.64,0.59,0.55,
                   0.65,0.57,0.57,0.57,0.60,0.67,0.62,0.58,0.62,0.56],
    "R@10":       [1.11,1.03,1.10,1.04,1.17,1.11,1.08,1.22,1.23,1.15,
                   1.29,1.18,1.13,1.13,1.21,1.30,1.18,1.13,1.20,1.16],
    "epoch_sec":  [650.1,632.0,622.8,592.6,576.7,586.8,585.5,570.2,570.7,577.5,
                   575.7,595.3,597.6,604.8,600.7,602.4,603.7,602.3,605.6,598.3],
}

RUN_B = {  # 4 GPU, B=64/GPU, total wall-clock = 7860 s
    "name": "Run B — DDP-4 (B=64/GPU, global=256)", "color": "tab:red",
    "epochs":     list(range(20)),
    "nce":        [6.545,5.351,5.498,5.512,5.645,5.689,5.748,5.852,6.189,6.109,
                   6.083,6.305,6.370,6.226,6.186,6.474,6.579,6.710,6.736,6.737],
    "R@1":        [0.10,0.10,0.07,0.11,0.11,0.11,0.12,0.11,0.12,0.11,
                   0.13,0.12,0.20,0.18,0.13,0.13,0.12,0.16,0.12,0.12],
    "R@5":        [0.54,0.53,0.40,0.57,0.51,0.47,0.62,0.60,0.64,0.57,
                   0.58,0.52,0.86,0.64,0.62,0.67,0.62,0.71,0.53,0.62],
    "R@10":       [1.13,1.08,0.92,1.07,1.02,1.00,1.20,1.21,1.18,1.14,
                   1.15,0.96,1.71,1.29,1.26,1.33,1.27,1.34,1.10,1.18],
    "epoch_sec":  [161.5,164.3,159.1,158.0,157.5,161.3,164.8,152.8,158.6,157.7,
                   152.6,150.5,154.9,159.5,152.9,147.6,150.7,152.4,152.2,151.1],
}

RUN_C = {  # 2 GPU, B=64/GPU, total wall-clock = 10839 s
    "name": "Run C — DDP-2 (B=64/GPU, global=128)", "color": "tab:green",
    "epochs":     list(range(20)),
    "nce":        [6.499,5.414,5.536,5.934,6.204,5.838,6.172,6.184,6.241,6.132,
                   6.093,6.238,6.280,6.532,6.246,6.500,6.709,6.709,6.738,6.739],
    "R@1":        [0.13,0.15,0.09,0.10,0.20,0.12,0.13,0.17,0.12,0.10,
                   0.12,0.14,0.16,0.13,0.18,0.12,0.16,0.17,0.18,0.15],
    "R@5":        [0.60,0.60,0.50,0.55,0.89,0.55,0.58,0.70,0.61,0.49,
                   0.59,0.63,0.68,0.60,0.78,0.56,0.72,0.90,0.83,0.75],
    "R@10":       [1.14,1.09,1.00,1.02,1.55,1.14,1.09,1.38,1.28,1.02,
                   1.16,1.21,1.36,1.22,1.49,1.22,1.31,1.70,1.69,1.54],
    "epoch_sec":  [323.3,316.2,308.6,302.5,297.0,296.8,297.8,303.8,299.6,301.5,
                   304.5,300.0,297.3,297.2,296.0,294.2,294.8,300.6,300.0,296.1],
}


def cumulative(seq):
    out = []
    s = 0.0
    for x in seq:
        s += x
        out.append(s)
    return out


def main():
    runs = [RUN_A, RUN_B, RUN_C]
    for r in runs:
        r["cum_sec"] = cumulative(r["epoch_sec"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── Panel A: train_nce vs epoch ───────────────────────────────────────
    axA = axes[0, 0]
    for r in runs:
        axA.plot(r["epochs"], r["nce"], "o-", color=r["color"],
                 linewidth=2, label=r["name"])
    axA.axhline(3.466, color="gray", linestyle=":", linewidth=1, label="ln(32) — random for B=32")
    axA.set_xticks(range(0, 20, 2))
    axA.set_xlabel("Epoch")
    axA.set_ylabel("train_nce")
    axA.set_title("Loss stabilisation — InfoNCE across runs")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="lower right", fontsize=9)

    # ── Panel B: R@1 vs epoch ─────────────────────────────────────────────
    axB = axes[0, 1]
    for r in runs:
        axB.plot(r["epochs"], r["R@1"], "o-", color=r["color"],
                 linewidth=2, label=r["name"])
    axB.axhline(100/951, color="gray", linestyle=":", linewidth=1,
                label="Random R@1 (1/951)")
    axB.set_xticks(range(0, 20, 2))
    axB.set_xlabel("Epoch")
    axB.set_ylabel("Validation R@1 (%)")
    axB.set_title("Convergence — R@1 vs epoch")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="upper left", fontsize=9)

    # ── Panel C: R@5 / R@10 vs epoch ──────────────────────────────────────
    axC = axes[1, 0]
    for r in runs:
        axC.plot(r["epochs"], r["R@5"], "o-", color=r["color"],
                 linewidth=2, label=f"{r['name']} (R@5)")
        axC.plot(r["epochs"], r["R@10"], "s--", color=r["color"],
                 linewidth=1.5, alpha=0.6, label=f"{r['name']} (R@10)")
    axC.set_xticks(range(0, 20, 2))
    axC.set_xlabel("Epoch")
    axC.set_ylabel("Recall (%)")
    axC.set_title("R@5 (solid) and R@10 (dashed) vs epoch")
    axC.grid(True, alpha=0.3)
    axC.legend(loc="upper left", fontsize=7, ncol=2)

    # ── Panel D: R@1 vs cumulative wall-clock ─────────────────────────────
    axD = axes[1, 1]
    for r in runs:
        axD.plot([s/60 for s in r["cum_sec"]], r["R@1"], "o-",
                 color=r["color"], linewidth=2, label=r["name"])
    axD.set_xlabel("Cumulative training wall-clock (min)")
    axD.set_ylabel("Validation R@1 (%)")
    axD.set_title("Time-to-recall — R@1 vs wall-clock minutes")
    axD.grid(True, alpha=0.3)
    axD.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "University-1652 — 20-epoch convergence parity (1 GPU vs DDP-2 vs DDP-4)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
