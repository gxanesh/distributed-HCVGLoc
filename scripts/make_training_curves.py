"""
make_training_curves.py
------------------------
Plot per-epoch loss components + R@K for the 4-GPU DDP University-1652 run.
Input:  experiments/stage1_univ1652_4gpu/logs/history.json
Output: results/plots/training_curves_univ1652_4gpu.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "experiments" / "stage1_univ1652_4gpu" / "logs" / "history.json"
OUT = ROOT / "results" / "plots" / "training_curves_univ1652_4gpu.png"


def main():
    with open(HISTORY) as f:
        h = json.load(f)
    epochs = [r["epoch"] for r in h]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # --- Panel A: total + NCE loss vs epoch ---
    axA = axes[0]
    axA.plot(epochs, [r["train_total"] for r in h], "o-", linewidth=2,
             color="tab:blue", label="train_total")
    axA.plot(epochs, [r["train_nce"] for r in h], "s-", linewidth=2,
             color="tab:orange", label="train_nce (InfoNCE)")
    axA.axhline(3.4657, color="gray", linestyle=":", linewidth=1,
                label="ln(32) = random NCE")
    axA.set_xticks(epochs)
    axA.set_xlabel("Epoch")
    axA.set_ylabel("Loss")
    axA.set_title("Training loss — University-1652, 4-GPU DDP (5 epochs)")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="upper right")

    # --- Panel B: R@K vs epoch ---
    axB = axes[1]
    axB.plot(epochs, [100 * r["val/University-1652/R@1"] for r in h],
             "o-", linewidth=2, color="tab:red", label="R@1")
    axB.plot(epochs, [100 * r["val/University-1652/R@5"] for r in h],
             "s-", linewidth=2, color="tab:green", label="R@5")
    axB.plot(epochs, [100 * r["val/University-1652/R@10"] for r in h],
             "^-", linewidth=2, color="tab:purple", label="R@10")
    # 951-gallery random baseline
    axB.axhline(100 / 951, color="gray", linestyle=":", linewidth=1,
                label="Random R@1 (1/951 gallery)")
    axB.set_xticks(epochs)
    axB.set_xlabel("Epoch")
    axB.set_ylabel("Recall (%)")
    axB.set_title("Validation R@K — University-1652 gallery (951 classes)")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="center right")

    # --- Panel C: LR + GRL λ vs epoch ---
    axC = axes[2]
    lrs = [r["lr"] for r in h]
    lambdas = [r["grl_lambda"] for r in h]
    axC.semilogy(epochs, lrs, "o-", linewidth=2, color="tab:blue",
                 label="Learning rate (log scale)")
    axC.set_xlabel("Epoch")
    axC.set_ylabel("LR (log)", color="tab:blue")
    axC.tick_params(axis="y", labelcolor="tab:blue")
    twin = axC.twinx()
    twin.plot(epochs, lambdas, "s--", linewidth=2, color="tab:red",
              label="GRL λ")
    twin.set_ylabel("GRL λ", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")
    axC.set_xticks(epochs)
    axC.set_title("Schedules — LR (cosine, T_max=5) and GRL λ (linear ramp)")
    axC.grid(True, alpha=0.3)
    lines1, labels1 = axC.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axC.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.suptitle(
        "Pillar 4 (partial) — University-1652 DDP training on 4× RTX 6000 Ada",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
