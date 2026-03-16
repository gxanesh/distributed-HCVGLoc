# distributed-HCVGLoc

**Distributed Training Infrastructure for Hierarchical Cross-View Geo-Localization**

> Graduate Distributed Systems Course Project · Missouri S&T · Spring 2026  
> Researcher: Ganesh (gs37r) · Advisor: Dr. Sanjay Madria  
> Funded by: Army Research Laboratory

---

## Overview

This repository implements and rigorously evaluates the **distributed PyTorch DDP training
pipeline** for **Stage 1 (Coarse Retrieval)** of the HCVGLoc framework — a
hierarchical cross-view geo-localization system for GPS-denied UAV navigation.

The project treats the 4× NVIDIA RTX Ada 6000 training cluster as a distributed system:

| Pillar | What We Measure |
|--------|----------------|
| **Scalability** | Throughput (img/s), speedup, parallel efficiency across 1/2/4 GPUs |
| **Communication Overhead** | AllReduce latency, compute-to-comm ratio, bandwidth sensitivity |
| **Fault Tolerance** | Mid-training failure recovery, checkpoint overhead vs. recovery time |
| **Convergence Parity** | Single-GPU vs DDP: loss curves, R@1/R@5, wall-clock time |

---

## Research Context — HCVGLoc Stage 1

Stage 1 performs **coarse satellite image retrieval**: given a UAV query image, retrieve
the top-K matching geo-referenced satellite tiles from a large-scale gallery.

```
UAV Query Image (384×384)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1: Coarse Retrieval                                       │
│                                                                  │
│  EfficientViT-B1  ──→  BiFPN  ──→  {C3, C4, C5} features        │
│        │                                                         │
│        ├──→  GLE (Slot Attention)     ──→  512-D descriptor      │
│        ├──→  GRL + Domain Classifier  ──→  domain logits (×4)   │
│        ├──→  Rotation Head            ──→  (sin θ, cos θ)        │
│        └──→  Uncertainty Head         ──→  σ² (aleatoric)        │
│                                                                  │
│  Loss:  L = L_InfoNCE + 0.5·L_JS + 0.1·L_domain                 │
│                       + 0.2·L_rot + 0.1·L_unc                   │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
  Top-K Satellite Candidates  ──→  Stage 2 (Fine 6-DOF Pose)
```

**Datasets:** University-1652 · VIGOR · CVUSA · CVACT  
**Hardware:** 4× NVIDIA RTX Ada 6000 (48 GB each), NCCL backend  
**Related repo:** [gxanesh/avgl_plus_plus](https://github.com/gxanesh/avgl_plus_plus)

---

## Repository Structure

```
distributed-HCVGLoc/
├── hcvgloc/
│   ├── models/
│   │   ├── backbone/efficientvit.py     # EfficientViT-B1, multi-scale outputs
│   │   └── stage1/
│   │       ├── bifpn.py                 # Bi-directional FPN neck
│   │       ├── geometric_layout.py      # Slot-attention descriptor (GLE)
│   │       ├── domain_adversarial.py    # GRL + domain classifier
│   │       ├── rotation_module.py       # Rotation-aware head
│   │       ├── uncertainty_head.py      # Aleatoric uncertainty
│   │       └── coarse_localizer.py      # Full Stage 1 model
│   ├── losses/
│   │   ├── infonce.py                   # Label-aware InfoNCE (multi-query safe)
│   │   ├── js_divergence.py             # JS divergence alignment
│   │   ├── domain_loss.py               # Adversarial domain loss
│   │   └── composite_loss.py            # Combined L_total
│   ├── datasets/
│   │   ├── base_dataset.py              # Abstract cross-view dataset
│   │   ├── cvusa.py                     # CVUSA dataloader
│   │   ├── cvact.py                     # CVACT dataloader
│   │   ├── vigor.py                     # VIGOR dataloader
│   │   ├── transforms.py                # Augmentation pipeline
│   │   └── sampler.py                   # Domain-balanced DDP sampler
│   └── utils/
│       ├── metrics.py                   # Recall@K, mAP evaluation
│       ├── logger.py                    # Rank-aware logging
│       └── dist_utils.py               # DDP helpers & timing
├── tools/
│   └── train_stage1.py                  # Main DDP training entry point
├── benchmarks/
│   ├── scaling_benchmark.py             # Throughput vs GPU count
│   └── communication_profiler.py        # AllReduce vs compute timing
├── fault_tolerance/
│   ├── checkpoint_manager.py            # Save/restore + overhead measurement
│   └── fault_injector.py               # Rank kill simulation
├── configs/
│   ├── stage1_baseline.yaml             # Single GPU baseline
│   ├── stage1_ddp_1gpu.yaml
│   ├── stage1_ddp_2gpu.yaml
│   └── stage1_ddp_4gpu.yaml
├── scripts/
│   ├── run_single_gpu.sh
│   ├── run_ddp_2gpu.sh
│   ├── run_ddp_4gpu.sh
│   └── setup_env.sh
└── results/plots/ results/tables/
```

---

## Quick Start

```bash
git clone https://github.com/gxanesh/distributed-HCVGLoc.git
cd distributed-HCVGLoc
bash scripts/setup_env.sh

# Single GPU baseline
bash scripts/run_single_gpu.sh

# 4-GPU DDP
bash scripts/run_ddp_4gpu.sh

# Scaling benchmark
python benchmarks/scaling_benchmark.py --gpus 1 2 4 --epochs 3

# Communication profiler
torchrun --nproc_per_node=4 benchmarks/communication_profiler.py

# Fault tolerance
python fault_tolerance/checkpoint_manager.py --benchmark
```

---

## Experiment Timeline

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | Mar 16–22 | DDP architecture doc, single-GPU baseline |
| 2 | Mar 23–29 | Scaling benchmark (1/2/4 GPU), profiling plots |
| 3 | Mar 30–Apr 5 | Communication overhead, `tc netem` latency simulation |
| 4 | Apr 6–12 | Fault injection, checkpoint recovery experiments |
| 5 | Apr 13–19 | Distributed vs. baseline comparison, convergence curves |
| 6 | Apr 20–30 | Final report, analysis, cleanup |

---

## License

MIT — see [LICENSE](LICENSE)
