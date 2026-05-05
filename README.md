# distributed-HCVGLoc

**Distributed Training Infrastructure for Hierarchical Cross-View Geo-Localization**

> Distributed Systems Theory and Analysis Course Project · Missouri S&T · Spring 2026  

---

## Overview

This repository implements and rigorously evaluates the **distributed PyTorch DDP training
pipeline** for Coarse Retrieval Stage of the HCVGLoc framework -- a
hierarchical cross-view geo-localization system for GPS-denied UAV navigation.

The project treats the 4× NVIDIA RTX Ada 6000 training cluster as a distributed system:

| Metrics | What is Measured |
|--------|----------------|
| **Scalability** | Throughput (img/s), speedup, parallel efficiency across 1/2/4 GPUs |
| **Communication Overhead** | AllReduce latency, compute-to-comm ratio, bandwidth sensitivity |
| **Fault Tolerance** | Mid-training failure recovery, checkpoint overhead vs. recovery time |
| **Convergence Parity** | Single-GPU vs DDP: loss curves, R@1/R@5, wall-clock time |

---

## Research Context — HCVGLoc Stage 1

Stage 1 performs **coarse satellite image retrieval**: given a UAV query image, retrieve
the top-K matching geo-referenced satellite tiles from a large-scale gallery.


**Datasets:** University-1652 · VIGOR · CVUSA · CVACT  
**Hardware:** 4× NVIDIA RTX Ada 6000 (48 GB each), NCCL backend  
**Related repo:** [gxanesh/avgl_plus_plus](https://github.com/gxanesh/avgl_plus_plus)

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

## License

MIT — see [LICENSE](LICENSE)
