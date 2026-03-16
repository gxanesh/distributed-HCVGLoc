# HCVGLoc Stage 1 — Distributed Training Architecture

## System Overview

```
                         ┌─────────────────────────────────────────────────┐
                         │         HCVGLoc Stage 1 — DDP Training          │
                         │              (4× RTX Ada 6000, NCCL)            │
                         └──────────────────────┬──────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
             ┌──────▼──────┐           ┌───────▼──────┐           ┌───────▼──────┐
             │   Rank 0    │           │   Rank 1     │           │   Rank 2-3   │
             │  GPU 0      │           │  GPU 1       │           │  GPU 2-3     │
             │  Shard 0    │           │  Shard 1     │           │  Shard 2-3   │
             └──────┬──────┘           └───────┬──────┘           └───────┬──────┘
                    │                           │                           │
             ┌──────▼──────────────────────────▼───────────────────────────▼──────┐
             │                    All-Reduce (NCCL ring / tree)                    │
             │            Gradient synchronization across all ranks                │
             └────────────────────────────────────────────────────────────────────┘
```

## Model Architecture (Stage 1)

```
Input Image [B, 3, 384, 384]
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  EfficientViT-B1 Backbone  (9.1M params, O(N) linear attention)              │
│                                                                               │
│   Stem (3→64, /4) → Stage1 (64→128, /8) → Stage2 (128→256, /16)            │
│                  → Stage3 (256→512, /32)                                      │
│                                                                               │
│   Outputs: C3 [B,128,48,48] · C4 [B,256,24,24] · C5 [B,512,12,12]          │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  BiFPN Neck  (3 layers, weighted bi-directional fusion)                       │
│                                                                               │
│   Top-down: P5→P4→P3  +  Bottom-up: P3→P4→P5                                │
│   All levels unified to 256-D                                                 │
│                                                                               │
│   Outputs: P3 [B,256,48,48] · P4 [B,256,24,24] · P5 [B,256,12,12]          │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
               ▼                    ▼                    ▼
  ┌────────────────────┐  ┌─────────────────┐  ┌────────────────────┐
  │  GLE (Slot Attn)  │  │  GRL + Domain   │  │  Rotation Head     │
  │  K=8 slots        │  │  Classifier     │  │  → (sinθ, cosθ)    │
  │  → 512-D desc     │  │  4 domains      │  │  + Uncertainty σ²  │
  └────────────────────┘  └─────────────────┘  └────────────────────┘
               │
               ▼
        L2-normalised 512-D descriptor
               │
               ▼
        InfoNCE + JS + domain + rot + unc loss
```

## DDP Data Flow

```
Training Dataset (CVUSA + VIGOR + CVACT)
        │
        ▼
DomainBalancedDistributedSampler
   ├── Rank 0: indices [0, 4, 8, ...]   (shard 0 of balanced list)
   ├── Rank 1: indices [1, 5, 9, ...]   (shard 1)
   ├── Rank 2: indices [2, 6, 10, ...]  (shard 2)
   └── Rank 3: indices [3, 7, 11, ...]  (shard 3)
        │
        ▼ (each rank independently)
   DataLoader (num_workers=8, pin_memory=True, persistent_workers=True)
        │
        ▼
   CoarseLocalizer.forward(query, sat)
        │
   Stage1Loss.forward(...)
        │
   loss.backward()  ←── DDP hooks: AllReduce gradients automatically
        │
   optimizer.step()
```

## Loss Function

```
L_total = 1.0 · L_InfoNCE      ← primary retrieval signal
        + 0.5 · L_JS            ← query/satellite distribution alignment
        + 0.1 · L_domain        ← domain adversarial (via GRL, ramped λ)
        + 0.2 · L_rotation      ← rotation consistency (sinθ, cosθ supervision)
        + 0.1 · L_uncertainty   ← aleatoric NLL calibration

InfoNCE temperature τ = 0.07   (fixed from earlier τ=0.02 overfitting)
GRL λ: 0.0 → 0.1 linearly over 20 epochs
```

## Distributed Systems Experiments

| Experiment | File | Key Metric |
|---|---|---|
| Scalability benchmark | `benchmarks/scaling_benchmark.py` | Throughput (img/s), speedup, efficiency |
| Communication profiling | `benchmarks/communication_profiler.py` | C2C ratio, AllReduce overhead % |
| Fault injection | `fault_tolerance/fault_injector.py` | Steps lost, detection latency |
| Checkpoint overhead | `fault_tolerance/checkpoint_manager.py` | Save/load ms, overhead vs frequency |
| Network degradation | `fault_tolerance/fault_injector.py --degrade_bandwidth` | Throughput vs added latency |

## Hardware Configuration

| Resource | Spec |
|---|---|
| GPUs | 4× NVIDIA RTX Ada 6000 (48 GB each) |
| GPU interconnect | PCIe (single node) |
| NCCL backend | Ring AllReduce |
| CPU | 64-core workstation |
| Storage | 8 TB NVMe — `/media/8TB-1/gs37r/` |
| OS | Ubuntu 22.04 |
| PyTorch | 2.2.0 + CUDA 12.1 |
