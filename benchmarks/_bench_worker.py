"""
_bench_worker.py
-----------------
Worker script launched by torchrun inside scaling_benchmark.py.
Runs a short training loop and writes throughput metrics to a JSON file.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.communication_profiler import ProxyStage1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/stage1_ddp_4gpu.yaml")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--result_file", type=str, default="results/tables/_bench_tmp.json")
    return p.parse_args()


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = ProxyStage1(embed_dim=512).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()

    dummy = torch.randn(args.batch_size, 3, 224, 224, device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = ddp_model(dummy)
            loss = out.norm()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Timed training
    dist.barrier()
    t0 = time.time()
    total_steps = 0

    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = ddp_model(dummy)
                loss = out.norm()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_steps += 1

    dist.barrier()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_imgs = total_steps * args.batch_size * world_size
    throughput = total_imgs / elapsed

    if rank == 0:
        result = {
            "num_gpus": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "total_steps": total_steps,
            "wall_time_sec": round(elapsed, 3),
            "samples_per_sec": round(throughput, 2),
            "epochs": args.epochs,
        }
        Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Throughput: {throughput:.1f} img/s | Wall time: {elapsed:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
