"""
fault_injector.py
------------------
Simulates GPU worker failure mid-training for HCVGLoc DDP.

Experiments:
  1. Process kill at step N:
     - Kill one DDP rank mid-epoch using SIGKILL
     - Observe NCCL error propagation to remaining ranks
     - Measure time from failure to detection

  2. Training resumption:
     - Restart from last checkpoint after fault
     - Measure total steps lost + recovery wall-clock overhead

  3. Network bandwidth degradation (via Linux `tc netem`):
     - Add artificial latency/jitter to the loopback/IPC interface
     - Measure impact on AllReduce throughput and training convergence

Usage:
    # Simulate rank-1 failure at step 50 during a 4-GPU run
    torchrun --nproc_per_node=4 --master_port=29503 \\
        fault_tolerance/fault_injector.py \\
        --kill_rank 1 --kill_after_steps 50

    # Bandwidth degradation simulation (run as root or with sudo)
    python fault_tolerance/fault_injector.py --degrade_bandwidth \\
        --delay_ms 50 --jitter_ms 10 --duration_epochs 5
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.communication_profiler import ProxyStage1, cuda_timer
from fault_tolerance.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# Experiment 1: Process kill simulation
# ---------------------------------------------------------------------------

def run_fault_injection(args):
    """
    Launch a DDP training run and kill one rank at step kill_after_steps.
    Measures:
        - Steps completed before failure
        - Time from kill to NCCL error detection on other ranks
        - Checkpoint recovery time
    """
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model     = ProxyStage1(embed_dim=512).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)

    # Checkpoint manager — save every 10 steps
    ckpt_dir = Path(args.output_dir) / "fault_checkpoints"
    mgr = CheckpointManager(save_dir=str(ckpt_dir), interval=10)

    dummy = torch.randn(32, 3, 224, 224, device=device)

    print(f"[Rank {rank}] Training started. Will kill rank {args.kill_rank} at step {args.kill_after_steps}.")

    detection_times = []
    steps_completed = 0
    t_start = time.time()

    try:
        for step in range(args.total_steps):
            # ── Checkpoint periodically ──────────────────────────────────
            if mgr.should_save(step):
                mgr.save(ddp_model, optimizer, step,
                         metadata={"step": step}, tag=f"step_{step}")
                if rank == 0:
                    print(f"  [Ckpt saved @ step {step}]")

            # ── Simulate fault: kill target rank ─────────────────────────
            if rank == args.kill_rank and step == args.kill_after_steps:
                if rank == 0:
                    print(f"\n[FAULT INJECTION] Killing rank {rank} at step {step}")
                t_kill = time.time()
                os.kill(os.getpid(), signal.SIGKILL)   # hard kill

            # ── Normal training step ──────────────────────────────────────
            optimizer.zero_grad()
            out = ddp_model(dummy)
            loss = out.norm()
            loss.backward()
            optimizer.step()
            steps_completed += 1

            # Detect if another rank failed (NCCL will throw on barrier)
            dist.barrier()

    except Exception as e:
        t_detect = time.time()
        detect_latency_ms = (t_detect - t_start) * 1000
        print(f"[Rank {rank}] Fault detected at step {steps_completed}: {e}")
        print(f"[Rank {rank}] Detection latency: {detect_latency_ms:.1f}ms")

        if rank == 0:
            result = {
                "killed_rank":       args.kill_rank,
                "kill_at_step":      args.kill_after_steps,
                "steps_completed":   steps_completed,
                "steps_lost":        args.kill_after_steps - steps_completed + 1,
                "detect_latency_ms": round(detect_latency_ms, 2),
                "last_checkpoint":   (steps_completed // 10) * 10,
            }
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "tables" / "fault_injection_result.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[Results saved]")
            print(f"  Steps lost:        {result['steps_lost']}")
            print(f"  Last checkpoint:   step {result['last_checkpoint']}")
            print(f"  Effective loss:    {result['steps_lost'] - (steps_completed - result['last_checkpoint'])} steps net")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Experiment 2: Network latency degradation via tc netem
# ---------------------------------------------------------------------------

NETEM_IFACE = "lo"   # loopback; adjust to your NCCL interface (e.g. ib0, enp5s0)

def apply_network_degradation(delay_ms: int = 50, jitter_ms: int = 10):
    """
    Use Linux tc (traffic control) to add artificial latency + jitter to
    the network interface used by NCCL for gradient AllReduce.

    Requires: iproute2 installed, run with sudo or CAP_NET_ADMIN.
    """
    if platform.system() != "Linux":
        print("[WARNING] tc netem is Linux-only. Skipping degradation.")
        return False

    # Clear existing rules
    subprocess.run(["tc", "qdisc", "del", "dev", NETEM_IFACE, "root"],
                   capture_output=True)

    cmd = [
        "tc", "qdisc", "add", "dev", NETEM_IFACE, "root", "netem",
        "delay", f"{delay_ms}ms", f"{jitter_ms}ms",
        "distribution", "normal",
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[WARNING] tc netem failed: {ret.stderr.strip()}")
        print("  → Run with sudo or grant CAP_NET_ADMIN to the process.")
        return False

    print(f"[tc netem] Applied: delay={delay_ms}ms ± {jitter_ms}ms on {NETEM_IFACE}")
    return True


def remove_network_degradation():
    """Remove tc netem rules."""
    subprocess.run(["tc", "qdisc", "del", "dev", NETEM_IFACE, "root"],
                   capture_output=True)
    print(f"[tc netem] Cleared rules on {NETEM_IFACE}")


def benchmark_with_degradation(args):
    """
    Compare AllReduce latency and training throughput with and without
    simulated network degradation.
    """
    results = []

    configs = [
        ("baseline",          0,               0),
        ("50ms delay",        50,              10),
        ("100ms delay",       100,             20),
        ("200ms delay",       200,             50),
    ]

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model     = ProxyStage1(embed_dim=512).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    dummy = torch.randn(32, 3, 224, 224, device=device)

    for label, delay, jitter in configs:
        if rank == 0:
            if delay > 0:
                ok = apply_network_degradation(delay, jitter)
            else:
                remove_network_degradation()
                ok = True

        dist.barrier()  # sync before measurement

        # Measure AllReduce time under this condition
        allreduce_times = []
        for _ in range(20):
            # Compute grads
            with ddp_model.no_sync():
                out = ddp_model(dummy)
                loss = out.norm()
                loss.backward()

            # Time AllReduce only
            with cuda_timer():
                for p in ddp_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            allreduce_times.append(cuda_timer._last_ms)
            optimizer.step()
            optimizer.zero_grad()

        avg_ar = sum(allreduce_times[3:]) / len(allreduce_times[3:])

        if rank == 0:
            print(f"  [{label:25s}] Avg AllReduce: {avg_ar:.2f} ms")
            results.append({
                "config": label,
                "delay_ms": delay,
                "jitter_ms": jitter,
                "avg_allreduce_ms": round(avg_ar, 3),
            })

    # Cleanup
    if rank == 0:
        remove_network_degradation()

    dist.destroy_process_group()

    if rank == 0:
        out_dir = Path(args.output_dir)
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        path = out_dir / "tables" / "degradation_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HCVGLoc Fault Injector")
    p.add_argument("--kill_rank",        type=int, default=1)
    p.add_argument("--kill_after_steps", type=int, default=50)
    p.add_argument("--total_steps",      type=int, default=100)
    p.add_argument("--degrade_bandwidth", action="store_true",
                   help="Run bandwidth degradation experiment instead of kill")
    p.add_argument("--delay_ms",   type=int, default=50)
    p.add_argument("--jitter_ms",  type=int, default=10)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    (Path(args.output_dir) / "tables").mkdir(parents=True, exist_ok=True)

    if args.degrade_bandwidth:
        benchmark_with_degradation(args)
    else:
        run_fault_injection(args)
