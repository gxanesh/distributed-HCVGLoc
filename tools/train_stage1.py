"""
train_stage1.py
----------------
Main DDP training entry point for HCVGLoc Stage 1 (Coarse Retrieval).

Launch commands:
    # Single GPU (baseline)
    python tools/train_stage1.py --config configs/stage1_ddp_1gpu.yaml

    # 4-GPU DDP (primary)
    torchrun --nproc_per_node=4 --master_port=29500 \
        tools/train_stage1.py --config configs/stage1_ddp_4gpu.yaml

    # Resume from checkpoint
    torchrun --nproc_per_node=4 \
        tools/train_stage1.py --config configs/stage1_ddp_4gpu.yaml \
        --resume experiments/stage1/latest.pt
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
import yaml

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcvgloc.models.stage1.coarse_localizer import CoarseLocalizer
from hcvgloc.losses.composite_loss import Stage1Loss
from hcvgloc.datasets.cvusa import CVUSADataset, CVUSAGallery
from hcvgloc.datasets.cvact import CVACTDataset, CVACTGallery
from hcvgloc.datasets.vigor import VIGORDataset, VIGORGallery
from hcvgloc.datasets.sampler import DomainBalancedDistributedSampler
from hcvgloc.datasets.transforms import (
    query_train_transform, sat_train_transform, val_transform
)
from hcvgloc.utils.metrics import (
    compute_recall_at_k, build_gallery_embeddings, build_query_embeddings
)
from hcvgloc.utils.dist_utils import (
    setup_distributed, teardown_distributed,
    is_main_process, get_rank, get_world_size,
    reduce_mean, all_reduce_dict,
    compute_grad_norm, print_rank0, format_eta, barrier, CUDATimer
)
from hcvgloc.utils.logger import TrainingLogger


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HCVGLoc Stage 1 — DDP Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only (no training)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir from config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def build_train_datasets(cfg: dict):
    """Build and concatenate training datasets from config."""
    q_tf = query_train_transform()
    s_tf = sat_train_transform()

    datasets = []
    domain_ids = []   # per-sample domain labels for balanced sampler

    data_cfg = cfg["data"]

    if "cvusa" in data_cfg.get("train_datasets", []):
        ds = CVUSADataset(
            root=data_cfg["cvusa_root"], split="train",
            query_transform=q_tf, sat_transform=s_tf,
        )
        domain_ids.extend([0] * len(ds))
        datasets.append(ds)
        print_rank0(f"  [CVUSA] train: {len(ds):,} pairs")

    if "vigor" in data_cfg.get("train_datasets", []):
        ds = VIGORDataset(
            root=data_cfg["vigor_root"], split="train", area="same",
            query_transform=q_tf, sat_transform=s_tf,
        )
        domain_ids.extend([1] * len(ds))
        datasets.append(ds)
        print_rank0(f"  [VIGOR] train: {len(ds):,} pairs")

    if "cvact" in data_cfg.get("train_datasets", []):
        ds = CVACTDataset(
            root=data_cfg["cvact_root"], split="train",
            query_transform=q_tf, sat_transform=s_tf,
        )
        domain_ids.extend([2] * len(ds))
        datasets.append(ds)
        print_rank0(f"  [CVACT] train: {len(ds):,} pairs")

    combined = ConcatDataset(datasets)
    return combined, domain_ids


def build_val_loaders(cfg: dict):
    """Build validation query + gallery loaders."""
    v_tf = val_transform()
    data_cfg = cfg["data"]
    val_bs = cfg["training"].get("val_batch_size", 128)
    nw = cfg["training"].get("num_workers", 8)

    val_loaders = {}

    if "cvusa" in data_cfg.get("val_datasets", []):
        q_ds = CVUSADataset(root=data_cfg["cvusa_root"], split="val",
                            query_transform=v_tf, sat_transform=v_tf)
        g_ds = CVUSAGallery(root=data_cfg["cvusa_root"], split="val", transform=v_tf)
        val_loaders["CVUSA"] = {
            "query":   DataLoader(q_ds, batch_size=val_bs, shuffle=False,
                                  num_workers=nw, pin_memory=True),
            "gallery": DataLoader(g_ds, batch_size=val_bs, shuffle=False,
                                  num_workers=nw, pin_memory=True),
        }

    if "cvact" in data_cfg.get("val_datasets", []):
        q_ds = CVACTDataset(root=data_cfg["cvact_root"], split="val",
                            query_transform=v_tf, sat_transform=v_tf)
        g_ds = CVACTGallery(root=data_cfg["cvact_root"], split="val", transform=v_tf)
        val_loaders["CVACT"] = {
            "query":   DataLoader(q_ds, batch_size=val_bs, shuffle=False,
                                  num_workers=nw, pin_memory=True),
            "gallery": DataLoader(g_ds, batch_size=val_bs, shuffle=False,
                                  num_workers=nw, pin_memory=True),
        }

    if "vigor" in data_cfg.get("val_datasets", []):
        q_ds = VIGORDataset(root=data_cfg["vigor_root"], split="val", area="cross",
                            query_transform=v_tf, sat_transform=v_tf)
        g_ds = VIGORGallery(root=data_cfg["vigor_root"], split="val", area="cross",
                            transform=v_tf)
        val_loaders["VIGOR-cross"] = {
            "query":   DataLoader(q_ds, batch_size=val_bs, shuffle=False,
                                  num_workers=nw, pin_memory=True),
            "gallery": DataLoader(g_ds, batch_size=val_bs, shuffle=False,
                                  num_workers=nw, pin_memory=True),
        }

    return val_loaders


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model, optimizer, scaler, scheduler,
    epoch: int, best_r1: float,
    output_dir: Path,
    is_best: bool = False,
):
    """Save full training state (rank 0 only)."""
    if not is_main_process():
        return

    state = {
        "epoch": epoch,
        "best_r1": best_r1,
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "latest.pt")
    if is_best:
        torch.save(state, output_dir / "best.pt")
    if epoch % 10 == 0:
        torch.save(state, output_dir / f"epoch_{epoch:03d}.pt")


def load_checkpoint(path: str, model, optimizer=None, scaler=None, scheduler=None):
    """Load checkpoint. Returns starting epoch and best_r1."""
    state = torch.load(path, map_location="cpu")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0) + 1, state.get("best_r1", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loaders: dict, device, logger: TrainingLogger, epoch: int):
    """Run Recall@K on all validation sets. Only rank 0 prints/logs results."""
    results = {}

    for dataset_name, loaders in val_loaders.items():
        q_feats, q_labels = build_query_embeddings(model, loaders["query"], device)
        g_feats, g_labels = build_gallery_embeddings(model, loaders["gallery"], device)

        scores = compute_recall_at_k(
            q_feats, g_feats, q_labels, g_labels,
            k_vals=[1, 5, 10],
        )

        if is_main_process():
            logger.info(
                f"  [{dataset_name}] "
                + " | ".join(f"{k}={v*100:.2f}%" for k, v in scores.items())
            )
        results[dataset_name] = scores

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer,
    scaler, scheduler, device,
    epoch: int, cfg: dict,
    logger: TrainingLogger,
    timer: CUDATimer,
) -> dict:
    """Train for one full epoch. Returns dict of averaged loss components."""
    model.train()
    rank = get_rank()
    total_steps = len(loader)
    log_interval = cfg["training"].get("log_interval", 50)

    # Update GRL lambda for this epoch
    grl_lambda = CoarseLocalizer.schedule_grl_lambda if hasattr(
        CoarseLocalizer, "schedule_grl_lambda"
    ) else None
    lam = model.module.domain_head.DomainAdversarialModule.schedule_lambda(
        epoch,
        warmup_epochs=cfg["training"].get("grl_warmup_epochs", 20),
        lambda_max=cfg["training"].get("grl_lambda_max", 0.1),
    ) if hasattr(model, "module") else 0.0

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.set_grl_lambda(lam)

    epoch_metrics = {k: 0.0 for k in ["total", "nce", "js", "domain", "rotation", "uncertainty"]}
    n_batches = 0
    epoch_start = time.time()

    for step, batch in enumerate(loader):
        query_imgs  = batch["query_img"].to(device, non_blocking=True)
        sat_imgs    = batch["sat_img"].to(device, non_blocking=True)
        labels      = batch["label"].to(device, non_blocking=True)
        domain_ids  = batch["domain_id"].to(device, non_blocking=True)

        with timer.measure("forward"):
            with autocast(enabled=cfg["training"].get("use_amp", True)):
                out = model(query_imgs, sat_imgs)

                # Build positive mask from labels
                pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))   # [B, B]

                losses = criterion(
                    query_desc=out.query_desc,
                    sat_desc=out.sat_desc,
                    pos_mask=pos_mask,
                    domain_logits=out.domain_logits,
                    domain_labels=domain_ids,
                    sin_cos_pred=out.sin_cos,
                    log_var=out.log_var,
                )

        with timer.measure("backward"):
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["training"].get("grad_clip", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["training"].get("grad_clip", 1.0)
                )
                optimizer.step()

        # Accumulate metrics
        for k in epoch_metrics:
            epoch_metrics[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
        n_batches += 1

        # Step-level logging
        if step % log_interval == 0 and is_main_process():
            elapsed = time.time() - epoch_start
            steps_per_sec = (step + 1) / elapsed
            eta_sec = (total_steps - step - 1) / max(steps_per_sec, 1e-6)
            imgs_per_sec = (step + 1) * query_imgs.size(0) * get_world_size() / elapsed

            logger.info(
                f"Epoch [{epoch:03d}] "
                f"Step [{step:04d}/{total_steps}] "
                f"Loss={losses['total'].item():.4f} "
                f"NCE={losses['nce'].item():.4f} "
                f"GRL_λ={lam:.4f} "
                f"FWD={timer.elapsed_ms('forward'):.1f}ms "
                f"BWD={timer.elapsed_ms('backward'):.1f}ms "
                f"Throughput={imgs_per_sec:.0f}img/s "
                f"ETA={format_eta(eta_sec)}"
            )
            logger.log_step(
                epoch * total_steps + step,
                {k: v / n_batches for k, v in epoch_metrics.items()},
                prefix="train",
            )

    # Average over batches
    avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

    # Reduce across ranks
    if get_world_size() > 1:
        tensors = {k: torch.tensor(v, device=device) for k, v in avg_metrics.items()}
        avg_metrics = all_reduce_dict(tensors)

    return avg_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # DDP setup
    rank, local_rank, world_size = setup_distributed(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    # Output directory
    output_dir = Path(args.output_dir or cfg.get("output_dir", "experiments/stage1"))
    exp_name = cfg.get("exp_name", "stage1")

    logger = TrainingLogger(
        log_dir=str(output_dir / "logs"),
        exp_name=exp_name,
        rank=rank,
        use_wandb=cfg.get("logging", {}).get("use_wandb", False),
        use_tensorboard=cfg.get("logging", {}).get("use_tensorboard", True),
        config=cfg,
    )

    print_rank0(f"\n{'='*60}")
    print_rank0(f"  HCVGLoc Stage 1 — Distributed Training")
    print_rank0(f"  world_size={world_size}  rank={rank}  device={device}")
    print_rank0(f"  config: {args.config}")
    print_rank0(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model", {})
    model = CoarseLocalizer(
        descriptor_dim=model_cfg.get("descriptor_dim", 512),
        fpn_out_ch=model_cfg.get("fpn_out_ch", 256),
        num_fpn_layers=model_cfg.get("num_fpn_layers", 3),
        num_slots=model_cfg.get("num_slots", 8),
        num_domains=model_cfg.get("num_domains", 4),
    ).to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,   # set True if heads are sometimes skipped
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print_rank0(f"Model parameters: {total_params:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_cfg = cfg.get("losses", {})
    criterion = Stage1Loss(
        temperature=loss_cfg.get("temperature", 0.07),
        lambda_nce=loss_cfg.get("lambda_nce", 1.0),
        lambda_js=loss_cfg.get("lambda_js", 0.5),
        lambda_domain=loss_cfg.get("lambda_domain", 0.1),
        lambda_rot=loss_cfg.get("lambda_rot", 0.2),
        lambda_unc=loss_cfg.get("lambda_unc", 0.1),
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    train_cfg = cfg["training"]
    raw_model = model.module if hasattr(model, "module") else model
    param_groups = raw_model.get_param_groups(
        base_lr=train_cfg.get("lr", 3e-4),
        backbone_lr_scale=train_cfg.get("backbone_lr_scale", 0.1),
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("epochs", 100),
        eta_min=train_cfg.get("min_lr", 1e-6),
    )
    scaler = GradScaler() if train_cfg.get("use_amp", True) else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_r1 = 0, 0.0
    if args.resume:
        start_epoch, best_r1 = load_checkpoint(
            args.resume, model, optimizer, scaler, scheduler
        )
        print_rank0(f"Resumed from {args.resume} → epoch {start_epoch}, best R@1={best_r1:.4f}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print_rank0("\nLoading datasets...")
    train_dataset, domain_ids = build_train_datasets(cfg)

    sampler = DomainBalancedDistributedSampler(
        domain_ids=domain_ids,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=train_cfg.get("seed", 42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loaders = build_val_loaders(cfg)
    print_rank0(f"Train batches/epoch: {len(train_loader)} "
                f"(global batch = {train_cfg.get('batch_size',32) * world_size})")

    # ── Eval only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        logger.info("Running evaluation only...")
        results = evaluate(model, val_loaders, device, logger, epoch=0)
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    timer = CUDATimer()
    print_rank0(f"\nStarting training: epochs {start_epoch} → {train_cfg['epochs']}\n")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        sampler.set_epoch(epoch)   # critical for correct DDP shuffling

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, scheduler, device,
            epoch, cfg, logger, timer,
        )
        epoch_time = time.time() - t0
        scheduler.step()

        # Validation (every val_interval epochs, or last epoch)
        val_interval = train_cfg.get("val_interval", 5)
        if (epoch + 1) % val_interval == 0 or epoch == train_cfg["epochs"] - 1:
            barrier()   # ensure all ranks finish training before validation
            val_results = evaluate(model, val_loaders, device, logger, epoch)

            # Use CVACT R@1 as primary metric (cross-area benchmark)
            primary_r1 = 0.0
            if "CVACT" in val_results:
                primary_r1 = val_results["CVACT"].get("R@1", 0.0)
            elif val_results:
                first = list(val_results.values())[0]
                primary_r1 = first.get("R@1", 0.0)

            is_best = primary_r1 > best_r1
            if is_best:
                best_r1 = primary_r1
                print_rank0(f"  ★ New best R@1 = {best_r1*100:.2f}%")

            save_checkpoint(
                model, optimizer, scaler, scheduler,
                epoch, best_r1, output_dir / "checkpoints",
                is_best=is_best,
            )

            # Epoch-level logging
            log_data = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "epoch_time_min": epoch_time / 60,
                "lr": scheduler.get_last_lr()[0],
                "grl_lambda": (model.module if hasattr(model, "module") else model).get_grl_lambda(),
            }
            for ds_name, scores in val_results.items():
                for metric, val in scores.items():
                    log_data[f"val/{ds_name}/{metric}"] = val

            logger.log_epoch(epoch, log_data)

    print_rank0(f"\n{'='*60}")
    print_rank0(f"  Training complete.  Best R@1 = {best_r1*100:.2f}%")
    print_rank0(f"  Checkpoints saved to: {output_dir}/checkpoints/")
    print_rank0(f"{'='*60}\n")

    logger.close()
    teardown_distributed()


if __name__ == "__main__":
    main()
