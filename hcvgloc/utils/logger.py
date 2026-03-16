"""
logger.py
----------
Rank-aware training logger for HCVGLoc distributed training.

Only rank 0 writes to disk and (optionally) to WandB / TensorBoard.
All ranks can log locally without I/O conflicts.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch


def setup_logger(
    name: str = "hcvgloc",
    log_dir: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """
    Create a logger that only writes files on rank 0.

    Args:
        name:    logger name
        log_dir: directory for log file (rank 0 only)
        rank:    current DDP rank
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — all ranks print INFO+
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler — rank 0 only
    if rank == 0 and log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"train_{ts}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class TrainingLogger:
    """
    Structured training logger with optional WandB + TensorBoard integration.
    Only active on rank 0.
    """

    def __init__(
        self,
        log_dir: str,
        exp_name: str,
        rank: int = 0,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        config: dict = None,
    ):
        self.rank = rank
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("hcvgloc.train", str(log_dir), rank)
        self._history: list = []

        if rank != 0:
            self._wandb = None
            self._tb = None
            return

        # Save config
        if config is not None:
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # TensorBoard
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.logger.info(f"TensorBoard logging → {self.log_dir}/tensorboard")
            except ImportError:
                self.logger.warning("TensorBoard not available; skipping.")

        # WandB
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project="hcvgloc-distributed", name=exp_name, config=config)
                self._wandb = wandb
                self.logger.info("WandB logging enabled.")
            except ImportError:
                self.logger.warning("WandB not installed; skipping.")

    def log_step(self, step: int, metrics: Dict[str, Any], prefix: str = "train"):
        """Log a dict of scalar metrics at a training step."""
        if self.rank != 0:
            return
        tagged = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if self._tb:
            for k, v in tagged.items():
                self._tb.add_scalar(k, v, step)
        if self._wandb:
            self._wandb.log(tagged, step=step)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch-level metrics and save to history JSON."""
        if self.rank != 0:
            return
        record = {"epoch": epoch, **metrics}
        self._history.append(record)
        with open(self.log_dir / "history.json", "w") as f:
            json.dump(self._history, f, indent=2)
        self.logger.info(
            f"Epoch {epoch:03d} | " +
            " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in metrics.items())
        )
        self.log_step(epoch, metrics, prefix="epoch")

    def info(self, msg: str):
        if self.rank == 0:
            self.logger.info(msg)

    def warning(self, msg: str):
        if self.rank == 0:
            self.logger.warning(msg)

    def close(self):
        if self._tb:
            self._tb.close()
        if self._wandb:
            self._wandb.finish()
