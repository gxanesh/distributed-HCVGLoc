"""
sampler.py
-----------
Domain-balanced DistributedSampler for HCVGLoc Stage 1.

Problem: When training with CVUSA + VIGOR + CVACT simultaneously, naive
random sampling produces imbalanced batches skewed toward the largest dataset.
This causes the GRL domain classifier to over-fit to majority domains.

Solution: DomainBalancedDistributedSampler builds batches with equal
representation from each domain, while still sharding correctly across DDP ranks.

Design:
    1. Group indices by domain_id.
    2. Each epoch, shuffle within each domain.
    3. Interleave: take ceil(N / num_domains) samples from each domain,
       then shard into world_size chunks.

Usage:
    sampler = DomainBalancedDistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    # Call sampler.set_epoch(epoch) at the start of each epoch!
"""

import math
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from typing import Iterator, List, Optional


class DomainBalancedDistributedSampler(Sampler):
    """
    Domain-balanced sampler compatible with DistributedDataParallel.

    Each batch (across the full world) contains an equal number of samples
    from each dataset domain. Indices are then sharded across DDP ranks.

    Args:
        dataset:        dataset with items that have a 'domain_id' field
        domain_ids:     list of domain_id per sample (len = len(dataset))
        num_replicas:   world_size (default: from dist.get_world_size())
        rank:           current rank (default: from dist.get_rank())
        shuffle:        whether to shuffle within each domain each epoch
        seed:           random seed for reproducibility
        drop_last:      drop last incomplete batch
    """

    def __init__(
        self,
        domain_ids: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Group indices by domain
        self.domain_to_indices = {}
        for idx, did in enumerate(domain_ids):
            self.domain_to_indices.setdefault(did, []).append(idx)

        self.num_domains = len(self.domain_to_indices)
        # Target: equal samples from each domain, rounded to largest domain
        max_per_domain = max(len(v) for v in self.domain_to_indices.values())
        self.total_per_domain = max_per_domain
        self.total_size = self.total_per_domain * self.num_domains

        # Per-replica size
        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        if drop_last:
            self.num_samples = math.floor(self.total_size / self.num_replicas)

    def set_epoch(self, epoch: int):
        """Must be called at the start of each epoch for correct shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Build balanced index list
        balanced_indices = []
        for domain_id in sorted(self.domain_to_indices.keys()):
            indices = self.domain_to_indices[domain_id]
            n = self.total_per_domain

            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # Oversample if needed to reach total_per_domain
            if len(indices) < n:
                repeat = math.ceil(n / len(indices))
                indices = (indices * repeat)[:n]
            else:
                indices = indices[:n]

            balanced_indices.extend(indices)

        # Global shuffle across domains (optional but helps)
        if self.shuffle:
            perm = torch.randperm(len(balanced_indices), generator=g).tolist()
            balanced_indices = [balanced_indices[i] for i in perm]

        # Pad to be divisible by num_replicas
        if not self.drop_last:
            pad = self.num_samples * self.num_replicas - len(balanced_indices)
            balanced_indices = balanced_indices + balanced_indices[:pad]
        else:
            n = (len(balanced_indices) // self.num_replicas) * self.num_replicas
            balanced_indices = balanced_indices[:n]

        # Shard: this rank gets every num_replicas-th element starting at rank
        assert len(balanced_indices) == self.num_samples * self.num_replicas
        rank_indices = balanced_indices[self.rank:len(balanced_indices):self.num_replicas]
        assert len(rank_indices) == self.num_samples

        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


class StandardDistributedSampler(Sampler):
    """
    Plain DistributedSampler (no domain balancing).
    Drop-in replacement for torch.utils.data.DistributedSampler
    with explicit epoch tracking.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples
