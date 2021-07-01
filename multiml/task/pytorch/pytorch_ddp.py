"""PytorchDDPTask module."""

import os
from abc import abstractmethod

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multiml import logger
from multiml.task.pytorch import PytorchBaseTask


class PytorchDDPTask(PytorchBaseTask):
    """Distributed data parallel (DDP) task for PyTorch model."""
    def __init__(self, **kwargs):
        """Initialize the pytorch DDP task."""
        super().__init__(**kwargs)
        self._data_parallel = False

    @logger.logging
    def execute(self):
        """Execute the pytorch DDP task.

        Multi processes are launched
        """
        world_size = len(self._gpu_ids)
        mp.spawn(self.execute_mp, args=(world_size, ), nprocs=world_size, join=True)

    @abstractmethod
    def execute_mp(self, rank, world_size):
        """User defined algorithms."""

    @staticmethod
    def setup(rank, world_size):
        """Setup multi processing."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
