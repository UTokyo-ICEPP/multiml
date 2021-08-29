"""PytorchDDPTask module."""

import os
from abc import abstractmethod

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from multiml import logger
from multiml.task.pytorch import PytorchBaseTask
from multiml.task.pytorch import pytorch_util as util


class PytorchDDPTask(PytorchBaseTask):
    """Distributed data parallel (DDP) task for PyTorch model."""
    def __init__(self, ddp=True, addr='localhost', port='12355', backend='nccl', **kwargs):
        """Initialize the pytorch DDP task."""
        super().__init__(**kwargs)
        self._ddp = ddp
        self._addr = addr
        self._port = port
        self._backend = backend
        self._data_parallel = False

    def compile_model(self, rank=None, world_size=None):
        """ Build model.
        """
        super().compile_model()
        if self._ddp and dist.is_initialized():
            self.ml.model = DDP(self.ml.model, device_ids=[self._device])

    def compile_device(self):
        """ Compile device.
        """
        pass

    def dump_model(self, extra_args=None):
        """ Dump current pytorch model.
        """
        if not self._ddp:
            super().dump_model(extra_args)
            return

        if not util.is_master_process():
            return

        args_dump_ml = dict(ml_type='pytorch')

        if extra_args is not None:
            args_dump_ml.update(extra_args)

        args_dump_ml['model'] = self.ml.model.module
        super().dump_model(args_dump_ml)

    @logger.logging
    def execute(self):
        """Execute the pytorch DDP task.

        Multi processes are launched
        """
        if not self._ddp:
            self.execute_mp()
            return

        if not isinstance(self._gpu_ids, list):
            raise ValueError('DDP mode enabled. Please provide gpu_ids.')

        world_size = len(self._gpu_ids)
        mp.spawn(self.execute_mp, args=(world_size, ), nprocs=world_size, join=True)

    @abstractmethod
    def execute_mp(self, rank, world_size):
        """User defined algorithms.

        Examples:
            >>> setup(rank, world_size)
            >>> # your algorithms
            >>> # ...
            >>> cleanup()
        """

    def setup(self, rank, world_size):
        """Setup multi processing."""
        if not self._ddp:
            return

        if self._pool_id is not None:
            counter, num_workers, num_jobs = self._pool_id
            self._port = str(int(self._port) + counter)

        os.environ['MASTER_ADDR'] = self._addr
        os.environ['MASTER_PORT'] = self._port
        dist.init_process_group(self._backend, rank=rank, world_size=world_size)

        self._device = rank

    def cleanup(self):
        """Cleanup multi processing."""
        if not self._ddp:
            return

        dist.destroy_process_group()
