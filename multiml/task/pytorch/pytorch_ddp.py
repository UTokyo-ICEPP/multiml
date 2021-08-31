"""PytorchDDPTask module."""

import os
from abc import abstractmethod

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
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

    def prepare_dataloader(self,
                           rank,
                           world_size,
                           data=None,
                           phase=None,
                           batch=False,
                           pin_memory=True,
                           preload=False,
                           callbacks=None):
        """Prepare dataloader.
        """
        if not self._ddp:
            return super().prepare_dataloader(data, phase, batch, pin_memory, preload, callbacks)

        dataset = self.get_dataset(data=data, phase=phase, preload=preload, callbacks=callbacks)
        dataloader_args = dict(dataset=dataset,
                               pin_memory=pin_memory,
                               num_workers=self._num_workers)

        if not batch:
            shuffle = True if phase in ('train', 'valid') else False

            if phase == 'train':
                self._sampler = self.get_distributed_sampler(phase, dataset, rank, world_size,
                                                             batch)
                return DataLoader(batch_size=self._get_batch_size(phase),
                                  sampler=self._sampler,
                                  **dataloader_args)
            else:
                return DataLoader(batch_size=self._get_batch_size(phase),
                                  shuffle=shuffle,
                                  **dataloader_args)
        else:
            if phase == 'train':
                sampler = self.get_distributed_sampler(phase, dataset, rank, world_size, batch)
                return DataLoader(sampler=sampler, batch_size=None, **dataloader_args)
            else:
                sampler = self.get_batch_sampler(phase, dataset)
                return DataLoader(sampler=sampler, batch_size=None, **dataloader_args)

    def get_distributed_sampler(self, phase, dataset, rank, world_size, batch=False):
        """ Get batch sampler.
        """
        if batch:
            from torchnlp.samplers import DistributedBatchSampler
            sampler = self.get_batch_sampler(phase, dataset)
            sampler = DistributedBatchSampler(sampler, num_replicas=world_size, rank=rank)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=world_size,
                                                                      rank=rank)
        return sampler

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
    def execute_mp(self, rank=None, world_size=None):
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

        # FIXME
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    def cleanup(self):
        """Cleanup multi processing."""
        if not self._ddp:
            return

        dist.destroy_process_group()
