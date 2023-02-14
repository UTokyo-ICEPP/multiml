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
    def __init__(self,
                 ddp=True,
                 addr='localhost',
                 port='12355',
                 backend='nccl',
                 find_unused_parameters=False,
                 **kwargs):
        """Initialize the pytorch DDP task."""
        super().__init__(**kwargs)
        self._ddp = ddp
        self._addr = addr
        self._port = port
        self._backend = backend
        self._find_unused_parameters = find_unused_parameters
        self._data_parallel = (not ddp) and (self._gpu_ids is not None)

    def compile_model(self, rank=None, world_size=None):
        """ Build model.
        """
        super().compile_model()
        if self._ddp and dist.is_initialized():
            if (not isinstance(self._device, int)) and ('cpu' in self._device.type):
                self.ml.model = DDP(self.ml.model)
            else:
                self.ml.model = DDP(self.ml.model,
                                    device_ids=[self._device],
                                    find_unused_parameters=self._find_unused_parameters)

    def compile_device(self):
        """ Compile device.
        """
        if not self._ddp:
            super().compile_device()

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
                           dataset_args=None,
                           dataloader_args=None):
        """Prepare dataloader.
        """
        if not self._ddp:
            return super().prepare_dataloader(data, phase, dataset_args, dataloader_args)

        dataset_args_tmp = dict(preload=False, callbacks=[])
        dataset_args_tmp.update(dataset_args)
        dataset = self.get_dataset(data=data, phase=phase, **dataset_args_tmp)

        dataloader_args_tmp = dict(dataset=dataset, pin_memory=True, num_workers=self._num_workers)
        dataloader_args_tmp.update(dataloader_args)

        dataset = self.get_dataset(data=data, phase=phase, **dataset_args_tmp)

        if not self._batch_sampler:
            shuffle = True if phase in ('train', 'valid') else False

            if phase == 'train':
                self._sampler = self.get_distributed_sampler(phase, dataset, rank, world_size,
                                                             self._batch_sampler)
                return DataLoader(batch_size=self._get_batch_size(phase),
                                  sampler=self._sampler,
                                  **dataloader_args_tmp)
            else:
                return DataLoader(batch_size=self._get_batch_size(phase),
                                  shuffle=shuffle,
                                  **dataloader_args_tmp)
        else:
            if phase == 'train':
                sampler = self.get_distributed_sampler(phase, dataset, rank, world_size,
                                                       self._batch_sampler)
                return DataLoader(sampler=sampler, batch_size=None, **dataloader_args_tmp)
            else:
                sampler = self.get_batch_sampler(phase, dataset)
                return DataLoader(sampler=sampler, batch_size=None, **dataloader_args_tmp)

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

    def fix_submodule(self, target):
        """ Fix given parameters of model.
        """
        if self._ddp:
            for param in self.ml.model.get_submodule('module.' + target).parameters():
                param.requires_grad = False
        else:
            for param in self.ml.model.get_submodule(target).parameters():
                param.requires_grad = False

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

        if 'cpu' not in self._device.type:
            self._device = self._gpu_ids[rank]

        # FIXME
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    def cleanup(self):
        """Cleanup multi processing."""
        if not self._ddp:
            return

        dist.destroy_process_group()
