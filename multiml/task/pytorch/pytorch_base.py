"""PytorchBaseTask module."""
import copy
import math
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import torch
from torch import optim, Tensor, LongTensor
from torch.nn.modules import loss as tl
from torch.utils.data import DataLoader

from multiml import logger, const
from multiml.task.basic import MLBaseTask
from multiml.task.pytorch import modules
from multiml.task.pytorch.datasets import StoreGateDataset, NumpyDataset
from multiml.task.pytorch import pytorch_util as util
from multiml.task.pytorch import pytorch_metrics as metrics


class PytorchBaseTask(MLBaseTask):
    """Base task for PyTorch model.

    Examples:
        >>> # your pytorch model
        >>> class MyPytorchModel(nn.Module):
        >>>     def __init__(self, inputs=2, outputs=2):
        >>>         super(MyPytorchModel, self).__init__()
        >>>
        >>>         self.fc1 = nn.Linear(inputs, outputs)
        >>>         self.relu = nn.ReLU()
        >>>
        >>>     def forward(self, x):
        >>>         return self.relu(self.fc1(x))
        >>>
        >>> # create task instance
        >>> task = PytorchBaseTask(storegate=storegate,
        >>>                        model=MyPytorchModel,
        >>>                        input_var_names=('x0', 'x1'),
        >>>                        output_var_names='outputs-pytorch',
        >>>                        true_var_names='labels',
        >>>                        optimizer='SGD',
        >>>                        optimizer_args=dict(lr=0.1),
        >>>                        loss='CrossEntropyLoss')
        >>> task.set_hps({'num_epochs': 5})
        >>> task.execute()
        >>> task.finalize()
    """
    def __init__(self,
                 device='cpu',
                 gpu_ids=None,
                 torchinfo=False,
                 amp=False,
                 torch_compile=False,
                 dataset_args=None,
                 dataloader_args=None,
                 batch_sampler=False,
                 metric_sample=1,
                 **kwargs):
        """Initialize the pytorch base task.

        Args:
            device (str or obj): pytorch device, e.g. 'cpu', 'cuda'.
            gpu_ids (list): GPU identifiers, e.g. [0, 1, 2]. ``data_parallel`` mode is enabled if
                ``gpu_ids`` is given.
            torchinfo (bool): show torchinfo summary after model compile.
            amp (bool): *(expert option)* enable amp mode.
            torch_compile (bool): *(expert option)* enable torch.compile.
            dataset_args (dict): args passed to default DataSet creation.
            dataloader_args (dict): args passed to default DataLoader creation.
            batch_sampler (bool): user batch_sampler or not.
            metric_sample (float or int): sampling ratio for running metrics.
        """
        super().__init__(**kwargs)

        if isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

        self._is_gpu = 'cuda' in self._device.type
        self._data_parallel = self._is_gpu and (gpu_ids is not None)

        logger.info(f'{self._name}: PyTorch device: {self._device}')

        self._gpu_ids = gpu_ids
        self._torchinfo = torchinfo
        self._amp = amp
        self._torch_compile = torch_compile
        self._dataset_args = dataset_args
        self._dataloader_args = dataloader_args
        self._batch_sampler = batch_sampler

        self._pbar_args = const.PBAR_ARGS
        self._metric_sample = metric_sample
        self._pred_index = None
        self._early_stopping = False
        self._sampler = None
        self._scaler = None

        if self._metrics is None:
            self._metrics = ['loss']

        if self._max_patience is not None:
            self._early_stopping = True

        if isinstance(self._dataset_args, dict) and ('train' not in self._dataset_args):
            self._dataset_args = dict(train=dataset_args.copy(),
                                      valid=dataset_args.copy(),
                                      test=dataset_args.copy())

        if self._dataset_args is None:
            self._dataset_args = dict(train={}, valid={}, test={})

        if isinstance(self._dataloader_args, dict) and ('train' not in self._dataloader_args):
            self._dataloader_args = dict(train=dataloader_args,
                                         valid=dataloader_args,
                                         test=dataloader_args)

        if self._dataloader_args is None:
            self._dataloader_args = dict(train={}, valid={}, test={})

    def compile(self):
        """Compile pytorch ml objects."""
        self.compile_device()
        super().compile()

    def compile_model(self):
        """Compile pytorch model.

        Compile model based on self._model type, which is usually set by ``__init__()`` or
        ``build_model() method. Compiled model is set to ``self.ml.model`` and moved to
        ``self._device``.
        """
        if self._model is None:
            return

        if self._is_gpu and (not torch.cuda.is_available()):
            raise ValueError(f'{self._device} is not available')

        self.ml.model = util.compile(self._model, self._model_args, modules)

        if self._torch_compile:
            self.ml.model = torch.compile(self.ml.model)

        if self.pred_var_names is not None:
            self._pred_index = self.get_pred_index()

        if self._load_weights:
            self.load_model()
            self.load_metadata()

        if self._data_parallel:
            self.ml.model = torch.nn.DataParallel(self.ml.model, device_ids=self._gpu_ids)
            self.ml.model.to(self._device)

        else:
            self.ml.model.to(self._device)

        if self._torchinfo:
            from torchinfo import summary
            summary(self.ml.model)

    def compile_optimizer(self):
        """Compile pytorch optimizer and scheduler.

        Compile optimizer based on self._optimizer type, which is usually set by ``__init__()``
        method. Compiled optimizer is set to ``self.ml.optimizer``.
        """
        if self._optimizer is None:
            return

        optimizer_args = copy.deepcopy(self._optimizer_args)
        if 'params' not in optimizer_args:
            optimizer_args['params'] = list(self.ml.model.parameters())

        if 'per_params' in optimizer_args:
            optimizer_args['params'] = []
            for per_param in optimizer_args['per_params']:
                if getattr(self.ml.model, 'module', False):
                    params = getattr(self.ml.model.module, per_param['params']).parameters()
                    per_param['params'] = params
                else:
                    params = getattr(self.ml.model, per_param['params']).parameters()
                    per_param['params'] = params

                optimizer_args['params'].append(per_param)
            del optimizer_args['per_params']

        self.ml.optimizer = util.compile(self._optimizer, optimizer_args, optim)

        if self._scheduler is not None:
            scheduler_args = copy.copy(self._scheduler_args)
            scheduler_args['optimizer'] = self.ml.optimizer
            self.ml.scheduler = util.compile(self._scheduler, scheduler_args, optim.lr_scheduler)

    def compile_loss(self):
        """Compile pytorch loss.

        Compile loss based on self._loss type, which is usually set by ``__init__()`` method.
        Compiled loss is set to ``self.ml.loss``.
        """

        if self._loss is None:
            return

        if 'weight' in self._loss_args:
            if isinstance(self._loss_args['weight'], list):
                weight = torch.tensor(self._loss_args['weight'])
                weight = weight.to(self._device)
                self._loss_args['weight'] = weight

        self.ml.loss = util.compile(self._loss, self._loss_args, tl)
        self.ml.loss_weights = self._loss_weights

        if isinstance(self.ml.loss, list):
            for ii, ml_loss in enumerate(self.ml.loss):
                self.ml.loss[ii] = ml_loss.to(self._device)
        else:
            self.ml.loss = self.ml.loss.to(self._device)

    def compile_device(self):
        """Compile device.

        This method is valid only for multiprocessing mode so far. Devices are set based on
        ``pool_id``.
        """
        if self._pool_id is None:
            return  # nothing to do so far

        if self._data_parallel:
            logger.warn(f'data_parallel with pool_id mode: gpu_ids {self._gpu_ids}')
            return

        if 'cuda' in self._device.type:
            counter, num_workers, num_jobs = self._pool_id
            cuda_id = num_workers[counter]

            if cuda_id >= torch.cuda.device_count():
                cuda_id = cuda_id % torch.cuda.device_count()

            logger.debug(f'Apply cuda_id:{cuda_id}')
            self._device = torch.device(f'cuda:{cuda_id}')

    def load_model(self):
        """Load pre-trained pytorch model weights.
        
        If model_path is given with ``:`` delimiter, e.g. my_model_path:features, only partial 
        weights are loaded.
        """
        model_path = super().load_model()

        # partial weights
        if ':' in model_path:
            model_path, *partial = model_path.split(':')
            model_dict = self.ml.model.state_dict()
            state_dict = torch.load(model_path)
            new_state_dict = {}
            for key, value in state_dict.items():
                for ipartial in partial:
                    if ipartial in key:
                        new_state_dict[key] = value
            model_dict.update(new_state_dict)
            self.ml.model.load_state_dict(model_dict)
        else:
            self.ml.model.load_state_dict(torch.load(model_path))

    def dump_model(self, extra_args=None):
        """Dump current pytorch model."""
        args_dump_ml = dict(ml_type='pytorch')

        if self._data_parallel:
            args_dump_ml['model'] = self.ml.model.module
        else:
            args_dump_ml['model'] = self.ml.model

        if extra_args is not None:
            args_dump_ml.update(extra_args)

        super().dump_model(args_dump_ml)

    def prepare_dataloaders(self, phases=None, dataset_args=None, dataloader_args=None, **kwargs):
        """Prepare dataloaders for all phases.

        Args:
            dataset_args (dict): args passed to get_dataset. Dict format must be
                {'train': {...}, 'valid': {...}, 'test': {...}}.
            dataloader_args (dict): args passed to prepare_dataloader. Dict format must be
                {'train': {...}, 'valid': {...}, 'test': {...}}.
            kwargs (dict): arbitrary args.
        """
        if dataset_args is None:
            dataset_args = self._dataset_args

        if dataloader_args is None:
            dataloader_args = self._dataloader_args

        dataloaders = {}

        if phases is None:
            phases = const.PHASES

        for phase in phases:
            dataloaders[phase] = self.prepare_dataloader(phase=phase,
                                                         dataset_args=dataset_args[phase],
                                                         dataloader_args=dataloader_args[phase],
                                                         **kwargs)

        return dataloaders

    def prepare_dataloader(self, data=None, phase=None, dataset_args=None, dataloader_args=None):
        """Prepare dataloader.

        If inputs are given, tensor_dataset() is called. If inputs are None, storegate_dataset with
        given phase is called.

        Args:
            data (ndarray): data passed to tensor_dataset().
            phase (str): phase passed to storegate_dataset().
            dataset_args(dict):
                preload (bool): If True, all data are preloaded in the initialization of Dataset class.
                callbacks (list): list of data augmentation functions.
            dataloader_args(dict):
                pin_memory (bool): pin_memory for DataLoader.
                batch (bool): batch sampler or not..

        Returns:
            DataLoader: Pytorch dataloader instance.
        """
        dataset_args_tmp = dict(preload=False, callbacks=[])
        dataset_args_tmp.update(dataset_args)
        dataset = self.get_dataset(data=data, phase=phase, **dataset_args_tmp)

        dataloader_args_tmp = dict(dataset=dataset, pin_memory=True, num_workers=self._num_workers)

        if self._num_workers > 0:
            dataloader_args_tmp['persistent_workers'] = True

        dataloader_args_tmp.update(dataloader_args)

        if not self._batch_sampler:
            shuffle = True if phase in ('train', 'valid') else False
            return DataLoader(batch_size=self._get_batch_size(phase, len(dataset)),
                              shuffle=shuffle,
                              **dataloader_args_tmp)
        else:
            sampler = self.get_batch_sampler(phase, dataset)
            return DataLoader(batch_size=None, sampler=sampler, **dataloader_args_tmp)

    def fit(self, train_data=None, valid_data=None, dataloaders=None, valid_step=1, dump=False):
        """Train model over epoch.

        This methods train and valid model over epochs by calling ``step_epoch()`` method.
        train and valid need to be provided by ``train_data`` and ``valid_data`` options, or
        ``dataloaders`` option.

        Args:
            train_data (ndarray): If ``train_data`` is given, data are converted
                to ``TendorDataset`` and set to ``dataloaders['train']``.
            valid_data (ndarray): If ``valid_data`` is given, data are converted
                to ``TendorDataset`` and set to ``dataloaders['valid']``.
            dataloaders (dict): dict of dataloaders, dict(train=xxx, valid=yyy).
            valid_step (int): step to process validation.
            dump (bool): If True, results are dumped together with model.

        Returns:
            list: history data of train and valid.
        """
        self.ml.validate('train')

        if dataloaders is None:
            dataloaders = dict(train=self.prepare_dataloader(train_data, 'train',
                                                             self._dataset_args['train'],
                                                             self._dataloader_args['train']),
                               valid=self.prepare_dataloader(valid_data, 'valid',
                                                             self._dataset_args['valid'],
                                                             self._dataloader_args['valid']))

        early_stopping = util.EarlyStopping(patience=self._max_patience)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._is_gpu)

        history = {'train': [], 'valid': []}

        for epoch in range(1, self._num_epochs + 1):
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)

            # train
            self.ml.model.train()
            result = self.step_epoch(epoch, 'train', dataloaders['train'])
            history['train'].append(result)

            # valid
            if (const.VALID in self.phases) and (epoch % valid_step == 0):
                self.ml.model.eval()
                result = self.step_epoch(epoch, 'valid', dataloaders['valid'])
                history['valid'].append(result)

                if early_stopping(result['loss'], self.ml.model):
                    break

            if self.ml.scheduler is not None:
                self.ml.scheduler.step()

        if self._early_stopping:
            self.ml.model.load_state_dict(early_stopping.best_model)

        if dump:
            self.dump_model(dict(result=history))

        return history

    def predict(self, data=None, dataloader=None, phase=None, label=False):
        """Predict model.

        This method predicts and returns results. Data need to be provided by ``data`` option, or
        setting property of ``dataloaders`` directory.

        Args:
            data (ndarray): If ``data`` is given, data are converted to
                ``TendorDataset`` and set to ``dataloaders['test']``.
            dataloader (obj): dataloader instance.
            phase (str): 'all' or 'train' or 'valid' or 'test' to specify dataloaders.
            label (bool): If True, returns metric results based on labels.

        Returns:
            ndarray or list: results of prediction.
        """
        self.ml.validate('test')
        self.ml.model.eval()

        if dataloader is None:
            if (phase in self._dataset_args) and (phase in self._dataloader_args):
                dataloader = self.prepare_dataloader(data, phase, self._dataset_args[phase],
                                                     self._dataloader_args[phase])
            else:
                dataloader = self.prepare_dataloader(data, phase, dict(), dict())

        results = self.step_epoch(0, 'test', dataloader, label)

        if label:
            return results
        else:
            return results['pred']

    def step_epoch(self, epoch, phase, dataloader, label=True):
        """Process model for given epoch and phase.

        ``ml.model``, ``ml.optimizer`` and ``ml.loss`` need to be set before calling this method,
        please see ``compile()`` method.

        Args:
            epoch (int): epoch numer.
            phase (str): *train* mode or *valid* mode.
            dataloader (obj): dataloader instance.
            label (bool): If True, returns metric results based on labels.

        Returns:
            dict: dict of result.
        """
        epoch_metric = metrics.EpochMetric(self._metrics, label, self.ml)
        num_batches = len(dataloader)
        pbar_args = dict(total=num_batches, disable=self._disable_tqdm())
        pbar_args.update(self._pbar_args)
        pbar_desc = self._get_pbar_description(epoch, phase)

        if self._metric_sample <= 1.0:
            metric_step = 1.0 // self._metric_sample
            metric_step = math.floor(metric_step)
        else:
            metric_step = num_batches / self._metric_sample
            metric_step = math.ceil(metric_step)

        results = {}
        with tqdm(**pbar_args) as pbar:
            pbar.set_description(pbar_desc)

            for ii, data in enumerate(dataloader):
                batch_result = self.step_batch(data, phase, label)

                if phase == 'test':
                    epoch_metric.pred(batch_result)

                if phase == 'train':
                    if (ii % metric_step == 0) or (ii == num_batches - 1):
                        results.update(epoch_metric(batch_result))
                        pbar_metrics = metrics.get_pbar_metric(results)
                        pbar.set_postfix(pbar_metrics)
                else:
                    results.update(epoch_metric(batch_result))
                    pbar_metrics = metrics.get_pbar_metric(results)
                    pbar.set_postfix(pbar_metrics)
                pbar.update(1)

        if self._verbose == 2:
            logger.info(f'{pbar_desc} {pbar_metrics}')

        if phase == 'test':
            results['pred'] = epoch_metric.all_preds()

        return results

    def step_batch(self, data, phase, label=True):
        """Process batch data and update weights.

        Args:
            data (obj): inputs and labels data.
            phase (str): *train* mode or *valid* mode or *test* mode.
            label (bool): If True, returns metric results based on labels.

        Returns:
            dict: dict of result.
        """
        inputs, labels = data  # FIXME: data without labels
        inputs = self.add_device(inputs, self._device)
        labels = self.add_device(labels, self._device)

        result = {'batch_size': util.inputs_size(inputs)}
        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast(self._is_gpu and self._amp):
                outputs_all = self.step_model(inputs)

                if self._pred_index is not None:
                    outputs = self._select_pred_data(outputs_all)
                else:
                    outputs = outputs_all

                if label:
                    loss_result = self.step_loss(outputs, labels)
                else:
                    loss_result = None

            if phase == 'train':
                self.step_optimizer(loss_result['loss'])

            elif phase == 'test':
                result['pred'] = outputs_all

            result.update(dict(outputs=outputs, labels=labels, loss=loss_result))

        return result

    def step_model(self, inputs):
        """Process model.

        Args:
            inputs (Tensor or list): inputs data passed to model.

        Returns:
            Tensor or list: outputs of model.
        """
        outputs = self.ml.model(inputs)

        return outputs

    def step_loss(self, outputs, labels):
        """Process loss function.

        Args:
            outputs (Tensor or list): predicted data by model.
            labels (Tensor or list): true data.

        Returns:
            dict: result of loss and subloss.
        """
        loss_result = {'loss': 0, 'subloss': []}

        if self.ml.multi_loss:
            for loss_fn, loss_w, output, label in zip(self.ml.loss, self.ml.loss_weights, outputs,
                                                      labels):
                if loss_w:
                    loss_tmp = loss_fn(output, label) * loss_w
                    loss_result['loss'] += loss_tmp
                    loss_result['subloss'].append(loss_tmp)

        else:
            if self.ml.loss_weights is None:
                loss_result['loss'] += self.ml.loss(outputs, labels)
            elif self.ml.loss_weights != 0.0:
                loss_result['loss'] += self.ml.loss(outputs, labels) * self.ml.loss_weights

        return loss_result

    def step_optimizer(self, loss):
        """Process optimizer.

        Args:
            loss (obf): loss value.
        """
        self.ml.optimizer.zero_grad(set_to_none=True)
        if self._is_gpu and self._amp:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.ml.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self.ml.optimizer.step()

    def get_dataset(self, data=None, phase=None, preload=False, callbacks=None):
        """Returns dataset from given ndarray data."""
        if data is None:
            if phase is None:
                data = self.get_input_true_data(phase)
                dataset = self.get_tensor_dataset(data)
            else:
                dataset = self.get_storegate_dataset(phase, preload=preload, callbacks=callbacks)
        else:
            dataset = self.get_tensor_dataset(data, callbacks=callbacks)

        return dataset

    @staticmethod
    def get_tensor_dataset(data, callbacks=None):
        """Returns tensor dataset from given ndarray data."""
        return NumpyDataset(*data, callbacks=callbacks)

    def get_storegate_dataset(self, phase, preload=False, callbacks=None):
        """Returns storegate dataset."""
        return StoreGateDataset(self.storegate,
                                phase,
                                self._device,
                                input_var_names=self.input_var_names,
                                true_var_names=self.true_var_names,
                                preload=preload,
                                callbacks=callbacks)

    def get_batch_sampler(self, phase, dataset):
        """Returns batch sampler."""
        sampler_args = dict(drop_last=False, batch_size=self._get_batch_size(phase))

        if phase in ('train', 'valid'):
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(dataset), **sampler_args)
        else:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(dataset), **sampler_args)
        return sampler

    def add_device(self, data, device):
        """Add data to device."""
        if isinstance(data, Tensor):
            return data.to(device)

        if isinstance(data, np.ndarray):
            if isinstance(data[0], (int, np.integer)):
                return LongTensor(data).to(device)
            return Tensor(data).to(device)

        if isinstance(data, list):
            return [self.add_device(idata, device) for idata in data]

        logger.debug(f'Unexpected data type: {type(data)} is added to device.')
        return data.to(device)

    def fix_submodule(self, target):
        """ Fix given parameters of model."""
        for param in self.ml.model.get_submodule(target).parameters():
            param.requires_grad = False

    ##########################################################################
    # Internal methods
    ##########################################################################
    def _select_pred_data(self, y_pred):
        if len(self._pred_index) == 1:
            return y_pred[self._pred_index[0]]
        else:
            return [y_pred[index] for index in self._pred_index]

    def _get_pbar_description(self, epoch, phase):
        if (self.job_id is None) and (self.trial_id is None):
            return f'Epoch [{epoch: >4}/{self._num_epochs}] {phase.ljust(5)}'
        elif (self.trial_id is None):
            return f'Epoch [{epoch: >4}/{self._num_epochs},{self.job_id: >2}] {phase.ljust(5)}'
        else:
            return f'Epoch [{epoch: >4}/{self._num_epochs},{self.job_id: >2},{self.trial_id+1: >2}] {phase.ljust(5)}'

    def _disable_tqdm(self):
        disable_tqdm = True
        if self._verbose is None:
            if logger.MIN_LEVEL <= logger.DEBUG:
                disable_tqdm = False
        elif self._verbose == 1:
            disable_tqdm = False
        return disable_tqdm

    def _get_batch_size(self, phase=None, num_dataset=None):
        if isinstance(self._batch_size, int):
            return self._batch_size

        if not isinstance(self._batch_size, dict):
            raise ValueError(f'batch_size is not known!! {self._batch_size}')

        if phase in self._batch_size:
            return self._batch_size[phase]

        if 'equal_length' not in self._batch_size['type']:
            raise ValueError(f'batch_size is not known!! {self._batch_size}')

        batch_size = num_dataset // self._batch_size['length']
        batch_size = batch_size if batch_size > 1.0 else 1

        log = f'phase={phase}, '
        log += f'dataset={num_dataset}, '
        log += f'length={self._batch_size["length"]}, '
        log += f'batch_size={batch_size}'
        logger.info(log)

        return batch_size
