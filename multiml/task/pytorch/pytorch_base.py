""" PytorchBaseTask module.
"""
import copy

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
    """ Base task for PyTorch model.

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
    def __init__(self, device='cpu', gpu_ids=None, amp=False, **kwargs):
        """ Initialize the pytorch base task.

        Args:
            device (str or obj): pytorch device, e.g. 'cpu', 'cuda'.
            gpu_ids (list): GPU identifiers, e.g. [0, 1, 2]. ``data_parallel``
                mode is enabled if ``gpu_ids`` is given.
            amp (bool): *(expert option)* enable amp mode.
        """
        super().__init__(**kwargs)

        if isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

        self._is_gpu = 'cuda' in self._device.type
        if self._is_gpu and (not torch.cuda.is_available()):
            raise ValueError(f'{self._device} is not available')

        self._data_parallel = self._is_gpu and (gpu_ids is not None)

        logger.info(f'{self._name}: PyTorch device: {self._device}')

        self._gpu_ids = gpu_ids
        self._amp = amp

        self._pred_index = None
        self._early_stopping = False
        self._scheduler = None
        self._scaler = None

        if self._metrics is None:
            self._metrics = ['loss']

        if self._max_patience is not None:
            self._early_stopping = True

    def compile_model(self):
        """ Compile pytorch model.

        Compile model based on self._model type, which is usually set by
        ``__init__()`` or ``build_model() method. Compiled model is set to
        ``self.ml.model`` and moved to ``self._device``.

        """
        if self._model is None:
            return

        self.ml.model = util.compile(self._model, self._model_args, modules)

        if self.pred_var_names is not None:
            self._pred_index = self.get_pred_index()

        if self._load_weights:
            self.load_model()
            self.load_metadata()

        if self._data_parallel:
            self.ml.model = torch.nn.DataParallel(self.ml.model,
                                                  device_ids=self._gpu_ids)
            self.ml.model.to(self._device)

        else:
            self.ml.model.to(self._device)

    def compile_optimizer(self):
        """ Compile pytorch optimizer.

        Compile optimizer based on self._optimizer type, which is usually set
        by ``__init__()`` method. Compiled optimizer is set to
        ``self.ml.optimizer``.
        """
        if self._optimizer is None:
            return

        optimizer_args = copy.copy(self._optimizer_args)
        if 'params' not in optimizer_args:
            optimizer_args['params'] = list(self.ml.model.parameters())

        self.ml.optimizer = util.compile(self._optimizer, optimizer_args,
                                         optim)

    def compile_loss(self):
        """ Compile pytorch loss.

        Compile loss based on self._loss type, which is usually set by
        ``__init__()`` method. Compiled loss is set to ``self.ml.loss``.
        """

        if self._loss is None:
            return

        self.ml.loss = util.compile(self._loss, self._loss_args, tl)
        self.ml.loss_weights = self._loss_weights

    def compile_device(self):
        """ Compile device.

        This method is valid only for multiprocessing mode so far. Devices are
        set based on ``pool_id``.
        """
        if self._pool_id is None:
            return  # nothing to do so far

        if self._data_parallel:
            raise ValueError(
                'data_parallel is not available with pool_id mode')

        if 'cuda' in self._device.type:
            cuda_id = self._pool_id

            if cuda_id >= torch.cuda.device_count():
                cuda_id = cuda_id % torch.cuda.device_count()
            logger.debug(f'Apply pool_id:{cuda_id} to pytorch device')
            self._device = torch.device(f'cuda:{cuda_id}')

    def load_model(self):
        """ Load pre-trained pytorch model weights.
        """
        model_path = super().load_model()
        self.ml.model.load_state_dict(torch.load(model_path))

    def dump_model(self, extra_args=None):
        """ Dump current pytorch model.
        """
        args_dump_ml = dict(ml_type='pytorch')

        if self._data_parallel:
            args_dump_ml['model'] = self.ml.model.module
        else:
            args_dump_ml['model'] = self.ml.model

        if extra_args is not None:
            args_dump_ml.update(extra_args)

        super().dump_model(args_dump_ml)

    def prepare_dataloader(self, data=None, phase=None):
        """ Prepare dataloader.

        If inputs are given, tensor_dataset() is called. If inputs are None,
        storegate_dataset with given phase is called.

        Args:
            data (ndarray): data passed to tensor_dataset().
            phase (str): phase passed to storegate_dataset().

        Returns:
            DataLoader: Pytorch dataloader instance.
        """
        if data is None:
            if phase is None:
                data = self.get_input_true_data(phase)
                dataset = self.get_tensor_dataset(data)
            else:
                dataset = self.get_storegate_dataset(phase)
        else:
            dataset = self.get_tensor_dataset(data)

        shuffle = True if phase in ('train', 'valid') else False

        return DataLoader(dataset,
                          batch_size=self._get_batch_size(phase, len(dataset)),
                          num_workers=self._num_workers,
                          shuffle=shuffle)

    def fit(self,
            train_data=None,
            valid_data=None,
            dataloaders=None,
            valid_step=1,
            sampler=None):
        """ Train model over epoch.

        This methods train and valid model over epochs by calling
        ``step_epoch()`` method. train and valid need to be provided by
        ``train_data`` and ``valid_data`` options, or ``dataloaders`` option.

        Args:
            train_data (ndarray): If ``train_data`` is given, data are converted
                to ``TendorDataset`` and set to ``dataloaders['train']``.
            valid_data (ndarray): If ``valid_data`` is given, data are converted
                to ``TendorDataset`` and set to ``dataloaders['valid']``.
            dataloaders (dict): dict of dataloaders, dict(train=xxx, valid=yyy).
            valid_step (int): step to process validation.
            sampler (obf): sampler to execute ``set_epoch()``.

        Returns:
            list: history data of train and valid.
        """
        self.ml.validate('train')

        if dataloaders is None:
            dataloaders = dict(
                train=self.prepare_dataloader(train_data, 'train'),
                valid=self.prepare_dataloader(valid_data, 'valid'))

        early_stopping = util.EarlyStopping(patience=self._max_patience)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._is_gpu)

        history = {'train': [], 'valid': []}

        for epoch in range(1, self._num_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

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

            if self._scheduler is not None:
                self._scheduler.step()

        if self._early_stopping:
            best_model = early_stopping.best_model.state_dict()
            self.ml.model.load_state_dict(copy.deepcopy(best_model))

        return history

    def predict(self, data=None, dataloader=None, phase=None, label=False):
        """ Predict model.

        This method predicts and returns results. Data need to be provided by
        ```data``` option, or setting property of ``dataloaders`` directory.

        Args:
            data (ndarray): If ``data`` is given, data are converted to 
                ``TendorDataset`` and set to ``dataloaders['test']``.
            dataloader (obj): dataloader instance.
            phase (str): 'all' or 'train' or 'valid' or 'test' to specify 
                dataloaders.
            label (bool): If True, returns metric results based on labels.

        Returns:
            ndarray or list: results of prediction.
        """
        self.ml.validate('test')
        self.ml.model.eval()

        if dataloader is None:
            dataloader = self.prepare_dataloader(data, phase)

        results = self.step_epoch(0, 'test', dataloader, label)

        if label:
            return results
        else:
            return results['pred']

    def step_epoch(self, epoch, phase, dataloader, label=True):
        """ Process model for given epoch and phase.

        ``ml.model``, ``ml.optimizer`` and ``ml.loss`` need to be set before
        calling this method, please see ``compile()`` method.

        Args:
            epoch (int): epoch numer.
            phase (str): *train* mode or *valid* mode.
            dataloader (obj): dataloader instance.
            label (bool): If True, returns metric results based on labels.

        Returns:
            dict: dict of result.
        """
        disable_tqdm = self._disable_tqdm()
        epoch_metric = metrics.EpochMetric(self._metrics, label, 
                                           self.true_var_names, self.ml)

        pbar_args = dict(
            total=len(dataloader),
            unit=' batch',
            ncols=150,
            bar_format=
            "{desc}: {percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]",
            disable=disable_tqdm)
        pbar_desc = f'Epoch [{epoch: >4}/{self._num_epochs}] {phase.ljust(5)}'

        results = {}
        with tqdm(**pbar_args) as pbar:
            pbar.set_description(pbar_desc)

            for data in dataloader:
                batch_result = self.step_batch(data, phase, label)
                results.update(epoch_metric(batch_result))

                if phase == 'test':
                    epoch_metric.pred(batch_result)

                pbar.set_postfix(metrics.get_pbar_metric(results))
                pbar.update(1)

        if self._verbose == 2:
            logger.info(f'{pbar_desc} {metrics.get_pbar_metric(results)}')

        if phase == 'test':
            results['pred'] = epoch_metric.all_preds()

        return results

    def step_batch(self, data, phase, label=True):
        """ Process batch data and update weights.

        Args:
            data (obj): inputs and labels data.
            phase (str): *train* mode or *valid* mode or *test* mode.
            label (bool): If True, returns metric results based on labels.

        Returns:
            dict: dict of result.
        """
        inputs, labels = data  #FIXME: data without labels
        inputs = self.add_device(inputs, self._device)
        labels = self.add_device(labels, self._device)

        result = {'batch_size': util.inputs_size(inputs)}
        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast(self._is_gpu and self._amp):
                outputs = self.step_model(inputs)
                if self._pred_index is not None:
                    outputs = self._select_pred_data(outputs)

                if label:
                    loss_result = self.step_loss(outputs, labels)
                else:
                    loss_result = None

            if phase == 'train':
                self.step_optimizer(loss_result['loss'])

            elif phase == 'test':
                result['pred'] = outputs

            batch_metric = metrics.BatchMetric(self._metrics, label)
            result.update(batch_metric(outputs, labels, loss_result))

        return result

    def step_model(self, inputs):
        """ Process model.

        Args:
            inputs (Tensor or list): inputs data passed to model.

        Returns:
            Tensor or list: outputs of model.
        """

        return self.ml.model(inputs)

    def step_loss(self, outputs, labels):
        """ Process loss function. 

        Args:
            outputs (Tensor or list): predicted data by model.
            labels (Tensor or list): true data.

        Returns:
            dict: result of loss and subloss.
        """
        loss_result = {'loss': 0, 'subloss': []}

        if self.ml.multi_loss:
            for loss_fn, loss_w, output, label in zip(self.ml.loss,
                                                      self.ml.loss_weights,
                                                      outputs, labels):
                if loss_w:
                    loss_tmp = loss_fn(output, label) * loss_w
                    loss_result['loss'] += loss_tmp
                    loss_result['subloss'].append(loss_tmp)

        else:
            if self.ml.loss_weights is None:
                loss_result['loss'] += self.ml.loss(outputs, labels)
            elif self.ml.loss_weights != 0.0:
                loss_result['loss'] += self.ml.loss(
                    outputs, labels) * self.ml.loss_weights

        return loss_result

    def step_optimizer(self, loss):
        """ Process optimizer. 

        Args:
            loss (obf): loss value.
        """
        self.ml.optimizer.zero_grad()
        if self._is_gpu and self._amp:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.ml.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self.ml.optimizer.step()

    def get_tensor_dataset(self, data):
        """ Returns dataset from given ndarray data. 
        """
        inputs, targets = data

        return NumpyDataset(inputs, targets)

    def get_storegate_dataset(self, phase):
        """ Returns storegate dataset. 
        """
        dataset = StoreGateDataset(self.storegate,
                                   phase,
                                   input_var_names=self.input_var_names,
                                   true_var_names=self.true_var_names)
        return dataset

    def add_device(self, data, device):
        """ Add data to device.
        """
        if isinstance(data, Tensor):
            return data.to(device)

        if isinstance(data, np.ndarray):
            if isinstance(data[0], (int, np.integer)):
                return LongTensor(data).to(device)
            return Tensor(data).to(device)

        if isinstance(data, list):
            return [self.add_device(idata, device) for idata in data]

        raise ValueError(
            f'data type {type(data)} is not supported. cannot add to device')

    ##########################################################################
    # Internal methods
    ##########################################################################
    def _select_pred_data(self, y_pred):
        if len(self._pred_index) == 1:
            return y_pred[self._pred_index[0]]
        else:
            return [y_pred[index] for index in self._pred_index]

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

        batch_length = num_dataset / self._batch_size['length']
        batch_length = batch_length if batch_length > 1.0 else 1
        batch_size = int(np.floor(batch_length))

        log = f'phase={phase}, '
        log += f'dataset={num_dataset}, '
        log += f'length={batch_length}, '
        log += f'batch_size={batch_size}'
        logger.info(log)

        return batch_size
