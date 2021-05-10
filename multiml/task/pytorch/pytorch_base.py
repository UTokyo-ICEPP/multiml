""" PytorchBaseTask module.
"""
import copy
import inspect

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
    def __init__(
            self,
            device='cpu',
            num_workers=0,
            gpu_ids=None,
            amp=False,  # expert option
            benchmark=False,  # expert option
            unpack_inputs=False,  # expert option
            view_as_outputs=False,  # expert option
            **kwargs):
        """ Initialize the pytorch base task.

        Args:
            device (str or obj): pytorch device, e.g. 'cpu', 'cuda'.
            num_workers (int): number of workers for dataloaders.
            gpu_ids (list): GPU identifiers, e.g. [0, 1, 2]. ``data_parallel``
                mode is enabled if ``gpu_ids`` is given.
            amp (bool): *(expert option)* enable amp mode.
            benchmark (bool): *(expert option)* enable cudnn.benchmark mode.
            unpack_inputs (bool): *(expert option)* uppack inputs when being
                passed to model, e.g. model(*inputs).
            view_as_outputs (bool): *(expert option)* view_as outputs when being
                passed to loss, e.g. loss(output.view_as(label), label).
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

        self._num_workers = num_workers
        self._gpu_ids = gpu_ids
        self._amp = amp
        self._benchmark = benchmark
        self._unpack_inputs = unpack_inputs
        self._view_as_outputs = view_as_outputs

        self._pred_index = None
        self._pass_training = False
        self._early_stopping = False
        self._scheduler = None
        self._scaler = None

        if self._metrics is None:
            self._metrics = ['loss']

        if self._max_patience is not None:
            self._early_stopping = True

        if self._benchmark:
            torch.backends.cudnn.benchmark = True

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
            optimizer_args['params'] = self.ml.model.parameters()

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

    def fit(self,
            train_data=None,
            valid_data=None,
            dataloaders=None,
            valid_step=1,
            sampler=None,
            **kwargs):
        """ Train model over epoch.

        This methods train and valid model over epochs by calling
        ``train_model()`` method. train and valid need to be provided by
        ``train_data`` and ``valid_data`` options, or ``dataloaders`` option.

        Args:
            train_data (ndarray): If ``train_data`` is given, data are converted
                to ``TendorDataset`` and set to ``dataloaders['train']``.
            valid_data (ndarray): If ``valid_data`` is given, data are converted
                to ``TendorDataset`` and set to ``dataloaders['valid']``.
            dataloaders (dict): dict of dataloaders, dict(train=xxx, valid=yyy).
            valid_step (int): step to process validation.
            sampler (obf): sampler to execute ``set_epoch()``.
            kwargs (dict): arbitrary args passed to ``train_model()``.

        Returns:
            list: history data of train and valid.
        """
        if dataloaders is None:
            dataloaders = dict(train=None, valid=None)

        if train_data is not None:
            train_dataset = self.get_tensor_dataset(train_data)
            dataloaders['train'] = DataLoader(train_dataset,
                                              batch_size=self._batch_size,
                                              num_workers=self._num_workers,
                                              shuffle=True)

            kwargs['input_index'] = train_dataset.input_slice
            kwargs['true_index'] = train_dataset.true_slice

        if valid_data is not None:
            valid_dataset = self.get_tensor_dataset(valid_data)
            dataloaders['valid'] = DataLoader(valid_dataset,
                                              batch_size=self._batch_size,
                                              num_workers=self._num_workers,
                                              shuffle=True)

            kwargs['input_index'] = valid_dataset.input_slice
            kwargs['true_index'] = valid_dataset.true_slice

        if dataloaders['train'] is None:
            #train_data = self.get_input_true_data('train')
            #train_dataset = self.get_tensor_dataset(train_data)
            train_dataset = self.get_storegate_dataset('train')
            dataloaders['train'] = DataLoader(train_dataset,
                                              batch_size=self._batch_size,
                                              num_workers=self._num_workers,
                                              shuffle=True)

            kwargs['input_index'] = train_dataset.input_slice
            kwargs['true_index'] = train_dataset.true_slice

        if dataloaders['valid'] is None:
            #valid_data = self.get_input_true_data('valid')
            #valid_dataset = self.get_tensor_dataset(valid_data)
            valid_dataset = self.get_storegate_dataset('valid')
            dataloaders['valid'] = DataLoader(valid_dataset,
                                              batch_size=self._batch_size,
                                              num_workers=self._num_workers,
                                              shuffle=True)

            kwargs['input_index'] = valid_dataset.input_slice
            kwargs['true_index'] = valid_dataset.true_slice

        early_stopping = util.EarlyStopping(patience=self._max_patience)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._is_gpu)

        history = {'train': [], 'valid': []}

        for epoch in range(1, self._num_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

            # train
            self.ml.model.train()
            result = self.train_model(epoch, 'train', dataloaders['train'],
                                      **kwargs)
            history['train'].append(result)

            # valid
            if (const.VALID in self.phases) and (epoch % valid_step == 0):
                self.ml.model.eval()
                result = self.train_model(epoch, 'valid', dataloaders['valid'],
                                          **kwargs)
                history['valid'].append(result)

                if self._early_stopping:
                    if early_stopping(result['running_loss'], self.ml.model):
                        break

            if self._scheduler is not None:
                self._scheduler.step()

        if self._early_stopping:
            best_model = early_stopping.best_model.state_dict()
            self.ml.model.load_state_dict(copy.deepcopy(best_model))

        return history

    def train_model(self,
                    epoch,
                    phase,
                    dataloader,
                    rank=None,
                    input_index=0,
                    true_index=1):
        """ Process model for given epoch and phase.

        ``ml.model``, ``ml.optimizer`` and ``ml.loss`` need to be set before
        calling this method, please see ``compile()`` method.

        Args:
            epoch (int): epoch numer.
            phase (str): *train* mode or *valid* mode.
            dataloader (obj): dataloader instance.
            rank (int): rank for distributed data parallel.
            input_index (int): index of input variables for dataloader outputs.
            true_index (int): index of true variable for dataloader outputs.

        Returns:
            dict: dict of result.
        """
        if self.ml.model is None:
            raise AttributeError('model is not defined')

        if self.ml.optimizer is None:
            raise AttributeError('optimizer is not defined')

        if self.ml.loss is None:
            raise AttributeError('loss is not defined')

        if rank is None:
            rank = self._device

        disable_tqdm = True
        if self._verbose is None:
            if logger.MIN_LEVEL <= logger.DEBUG:
                disable_tqdm = False
        elif self._verbose == 1:
            disable_tqdm = False

        sig = inspect.signature(self.ml.model.forward)
        if 'training' in sig.parameters:
            self._pass_training = True
        else:
            self._pass_training = False

        epoch_loss, epoch_corrects, total = 0.0, 0, 0
        results = {}
        pbar_args = dict(total=len(dataloader),
                         unit='batch',
                         ncols=130,
                         disable=disable_tqdm)
        pbar_desc = f'Epoch [{epoch}/{self._num_epochs}] ({phase.ljust(5)})'

        with tqdm(**pbar_args) as pbar:
            pbar.set_description(pbar_desc)

            for data in dataloader:
                inputs = self.add_device(data[input_index], rank)
                labels = self.add_device(data[true_index], rank)

                batch_result = self.step_train(inputs, labels, phase)

                inputs_size = util.inputs_size(inputs)
                total += inputs_size
                epoch_loss += batch_result['loss'] * inputs_size
                running_loss = epoch_loss / total
                results['loss'] = f'{running_loss:.2e}'

                if 'acc' in self._metrics:
                    accuracy = batch_result['acc'] / total
                    results['acc'] = f'{accuracy:.2e}'

                if 'lrs' in self._metrics:
                    lrs = [f"{float(lr):.2e}" for lr in batch_result['lrs']]
                    results['lrs'] = f'{lrs}'

                pbar.set_postfix(results)
                pbar.update(1)

        if self._verbose == 2:
            logger.info(f'{pbar_desc} {results}')

        results['running_loss'] = running_loss
        return results

    def step_train(self, inputs, labels, phase):
        """ Process batch data and update weights.

        Args:
            inputs (obj): input tensor data.
            labels (obj): true tensor data.
            phase (str): *train* mode or *valid* mode.

        Returns:
            dict: dict of result.
        """
        result = {}
        self.ml.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast(self._is_gpu and self._amp):
                outputs = self._step_model(inputs, True)
                if self._pred_index is not None:
                    outputs = self._select_pred_data(outputs)

                loss = self._step_loss(outputs, labels)

            if phase == 'train':
                self._step_optimizer(loss)

            result['loss'] = loss.item()

            if 'acc' in self._metrics:
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == labels.data)
                result['acc'] = corrects.item()

            if 'lrs' in self._metrics:
                lrs = [param['lr'] for param in self.ml.optimizer.param_groups]
                result['lrs'] = lrs

        return result

    def predict(self,
                data=None,
                dataloader=None,
                phase=None,
                input_index=0,
                argmax=None):
        """ Predict model.

        This method predicts and returns results. Data need to be provided by
        ```data``` option, or setting property of ``dataloaders`` directory.

        Args:
            data (ndarray): If ``data`` is given, data are converted to 
                ``TendorDataset`` and set to ``dataloaders['test']``.
            dataloader (obj): dataloader instance.
            phase (str): 'all' or 'train' or 'valid' or 'test' to specify 
                dataloaders.
            input_index (int): index of input variables for dataloader outputs.
            argmax (int): apply ``np.argmax`` to resuls.

        Returns:
            ndarray or list: results of prediction.
        """
        if self.ml.model is None:
            raise AttributeError('model is not defined')

        self.ml.model.eval()

        sig = inspect.signature(self.ml.model.forward)
        if 'training' in sig.parameters:
            self._pass_training = True
        else:
            self._pass_training = False

        if data is not None:
            dataset = self.get_tensor_dataset(data)
            dataloader = DataLoader(dataset,
                                    batch_size=self._batch_size,
                                    num_workers=self._num_workers,
                                    shuffle=False)
            input_index = dataset.input_slice

        if dataloader is None:
            data = self.get_input_true_data(phase)
            dataset = self.get_tensor_dataset(data)
            dataloader = DataLoader(dataset,
                                    batch_size=self._batch_size,
                                    num_workers=self._num_workers,
                                    shuffle=False)
            input_index = dataset.input_slice

        results = []

        with torch.no_grad():
            for data in dataloader:
                inputs = self.add_device(data[input_index], self._device)

                with torch.cuda.amp.autocast(self._is_gpu and self._amp):
                    outputs = self._step_model(inputs, False)

                    if isinstance(outputs, Tensor):
                        results.append(outputs.cpu().numpy())
                    else:
                        if results:
                            for index, output_obj in enumerate(outputs):
                                output_obj = output_obj.cpu().numpy()
                                results[index].append(output_obj)
                        else:
                            for output_obj in outputs:
                                output_obj = output_obj.cpu().numpy()
                                results.append([output_obj])

        if isinstance(results[0], list):
            results = [np.concatenate(result, 0) for result in results]
        else:
            results = np.concatenate(results, 0)

            if argmax:
                results = np.argmax(results, axis=argmax)

        return results

    def get_tensor_dataset(self, data):
        """ Returns dataset from given ndarray data. 
        """
        inputs, targets = data

        inputs = self.add_device(inputs, 'cpu')
        targets = self.add_device(targets, 'cpu')

        dataset = NumpyDataset(inputs, targets)

        return dataset

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

    def get_inputs(self):
        if isinstance(self.input_var_names, list):
            return [len(self.input_var_names)]
        else:
            return [1]

    ##########################################################################
    # Internal methods
    ##########################################################################
    def _step_model(self, inputs, training):
        if self._pass_training:
            forward_args = dict(training=training)
        else:
            forward_args = {}

        if isinstance(inputs, list):
            if self._unpack_inputs:
                outputs = self.ml.model(*inputs, **forward_args)
            else:
                outputs = self.ml.model(inputs, **forward_args)
        else:
            outputs = self.ml.model(inputs, **forward_args)

        return outputs

    def _step_loss(self, outputs, labels):
        loss = 0.0

        if isinstance(outputs, list):
            for loss_fn, loss_w, output, label in zip(self.ml.loss,
                                                      self.ml.loss_weights,
                                                      outputs, labels):
                if loss_w:
                    if self._view_as_outputs:
                        output = output.view_as(label)
                    loss += loss_fn(output, label) * loss_w
        else:
            if self._view_as_outputs:
                outputs = outputs.view_as(labels)
            if self.ml.loss_weights is None:
                loss += self.ml.loss(outputs, labels)
            elif self.ml.loss_weights != 0.0:
                loss += self.ml.loss(outputs, labels) * self.ml.loss_weights

        return loss

    def _step_optimizer(self, loss):
        if self._is_gpu and self._amp:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.ml.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self.ml.optimizer.step()

    def _select_pred_data(self, y_pred):
        if len(self._pred_index) == 1:
            return y_pred[self._pred_index[0]]
        else:
            return [y_pred[index] for index in self._pred_index]
