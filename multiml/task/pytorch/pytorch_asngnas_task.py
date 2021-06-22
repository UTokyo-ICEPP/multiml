import copy
from multiml import logger
from ..basic import ModelConnectionTask
from . import PytorchBaseTask
from multiml.task.pytorch import pytorch_util as util
import torch
import numpy as np
from tqdm import tqdm


class PytorchASNGNASTask(ModelConnectionTask, PytorchBaseTask):
    def __init__(self, lam, delta_init_factor, **kwargs):
        """

        Args:
            subtasks (list): list of task instances.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        # self._subtasks = subtasks

        # TODO : get task information instead of submodels[0]
        self._proxy_model = self._subtasks[0]
        # self._task_id = self._subtasks[0].task_id
        self._task_id = 'ASNG-NAS'
        self._input_var_names = self._proxy_model._input_var_names
        self._output_var_names = self._proxy_model._output_var_names
        self._true_var_names = self._proxy_model._true_var_names

        self._loss = self._proxy_model.ml._loss
        self._loss_weights = self._proxy_model.ml._loss_weights
        self._optimizer = self._proxy_model._optimizer
        self._num_epochs = self._proxy_model._num_epochs
        # self._batch_size = self._proxy_model._batch_size
        self.task_ids = [subtask.task_id for subtask in self._subtasks]
        self.subtask_ids = [subtask.subtask_ids for subtask in self._subtasks]

        self.lam = lam
        self.delta_init_factor = delta_init_factor
        print(self._batch_size)

    def build_model(self):
        from multiml.task.pytorch.modules import ASNGModel
        models = [subtask.ml.model for subtask in self._subtasks]

        self._model = ASNGModel(
            self.lam,
            self.delta_init_factor,
            models,
            input_var_index=self._input_var_index,
            output_var_index=self._output_var_index,
        )

    def set_most_likely(self):
        self.ml.model.set_most_likely()

    def best_model(self):
        c_cat, c_int = self.ml.model.get_most_likely()

        best_task_ids = []
        best_subtask_ids = []
        for idx, best_idx in enumerate(c_cat.argmax(axis=1)):
            best_task_ids.append(self.task_ids[idx])
            best_subtask_ids.append(self.subtask_ids[idx][best_idx])

        return best_task_ids, best_subtask_ids

    def get_most_likely(self):
        return self.ml.model.get_most_likely()

    def get_thetas(self):
        return self.ml.model.asng.get_thetas()

    def fit(self,
            train_data=None,
            valid_data=None,
            dataloaders=None,
            valid_step=1,
            sampler=None,
            rank=None,
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

        dataloaders = dict(train=self.prepare_dataloader(train_data, 'train'),
                           valid=self.prepare_dataloader(valid_data, 'valid'))

        test_data = self.get_input_true_data(phase=None)
        test_dataloader = self.prepare_dataloader(test_data, phase='test')

        early_stopping = util.EarlyStopping(patience=self._max_patience)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._is_gpu)

        if self.ml.model is None:
            raise AttributeError('model is not defined')

        if self.ml.optimizer is None:
            raise AttributeError('optimizer is not defined')

        if self.ml.loss is None:
            raise AttributeError('loss is not defined')

        history = {'train': [], 'valid': [], 'test': []}
        losses = np.zeros(self.asng().get_lambda())

        if 'lr' in self._metrics:
            lr = [f'{p["lr"]:.2e}' for p in self.ml.optimizer.param_groups]
            results['lr'] = f'{lr}'

        disable_tqdm = True
        if self._verbose is None:
            if logger.MIN_LEVEL <= logger.DEBUG:
                disable_tqdm = False
        elif self._verbose == 1:
            disable_tqdm = False
        disable_tqdm = False

        results_train = training_results(self._metrics, self.ml.multi_loss,
                                         len(self.true_var_names))
        results_valid = training_results(self._metrics, self.ml.multi_loss,
                                         len(self.true_var_names))
        results_test = training_results(self._metrics, self.ml.multi_loss,
                                        len(self.true_var_names))

        logger.info(
            f'dataset(train/valid/test) len is {len(dataloaders["train"])}/{len(dataloaders["valid"])}/{len(test_dataloader)}'
        )

        dummy = 0.0

        for epoch in range(1, self._num_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

            pbar_args = dict(
                total=min(len(dataloaders['train']), len(
                    dataloaders['valid'])) + len(test_dataloader),
                unit=' batch',
                ncols=250,
                bar_format=
                "{desc}: {percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]",
                disable=disable_tqdm)
            pbar_desc = f'Epoch[{epoch: >4}/{self._num_epochs}] ASNG'

            # training
            with tqdm(**pbar_args) as pbar:
                pbar.set_description(pbar_desc)

                for train_data, valid_data in zip(dataloaders['train'],
                                                  dataloaders['valid']):
                    ### train
                    self.ml.model.train()
                    loss_train, batch_result = self._step_train(
                        True, train_data, rank)
                    self._step_optimizer(loss_train)
                    results_train.update_results(batch_result)

                    ### validation -> theta update
                    self.ml.model.eval()
                    loss_valid, batch_result = self._step_train(
                        False, valid_data, rank)
                    self.ml.model.update_theta(
                        np.array(loss_valid),
                        range_restriction=True)  # FIXME :
                    results_valid.update_results(batch_result)

                    loss_train = results_train.get_running_loss()
                    loss_valid = results_valid.get_running_loss()

                    theta_cats, theta_ints = self.ml.model.get_thetas()
                    theta_cat0 = '/'.join(f'{v:.2f}' for v in theta_cats[0])
                    theta_cat1 = '/'.join(f'{v:.2f}' for v in theta_cats[1])

                    pbar.set_postfix(
                        loss=f'{loss_train:.2e}/{loss_valid:.2e}/{dummy:.2e}',
                        c0=theta_cat0,
                        c1=theta_cat1)
                    #pbar.set_postfix( train = f'{loss_train:.2e}', valid = f'{loss_valid:.2e}', test = f'{dummy:.2e}')
                    pbar.update(1)

                # test
                for test_data in test_dataloader:

                    ### validation -> theta update
                    self.ml.model.eval()
                    loss_test, batch_result = self._step_train(
                        False, test_data, rank)
                    results_test.update_results(batch_result)

                    loss_test = results_test.get_running_loss()

                    pbar.set_postfix(
                        loss=
                        f'{loss_train:.2e}/{loss_valid:.2e}/{loss_test:.2e}',
                        c0=theta_cat0,
                        c1=theta_cat1)
                    # pbar.set_postfix( train = f'{loss_train:.2e}', valid = f'{loss_valid:.2e}', test = f'{loss_test:.2e}')
                    pbar.update(1)

            history['train'].append(results_train.get_results())
            history['valid'].append(results_valid.get_results())
            history['test'].append(results_test.get_results())

            if self._early_stopping:
                if early_stopping(results_test.get_running_loss(),
                                  self.ml.model):
                    logger.info(f'early stopping... at epoch = {epoch}')
                    break

            if self._scheduler is not None:
                self._scheduler.step()

        if self._early_stopping:
            best_model = early_stopping.best_model.state_dict()
            self.ml.model.load_state_dict(copy.deepcopy(best_model))

        return history

    def _step_train(self, is_train, data, rank):
        if rank is None:
            rank = self._device

        self.ml.model.train()
        inputs_data, labels_data = data
        inputs = self.add_device(inputs_data, rank)
        labels = self.add_device(labels_data, rank)

        self.ml.optimizer.zero_grad()

        result = {}
        result['batch_size'] = util.inputs_size(inputs_data)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(self._is_gpu and self._amp):
                outputs = self.step_model(inputs)
                loss = []
                subloss = []
                for output in outputs:
                    if self._pred_index is not None:
                        output = self._select_pred_data(output)
                    loss_result = self.step_loss(output, labels)
                    loss.append(loss_result['loss'])
                    subloss.append(loss_result['subloss'])

                result['loss'] = np.mean([l.item() for l in loss])

            if 'subloss' in self._metrics:
                result['subloss'] = list(
                    np.mean([[s.item() for s in sub] for sub in subloss],
                            axis=0))

            if 'acc' in self._metrics:
                result['acc'] = []
                if self.ml.multi_loss:
                    for output in outputs:
                        acc = []
                        for out, label in zip(output, labels):
                            _, preds = torch.max(out, 1)
                            corrects = torch.sum(preds == label.data)
                            acc.append(corrects.item())
                        results['acc'].append(acc)
                    results['acc'] = list(np.mean(results['acc'], axis=0))
                else:
                    for output in outputs:
                        _, preds = torch.max(output, 1)
                        corrects = torch.sum(preds == labels.data)
                        result['acc'].append(corrects.item())
                    results['acc'] = np.mean(results['acc'])
        return loss, result

    def step_epoch(self, epoch, phase, dataloader, label):
        self.ml.model.set_most_likely()
        outputs = super().step_epoch(epoch, phase, dataloader, label)
        return outputs

    def _step_optimizer(self, losses):
        loss = 0
        for l in losses:
            loss += l / len(losses)

        if self._is_gpu and self._amp:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.ml.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self.ml.optimizer.step()

    def finalize(self):
        c_cat, c_int = self.asng().most_likely_value()
        c_idx = c_cat.argmax(axis=1)

        # result['task_ids'].append(task_id)
        # result['subtask_ids'].append(subtask_id)
        # result['subtask_hps'].append(hps)
        # result['metric_value'] = metric

    # def get_input_true_data(self, phase):
    #     return self._proxy_model.get_input_true_data(phase)

    # def get_storegate_dataset(self, phase):
    #     return self._proxy_model.get_storegate_dataset(phase)

    # def get_inputs(self):
    #     return self._proxy_model.get_inputs()

    def get_submodel_names(self):
        return [v.subtask_id for v in self._subtasks]

    def get_submodel(self, i_models):
        return self._subtasks[i_models]

    def asng(self):
        return self.ml.model.asng


class training_results:
    def __init__(self, _metrics, is_multi_loss, len_true_var_names):
        self.epoch_loss = 0.0
        self.epoch_corrects = 0.0
        self.total = 0.0
        self.is_multi_loss = is_multi_loss
        self._metrics = _metrics
        if self.is_multi_loss:
            self.epoch_subloss = [0.0] * len_true_var_names
            self.epoch_corrects = [0] * len_true_var_names

    def get_results(self):
        results = self.results
        return results

    def get_running_loss(self):
        rl = self.running_loss.item()
        return rl

    def update_results(self, batch_result):
        self.results = {}

        self.total += batch_result['batch_size']
        self.epoch_loss = batch_result['loss'] * batch_result['batch_size']
        self.running_loss = self.epoch_loss / self.total
        self.results['loss'] = f'{self.running_loss:.2e}'

        if 'subloss' in self._metrics:
            self.results['subloss'] = []
            for index, subloss in enumerate(batch_result['subloss']):
                epoch_subloss[index] += subloss * inputs_size
                running_subloss = self.epoch_subloss[index] / self.total
                self.results['subloss'].append(f'{running_subloss:.2e}')

        if 'acc' in self._metrics:
            if self.is_multi_loss:
                self.results['acc'] = []
                for index, acc in enumerate(batch_result['acc']):
                    self.epoch_corrects[index] += acc
                    accuracy = self.epoch_corrects[index] / self.total
                    self.results['acc'].append(f'{accuracy:.2e}')
            else:
                self.epoch_corrects += batch_result['acc']
                accuracy = self.epoch_corrects / self.total
                self.results['acc'] = f'{accuracy:.2e}'
