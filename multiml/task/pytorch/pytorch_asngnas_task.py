
import copy
from multiml import logger
from ..basic import ModelConnectionTask
from . import PytorchBaseTask
from multiml.task.pytorch import pytorch_util as util
import torch
import numpy as np
from tqdm import tqdm


class PytorchASNGNASTask(ModelConnectionTask, PytorchBaseTask):
    def __init__(self, **kwargs):
        """

        Args:
            subtasks (list): list of task instances.
            **kwargs: Arbitrary keyword arguments.
        """
        print('PytorchASNGNASTask.__init__')
        super().__init__(**kwargs)
        # self._subtasks = subtasks
        
        # TODO : get task information instead of submodels[0]
        for k in self._subtasks : 
            print(f'k is {k._task_id}')
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
        self._batch_size = self._proxy_model._batch_size
        for m in self._subtasks : 
            print(f'm is {m._task_id}')
        
        
    def build_model(self):
        print('PytorchASNGNASTask.build_model')
        from multiml.task.pytorch.modules import ASNGModel
        
        for subtask in self._subtasks : 
            print(f'subtask._name is {subtask._name}')

        models = [subtask.ml.model for subtask in self._subtasks ]
        self._model = ASNGModel(models,
                                input_var_index=self._input_var_index,
                                output_var_index=self._output_var_index
                                )
                                
        
            
    def set_most_likely(self):
        self.c_cat, self.c_int = self.asng.most_likely_value() # best 
        
    def get_most_likely(self):
        return self.c_cat, self._cint 
    
    def get_thetas(self):
        return self.asng.get_thetas()
    
    def get_best_combination(self):
        cat_idx = self.c_cat.argmax(axis = 1)
        for i, idx in enumerate(cat_idx) : 
            self.ml.model._subtasks[i]
            
            
        
    def fit(self,
            train_data=None,
            valid_data=None,
            dataloaders=None,
            valid_step=1,
            sampler=None,
            input_index=0,
            true_index=1,
            rank = None,
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
        

        dataloaders = self.prepare_dataloaders( train_data, valid_data, dataloaders )
        early_stopping = util.EarlyStopping(patience=self._max_patience)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._is_gpu)

        if self.ml.model is None:
            raise AttributeError('model is not defined')

        if self.ml.optimizer is None:
            raise AttributeError('optimizer is not defined')

        if self.ml.loss is None:
            raise AttributeError('loss is not defined')

        history = {'train': [], 'valid': []}
        losses = np.zeros(self.asng().get_lambda() )
        
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
        
        
        results_train = training_results( self._metrics, self.ml.multi_loss, len(self.true_var_names) )
        results_valid = training_results( self._metrics, self.ml.multi_loss, len(self.true_var_names) )
        
        logger.info(f'dataset(train/valid) length is {len(dataloaders["train"])}/{len(dataloaders["valid"])}')
        
        for epoch in range(1, self._num_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

            
            pbar_args = dict(total = min(len(dataloaders['train']), len(dataloaders['valid']) ),
                            unit=' batch',
                            ncols=150,
                            bar_format="{desc}: {percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]",
                            disable=disable_tqdm)
            pbar_desc = f'Epoch [{epoch: >4}/{self._num_epochs}] ASNG-NAS'

            with tqdm(**pbar_args) as pbar:
                pbar.set_description(pbar_desc)
            
                for train_data, valid_data in zip( dataloaders['train'], dataloaders['valid']) : 
                    ### train
                    self.ml.model.train()
                    loss_train, batch_result = self._step_train(True, train_data[input_index], train_data[true_index], rank )
                    self._step_optimizer(loss_train)
                    results_train.update_results( batch_result, util.inputs_size(train_data[input_index]))
                    
                    ### validation -> theta update
                    self.ml.model.eval()
                    loss_valid, batch_result = self._step_train(False, valid_data[input_index], valid_data[true_index], rank )
                    self.ml.model.update_theta( np.array(loss_valid), range_restriction = True ) # FIXME : 
                    results_valid.update_results( batch_result, util.inputs_size(valid_data[input_index]))
                    
                    loss_train = results_train.get_running_loss()
                    loss_valid = results_valid.get_running_loss()
                    
                    pbar.set_postfix( tloss = f'{loss_train:.3e}', vloss = f'{loss_valid:.3e}')
                    pbar.update(1)
                
            history['train'].append(results_train.get_results())
            history['valid'].append(results_valid.get_results())
            
            if self._early_stopping:
                if early_stopping( results_valid.get_running_loss(), self.ml.model):
                    break

            if self._scheduler is not None:
                self._scheduler.step()

        if self._early_stopping:
            best_model = early_stopping.best_model.state_dict()
            self.ml.model.load_state_dict(copy.deepcopy(best_model))
        
        return history
    
    def _step_train( self, is_train, inputs_data, labels_data, rank):
        if rank is None:
            rank = self._device
        
        self.ml.model.train()
        inputs = self.add_device(inputs_data, rank)
        labels = self.add_device(labels_data, rank)
        
        self.ml.optimizer.zero_grad()
        
        
        result = {}
        with torch.set_grad_enabled( is_train ) : 
            with torch.cuda.amp.autocast(self._is_gpu and self._amp):
                outputs = self._step_model(inputs, False )
                loss = []
                subloss = []
                for output in outputs : 
                    if self._pred_index is not None:
                        output = self._select_pred_data(output)
                    _loss, _subloss = self._step_loss(output, labels)
                    loss.append(_loss)
                    subloss.append(_subloss)
                
                result['loss'] = np.mean([l.item() for l in loss ])
                
            if 'subloss' in self._metrics:
                result['subloss'] = list(np.mean([[ s.item() for s in sub ] for sub in subloss], axis=0))
            
            if 'acc' in self._metrics:
                result['acc'] = []
                if self.ml.multi_loss:
                    for output in outputs : 
                        acc = []
                        for out, label in zip(output, labels):
                            _, preds = torch.max(out, 1)
                            corrects = torch.sum(preds == label.data)
                            acc.append(corrects.item())
                        results['acc'].append(acc)
                    results['acc'] = list(np.mean(results['acc'], axis=0))
                else:
                    for output in outputs : 
                        _, preds = torch.max(output, 1)
                        corrects = torch.sum(preds == labels.data)
                        result['acc'].append(corrects.item())
                    results['acc'] = np.mean(results['acc'])
        return loss, result
    
    def _step_optimizer(self, losses):
        loss = 0
        for l in losses : 
            loss += l / len(losses)
        
        if self._is_gpu and self._amp:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.ml.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self.ml.optimizer.step()
    
    def finalize(self) : 
        c_cat, c_int = self.asng().most_likely_value()
        c_idx = c_cat.argmax(axis=1)
        
        # result['task_ids'].append(task_id)
        # result['subtask_ids'].append(subtask_id)
        # result['subtask_hps'].append(hps)
        # result['metric_value'] = metric


    def get_input_true_data(self, phase):
        return self._proxy_model.get_input_true_data(phase)

    def get_storegate_dataset(self, phase):
        return self._proxy_model.get_storegate_dataset(phase)

    def get_submodel_names(self):
        return [v.subtask_id for v in self._subtasks]

    def get_inputs(self):
        return self._proxy_model.get_inputs()

    def get_submodel(self, i_models):
        return self._subtasks[i_models]
    
    def asng(self) : 
        return self.ml.model.asng 

class training_results: 
    def __init__(self, _metrics, is_multi_loss, len_true_var_names ) : 
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
        
    def update_results(self, batch_result, input_size ): 
        self.results = {}
        
        self.total += input_size 
        self.epoch_loss = batch_result['loss'] * input_size
        self.running_loss = self.epoch_loss / self.total 
        self.results['loss'] = f'{self.running_loss:.2e}'
        
        if 'subloss' in self._metrics:
            self.results['subloss'] = []
            for index, subloss in enumerate(batch_result['subloss']):
                epoch_subloss[index] += subloss * inputs_size
                running_subloss = self.epoch_subloss[index] / self.total
                self.results['subloss'].append(f'{running_subloss:.2e}')
            
        if 'acc' in self._metrics:
            if self.is_multi_loss :
                self.results['acc'] = []
                for index, acc in enumerate(batch_result['acc']):
                    self.epoch_corrects[index] += acc
                    accuracy = self.epoch_corrects[index] / self.total
                    self.results['acc'].append(f'{accuracy:.2e}')
            else:
                self.epoch_corrects += batch_result['acc']
                accuracy = self.epoch_corrects / self.total
                self.results['acc'] = f'{accuracy:.2e}'
