import copy
from multiml import logger, const
from ..basic import ModelConnectionTask
from . import PytorchBaseTask
from multiml.task.pytorch import pytorch_util as util
import torch
import numpy as np
from tqdm import tqdm


class PytorchYotoConnectionTask(ModelConnectionTask, PytorchBaseTask):
    def __init__(self, loss_merge_f, yoto_model_args, lambda_to_weight, *args, **kwargs):
            
        """

        Args:
            subtasks (list): list of task instances.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self._task_id = 'Yoto-connection'
        self.yoto_model_args = yoto_model_args
        self._loss_merge_f = loss_merge_f
        # idk this is good or not, just evaluate just one lambdas for validation step
        self.valid_lambdas = self.yoto_model_args['valid_lambdas']
        self.lambda_to_weight = lambda_to_weight 
        self._running_step = 10
    
    def build_model(self):
        from multiml.task.pytorch.modules import YotoConnectionModel
        models = [subtask.ml.model for subtask in self._subtasks]

        self._model = YotoConnectionModel(
            layer_hiddens = self.yoto_model_args['layer_hiddens'],
            common_hiddens = self.yoto_model_args['common_hiddens'],
            dropout = self.yoto_model_args['dropout'],
            sampler_args = self.yoto_model_args['sampler_args'], 
            lambda_to_weight = self.lambda_to_weight,
            device = self._device, 
            batch_size = self._batch_size,
            models=models,
            input_var_index=self._input_var_index,
            output_var_index=self._output_var_index,
        )
        
    
    def fit(self, train_data = None, valid_data = None, dataloaders = None, valid_step = 4) : 
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

        Returns:
            list: history data of train and valid.
        """
        self.ml.validate('train')

        if dataloaders is None:
            dataloaders = dict(train = self.prepare_dataloader(train_data, 'train'), valid = self.prepare_dataloader(valid_data, 'valid'))

        early_stopping = util.EarlyStopping(patience = self._max_patience)
        self._scaler = torch.cuda.amp.GradScaler(enabled = self._is_gpu)

        history = {'train': [], 'valid': []}
        
        torch.backends.cudnn.benchmark = True
        
        
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
                
                self.ml.model.set_lambdas( None ) 
                
                result = self.step_epoch(epoch, 'valid', dataloaders['valid'])
                history['valid'].append(result)

                if early_stopping(result['loss'], self.ml.model):
                    break

            if self.ml.scheduler is not None:
                self.ml.scheduler.step()

        if self._early_stopping:
            best_model = early_stopping.best_model.state_dict()
            self.ml.model.load_state_dict(copy.deepcopy(best_model))

        return history
        
    def step_loss(self, outputs, labels) : 
        loss_weights = self.ml.model.get_loss_weight()
        loss_result = {'loss': 0, 'subloss': []}
        loss_tmps = 0
        for loss_fn, loss_w, output, label in zip(self.ml.loss, loss_weights, outputs, labels):
            loss_tmp  = loss_fn(output, label)
            loss_tmp = self._loss_merge_f( loss_tmp * loss_w )
            loss_result['loss'] += loss_tmp
            loss_result['subloss'].append(loss_tmp)
        
        return loss_result


