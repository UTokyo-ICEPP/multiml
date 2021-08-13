from multiml import logger
from multiml.task.pytorch import PytorchYotoConnectionTask

from . import PytorchConnectionRandomSearchAgent
from multiml.task.pytorch.datasets import StoreGateDataset, NumpyDataset
import numpy as np


class PytorchYotoConnectionAgent(PytorchConnectionRandomSearchAgent):
    """Agent packing subtasks using Pytorch Yoto Model."""
    def __init__(self, loss_merge_f, eval_lambdas, lambda_to_weight, verbose, yoto_model_args, 
                 num_epochs = 100, max_patience = 10, batch_size = 128, 
                 optimizer=None,
                 optimizer_args=None,
                 scheduler=None,
                 scheduler_args=None,
                 amp = False,
                 num_workers = 2,
                 **kwargs):
        """

        Args:
            training_choiceblock_model (bool): Training choiceblock model after connecting submodels
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self.do_pretraining = kwargs['do_pretraining']
        self._verbose = verbose
        self._loss_merge_f = loss_merge_f
        self._eval_lambdas = eval_lambdas
        self._lambda_to_weight = lambda_to_weight
        self._yoto_model_args = yoto_model_args
        self._num_epochs = num_epochs
        self._max_patience = max_patience
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._scheduler = scheduler
        self._scheduler_args = scheduler_args
        self._amp = amp
        self._num_workers = num_workers
        
        
        # this variable will be set in _build_block funciton

    @logger.logging
    def execute(self):
        """Execute"""

        yoto_block_list = self._build_task_block_list()

        yoto_task = PytorchYotoConnectionTask(
            loss_merge_f = self._loss_merge_f, 
            yoto_model_args = self._yoto_model_args, 
            lambda_to_weight = self._lambda_to_weight,
            subtasks=yoto_block_list,
            variable_mapping=self._connectiontask_args["variable_mapping"],
            saver=self._saver,
            device=self._connectiontask_args['device'],
            gpu_ids=None,
            amp= self._amp,  # expert option
            pin_memory = True,
            num_workers = self._num_workers, 
            
            metrics=self._connectiontask_args["metrics"],
            verbose=self._verbose,
            num_epochs=self._num_epochs,
            batch_size=self._batch_size,
            max_patience=self._max_patience,
            loss_weights=self._loss_weights,
            optimizer=self._optimizer,
            optimizer_args=self._optimizer_args,
            scheduler=self._scheduler,
            scheduler_args=self._scheduler_args,
        )

        self._task_scheduler.add_task(task_id='Yoto-Connection', add_to_dag=False)
        self._task_scheduler.add_subtask('Yoto-Connection', 'main-task', env = yoto_task)
        yoto_subtask = self._task_scheduler.get_subtask('Yoto-Connection', 'main-task')

        if not self._connectiontask_args["load_weights"]:
            unique_id = yoto_task.get_unique_id()
            self.saver.dump_ml(unique_id, ml_type='pytorch', model = yoto_task.ml.model )

        # Save model ordering (model index)
        self._saver.add(f'Yoto-Connection', 'Yoto-Connection')
        yoto_subtask.env.verbose = self._verbose

        self._execute_subtask(yoto_subtask, is_pretraining=False)
        
        test_metrics = []
        for lambdas in self._eval_lambdas : 
            yoto_subtask.env.ml.model.set_lambdas(lambdas)
            pred_result = yoto_subtask.env.predict(label = True)
            data = pred_result.pop('pred')
            yoto_subtask.env.storegate.update_data( data = data, 
                                                    var_names = yoto_subtask.env._output_var_names,
                                                    phase = 'auto')
            self._metric._storegate = yoto_subtask.env._storegate
            test_metric = self._metric.calculate()
            pred_result['test_metric'] = test_metric
            pred_result['lambdas'] = lambdas
            
            test_metrics.append( pred_result )
            
        self.result = dict( 
            task_ids = self.task_ids,
            subtask_ids = self.subtask_ids,
            subtask_hps = [],
            metric_value = test_metrics )
        
        self.results_json = dict( metric_values = test_metrics )
        
    def _build_task_block_list(self):
        yoto_block_list = []
        self._loss_weights = []

        self.task_ids = []
        self.subtask_ids = []
        
        for task_idx, task_id in enumerate(self._task_scheduler.get_sorted_task_ids()):
            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id, is_grid_hps=False)
            
            self.task_ids.append( task_id )
            for subtask_idx, subtask in enumerate(subtasktuples):
                subtask_env = subtask.env
                subtask_hps = subtask.hps
                subtask_env.set_hps(subtask_hps)
                
                self.subtask_ids.append( subtask.env.subtask_id) 

                if self.do_pretraining:
                    logger.info(f'pretraining of {subtask_env.subtask_id} is starting...')
                    self._execute_subtask(subtask, is_pretraining=True)
                else:
                    subtask.env.storegate = self._storegate
                    subtask.env.saver = self._saver
                    subtask.env.compile()

                if '_model_fit' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env._model_fit, False)
            
            
            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'yoto_block_{task_id}_submodel_params', params_list)
            
            # build task block
            subtasks = [v.env for v in subtasktuples]

            if len(subtasks) == 1:
                yoto_block_subtask = subtasktuples[0]

            else:
                # yoto_block_subtask = self._build_block_task(subtasks, task_id, is_pretraining = False)
                raise ValueError(f'Yoto support only sequential subtask format!!')
            
            yoto_task_id = yoto_block_subtask.task_id
            self._loss_weights.append(1.0)
            yoto_block_list.append(yoto_block_subtask.env)

        return yoto_block_list

