from multiml import logger
from multiml.task.pytorch import PytorchASNGNASTask
from multiml.task.pytorch import PytorchASNGChoiceBlockTask

from . import PytorchConnectionRandomSearchAgent
from multiml.task.pytorch.datasets import StoreGateDataset, NumpyDataset
import numpy as np


class PytorchASNGNASAgent(PytorchConnectionRandomSearchAgent):
    """Agent packing subtasks using Pytorch ASNG-NAS Model."""
    def __init__(
            self,
            verbose=1,
            num_epochs=1000,
            max_patience=5,
            batch_size={
                'type': 'equal_length',
                'length': 500,
                'test': 100
            },
            asng_args={
                'lam': 2,
                'delta': 0.0,
                'alpha': 1.5,
                'clipping_value': None,
                'range_restriction': True
            },
            optimizer=None,
            optimizer_args=None,
            scheduler=None,
            scheduler_args=None,
            **kwargs):
        """

        Args:
            training_choiceblock_model (bool): Training choiceblock model after connecting submodels
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self.do_pretraining = kwargs['do_pretraining']
        self._verbose = verbose
        self._num_epochs = num_epochs

        self.asng_args = asng_args

        self.batch_size = batch_size
        self._max_patience = max_patience
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._scheduler = scheduler
        self._scheduler_args = scheduler_args

        # this variable will be set in _build_block funciton
        self._loss_weights = {}

    @logger.logging
    def execute(self):
        """Execute Currently, only categorical ASNG NAS is implemented."""

        asng_block_list, task_ids = self._build_task_block_list()
        
        asng_task = PytorchASNGNASTask(
            asng_args=self.asng_args,
            subtasks=asng_block_list,
            variable_mapping=self._connectiontask_args["variable_mapping"],
            saver=self._saver,
            device=self._connectiontask_args['device'],
            gpu_ids=None,
            amp=False,  # expert option
            metrics=self._connectiontask_args["metrics"],
            verbose=self._verbose,
            num_epochs=self._num_epochs,
            batch_size=self.batch_size,
            max_patience=self._max_patience,
            loss_weights=self._loss_weights,
            optimizer=self._optimizer,
            optimizer_args=self._optimizer_args,
            scheduler=self._scheduler,
            scheduler_args=self._scheduler_args,
        )

        self._task_scheduler.add_task(task_id='ASNG-NAS', add_to_dag=False)
        self._task_scheduler.add_subtask('ASNG-NAS', 'main-task', env=asng_task)
        asng_subtask = self._task_scheduler.get_subtask('ASNG-NAS', 'main-task')

        if not self._connectiontask_args["load_weights"]:
            unique_id = asng_task.get_unique_id()
            self.saver.dump_ml(unique_id, ml_type='pytorch', model=asng_task.ml.model)

        # Save model ordering (model index)
        submodel_names = asng_subtask.env.get_submodel_names()
        self._saver.add(f'ASNG-NAS_{submodel_names}', submodel_names)
        asng_subtask.env.verbose = self._verbose

        self._execute_subtask(asng_subtask, is_pretraining=False)

        # check best model
        asng_task.set_most_likely()

        # re-train
        best_choice = asng_task.best_choice()
        best_theta  = asng_task.get_thetas()
        
        print(best_theta)
        best_subtasks = []
        
        best_subtask_ids = []
        for task_id, value in best_choice.items():
            idx = 0 if len(task_ids[task_id]) == 1 else value[task_id]
            best_subtasks.append( self._task_scheduler.get_subtask(task_id, task_ids[task_id][idx] ) )
            best_subtask_ids.append( [task_id, task_ids[task_id][idx]] )
        
        best_combination_task = self._build_connected_models(
            subtasks=[t.env for t in best_subtasks],
            job_id='ASNG-NAS-Final',
            use_task_scheduler=True
            )
        best_combination_task.env._phases = ['train', 'valid']
        self._execute_subtask(best_combination_task, is_pretraining=False)
        
        ### evaluate
        # make results for json output
        # seed, nevents, walltime will be set at outside
        
        pred_result = best_combination_task.env.predict( label = True )

        best_combination_task.env._storegate.update_data(data = pred_result['pred'],
            var_names = best_combination_task.env._output_var_names,
            phase = 'auto')
        self._metric._storegate = best_combination_task.env._storegate
        test_metric = self._metric.calculate()
        
        
        self.result = dict(task_ids=[v1 for v1, v2 in best_subtask_ids], 
                           subtask_ids=[v2 for v1, v2 in best_subtask_ids], 
                           subtask_hps=[ best_choice[v1] for v1, v2 in best_subtask_ids ],
                           metric_value=test_metric)

        test_result = dict(model_name='ASNG-NAS-Final', best_theta = best_theta, metric=test_metric)
        self._saver.add(f"results.ASNG-NAS-Final", test_result)
        
        results_json = {'agent': 'ASNG-NAS', 'tasks': {}, 'loss_test' : pred_result['loss'], 
                        'subloss_test' : pred_result['subloss'], 'metric_test' : test_metric }
        
        for task_idx, (task_id, value) in enumerate(best_choice.items()):
            results_json['tasks'][task_id] = { }
            results_json['tasks'][task_id]['weight'] = best_combination_task.env.ml.loss_weights[task_idx]
            
            if len(task_ids[task_id]) == 1 : 
                subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id, is_grid_hps = False)

                results_json['tasks'][task_id]['hps'] = subtasktuples[0].hps 
                results_json['tasks'][task_id]['hps_selected'] = { key : val[value[key]] for key, val in subtasktuples[0].hps.items() if key in value.keys()}

            else : 
                results_json['tasks'][task_id]['models'] = task_ids[task_id] 
                results_json['tasks'][task_id]['model_selected'] = task_ids[task_id][value[task_id]]
        
        logger.info(f'best combination is {best_subtask_ids}')

        self.results_json = results_json

    def _build_task_block_list(self):
        task_ids = {}
        asng_block_list = []
        
        
        for task_idx, task_id in enumerate(self._task_scheduler.get_sorted_task_ids()):
            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id, is_grid_hps = False)
            
            
            for subtask_idx, subtask in enumerate(subtasktuples):
                subtask_env = subtask.env
                subtask_hps = subtask.hps
                subtask_env.set_hps( {'hps' : subtask_hps } )
                
                
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

            l = ', '.join(subtask.env.subtask_id for subtask in subtasktuples)
            logger.info(f'{l}')
            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'asng_block_{task_id}_submodel_params', params_list)
            
            
            # build asng task block
            subtasks = [v.env for v in subtasktuples]
            task_ids[task_id] = [v.env.subtask_id for v in subtasktuples]
            
            if len(subtasks) == 1 : 
                asng_block_subtask = subtasktuples[0]
                
            else : 
                asng_block_subtask = self._build_block_task(subtasks, task_id, is_pretraining=False)
            
            asng_task_id = asng_block_subtask.task_id 
            self._loss_weights[asng_task_id] = self._connectiontask_args['loss_weights'][task_id]
            
            asng_block_list.append(asng_block_subtask.env)
            
            
        return asng_block_list, task_ids

    def _build_block_task(self, subtasks, task_id, is_pretraining):

        asng_block = PytorchASNGChoiceBlockTask(
            job_id=f'ASNG-NAS-Block-{task_id}',
            subtasks=subtasks,
            saver=self._saver,
            device=self._connectiontask_args['device'],
            load_weights=self._connectiontask_args['load_weights'],
            
            # gpu_ids=None,
            # amp=False,  # expert option
            # metrics=self._connectiontask_args["metrics"],
            # verbose=self._verbose,
            # num_epochs=self._num_epochs,
            # batch_size=self.batch_size,
            # max_patience=self._max_patience,
            # loss_weights=self._loss_weights,
            # optimizer=self._optimizer,
            # optimizer_args=self._optimizer_args,
            # scheduler=self._scheduler,
            # scheduler_args=self._scheduler_args,
        )
        # asng_task_id = 'ASNG-NAS-' + task_id
        asng_task_id = task_id
        
        self._task_scheduler.add_task(task_id=asng_task_id)
        self._task_scheduler.add_subtask(asng_task_id, 'BlockTask', env=asng_block)
        asng_block_subtask = self._task_scheduler.get_subtask(asng_task_id, 'BlockTask')

        if is_pretraining:
            self._execute_subtask(asng_block_subtask, is_pretraining=True)
        else:
            asng_block_subtask.env.storegate = self._storegate
            asng_block_subtask.env.saver = self._saver
            asng_block_subtask.env.compile()

        if not self._connectiontask_args['load_weights']:
            unique_id = asng_block.get_unique_id()
            self.saver.dump_ml(unique_id, ml_type='pytorch', model=asng_block.ml.model)

        submodel_names = asng_block_subtask.env.get_submodel_names()
        self._saver.add(f'asng_block_{task_id}_submodel_names', submodel_names)

        return asng_block_subtask
