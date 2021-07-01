from multiml import logger
from multiml.task.pytorch import PytorchASNGNASTask
from multiml.task.pytorch import PytorchASNGNASBlockTask

from . import PytorchConnectionRandomSearchAgent
from multiml.task.pytorch.datasets import StoreGateDataset, NumpyDataset
import numpy as np


class PytorchASNGNASAgent(PytorchConnectionRandomSearchAgent):
    """ Agent packing subtasks using Pytorch ASNG-NAS Model
    """
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
            asng_args,
            #lam=2, delta_init_factor=1, alpha = 1.5, clipping_value = None,
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
        self.clipping_value = clipping_value
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._scheduler = scheduler
        self._scheduler_args = scheduler_args
        self.loss_weights = {}
        self.is_connected_categorical = False

    @logger.logging
    def execute(self):
        """ Execute
        Currently, only categorical ASNG NAS is implemented.
        """

        asng_block_list = None
        if self.is_connected_categorical:
            asng_block_list, task_ids = self._build_connected_task_block_list()
        else:
            asng_block_list, task_ids = self._build_disconnected_task_block_list(
            )

        asng_task = PytorchASNGNASTask(
            asng_args=self.asng_args,
            subtasks=asng_block_list,
            auto_ordering=False,
            use_multi_loss=self._connectiontask_args["use_multi_loss"],
            variable_mapping=self._connectiontask_args["variable_mapping"],
            saver=self._saver,
            device=self._connectiontask_args['device'],
            gpu_ids=None,
            amp=False,  # expert option
            benchmark=False,  # expert option
            unpack_inputs=self._connectiontask_args["unpack_inputs"],
            view_as_outputs=False,  # expert option
            metrics=self._connectiontask_args["metrics"],
            verbose=self._verbose,
            num_epochs=self._num_epochs,
            batch_size=self.batch_size,
            max_patience=self._max_patience,
            loss_weights=self.loss_weights,
            optimizer=self._optimizer,
            optimizer_args=self._optimizer_args,
            scheduler=self._scheduler,
            scheduler_args=self._scheduler_args,
        )

        self._task_scheduler.add_task(task_id='ASNG-NAS', add_to_dag=False)
        self._task_scheduler.add_subtask('ASNG-NAS',
                                         'main-task',
                                         env=asng_task)
        asng_subtask = self._task_scheduler.get_subtask(
            'ASNG-NAS', 'main-task')

        if not self._connectiontask_args["load_weights"]:
            unique_id = asng_task.get_unique_id()
            self.saver.dump_ml(unique_id,
                               ml_type='pytorch',
                               model=asng_task.ml.model)

        # Save model ordering (model index)
        submodel_names = asng_subtask.env.get_submodel_names()
        self._saver.add(f'ASNG-NAS_{submodel_names}', submodel_names)
        asng_subtask.env.verbose = self._verbose

        self._execute_subtask(asng_subtask, is_pretraining=False)

        # check best model
        asng_task.set_most_likely()

        # re-train
        best_task_ids, best_subtask_ids = asng_task.best_model()
        best_subtasks = [
            self._task_scheduler.get_subtask(task_id, subtask_id)
            for task_id, subtask_id in zip(task_ids, best_subtask_ids)
        ]
        best_combination_task = self._build_connected_models(
            subtasks=[t.env for t in best_subtasks],
            job_id='ASNG-NAS-Final',
            use_task_scheduler=True)
        best_comb = '+'.join(s for s in best_subtask_ids)

        self._execute_subtask(best_combination_task, is_pretraining=False)
        self._metric.storegate = self._storegate
        metric = self._metric.calculate()

        ### evaluate
        # make results for json output
        # seed, nevents, walltime will be set at outside
        results_json = {'agent': 'ASNG-NAS', 'tasks': {}}

        # best_combination_task.env._unpack_inputs = True

        c_cat, c_int = asng_task.get_most_likely()
        theta_cat, theta_int = asng_task.get_thetas()
        cat_idx = c_cat.argmax(axis=1)

        pred, loss = best_combination_task.env.predict_and_loss()
        best_combination_task.env._storegate.update_data(
            data=pred,
            var_names=best_combination_task.env._output_var_names,
            phase='auto')
        self._metric._storegate = best_combination_task.env._storegate
        test_metric = self._metric.calculate()

        self.result = dict(task_ids=['ASNG-NAS-Final'],
                           subtask_ids=best_subtask_ids,
                           subtask_hps=[None],
                           metric_value=test_metric)

        test_result = dict(model_name='ASNG-NAS-Final',
                           cat_idx=cat_idx,
                           metric=test_metric)
        self._saver.add(f"results.ASNG-NAS-Final", test_result)

        results_json['loss_test'] = loss['loss']
        results_json['subloss_test'] = loss['subloss']
        results_json['metric_test'] = test_metric

        for task_idx, task_id in enumerate(task_ids):
            results_json['tasks'][task_id] = {}
            results_json['tasks'][task_id][
                'weight'] = best_combination_task.env.ml.loss_weights[task_idx]
            results_json['tasks'][task_id]['models'] = []
            results_json['tasks'][task_id]['theta_cat'] = []

            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id)
            for subtask_idx, subtask in enumerate(subtasktuples):
                this_id = subtask.subtask_id.split('-')[
                    -1]  # FIXME : hard coded
                theta = theta_cat[task_idx, subtask_idx]
                results_json['tasks'][task_id]['models'].append(this_id)
                results_json['tasks'][task_id]['theta_cat'].append(theta)

                if subtask_idx == cat_idx[task_idx]:
                    results_json['tasks'][task_id]['model_selected'] = this_id

                if theta_cat is not None:
                    logger.info(f'  theta_cat is {this_id: >20} : {theta:.3e}')
                else:
                    logger.info(f'theta_cat is None')

        if theta_int is not None:
            for theta, job_id in zip(theta_int.tolist(), ):
                for t, j in zip(theta, job_id):
                    logger.info(f'  theta_cat is {j: >20} : {t:.3e}')
        else:
            logger.info(f'theta_int is None')

        logger.info(f'best cat_idx is {cat_idx}')
        logger.info(f'best combination is {best_comb}')

        self.results_json = results_json

    def _build_disconnected_task_block_list(self):
        task_ids = []
        asng_block_list = []

        for task_idx, task_id in enumerate(
                self._task_scheduler.get_sorted_task_ids()):
            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id)

            for subtask_idx, subtask in enumerate(subtasktuples):
                subtask_env = subtask.env
                subtask_hps = subtask.hps
                subtask_env.set_hps(subtask_hps)

                if self.do_pretraining:
                    logger.info(
                        f'pretraining of {subtask_env.subtask_id} is starting...'
                    )
                    self._execute_subtask(subtask, is_pretraining=True)
                else:
                    subtask.env.storegate = self._storegate
                    subtask.env.saver = self._saver
                    subtask.env.compile()

                if '_model_fit' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env._model_fit,
                                                  False)

            l = ', '.join(subtask.env.subtask_id for subtask in subtasktuples)
            logger.info(f'{l}')
            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'asng_block_{task_id}_submodel_params',
                            params_list)

            # build asng task block
            subtasks = [v.env for v in subtasktuples]
            asng_block_subtask = self._build_block_task(subtasks,
                                                        task_id,
                                                        is_pretraining=False)
            asng_block_list.append(asng_block_subtask.env)
            task_ids.append(task_id)

        return asng_block_list, task_ids

    def _build_connected_task_block_list(self):
        task_ids = []
        asng_block_list = []
        all_subtasks = []

        for task_idx, task_id in enumerate(
                self._task_scheduler.get_sorted_task_ids()):
            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id)

            for subtask_idx, subtask in enumerate(subtasktuples):
                subtask_env = subtask.env
                subtask_hps = subtask.hps
                subtask_env.set_hps(subtask_hps)

                logger.info(f'{subtask_env.subtask_id}')
                if self.do_pretraining:
                    logger.info(
                        f'pretraining of {subtask_env.subtask_id} is starting...'
                    )
                    self._execute_subtask(subtask, is_pretraining=True)
                else:
                    subtask.env.storegate = self._storegate
                    subtask.env.saver = self._saver
                    subtask.env.compile()

                if '_model_fit' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env._model_fit,
                                                  False)

                all_subtasks.append(subtask)

            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'asng_block_{task_id}_submodel_params',
                            params_list)
            task_ids.append(task_id)

        all_combinations = list(itertools.product(*all_subtasks))
        connected_subtasks = []

        for combination in all_combinations:
            connected_subtask = self._build_connected_models(
                subtasks=[subtask.env for subtask in combination],
                jobid='',
                use_task_scheduler=True)
            if self.do_pretraining:
                logger.info(
                    f'pretraining of {subtask_env.subtask_id} is starting...')
                self._execute_subtask(sconnected_subtask, is_pretraining=True)
            else:
                connected_subtask.env.storegate = self._storegate
                connected_subtask.env.saver = self._saver
                connected_subtask.env.compile()

            connected_subtasks.append(connected_subtask)

        # build asng task block
        task_id = 'Flatten'
        asng_block_subtask = self._build_block_task(connected_subtasks,
                                                    task_id,
                                                    is_pretraining=False)
        asng_block_list.append(asng_block_subtask.env)
        return asng_block_list, task_ids

    def _build_block_task(self, subtasks, task_id, is_pretraining):

        asng_block = PytorchASNGNASBlockTask(
            subtasks=subtasks,
            job_id=f'ASNG-NAS-Block-{task_id}',
            saver=self._saver,
            load_weights=self._connectiontask_args['load_weights'],
            unpack_inputs=False,
        )
        asng_task_id = 'ASNG-NAS-' + task_id

        self.loss_weights[asng_task_id] = self._connectiontask_args[
            'loss_weights'][task_id]

        self._task_scheduler.add_task(task_id=asng_task_id)
        self._task_scheduler.add_subtask(asng_task_id,
                                         'BlockTask',
                                         env=asng_block)
        asng_block_subtask = self._task_scheduler.get_subtask(
            asng_task_id, 'BlockTask')

        if is_pretraining:
            self._execute_subtask(asng_block_subtask, is_pretraining=True)
        else:
            asng_block_subtask.env.storegate = self._storegate
            asng_block_subtask.env.saver = self._saver
            asng_block_subtask.env.compile()

        if not self._connectiontask_args['load_weights']:
            unique_id = asng_block.get_unique_id()
            self.saver.dump_ml(unique_id,
                               ml_type='pytorch',
                               model=asng_block.ml.model)

        submodel_names = asng_block_subtask.env.get_submodel_names()
        self._saver.add(f'asng_block_{task_id}_submodel_names', submodel_names)

        return asng_block_subtask
