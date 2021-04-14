from multiml import logger
from multiml.task.keras import EnsembleTask
from multiml.agent.basic.random_search import resulttuple

from . import KerasConnectionRandomSearchAgent


class KerasEnsembleAgent(KerasConnectionRandomSearchAgent):
    """ Agent packing subtasks using Keras EnsembelModel
    """
    def __init__(self, ensembletask_args={}, **kwargs):
        """

        Args:
            ensembletask_args (dict): args for EnsembleTask
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._ensembletask_args = ensembletask_args

        phases = self._ensembletask_args['phases']
        if phases is None or 'train' in phases:
            raise ValueError(
                "Not Implemented the case of phases = ['train'] in ensembletask"
            )

    @logger.logging
    def execute(self):
        """ Execute
        """
        ensemble_list = []  # list of ensembled subtasks

        tasks_list = self._task_scheduler.get_sorted_task_ids()
        for subtasktuples, task_id in zip(self._task_prod, tasks_list):
            # Execute sub-models
            for subtasktuple in subtasktuples:
                subtask_env = subtasktuple.env
                subtask_hps = subtasktuple.hps

                subtask_env.set_hps(subtask_hps)
                self._execute_subtask(subtasktuple, is_pretraining=True)

                if '_model' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env.ml.model, False)

            # Save model hyperparameters
            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'ensemble_{task_id}_submodel_params', params_list)

            # Not apply ensemble for single submodel set
            if len(subtasktuples) == 1:
                ensemble_list.append(subtasktuples[0].env)
                continue

            subtasks = [subtasktuple.env for subtasktuple in subtasktuples]
            ensemble = self._build_ensemble_task(subtasks, task_id)
            ensemble_list.append(ensemble.env)

        # Connecting each ensemble model
        subtask = self._build_connected_models(ensemble_list)
        self._execute_subtask(subtask, is_pretraining=False)

        # Model summary to show trainable/non-trainable variables
        if logger.MIN_LEVEL <= logger.DEBUG:
            subtask.env.ml.model.summary()

        self._metric.storegate = self._storegate
        metric = self._metric.calculate()
        self.result = resulttuple('connection', 'connection_ensemble', [],
                                  metric)

        # Save ensemble weights
        self._save_ensemble_weights(subtask, "connection_ensemble")

    def _build_ensemble_task(self, subtasks, task_id=None):
        # Merge sub-tasks with ensemble
        subtask_id = task_id + '_ensemble'
        ensemble = EnsembleTask(
            name=subtask_id,
            subtasks=subtasks,
            saver=self._saver,
            **self._ensembletask_args,
        )
        ensemble.task_id = task_id
        ensemble.subtask_id = subtask_id

        self._task_scheduler.add_task(task_id=task_id, add_to_dag=False)
        self._task_scheduler.add_subtask(task_id, subtask_id, env=ensemble)
        subtask = self._task_scheduler.get_subtask(task_id, subtask_id)

        self._execute_subtask(subtask, is_pretraining=True)

        # Save model ordering (model index)
        submodel_names = subtask.env.get_submodel_names()
        self._saver.add(f'ensemble_{task_id}_submodel_names', submodel_names)

        return subtask

    def _save_ensemble_weights(self, task, prefix):
        # Save ensemble weights
        for var in task.env._model._get_variables():
            if "_ensemble_weights/" in var.name:
                varname = var.name.split('/')[1]
                self._saver.add(f'{prefix}_{varname}', var.numpy().reshape(-1))
