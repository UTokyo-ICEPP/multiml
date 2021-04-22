from abc import abstractmethod

from multiml import logger
from multiml.agent.basic.random_search import resulttuple

from ..basic import RandomSearchAgent


class ConnectionRandomSearchAgent(RandomSearchAgent):
    """ Agent executing with only one possible hyperparameter combination
    """
    def __init__(self,
                 freeze_model_weights=False,
                 do_pretraining=True,
                 connectiontask_name=None,
                 connectiontask_args={},
                 **kwargs):
        """

        Args:
            freeze_model_weights (bool): Fix model trainable weights after pre-training
            do_pretraining (bool): run pre-training before model connection
            connectiontask_name (str): Task name for ModelConnectionTask
            connectiontask_args (dict): args for ModelConnectionTask
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._freeze_model_weights = freeze_model_weights
        self._do_pretraining = do_pretraining
        self._connectiontask_name = connectiontask_name

        self._connectiontask_args = connectiontask_args

    @logger.logging
    def execute(self):
        """ Execute
        """
        # Select first candidate of all combinations
        subtasktuples = self.task_scheduler[0]

        # Execute sub-models
        for subtasktuple in subtasktuples:
            subtask_env = subtasktuple.env
            subtask_hps = subtasktuple.hps

            subtask_env.set_hps(subtask_hps)
            self._execute_subtask(subtasktuple, is_pretraining=True)

            if '_model_fit' in dir(subtask_env):
                if self._freeze_model_weights:
                    self._set_trainable_flags(subtask_env._model_fit, False)

        subtasks = [v.env for v in subtasktuples]
        task_ids = [v.task_id for v in subtasktuples]
        subtask_ids = [v.subtask_id for v in subtasktuples]
        subtask_hps = [v.hps for v in subtasktuples]

        # Build and train model connection tasks
        subtask = self._build_connected_models(subtasks)
        self._execute_subtask(subtask, is_pretraining=False)

        self._metric.storegate = self._storegate
        metric = self._metric.calculate()
        self.result = resulttuple(task_ids, subtask_ids, subtask_hps, metric)

    def _build_connected_models(self,
                                subtasks,
                                job_id=None,
                                use_task_scheduler=True):
        task_id = 'connection'
        subtask_id = self.connectiontask_name_with_jobid(job_id)

        subtask = self._ModelConnectionTask(
            name=subtask_id,
            saver=self._saver,
            subtasks=subtasks,
            **self._connectiontask_args,
        )
        subtask.task_id = task_id
        subtask.subtask_id = subtask_id

        if job_id is not None:
            subtask.set_hps({"job_id": job_id})
            if isinstance(subtask._save_weights, str):
                subtask._save_weights += f'__{job_id}'
            if isinstance(subtask._load_weights, str):
                subtask._load_weights += f'__{job_id}'

        if use_task_scheduler:
            self._task_scheduler.add_task(task_id=task_id, add_to_dag=False)
            self._task_scheduler.add_subtask(task_id, subtask_id, env=subtask)
            task = self._task_scheduler.get_subtask(task_id, subtask_id)
            return task
        else:
            from multiml.hyperparameter import Hyperparameters
            from multiml.task_scheduler import subtasktuple
            return subtasktuple(task_id, subtask_id, subtask,
                                Hyperparameters())

    def _execute_subtask(self, subtask, is_pretraining):
        if is_pretraining is True and self._do_pretraining is False:
            phases = subtask.env._phases
            subtask.env._phases = ['test']

            super()._execute_subtask(subtask)

            subtask.env._phases = phases
        else:
            super()._execute_subtask(subtask)

    @abstractmethod
    def _set_trainable_flags(model, do_trainable):
        model.trainable = do_trainable
        # Maybe below lienes are unnecessary...
        for var in model.layers:
            var.trainable = do_trainable

    def connectiontask_name_with_jobid(self, job_id):
        """ Returns a formatted name for ConnectionTask. job_id is used as a suffix.

        Returns:
            str: formatted name for the ConnectionTask
        """
        if self._connectiontask_name is None:
            subtask_id = 'connection'
        else:
            subtask_id = self._connectiontask_name

        if job_id is not None:
            subtask_id += f".{job_id}"
        return subtask_id
