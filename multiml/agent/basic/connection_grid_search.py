import os
import json

from multiml import logger

from ..basic import GridSearchAgent
from . import ConnectionRandomSearchAgent


class ConnectionGridSearchAgent(GridSearchAgent, ConnectionRandomSearchAgent):
    """Agent executing with all possible hyperparameter combination."""
    def __init__(self, reuse_pretraining=False, **kwargs):
        """Initialize.

        Args:
            reuse_pretraining (bool): Use common model weights for pretraining
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._history_agent = []

        self._reuse_pretraining = reuse_pretraining

    @logger.logging
    def execute(self):
        """Execute."""

        cache_model = dict()
        # Iterate all combinations
        for counter, subtasktuples in enumerate(self.task_scheduler):
            logger.counter(counter, len(self.task_scheduler))
            result_task_ids = []
            result_subtask_ids = []
            result_subtask_hps = []

            for subtasktuple in subtasktuples:
                task_id = subtasktuple.task_id
                subtask_id = subtasktuple.subtask_id
                subtask_env = subtasktuple.env
                subtask_hps = subtasktuple.hps.copy()
                hps_hash = subtask_id + " : " + json.dumps(subtask_hps, sort_keys=True)
                logger.info(f'subtask is {subtask_id}')

                if self._reuse_pretraining and (hps_hash in cache_model):
                    # Set hyperparameter
                    subtask_hps['job_id'] = cache_model[hps_hash]
                    subtask_env.set_hps(subtask_hps)

                    if 'load_weights' in self._connectiontask_args and isinstance(
                            self._connectiontask_args['load_weights'], dict):
                        model_path = self._connectiontask_args['load_weights'][
                            f"{subtask_env.name}__{subtask_hps['job_id']}"]
                    else:
                        unique_id = subtask_env.get_unique_id()

                        # for compatibility
                        if unique_id not in self.saver.keys():
                            load_config = self._saver.load_ml(subtask_env.name,
                                                              suffix=subtask_hps['job_id'])
                        else:
                            load_config = self._saver.load_ml(unique_id)

                        if 'model_path' not in load_config:
                            raise ValueError(
                                f"model_path is missing. subtaskname = {subtask_env.name} / job_id = {subtask_hps['job_id']} is not saved correctly."
                            )
                        model_path = load_config['model_path']

                    from multiml.task.keras import KerasBaseTask
                    if isinstance(
                            subtask_env,
                            KerasBaseTask) and subtask_env._trainable_model and not os.path.exists(
                                f'{model_path}.index'):
                        raise ValueError(f'model weight does not exist. model_path = {model_path}')

                    load_weights = subtask_env._load_weights
                    phases = subtask_env._phases
                    subtask_env._load_weights = model_path
                    subtask_env._phases = []

                    self._execute_subtask(subtasktuple, is_pretraining=True)

                    subtask_env._load_weights = load_weights
                    subtask_env._phases = phases
                else:
                    # Set hyperparameter
                    subtask_hps['job_id'] = counter
                    subtask_env.set_hps(subtask_hps)

                    cache_model[hps_hash] = subtask_hps['job_id']

                    self._execute_subtask(subtasktuple, is_pretraining=True)

                if '_model_fit' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env.ml.model, False)

                result_task_ids.append(task_id)
                result_subtask_ids.append(subtask_id)
                result_subtask_hps.append(subtask_hps)

            # Build and train model connection tasks
            subtasks = [v.env for v in subtasktuples]
            subtask = self._build_connected_models(subtasks, job_id=counter)
            self._execute_subtask(subtask, is_pretraining=False)

            self._metric.storegate = self._storegate
            metric = self._metric.calculate()

            result = dict(task_ids=result_task_ids,
                          subtask_ids=result_subtask_ids,
                          subtask_hps=result_subtask_hps,
                          metric_value=metric)

            self._history.append(result)
            self._history_agent.append({"job_id": counter})

    @logger.logging
    def finalize(self):
        """Finalize."""
        super().finalize()

        metrics = [result['metric_value'] for result in self._history]
        if self._metric_type == 'max':
            index = metrics.index(max(metrics))
        elif self._metric_type == 'min':
            index = metrics.index(min(metrics))
        else:
            raise ValueError(f'{self._metric_type} is not valid option.')
        self._best_result_config = self._history_agent[index]

    def get_best_result(self):
        """Returns the best combination as a result of agent execution.

        Returns:
            dict: best result
            dict: auxiliary value of the best result
        """
        return self._result, self._best_result_config
