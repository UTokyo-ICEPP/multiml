""" SequentialAgent module.

Attributes:
    resulttuple (namedtuple): namedtuple contains lists of results.
"""
from collections import namedtuple

from multiml import logger
from multiml.agent.basic import BaseAgent

resulttuple = namedtuple(
    'resulttuple', ('task_ids', 'subtask_ids', 'subtask_hps', 'metric_value'))


class SequentialAgent(BaseAgent):
    """ Agent execute sequential tasks.
    """
    def __init__(self,
                 differentiable=None,
                 diff_pretrain=False,
                 diff_task_args=None,
                 **kwargs):
        """ Initialize sequential agent.

        Args:
            differentiable (str): ``keras`` or ``pytorch``. If differentiable
                is given, ``ConnectionTask()`` is created based on sequential
                tasks. If differentiable is None (default), sequential tasks
                are executed step by step.
            diff_pretrain (bool): If True, each subtask is trained before
                creating `ConnectionTask()``.
            diff_task_args (dict): arbitrary args passed to ``ConnectionTask()``.
        """
        if diff_task_args is None:
            diff_task_args = {}

        super().__init__(**kwargs)
        self._result = None
        self._differentiable = differentiable
        self._diff_pretrain = diff_pretrain
        self._diff_task_args = diff_task_args

    @property
    def result(self):
        """ Return result of execution.
        """
        return self._result

    @result.setter
    def result(self, result):
        """ Set result of execution.
        """
        self._result = result

    @logger.logging
    def execute(self):
        """ Execute sequential agent.
        """
        if len(self.task_scheduler) != 1:
            raise ValueError(
                'Multiple sutasks or hyperparameters are defined.')

        subtasktuples = self.task_scheduler[0]
        if self._differentiable is None:
            self._result = self.execute_pipeline(subtasktuples, 0)

        else:
            self._result = self.execute_connection_model(subtasktuples, 0)

    @logger.logging
    def finalize(self):
        """ Finalize sequential agent.
        """
        if self._result is None:
            logger.warn(f'No result at finalize of {self.__class__.__name__}')

        else:
            self._print_result(self._result)

    def execute_pipeline(self, subtasktuples, counter):
        """ Execute pipeline.
        """
        result_task_ids = []
        result_subtask_ids = []
        result_subtask_hps = []

        for subtasktuple in subtasktuples:
            task_id = subtasktuple.task_id
            subtask_id = subtasktuple.subtask_id
            subtask_env = subtasktuple.env
            subtask_hps = subtasktuple.hps.copy()
            subtask_hps['job_id'] = counter

            subtask_env.saver = self._saver
            subtask_env.set_hps(subtask_hps)
            self._execute_subtask(subtasktuple)

            result_task_ids.append(task_id)
            result_subtask_ids.append(subtask_id)
            result_subtask_hps.append(subtask_hps)

        self._metric.storegate = self._storegate
        metric = self._metric.calculate()

        return resulttuple(result_task_ids, result_subtask_ids,
                           result_subtask_hps, metric)

    def execute_connection_model(self, subtasktuples, counter):
        """ Execute connection model.
        """
        result_task_ids = []
        result_subtask_ids = []
        result_subtask_hps = []

        if self._diff_pretrain:
            for subtasktuple in subtasktuples:
                task_id = subtasktuple.task_id
                subtask_id = subtasktuple.subtask_id
                subtask_env = subtasktuple.env
                subtask_hps = subtasktuple.hps.copy()
                subtask_hps['job_id'] = counter

                subtask_env.saver = self._saver
                subtask_env.set_hps(subtask_hps)
                self._execute_subtask(subtasktuple)

                result_task_ids.append(task_id)
                result_subtask_ids.append(subtask_id)
                result_subtask_hps.append(subtask_hps)

        subtasks = [v.env for v in subtasktuples]

        if self._differentiable == 'keras':
            from multiml.task.keras import ModelConnectionTask

            subtask = ModelConnectionTask(
                subtasks=subtasks,
                **self._diff_task_args,
            )

        elif self._differentiable == 'pytorch':
            from multiml.task.pytorch import ModelConnectionTask

            subtask = ModelConnectionTask(
                subtasks=subtasks,
                **self._diff_task_args,
            )

        else:
            raise ValueError(
                f'differentiable: {self._differentiable} is not supported.')

        from multiml.hyperparameter import Hyperparameters
        from multiml.task_scheduler import subtasktuple

        task_id = 'connection-' + self._differentiable
        subtask_id = subtask.get_unique_id()
        hps = Hyperparameters()
        self._execute_subtask(subtasktuple(task_id, subtask_id, subtask, hps))

        self._metric.storegate = self._storegate
        metric = self._metric.calculate()

        return resulttuple(result_task_ids + [task_id],
                           result_subtask_ids + [subtask_id],
                           result_subtask_hps + [hps], metric)

    def _execute_subtask(self, subtask):
        """ Execute subtask.
        """
        subtask.env.storegate = self._storegate
        subtask.env.saver = self._saver
        subtask.env.execute()
        subtask.env.finalize()

    def _print_result(self, result):
        """ Show result.
        """
        logger.header2('Result')
        for task_id, subtask_id, subtask_hp in zip(result.task_ids,
                                                   result.subtask_ids,
                                                   result.subtask_hps):
            logger.info(f'task_id {task_id} and subtask_id {subtask_id} with:')
            if subtask_hp is None or len(subtask_hp) == 0:
                logger.info('  No hyperparameters')
            else:
                for key, value in subtask_hp.items():
                    logger.info(f'  {key} = {value}')
        logger.info(f'Metric ({self._metric.name}) is {result.metric_value}')
        logger.header2('')
