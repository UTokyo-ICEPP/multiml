""" SequentialAgent module.
"""
from collections import namedtuple

from multiml import logger
from multiml.agent.basic import BaseAgent


class SequentialAgent(BaseAgent):
    """ Agent execute sequential tasks.

    Examples:
        >>> task0 = your_task0
        >>> task1 = your_task1
        >>> task2 = your_task2
        >>>
        >>> agent = SequentialAgent(storegate=storegate,
        >>>                         task_scheduler=[task0, task1, task2],
        >>>                         metric=your_metric)
        >>> agent.execute()
        >>> agent.finalize()
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
        self._result = self.execute_subtasktuples(subtasktuples, 0)

    @logger.logging
    def finalize(self):
        """ Finalize sequential agent.
        """
        if self._result is None:
            logger.warn(f'No result at finalize of {self.__class__.__name__}')

        else:
            header = f'Result of {self.__class__.__name__}'
            names, data = self._print_result(self._result)
            logger.table(header=header, names=names, data=data, max_length=40)
            self.saver['result'] = self._result

    def execute_subtasktuples(self, subtasktuples, counter):
        """ Execute given subtasktuples.
        """
        if self._differentiable is None:
            return self.execute_pipeline(subtasktuples, counter)

        else:
            return self.execute_differentiable(subtasktuples, counter)

    def execute_pipeline(self, subtasktuples, counter):
        """ Execute pipeline.
        """
        result = {
            'task_ids': [],
            'subtask_ids': [],
            'subtask_hps': [],
            'metric_value': None
        }

        for subtasktuple in subtasktuples:
            task_id = subtasktuple.task_id
            subtask_id = subtasktuple.subtask_id
            subtask_env = subtasktuple.env
            subtask_hps = subtasktuple.hps.copy()

            subtask_env.saver = self._saver
            subtask_env.job_id = counter
            subtask_env.set_hps(subtask_hps)
            self._execute_subtask(subtasktuple)

            result['task_ids'].append(task_id)
            result['subtask_ids'].append(subtask_id)
            result['subtask_hps'].append(subtask_hps)

        self._metric.storegate = self._storegate
        result['metric_value'] = self._metric.calculate()

        return result

    def execute_differentiable(self, subtasktuples, counter):
        """ Execute connection model.
        """
        result = {
            'task_ids': [],
            'subtask_ids': [],
            'subtask_hps': [],
            'metric_value': None
        }

        if self._diff_pretrain:
            for subtasktuple in subtasktuples:
                task_id = subtasktuple.task_id
                subtask_id = subtasktuple.subtask_id
                subtask_env = subtasktuple.env
                subtask_hps = subtasktuple.hps.copy()

                subtask_env.saver = self._saver
                subtask_env.job_id = counter
                subtask_env.set_hps(subtask_hps)
                self._execute_subtask(subtasktuple)

                result['task_ids'].append(task_id)
                result['subtask_ids'].append(subtask_id)
                result['subtask_hps'].append(subtask_hps)

        subtasks = [v.env for v in subtasktuples]
        self._diff_task_args['auto_ordering'] = False

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

        result['task_ids'].append(task_id)
        result['subtask_ids'].append(subtask_id)
        result['subtask_hps'].append(hps)
        result['metric_value'] = metric

        return result

    def _execute_subtask(self, subtask, is_skip=False):
        """ Execute subtask.
        """

        subtask.env.storegate = self._storegate
        subtask.env.saver = self._saver
        subtask.env.execute()
        subtask.env.finalize()

    def _print_result(self, result):
        """ Returns print result.
        """
        metric_name = self._metric.name
        names = ['task_id', 'subtask_id', 'hps', f'metric({metric_name})']
        data = []

        for task_id, subtask_id, subtask_hp in zip(result['task_ids'],
                                                   result['subtask_ids'],
                                                   result['subtask_hps']):
            if subtask_hp is None or len(subtask_hp) == 0:
                data.append([task_id, subtask_id, 'no hyperparameters'])
            else:
                for index, (key, value) in enumerate(subtask_hp.items()):
                    if index == 0:
                        data.append([task_id, subtask_id, f'{key} = {value}'])
                    else:
                        data.append(['', '', f'{key} = {value}'])

        metric_data = []
        for index, idata in enumerate(data):
            if index == 0:
                metric_data.append(idata + [f'{result["metric_value"]}'])
            else:
                metric_data.append(idata + [''])

        return names, metric_data
