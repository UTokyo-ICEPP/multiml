""" RandomSearchAgent module.

Attributes:
    resulttuple (namedtuple): namedtuple contains lists of results.
"""
import random
import itertools
from collections import namedtuple

from multiml import logger
from multiml.agent.basic import BaseAgent

resulttuple = namedtuple(
    'resulttuple', ('task_ids', 'subtask_ids', 'subtask_hps', 'metric_value'))


class RandomSearchAgent(BaseAgent):
    """ Agent executing with only one possible hyper parameter combination.
    """
    def __init__(self,
                 samplings=[0],
                 seed=0,
                 metric_type=None,
                 dump_all_results=False,
                 **kwargs):
        """ Initialize simple agent.

        Args:
            samplings (int or list): If int, number of random samplings.
                If list, indexes of combination. 
            seed (int): seed of random samplings.
            metric_type (str): 'min' or 'max' for indicating direction of
                metric optimization. If it is None, ``type`` is retrieved from
                metric class instance.
            dump_all_results (bool): dump all results or not.
        """
        super().__init__(**kwargs)
        self._result = None  # backward compatibility
        self._history = []
        self._samplings = samplings
        self._seed = seed
        self._dump_all_results = dump_all_results
        self._task_prod = self.task_scheduler.get_all_subtasks_with_hps()

        if metric_type is None:
            self._metric_type = self._metric.type
        else:
            self._metric_type = metric_type

        random.seed(seed)

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
        """ Execute simple agent.
        """
        if isinstance(self._samplings, int):
            samples = range(0, len(self.task_scheduler))
            self._samplings = random.sample(samples, k=self._samplings)

        for counter, index in enumerate(self._samplings):
            subtasktuples = self.task_scheduler[index]
            self._history.append(self.execute_pipeline(subtasktuples, counter))

    @logger.logging
    def finalize(self):
        """ Finalize grid scan agent.
        """
        # backward compatibility
        if self._result is not None:
            self.finalize_legacy()
            return

        if self._dump_all_results:
            logger.header1('All results')
            for result in self._history:
                self._print_result(result)
            logger.header1('Best results')

        metrics = [result.metric_value for result in self._history]
        if self._metric_type == 'max':
            index = metrics.index(max(metrics))
        elif self._metric_type == 'min':
            index = metrics.index(min(metrics))
        else:
            raise NotImplementedError(
                f'{self._metric_type} is not valid option.')
        best_result = self._history[index]
        self._print_result(best_result)

        scan_history = [{
            "task_ids": result.task_ids,
            "subtask_ids": result.subtask_ids,
            "subtask_hps": result.subtask_hps,
            "metric_value": result.metric_value
        } for result in self._history]

        self._saver.add('scan_history', scan_history)

        self._saver.add('result.task_ids', best_result.task_ids)
        self._saver.add('result.subtask_ids', best_result.subtask_ids)
        self._saver.add('result.subtask_hps', best_result.subtask_hps)
        self._saver.add('result.metric_value', best_result.metric_value)

        self._best_result = best_result

        self._saver.save()

    @logger.logging
    def finalize_legacy(self):
        """ Finalize simple agent. Result is shown and saved.
        """
        self._print_result(self._result)

        self._saver.add('result.task_ids', self._result.task_ids)
        self._saver.add('result.subtask_ids', self._result.subtask_ids)
        self._saver.add('result.subtask_hps', self._result.subtask_hps)
        self._saver.add('result.metric_value', self._result.metric_value)

        self._saver.save()

    def get_best_result(self):
        """ Return the best result.
        """
        return self._best_result

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
