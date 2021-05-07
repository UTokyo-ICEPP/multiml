""" RandomSearchAgent module.

Attributes:
    resulttuple (namedtuple): namedtuple contains lists of results.
"""
import random

from multiml import logger
from multiml.agent.basic import SequentialAgent


class RandomSearchAgent(SequentialAgent):
    """ Agent executing random search..
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
    def history(self):
        """ Return history of execution.
        """
        return self._history

    @history.setter
    def history(self, history):
        """ Set history of execution.
        """
        self._hisotry = history

    @logger.logging
    def execute(self):
        """ Execute simple agent.
        """
        if isinstance(self._samplings, int):
            samples = range(0, len(self.task_scheduler))
            self._samplings = random.choices(samples, k=self._samplings)

        for counter, index in enumerate(self._samplings):
            subtasktuples = self.task_scheduler[index]
            result = self.execute_subtasktuples(subtasktuples, counter)
            self._history.append(result)

    @logger.logging
    def finalize(self):
        """ Finalize grid scan agent.
        """
        if self._result is None:
            metrics = [result.metric_value for result in self._history]

            if self._metric_type == 'max':
                index = metrics.index(max(metrics))
            elif self._metric_type == 'min':
                index = metrics.index(min(metrics))
            else:
                raise NotImplementedError(
                    f'{self._metric_type} is not valid option.')

            self._result = self._history[index]

        # print results
        if self._dump_all_results:
            logger.header1('All results')
            for result in self._history:
                self._print_result(result)

        logger.header1('Best results')
        self._print_result(self._result)
