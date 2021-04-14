""" BaseAgent module.
"""

from multiml import Saver, TaskScheduler, logger
from multiml.agent import Agent


class BaseAgent(Agent):
    """ Base class of agent.

    All agent class need to inherit this ``BaseAgent`` class.
    """
    def __init__(self,
                 saver=None,
                 storegate=None,
                 task_scheduler=None,
                 metric=None,
                 metric_args=None):
        """ Initialize base agent.

        Args:
            saver (Saver): ``Saver`` class instance. If ``saver`` is None,
                ``Saver`` class instance is created without any args.
            storegate (Storegate): ``Storegate`` class instance.
            task_scheduler (TaskScheduler or list): ``TaskScheduler`` class
                instance. If *ordered tasks* (list) are given, ``TaskScheduler``
                is initialized with *ordered tasks*. Please see
                ``TaskScheduler`` class for details.
            metric (str or BaseMetric): str or Metric class instance. If str
                is given, Metric class is searched from multiml.agent.metri,
                and initialized with ``metric_args`` below.
            metric_args (dict): arbitrary args of Metric class. This option is
                valid only if ``metric`` is str.
        """
        if metric_args is None:
            metric_args = {}

        if saver is None:
            saver = Saver()

        if isinstance(metric, str):
            if not metric.endswith('Metric'):
                metric += 'Metric'
            import multiml.agent.metric as metrics
            metric = getattr(metrics, metric)(**metric_args)

        if isinstance(task_scheduler, list):
            task_scheduler = TaskScheduler(task_scheduler)
            task_scheduler.show_info()

        self._saver = saver
        self._storegate = storegate
        self._task_scheduler = task_scheduler
        self._metric = metric

    def __repr__(self):
        result = f'{self.__class__.__name__}(metric={self._metric})'
        return result

    @logger.logging
    def execute(self):
        """ Execute base agent. Users need to implement algorithms.
        """

    @logger.logging
    def finalize(self):
        """ Finalize base agent.
        """

    @logger.logging
    def execute_finalize(self):
        """ Execute and finalize base agent.
        """
        self.execute()
        self.finalize()

    @property
    def storegate(self):
        """ Return storegate of base agent.
        """
        return self._storegate

    @storegate.setter
    def storegate(self, storegate):
        """ Set storegate to base agent.
        """
        self._storegate = storegate

    @property
    def saver(self):
        """ Return saver of base agent.
        """
        return self._saver

    @saver.setter
    def saver(self, saver):
        """ Set saver to base agent.
        """
        self._saver = saver

    @property
    def task_scheduler(self):
        """ Return task_scheduler of base agent.
        """
        return self._task_scheduler

    @task_scheduler.setter
    def task_scheduler(self, task_scheduler):
        """ Set task_scheduler to base agent.
        """
        self._task_scheduler = task_scheduler

    @property
    def metric(self):
        """ Return metric of base agent.
        """
        return self._metric

    @metric.setter
    def metric(self, metric):
        """ Set metric to base agent.
        """
        self._metric = metric
