"""GridSearchAgent module."""

import multiprocessing as mp

from multiml import logger
from multiml.agent.basic.random_search import RandomSearchAgent


class GridSearchAgent(RandomSearchAgent):
    """Agent scanning all possible subtasks and hyper parameters."""
    def __init__(self, **kwargs):
        """Initialize grid scan agent.

        Args:
            kwargs (dict): arbitrary kwargs passed to ``RandomSearchAgent`` class.
        """
        super().__init__(**kwargs)

    @logger.logging
    def execute(self):
        """Execute grid scan agent."""
        if not self._multiprocessing:
            for counter, subtasktuples in enumerate(self.task_scheduler):
                self._storegate.compile()
                result = self.execute_subtasktuples(subtasktuples, counter)
                self._history.append(result)

                logger.counter(counter + 1,
                               len(self.task_scheduler),
                               divide=1,
                               message=f'metric={result["metric_value"]}')

        else:  # multiprocessing
            if self._storegate.backend not in ('numpy', 'hybrid'):
                raise NotImplementedError(
                    'multiprocessing is supported for only numpy and hybrid backend')

            ctx = mp.get_context(self._context)
            queue = ctx.Queue()
            args = []

            for counter, subtasktuples in enumerate(self.task_scheduler):
                args.append([subtasktuples, counter])

                if len(args) == len(self._num_workers):
                    self.execute_jobs(ctx, queue, args)
                    args = []

            if len(args) > 0:
                self.execute_jobs(ctx, queue, args)
