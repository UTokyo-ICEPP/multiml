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
            for job_id, subtasktuples in enumerate(self.task_scheduler):
                if self._num_trials is None:
                    result = self.execute_subtasktuples(subtasktuples, job_id)
                    self._history.append(result)

                else:
                    for trial_id in range(self._num_trials):
                        result = self.execute_subtasktuples(subtasktuples, job_id, trial_id)
                        self._history.append(result)

        else:  # multiprocessing
            if self._storegate.backend not in ('numpy', 'hybrid'):
                raise NotImplementedError(
                    'multiprocessing is supported for only numpy and hybrid backend')

            ctx = mp.get_context(self._context)
            queue = ctx.Queue()
            args = []

            for job_id, subtasktuples in enumerate(self.task_scheduler):
                if self._num_trials is None:
                    args.append([subtasktuples, job_id, None])
              
                else:
                    for trial_id in range(self._num_trials):
                        args.append([subtasktuples, job_id, trial_id])

            self.execute_pool_jobs(ctx, queue, args)
