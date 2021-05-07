""" GridSearchAgent module.
"""

import multiprocessing as mp

from multiml import logger
from multiml.agent.basic.random_search import RandomSearchAgent
from multiml.agent.basic.sequential import resulttuple


class GridSearchAgent(RandomSearchAgent):
    """ Agent scanning all possible subtasks and hyper parameters.
    """
    def __init__(self, num_workers=None, **kwargs):
        """ Initialize grid scan agent.

        Args:
            num_workers (int): number of workers for multiprocessing. If
                ``num_workers`` is given, multiprocessing is enabled.
            kwargs (dict): arbitrary kwargs passed to ``RandomSearchAgent`` class.
        """
        super().__init__(**kwargs)
        self._num_workers = num_workers
        self._multiprocessing = False

        if self._num_workers is not None:
            self._multiprocessing = True

    @logger.logging
    def execute(self):
        """ Execute grid scan agent.
        """
        if not self._multiprocessing:
            for counter, subtasktuples in enumerate(self.task_scheduler):
                self._storegate.compile()
                result = self.execute_pipeline(subtasktuples, counter)
                self._history.append(result)

                logger.counter(counter + 1,
                               len(self.task_scheduler),
                               divide=1,
                               message=f'metric={result.metric_value}')

        else:  # multiprocessing
            if self._storegate.backend not in ('numpy', 'hybrid'):
                raise NotImplementedError(
                    'multiprocessing is supported for only numpy and hybrid backend'
                )

            ctx = mp.get_context('spawn')
            queue = ctx.Queue()
            args = []

            for counter, subtasktuples in enumerate(self.task_scheduler):
                args.append([subtasktuples, counter])

                if len(args) == self._num_workers:
                    self.execute_jobs(ctx, queue, args)
                    args = []
                    logger.counter(counter + 1,
                                   len(self.task_scheduler),
                                   divide=1)

            self.execute_jobs(ctx, queue, args)

    def execute_jobs(self, ctx, queue, args):
        """ (expert method) Execute multiprocessing jobs.
        """
        jobs = []

        for pool_id, pargs in enumerate(args):
            process = ctx.Process(target=self.execute_wrapper,
                                  args=(pool_id, queue, *pargs),
                                  daemon=False)
            jobs.append(process)
            process.start()

        for job in jobs:
            job.join()

        while not queue.empty():
            self._history.append(queue.get())

    def execute_wrapper(self, pool_id, queue, subtasktuples, counter):
        """ (expert method) Wrapper method to execute multiprocessing pipeline.
        """
        self._saver.set_mode('dict')
        if self._storegate.backend == 'hybrid':
            self._storegate.set_mode('numpy')

        for subtasktuple in subtasktuples:
            subtasktuple.env.pool_id = pool_id

        result = self.execute_pipeline(subtasktuples, counter)
        queue.put(result)
