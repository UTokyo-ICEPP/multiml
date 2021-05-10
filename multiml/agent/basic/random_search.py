""" RandomSearchAgent module.
"""
import random
import multiprocessing as mp

from multiml import logger
from multiml.agent.basic import SequentialAgent


class RandomSearchAgent(SequentialAgent):
    """ Agent executing random search..
    """
    def __init__(self,
                 samplings=[0],
                 seed=0,
                 metric_type=None,
                 num_workers=None,
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
            num_workers (int): number of workers for multiprocessing. If
                ``num_workers`` is given, multiprocessing is enabled.
            dump_all_results (bool): dump all results or not.
        """
        super().__init__(**kwargs)
        self._history = []
        self._samplings = samplings
        self._seed = seed
        self._dump_all_results = dump_all_results
        self._task_prod = self.task_scheduler.get_all_subtasks_with_hps()

        self._num_workers = num_workers
        self._multiprocessing = False

        if self._num_workers is not None:
            self._multiprocessing = True

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
        self._history = history

    @logger.logging
    def execute(self):
        """ Execute simple agent.
        """
        if isinstance(self._samplings, int):
            samples = range(0, len(self.task_scheduler))
            self._samplings = random.choices(samples, k=self._samplings)

        if not self._multiprocessing:
            for counter, index in enumerate(self._samplings):
                subtasktuples = self.task_scheduler[index]
                result = self.execute_subtasktuples(subtasktuples, counter)
                self._history.append(result)

        else:  # multiprocessing
            if self._storegate.backend not in ('numpy', 'hybrid'):
                raise NotImplementedError(
                    'multiprocessing is supported for only numpy and hybrid backend'
                )

            ctx = mp.get_context('spawn')
            queue = ctx.Queue()
            args = []

            for counter, index in enumerate(self._samplings):
                subtasktuples = self.task_scheduler[index]
                args.append([subtasktuples, counter])

                if len(args) == self._num_workers:
                    self.execute_jobs(ctx, queue, args)
                    args = []
                    logger.counter(counter + 1,
                                   len(self.task_scheduler),
                                   divide=1)

            self.execute_jobs(ctx, queue, args)

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
            header = f'All result of {self.__class__.__name__}'
            data = []
            for result in self._history:
                names, tmp_data = self._print_result(result)
                data += tmp_data
                data += '-'
            logger.table(header=header, names=names, data=data, max_length=40)

        header = f'Best result of {self.__class__.__name__}'
        names, data = self._print_result(self._result)
        logger.table(header=header, names=names, data=data, max_length=40)

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

        result = self.execute_subtasktuples(subtasktuples, counter)
        queue.put(result)
