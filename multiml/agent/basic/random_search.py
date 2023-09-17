"""RandomSearchAgent module."""
import random
import time
import copy
from tqdm import tqdm
import multiprocessing as mp

from multiml import logger
from multiml.agent.basic import SequentialAgent


class RandomSearchAgent(SequentialAgent):
    """Agent executing random search.."""
    def __init__(self,
                 samplings=None,
                 seed=0,
                 metric_type=None,
                 **kwargs):
        """Initialize simple agent.

        Args:
            samplings (int or list): If int, number of random samplings. If list, indexes of
                combination.
            seed (int): seed of random samplings.
            metric_type (str): 'min' or 'max' for indicating direction of metric optimization.
                If it is None, ``type`` is retrieved from metric class instance.
        """
        super().__init__(**kwargs)
        if samplings is None:
            samplings = [0]

        self._samplings = samplings
        self._seed = seed
        self._task_prod = self.task_scheduler.get_all_subtasks_with_hps()

        if metric_type is None:
            self._metric_type = self._metric.type
        else:
            self._metric_type = metric_type

        random.seed(seed)

    @logger.logging
    def execute(self):
        """Execute simple agent."""
        if isinstance(self._samplings, int):
            samples = range(0, len(self.task_scheduler))
            self._samplings = random.choices(samples, k=self._samplings)

        if not self._multiprocessing:
            for job_id, index in enumerate(self._samplings):
                subtasktuples = self.task_scheduler[index]

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

            for job_id, index in enumerate(self._samplings):
                subtasktuples = self.task_scheduler[index]

                if self._num_trials is None:
                    args.append([subtasktuples, job_id, None])
                else:
                    for trial_id in range(self._num_trials):
                        args.append([subtasktuples, job_id, trial_id])

            self.execute_pool_jobs(ctx, queue, args)

    @logger.logging
    def finalize(self):
        """Finalize random search agent."""
        if (self._result is None) and (self._history):
            metric_values_dict = {}
            results_dict = {}

            for history in self._history:
                job_id = history['job_id']

                if job_id in metric_values_dict:
                    metric_values_dict[job_id].append(history['metric_value'])
                else:
                    metric_values_dict[job_id] = [history['metric_value']]
                    results_dict[job_id] = history

            metric_value_dict = {}
            for key, value in metric_values_dict.items():
                metric_value_dict[key] = sum(value) / len(value)
                results_dict[key]['metric_value'] = sum(value) / len(value)

                if self._num_trials is not None:
                    results_dict[key]['metric_values'] = value
      
            if self._metric_type == 'max':
                job_id = max(metric_value_dict, key=metric_value_dict.get)
            elif self._metric_type == 'min':
                job_id = min(metric_value_dict, key=metric_value_dict.get)
            else:
                raise NotImplementedError(f'{self._metric_type} is not valid option.')

            self._result = results_dict[job_id]

            # print results
            if self._dump_all_results:
                header = f'All result of {self.__class__.__name__}'
                data = []
                for job_id, result in results_dict.items():
                    names, tmp_data = self._print_result(result)
                    data += tmp_data + ['-']
                logger.table(header=header, names=names, data=data, max_length=40)
                self.saver['history'] = list(results_dict.values())

        super().finalize()

