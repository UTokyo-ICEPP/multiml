"""SequentialAgent module."""
import copy
import time
from multiml import logger
from multiml.agent.basic import BaseAgent

import numpy as np

from tqdm import tqdm
import multiprocessing as mp


class SequentialAgent(BaseAgent):
    """Agent execute sequential tasks.

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
                 num_trials=None,
                 dump_all_results=False,
                 num_workers=None,
                 context='spawn',
                 disable_tqdm=True,
                 **kwargs):
        """Initialize sequential agent.

        Args:
            differentiable (str): ``keras`` or ``pytorch``. If differentiable is given,
                ``ConnectionTask()`` is created based on sequential tasks. If differentiable is
                None (default), sequential tasks are executed step by step.
            diff_pretrain (bool): If True, each subtask is trained before creating
                `ConnectionTask()``.
            diff_task_args (dict): arbitrary args passed to ``ConnectionTask()``.
            num_trials (ine): number of trials. Average value of trials is used as final metric.
            dump_all_results (bool): dump all results or not.
            num_workers (int or list): number of workers for multiprocessing or lsit of GPU ids.
                If ``num_workers`` is given, multiprocessing is enabled.
            context (str): fork (default) or spawn.
            disable_tqdm (bool): enable tqdm bar.
        """
        if diff_task_args is None:
            diff_task_args = {}

        super().__init__(**kwargs)
        self._result = None
        self._history = []
        self._differentiable = differentiable
        self._diff_pretrain = diff_pretrain
        self._diff_task_args = diff_task_args
        self._num_trials = num_trials
        self._dump_all_results = dump_all_results
        self._num_workers = num_workers
        self._disable_tqdm = disable_tqdm

        self._context = context
        self._multiprocessing = False

        if self._num_workers is not None:
            self._multiprocessing = True

            if isinstance(self._num_workers, int):
                self._num_workers = list(range(self._num_workers))

    @property
    def result(self):
        """Return result of execution."""
        return self._result

    @result.setter
    def result(self, result):
        """Set result of execution."""
        self._result = result

    @property
    def history(self):
        """Return history of execution."""
        return self._history

    @history.setter
    def history(self, history):
        """Set history of execution."""
        self._history = history

    @logger.logging
    def execute(self):
        """Execute sequential agent."""
        if len(self.task_scheduler) != 1:
            raise ValueError('Multiple sutasks or hyperparameters are defined.')

        subtasktuples = self.task_scheduler[0]

        if self._num_trials is None:
            self._result = self.execute_subtasktuples(subtasktuples, 0)

        else:
            if self._multiprocessing:
                if self._storegate.backend not in ('numpy', 'hybrid'):
                    raise NotImplementedError(
                        'multiprocessing is supported for only numpy and hybrid backend')

                ctx = mp.get_context(self._context)
                queue = ctx.Queue()
                args = []

                for trial_id in range(self._num_trials):
                    args.append([subtasktuples, 0, trial_id])

                self.execute_pool_jobs(ctx, queue, args)

            else:
                for trial_id in range(self._num_trials):
                    result = self.execute_subtasktuples(subtasktuples, 0, trial_id)
                    self._history.append(result)

    @logger.logging
    def finalize(self):
        """Finalize sequential agent."""
        if (self._result is None) and (not self._history):
            logger.warn(f'No result at finalize of {self.__class__.__name__}')

        elif self._result:
            pass

        elif self._history:
            metric_values = []
            for history in self._history:
                metric_values.append(history['metric_value'])
            metric_value = sum(metric_values) / self._num_trials
            self._result = self._history[0]

            self._result['metric_value'] = metric_value
            self._result['metric_values'] = metric_values

        header = f'Result of {self.__class__.__name__}'
        names, data = self._print_result(self._result)
        logger.table(header=header, names=names, data=data, max_length=40)
        self.saver['result'] = self._result

    def execute_jobs(self, ctx, queue, args):
        """(expert method) Execute multiprocessing jobs."""
        jobs = []

        for pargs in args:
            process = ctx.Process(target=self.execute_wrapper, args=(queue, *pargs), daemon=False)
            jobs.append(process)
            process.start()

        for job in jobs:
            job.join()

        while not queue.empty():
            self._history.append(queue.get())

    def execute_pool_jobs(self, ctx, queue, args):
        """(expert method) Execute multiprocessing pool jobs."""
        jobs = copy.deepcopy(args)

        pool = [0] * len(self._num_workers)
        num_jobs = len(jobs)
        all_done = False

        pbar_args = dict(ncols=80, total=num_jobs, disable=self._disable_tqdm)
        with tqdm(**pbar_args) as pbar:
            while not all_done:
                time.sleep(1)

                if len(jobs) == 0:
                    done = True
                    for ii, process in enumerate(pool):
                        if (process != 0) and (process.is_alive()):
                            done = False
                    all_done = done

                else:
                    for ii, process in enumerate(pool):
                        if len(jobs) == 0:
                            continue

                        if (process == 0) or (not process.is_alive()):
                            time.sleep(0.05)
                            job_arg = jobs.pop(0)
                            pool[ii] = ctx.Process(target=self.execute_wrapper,
                                                   args=(queue, *job_arg, ii),
                                                   daemon=False)
                            pool[ii].start()
                            pbar.update(1)

                            if self._disable_tqdm:
                                logger.info(f'launch process ({num_jobs - len(jobs)}/{num_jobs})')

                while not queue.empty():
                    self._history.append(queue.get())

        while not queue.empty():
            self._history.append(queue.get())

    def execute_wrapper(self, queue, subtasktuples, job_id, trial_id, cuda_id):
        """(expert method) Wrapper method to execute multiprocessing pipeline."""
        for subtasktuple in subtasktuples:
            subtasktuple.env.pool_id = (cuda_id, self._num_workers, len(self.task_scheduler))

        result = self.execute_subtasktuples(subtasktuples, job_id, trial_id)

        if self._dump_all_results:
            history_id = f'history_{job_id}'
            if trial_id is not None:
                history_id += f'_{trial_id}'

            self.saver[history_id] = result

        queue.put(result)

    def execute_subtasktuples(self, subtasktuples, job_id, trial_id=None):
        """Execute given subtasktuples."""
        if self._differentiable is None:
            fn_execute = self.execute_pipeline
        else:
            fn_execute = self.execute_differentiable

        return fn_execute(subtasktuples, job_id, trial_id)

    def execute_pipeline(self, subtasktuples, job_id, trial_id=None):
        """Execute pipeline."""
        result = {
            'task_ids': [],
            'subtask_ids': [],
            'job_id': None,
            'trial_id': None,
            'subtask_hps': [],
            'metric_value': None
        }

        result['job_id'] = job_id
        result['trial_id'] = trial_id

        for subtasktuple in subtasktuples:
            task_id = subtasktuple.task_id
            subtask_id = subtasktuple.subtask_id
            subtask_env = subtasktuple.env
            subtask_hps = copy.deepcopy(subtasktuple.hps)

            subtask_env.subtask_id = subtask_id
            subtask_env.saver = self._saver
            subtask_env.storegate = self._storegate
            subtask_env.job_id = job_id
            subtask_env.trial_id = trial_id
            subtask_env.set_hps(subtask_hps)
            self._execute_subtask(subtasktuple)

            result['task_ids'].append(task_id)
            result['subtask_ids'].append(subtask_id)
            result['subtask_hps'].append(subtask_hps)

        self._metric.storegate = self._storegate
        result['metric_value'] = self._metric.calculate()

        return result

    def execute_differentiable(self, subtasktuples, job_id, trial_id=None):
        """Execute connection model."""
        result = {
            'task_ids': [],
            'subtask_ids': [],
            'job_id': None,
            'trial_id': None,
            'subtask_hps': [],
            'metric_value': None
        }

        result['job_id'] = job_id
        result['trial_id'] = trial_id

        if self._diff_pretrain:
            for subtasktuple in subtasktuples:
                task_id = subtasktuple.task_id
                subtask_id = subtasktuple.subtask_id
                subtask_env = subtasktuple.env
                subtask_hps = copy.deepcopy(subtasktuple.hps)

                subtask_env.subtask_id = subtask_id
                subtask_env.saver = self._saver
                subtask_env.storegate = self._storegate
                subtask_env.job_id = job_id
                subtask_env.trial_id = trial_id
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
            raise ValueError(f'differentiable: {self._differentiable} is not supported.')

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
        """Execute subtask."""

        subtask.env.storegate = self._storegate
        subtask.env.saver = self._saver
        subtask.env.execute()
        subtask.env.finalize()

    def _print_result(self, result):
        """Returns print result."""
        metric_name = self._metric.name
        names = ['task_id', 'subtask_id', 'hps', f'metric({metric_name})']
        data = []

        for task_id, subtask_id, subtask_hp in zip(result['task_ids'], result['subtask_ids'],
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
                metric_value = f'{result["metric_value"]}'
                if 'metric_values' in result:
                    metric_std = np.array(result['metric_values'])
                    metric_std = metric_std.std()
                    metric_value += f' +- {metric_std}'

                metric_data.append(idata + [f'{metric_value}'])
            else:
                metric_data.append(idata + [''])

        return names, metric_data
