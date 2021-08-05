"""TaskScheduler module.

In the multiml framework, *task* describes each step of pipeline, and *subtask* describes component
of *task* with different type of approarchs, e.g. different type of ML models. The following scheme
shows the case that the multiml consists of two steps, and three subtasks are defined for each step:

>>> task0 (subtask0, subtask1, subtask2) -> task0 (subtask3, subtask4, subtask5)

TaskScheduler class manages dependencies of *task*, and stoers *subtask* class instances and thier
hyperparameters.

Attributes:
    tasktuple (namedtuple): namedtuple of *task*, which consists of ``task_id`` and ``subtasks``,
        ``task_id`` is unique identifier of *task*, and ``subtasks`` is a list of ``subtasktuple``
        described below.
    subtasktuple (namedtuple): namedtuple of *subtask*, which consists of
        ``task_id``, ``subtask_id``, ``env`` and ``hps``. ``subtask_id`` is unique identifier of
        *subtask*. ``env`` is class instance of *subtask*. ``hps`` is class instance of
        Hyperparameters.
"""
import itertools
from collections import namedtuple

import networkx as nx

from multiml import logger
from multiml.task.basic import BaseTask

tasktuple = namedtuple('tasktuple', ('task_id', 'subtasks'))
subtasktuple = namedtuple('subtasktuple', ('task_id', 'subtask_id', 'env', 'hps'))


class TaskScheduler:
    """Task management class for multiml execution.

    Manage tasks and subtasks. Ordering of tasks are controlled by DAG by providing parents and
    childs dependencies.

    Examples:
        >>> subtask = MyTask()
        >>> task_scheduler = TaskScheduler()
        >>> task_scheduler.add_task('task_id')
        >>> task_scheduler.add_subtask('task_id', 'subtask_id', subtask)
        >>> task_scheduler.get_sorted_task_ids()
    """
    def __init__(self, ordered_tasks=None):
        """Initialize the TaskScheduler and reset DAG.

        ``ordered_tasks`` option provides a shortcut of registering ordered task and subtask.
        Please see ``add_ordered_tasks()`` and ``add_ordered_subtasks()`` methods for details.
        If task dependencies are complex, please add task and subtask using ``add_task()`` and
        ``add_subtask()`` methods.

        Args:
            ordered_tasks (list): list of ordered task_ids, or list of ordered subtasks. If given
                value is list of str, ``add_ordered_tasks()`` is called to register task_ids. If
                given value is list of other types, ``add_ordered_subtasks()`` is called to
                register subtasks.

        Examples:
            >>> # ordered task_ids
            >>> task_scheduler = TaskScheduler(['task0', 'task1'])
            >>> task_scheduler.add_subtask('task0', 'subtask0', env)
        """
        self._dag = nx.DiGraph()
        self._dag_index = {}
        self._tasktuples = {}

        if isinstance(ordered_tasks, list):
            if isinstance(ordered_tasks[0], str):
                self.add_ordered_tasks(ordered_tasks)

            else:
                self.add_ordered_subtasks(ordered_tasks)

    def __repr__(self):
        return 'TaskScheduler()'

    def __len__(self):
        """Returns number of all grid combination.

        Returns:
            int: the number of all grid combination.
        """
        return len(list(itertools.product(*self.get_all_subtasks_with_hps())))

    def __getitem__(self, item):
        """Returns ``subtasktuples`` by index.

        Args:
            item (int): Index between 0 to len(task_scheduler).

        Examples:
            >>> task_scheduler[0]
        """
        return self.get_subtasks_pipeline(item)

    ##########################################################################
    # Public user APIs
    ##########################################################################
    def add_task(self, task_id, parents=None, children=None, subtasks=None, add_to_dag=True):
        """Register task and add the relation between tasks.

        If ``subtasks`` is provided as a list of dict, subtasks are also registered to given
        ``task_id``. To specify dependencies of tasks, ``parents`` or/and ``children`` need to be
        set, and ``add_to_dag`` must be True.

        Args:
            task_id (str): unique task identifier
            parents (list or str): list of parent task_ids, or str of parent task_id.
            children (list or str): list of child task_ids. or str of child task_id.
            subtasks (list): list of dict of subtasks with format of
                {'subtask_id': subtask_id, 'env': env, 'hps': hps}
            add_to_dag (bool): add task to DAG or not. To obtain task dependencies, e.g. ordered
                tasks, task need to be added to DAG.
        """
        if parents is None:
            parents = []

        if children is None:
            children = []

        if subtasks is None:
            subtasks = []

        self._add_task(task_id, add_to_dag)

        if isinstance(subtasks, dict):
            subtasks = [subtasks]

        for subtask in subtasks:
            self.add_subtask(task_id, subtask['subtask_id'], subtask['env'], subtask['hps'])

        if (not add_to_dag) and (parents != [] or children != []):
            raise ValueError("add_to_dag must be True to register ordering")

        parents = self._str_to_list(parents)
        children = self._str_to_list(children)

        for parent in parents:
            self._add_task(parent, add_to_dag)
            self._dag.add_edge(self._dag_index[parent], self._dag_index[task_id])

        for child in children:
            self._add_task(child, add_to_dag)
            self._dag.add_edge(self._dag_index[task_id], self._dag_index[child])

    def add_ordered_tasks(self, ordered_tasks):
        """Register ordered tasks.

        For example, if ``ordered_tasks`` is ['task0', 'task1'], 'task0' and 'task0' are registered
        with dependency of 'task0 (parent)' -> 'task1 (child)'.

        Args:
            ordered_tasks (list): list of task_ids
        """
        for index, task_id in enumerate(ordered_tasks):
            parents = None

            if (index - 1) >= 0:
                parents = [ordered_tasks[index - 1]]

            self.add_task(task_id, parents=parents)

    def add_ordered_subtasks(self, ordered_tasks):
        """Register ordered subtasks.

        ``ordered_tasks`` need to be a format of [task0, task1...], where e.g. task0 is a list of
        tuples [('subtask0', env0, hps0), ('subtask1', env0, hps0)...]. ``task_id`` is
        automatically set with 'step0', 'step1'... For the examples below, scheme of pipeline is:

        >>> step0 (subtask0, subtask1) -> step1 (subtask2, subtask2)

        Args:
            ordered_tasks (list): list of subtasks. Please see examples below.

        Examples:
           >>> # ordered tasks with subtask_id and hyperparameters
           >>> step0 = [('subtask0', env0, hps0), ('subtask1', env1, hps1)]
           >>> step1 = [('subtask2', env2, hps2), ('subtask3', env3, hps3)]
           >>> steps = [step0, step1]
           >>> task_scheduler.add_ordered_subtasks(steps)
           >>>
           >>> # ordered tasks with hyperparameters (subtask_id will be class name)
           >>> step0 = [(env0, hps0), (env1, hps1)]
           >>> step1 = [(env2, hps2), (env3, hps3)]
           >>> steps = [step0, step1]
           >>> task_scheduler.add_ordered_subtasks(steps)
           >>>
           >>> # ordered tasks without hyperparameters
           >>> steps = [env0, env1]
           >>> task_scheduler.add_ordered_subtasks(steps)
        """
        for index, subtasks in enumerate(ordered_tasks):
            task_id = f'step{index}'
            parents = None

            if (index - 1) >= 0:
                parents = f'step{index-1}'

            self.add_task(task_id, parents=parents)

            if not isinstance(subtasks, list):
                subtasks = [subtasks]

            for subtask in subtasks:
                if isinstance(subtask, tuple) and len(subtask) == 2:
                    env, hps = subtask
                    self.add_subtask(task_id, env=env, hps=hps)

                elif isinstance(subtask, tuple) and len(subtask) == 3:
                    subtask_id, env, hps = subtask
                    self.add_subtask(task_id, subtask_id=subtask_id, env=env, hps=hps)

                else:
                    self.add_subtask(task_id, env=subtask)

    def add_subtask(self, task_id, subtask_id=None, env=None, hps=None):
        """Register a subtask to given task.

        Need to register the corresponding task before calling this method.

        Args:
            task_id (str): unique task identifier.
            subtask_id (str): unique subtask identifier.
            env (BaseTask): user defined subtask class instance. subtask class need to inherited
                from BaseTask class.
            hps (dict or Hyperparameters): user defined Hyperparameters class instance or dict.
                If hps is dict, dict is converted to Hyperparameters class instance automatically.
        """
        from multiml import Hyperparameters

        if env is None:
            raise ValueError("subtask object is required.")

        if not isinstance(env, BaseTask):
            raise ValueError('env must inherit BaseTask class.')

        if subtask_id is None:
            subtask_id = env.name

        if task_id not in self._tasktuples:
            raise KeyError(f"{task_id} is not registered in task_scheduler.")

        for subtask in self._tasktuples[task_id].subtasks:
            if subtask.subtask_id == subtask_id:
                raise ValueError(f"{subtask_id} is already registered in {task_id}.")

        if hps is None:
            hps = Hyperparameters()

        if isinstance(hps, dict):
            hps = Hyperparameters(hps)

        env.task_id = task_id
        env.subtask_id = subtask_id

        self._tasktuples[task_id].subtasks.append(subtasktuple(task_id, subtask_id, env, hps))

    def get_subtasks(self, task_id):
        """Returns subtasks of tasktuple for given task_id.

        Args:
            task_id (str): unique task identifier.

        Returns:
            list: list of subtasktuples for given ``task_id``.
        """
        return self._tasktuples[task_id].subtasks

    def get_subtask_ids(self, task_id):
        """Returns subtask_ids by task_id.

        Args:
            task_id (str): unique task identifier.

        Returns:
            list: list of subtask_ids for given ``task_id``.
        """
        subtasks = self.get_subtasks(task_id)
        return [subtask.subtask_id for subtask in subtasks]

    def get_subtask(self, task_id, subtask_id):
        """Returns subtasktuple for given task_id and subtask_id.

        Args:
            task_id (str): unique task identifier.
            subtask_id (str): unique subtask identifier.

        Returns:
            subtasktuple: ``subtasktuple`` for given ``task_id`` and ``subtask_id``.
        """
        subtasks = self.get_subtasks(task_id)
        for subtask in subtasks:
            if subtask.subtask_id == subtask_id:
                return subtask

        raise ValueError(f'{subtask_id} is not found in {task_id}.')

    def get_parents_task_ids(self, task_id):
        """Returns parent task_ids for given task_id.

        Args:
            task_id (str): unique task identifier.

        Returns:
            list: list of parent ``task_ids`` for given ``task_ids``.
        """
        if not self._in_dag(task_id):
            return []

        parents = list(self._dag.predecessors(self._dag_index[task_id]))
        parents = [self._dag.nodes[parent]['name'] for parent in parents]
        return parents

    def get_children_task_ids(self, task_id):
        """Returns child task_ids for given task_id.

        Args:
            task_id (str): unique task identifier.

        Returns:
            list: list of child ``task_ids`` for given ``task_id``.
        """
        if not self._in_dag(task_id):
            return []

        children = list(self._dag.successors(self._dag_index[task_id]))
        children = [self._dag.nodes[child]['name'] for child in children]
        return children

    def get_sorted_task_ids(self):
        """Returns topologically sorted task_ids.

        Returns:
            list: a list of topologically sorted ``task_ids``.
        """
        tasks = list(nx.topological_sort(self._dag))
        tasks = [self._dag.nodes[task]['name'] for task in tasks]
        return tasks

    def get_subtasks_with_hps(self, task_id, is_grid_hps=True):
        """Returns all combination of subtask_ids and hps for given task_id.

        Args:
            task_id (str): unique task identifier.

        Returns:
            list:
                list of modified subtasktuples. Modified subtasktuple format is .task_id: task_id,
                .subtask_id: subtask_id, .env: subtask class instance, .hps: *dictionary of hps*.
        """
        results = []
        for subtask in self.get_subtasks(task_id):
            if is_grid_hps:
                grid_hps = subtask.hps.get_grid_hps()

                if not grid_hps:
                    results.append(
                        subtasktuple(subtask.task_id, subtask.subtask_id, subtask.env, {}))
                    continue

                for hps in grid_hps:
                    results.append(
                        subtasktuple(subtask.task_id, subtask.subtask_id, subtask.env, hps))
            else:
                hps = subtask.hps.get_all_hps()
                results.append(subtasktuple(subtask.task_id, subtask.subtask_id, subtask.env, hps))

        return results

    def get_all_subtasks_with_hps(self):
        """Returns all combination of subtask_ids and hps for all task_ids.

        Returns:
            list: list of ``get_subtasks_with_hps()`` for each ``task_id``.
        """
        all_combs = []
        for task_id in self.get_sorted_task_ids():
            subtasktuples = self.get_subtasks_with_hps(task_id)
            all_combs.append(subtasktuples)
        return all_combs

    def get_subtasks_pipeline(self, index):
        """Returns modified subtasktuples for given index.

        Returns:
            list: list of modified subtasktuples.
        """
        all_combs = self.get_all_subtasks_with_hps()

        # TODO: combination explosion
        return list(itertools.product(*all_combs))[index]

    def show_info(self):
        """Show information of registered tasks and subtasks."""
        header = f'TaskScheduler: total combination {len(self)}'
        names = ['task_id', 'subtask_id', 'hps', 'DAG', 'parents', 'children']
        data = []

        for task_id in self._tasktuples:
            in_dag = f'{self._in_dag(task_id)}'
            parents = f'{self.get_parents_task_ids(task_id)}'
            children = f'{self.get_children_task_ids(task_id)}'

            for index, subtask in enumerate(self._tasktuples[task_id].subtasks):
                subtask_id = subtask.subtask_id
                hp_name = f'{subtask.hps.get_hp_names()}'

                if index == 0:
                    data.append([task_id, subtask_id, hp_name, in_dag, parents, children])
                else:
                    data.append(['', subtask_id, hp_name, in_dag, parents, children])
            data.append('-')

        logger.table(header=header, names=names, data=data)

    ##########################################################################
    # Internal methods
    ##########################################################################
    def _add_task(self, task_id, add_to_dag):
        """(private) add a task to the task scheduler."""
        if task_id not in self._tasktuples:
            if add_to_dag:
                self._dag_index[task_id] = len(self._dag)
                self._dag.add_node(len(self._dag), name=task_id)
            self._tasktuples[task_id] = tasktuple(task_id, [])

    @staticmethod
    def _str_to_list(string):
        """(private) convert string to a list to allow common process."""
        if isinstance(string, str):
            if string == '':
                return []
            return [string]
        return string

    def _in_dag(self, task_id):
        """Returns if task_id is registered in DAG or not."""
        return bool(task_id in self._dag_index)
