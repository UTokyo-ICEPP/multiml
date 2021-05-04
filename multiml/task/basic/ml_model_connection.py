""" ModelConnectionTask module.
"""
import numpy as np

from multiml import logger
from multiml.task.basic import MLBaseTask


class ModelConnectionTask(MLBaseTask):
    """ Build a single task connecting with multiple tasks.

    ``ModelConnectionTask`` connects multiple ML tasks considering the
    input/output variables and dependencies of the tasks, then builds a single
    task. ML model of component tasks are trained diferentially, thus
    each ML model must be implemented by the same deep
    learning library, i.e. Keras or Pytorch. Each subtask must contain
      * ``input_var_names``, `output_var_names`` and `true_var_names`,
      * loss function,
    to compile subtask dependencies and data I/O formats. The following
    examples shows a workflow and its attributes, which are automatically
    compiled:

    Examples:
        >>> (input0, input1, input2)
        >>>      |   |        |
        >>>   [subtask0]      |
        >>>       |           |
        >>>   (output0)       |
        >>>       |           |
        >>>   [subtask1]------+
        >>>       |
        >>>   (output1)
        >>> 
        >>> input_var_names = ['input0', 'input1', 'input2']
        >>> output_var_names = ['output0', 'output1']
        >>> input_var_index = [[0, 1], [-1, 2]]
        >>> output_var_index = [[0], [1]]

    Examples:
        >>> task = ModelConnectionTask(subtasks=[your_subtask0, your_subtask2],
        >>>                            optimizer='SGD')
        >>> task.execute()
    """
    def __init__(self,
                 subtasks,
                 use_multi_loss=False,
                 variable_mapping=None,
                 **kwargs):
        """ Constructor of ModelConnectionTask.

        Args:
            subtasks (list): list must contains ordered instance objects
                inherited from ``MLBaseTask``.
            use_multi_loss (bool): If False, intermediate losses are not
                considered in training steps.
            variable_mapping (list(str, str)): Input variables are replaced
                following this list. Used for the case that the input variables
                change from pre-training to main-training (with model connecting).
            **kwargs: Arbitrary keyword arguments passed to ``MLBaseTask``.
        """
        super().__init__(**kwargs)

        self._subtasks = subtasks
        self._use_multi_loss = use_multi_loss
        self._variable_mapping = variable_mapping

        self._cache_var_names = None
        self._input_var_index = None
        self._output_var_index = None

        if self._input_var_names is not None:
            logger.warn(
                'input_var_names is given but it will be set automatically ')
            self._input_var_names = None

        if self._output_var_names is not None:
            logger.warn(
                'output_var_names is given but it will be set automatically ')
            self._output_var_names = None

    def compile(self):
        """ Compile subtasks and this task.
        """
        for subtask in self._subtasks:
            subtask.compile()

        self.compile_index()
        super().compile()

    def compile_loss(self):
        """ Compile loss.

        Loss functions are retrieved from subtasks, thus each subtask must
        contain ``loss``. ``loss_weights`` are set according to options:
          * If ``use_multi_loss`` is False, only loss of the last subtask is
            considered with weight = 1.0,
          * If ``use_multi_loss`` is True and ``loss_weights`` is given
            explicitly, given ``loss_weights`` is used.
          * If ``use_multi_loss`` is True and ``loss_weights`` is None,
            ``loss_weights`` is retrieved from each subtask.
        """
        if self._use_multi_loss:
            self.ml.loss = []

            for index, subtask in enumerate(self._subtasks):
                self.ml.loss.append(subtask.ml.loss)

            if self._loss_weights is None:
                self.ml.loss_weights = []

                for index, subtask in enumerate(self._subtasks):
                    loss_weights = subtask.ml.loss_weights

                    if loss_weights is None:
                        self.ml.loss_weights.append(1.0)
                    elif isinstance(loss_weights, (int, float, list)):
                        self.ml.loss_weights.append(loss_weights)

                    else:
                        raise ValueError(
                            f'loss_weights {loss_weights} is not supported')

            elif isinstance(self._loss_weights, list):
                self.ml.loss_weights = self._loss_weights

            elif isinstance(self._loss_weights, dict):
                self.ml.loss_weights = []

                for index, subtask in enumerate(self._subtasks):
                    loss_weights = self._loss_weights[subtask.task_id]
                    self.ml.loss_weights.append(loss_weights)

        else:
            self.ml.loss = []
            self.ml.loss_weights = []

            for subtask in self._subtasks:
                if isinstance(subtask.ml.loss, list):
                    self.ml.loss.append([None] * len(subtask.ml.loss))
                    self.ml.loss_weights.append([0.0] * len(subtask.ml.loss))
                else:
                    self.ml.loss.append(None)
                    self.ml.loss_weights.append(0.0)

            self.ml.loss[-1] = self._subtasks[-1].ml.loss
            self.ml.loss_weights[-1] = self._subtasks[-1].ml.loss_weights

        # TODO: special treatment for ensemble tasks
        new_loss = []
        new_loss_weights = []

        for index, loss in enumerate(self.ml.loss):
            if isinstance(loss, list):
                new_loss += loss
                new_loss_weights += self.ml.loss_weights[index]

            else:
                new_loss.append(loss)
                new_loss_weights.append(self.ml.loss_weights[index])

        if len(new_loss) == len(new_loss_weights):
            self.ml.loss = new_loss
            self.ml.loss_weights = new_loss_weights
        else:
            raise ValueError(
                'Length of loss and loss_weights is not consistent.')

    def compile_index(self):
        """ Compile subtask dependencies and I/O variables.
        """
        self.set_ordered_subtasks()
        self.set_output_var_index()
        self.set_input_var_index()

    def get_input_true_data(self, phase=None):
        """ Returns input and true data retrieved from storegate.

        The list of input data contains data for each *variable*, thus the
        length is equal to the length of ``input_var_names``. The list of
        true data contains data for each *subtask*, thus the length is equal to
        the length of ``subtasks``.

        Returns:
            (list, list): list of input data, and list of true data.

        """
        input_ret = []
        true_ret = []

        for var_name in self.input_var_names:
            input_data = self.storegate.get_data(var_names=var_name,
                                                 phase=phase)
            input_ret.append(input_data)

        for subtask in self._subtasks:
            true_var_names = subtask.true_var_names
            true_data = self.storegate.get_data(var_names=true_var_names,
                                                phase=phase)

            # TODO: special treatment for ensemble tasks
            if isinstance(subtask.ml.loss, list):
                true_ret += [true_data] * len(subtask.ml.loss)
            else:
                true_ret.append(true_data)

        return input_ret, true_ret

    def set_output_var_index(self):
        """ Set output_var_names and output_var_index.
        """
        self._cache_var_names = []
        self._output_var_index = []
        self._output_var_names = []

        for subtask in self._subtasks:

            output_index = []
            output_var_names = subtask.output_var_names

            if output_var_names is None:
                continue

            if isinstance(output_var_names, list):
                self._output_var_names += output_var_names
            else:
                self._output_var_names.append(output_var_names)

            if isinstance(output_var_names, str):
                output_var_names = (output_var_names, )

            for output_var_name in output_var_names:
                if output_var_name in self._cache_var_names:
                    logger.error(
                        f'output_var_name: {output_var_name} is duplicated.')
                else:
                    self._cache_var_names.append(output_var_name)
                    output_index.append(
                        self._cache_var_names.index(output_var_name))

            self._output_var_index.append(output_index)

    def set_input_var_index(self):
        """ Set input_var_names and input_var_index.
        """
        self._input_var_index = []
        self._input_var_names = []

        for subtask in self._subtasks:
            input_index = []
            input_var_names = subtask.input_var_names

            if input_var_names is None:
                continue

            if isinstance(input_var_names, str):
                input_var_names = (input_var_names, )

            input_var_names = self._apply_variable_mapping(input_var_names)

            # try tuple matching
            if input_var_names in self._cache_var_names:
                index = self._cache_var_names.index(input_var_names)
                index = (index + 1) * -1
                input_index.append(index)

            else:
                for input_var_name in input_var_names:
                    if input_var_name in self.input_var_names:
                        input_index.append(
                            self.input_var_names.index(input_var_name))

                    elif input_var_name in self._cache_var_names:
                        index = self._cache_var_names.index(input_var_name)
                        index = (index + 1) * -1
                        input_index.append(index)

                    else:
                        self.input_var_names.append(input_var_name)
                        input_index.append(
                            self.input_var_names.index(input_var_name))

            if isinstance(input_var_names, tuple):
                self._input_var_index.append(tuple(input_index))
            else:
                self._input_var_index.append(input_index)

    def set_ordered_subtasks(self):
        """ Order subtasks based on input_var_names and output_var_names.
        """
        import networkx as nx

        def _flatten(data):
            from collections.abc import Iterable
            for v in data:
                if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
                    yield from _flatten(v)
                else:
                    yield v

        # dag built with subtasks and input/output names
        dag_sub = nx.DiGraph()
        for i_subtask, subtask in enumerate(self._subtasks):
            # Add subtask name
            dag_sub.add_node(i_subtask)

            # Add variable name
            input_var_names = subtask.input_var_names
            if isinstance(input_var_names, str):
                input_var_names = [input_var_names]

            input_var_names = self._apply_variable_mapping(input_var_names)

            for var in _flatten(input_var_names):
                if not dag_sub.has_node('var_' + var):
                    dag_sub.add_node('var_' + var)
                dag_sub.add_edge('var_' + var, i_subtask)

            output_var_names = subtask.output_var_names
            if isinstance(output_var_names, str):
                output_var_names = [output_var_names]

            for var in _flatten(output_var_names):
                if not dag_sub.has_node('var_' + var):
                    dag_sub.add_node('var_' + var)
                dag_sub.add_edge(i_subtask, 'var_' + var)

        # dag built with subtasks
        dag = nx.DiGraph()
        for i_subtask, subtask in enumerate(self._subtasks):
            dag.add_node(i_subtask)

        for node in nx.topological_sort(dag_sub):
            if isinstance(node, int):
                predecessors = set([
                    u for v in dag_sub.predecessors(node)
                    for u in dag_sub.predecessors(v)
                ])
                for v in predecessors:
                    dag.add_edge(v, node)

                successors = set([
                    u for v in dag_sub.successors(node)
                    for u in dag_sub.successors(v)
                ])
                for v in successors:
                    dag.add_edge(node, v)

        new_subtasks = []
        for i_subtask in nx.topological_sort(dag):
            new_subtasks.append(self._subtasks[i_subtask])

        self._subtasks = new_subtasks

    def _apply_variable_mapping(self, input_vars):
        """ Convert variable name by given mapping.
        """
        if self._variable_mapping is None:
            return input_vars

        is_tuple = False
        if isinstance(input_vars, tuple):
            is_tuple = True

        ret = []
        if isinstance(input_vars[0], list):
            for v in input_vars:
                ret.append(self._apply_variable_mapping(v))
            return ret

        for input_var in input_vars:
            for (v_from, v_to) in self._variable_mapping:
                if v_from == input_var:
                    input_var = v_to
                    break
            ret.append(input_var)

        if is_tuple:
            ret = tuple(ret)

        return ret
