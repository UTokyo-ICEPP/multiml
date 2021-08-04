"""ModelConnectionTask module."""
from multiml import logger
from multiml.task.basic import MLBaseTask


class ModelConnectionTask(MLBaseTask):
    """Build a single task connecting with multiple tasks.

    ``ModelConnectionTask`` connects multiple ML tasks considering the input/output variables and
    dependencies of the tasks, then builds a single task. ML model of component tasks are trained
    diferentially, thus each ML model must be implemented by the same deep learning library,
    i.e. Keras or Pytorch. Each subtask must contain
      * ``input_var_names``, `output_var_names`` and `true_var_names`,
      * loss function,
    to compile subtask dependencies and data I/O formats. The following examples shows a workflow
    and its attributes, which are automatically compiled:

    Examples:
        >>> '''
        >>> (input0, input1, input2)
        >>>      |   |        |
        >>>   [subtask0]      |
        >>>       |           |
        >>>   (output0)       |
        >>>       |           |
        >>>   [subtask1]------+
        >>>       |
        >>>   (output1)
        >>> '''
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
    def __init__(self, subtasks, loss_weights=None, variable_mapping=None, **kwargs):
        """Constructor of ModelConnectionTask.

        Args:
            subtasks (list): list must contains ordered instance objects inherited
                from ``MLBaseTask``.
            loss_weights (list or dict or str): list of loss weights for each task. ``last_loss``
                and ``flat_loss`` are also allowed.
            variable_mapping (list(str, str)): Input variables are replaced following this list.
                Used for the case that the input variables change from pre-training to
                main-training (with model connecting).
            **kwargs: Arbitrary keyword arguments passed to ``MLBaseTask``.
        """
        super().__init__(**kwargs)

        if len(subtasks) <= 1:
            raise ValueError('Please provide at least two subtasks.')

        self._subtasks = subtasks
        self._loss_weights = loss_weights
        self._variable_mapping = variable_mapping

        self._cache_var_names = None
        self._input_var_index = None
        self._output_var_index = None

        if self._input_var_names is not None:
            logger.warn('input_var_names is given but it will be set automatically ')
            self._input_var_names = None

        if self._output_var_names is not None:
            logger.warn('output_var_names is given but it will be set automatically ')
            self._output_var_names = None

        if self._pred_var_names is not None:
            logger.warn('pred_var_names is given but it will be set automatically ')
            self._pred_var_names = None

    def compile(self):
        """Compile subtasks and this task."""
        for subtask in self._subtasks:
            subtask.compile()

        super().compile()

    def compile_loss(self):
        """Compile loss and loss_weights.

        Loss functions are retrieved from subtasks, thus each subtask must contain ``loss``.
        """
        self.ml.loss = []
        self.ml.loss_weights = []

        n_subtasks = len(self._subtasks)

        # Define loss weights for each task
        if (self._loss_weights is None) or (self._loss_weights == 'last_loss'):
            task_weights = [0.] * (n_subtasks - 1) + [1.0]

        elif self._loss_weights == 'flat_loss':
            task_weights = [1.0] * n_subtasks
            
        elif isinstance(self._loss_weights, list):
            task_weights = self._loss_weights
            
        elif isinstance(self._loss_weights, dict):
            task_weights = [self._loss_weights[s.task_id] for s in self._subtasks]
            
        else:
            raise ValueError(f'Unknown loss_weights: {self._loss_weights} is given')

        # Collect loss and weights for the loss from each subtask
        
        for subtask, task_weight in zip(self._subtasks, task_weights):
            if task_weight > 0.:
                lws = subtask.ml.loss_weights
                if lws is None:
                    lws = 1.0

                if isinstance(subtask.ml.loss, list):
                    self.ml.loss += subtask.ml.loss
                    self.ml.loss_weights += [lw * task_weight for lw in lws]
                else:
                    self.ml.loss.append(subtask.ml.loss)
                    self.ml.loss_weights.append(lws * task_weight)

            else:
                # Dummy loss which is not used in backpropagation
                if isinstance(subtask.ml.loss, list):
                    self.ml.loss += [None] * len(subtask.ml.loss)
                    self.ml.loss_weights += [0.0] * len(subtask.ml.loss)
                else:
                    self.ml.loss.append(None)
                    self.ml.loss_weights.append(0.0)

        if len(self.ml.loss) != len(self.ml.loss_weights):
            error_log = 'loss length is inconsistent: '
            error_log += f'tasl_weights {task_weights}, '
            error_log += f'loss {self.ml.loss}, '
            error_log += f'loss_weights {self.ml.loss_weights}'
            raise ValueError(error_log)

    def compile_var_names(self):
        """Compile subtask dependencies and I/O variables."""
        self.set_output_var_index()
        self.set_input_var_index()

        self.true_var_names = []
        for subtask in self._subtasks:
            true_var_names = subtask.true_var_names

            if isinstance(true_var_names, list):
                self.true_var_names += true_var_names
            else:
                self.true_var_names.append(true_var_names)

        super().compile_var_names()
        
    def set_output_var_index(self):
        """Set output_var_names and output_var_index."""
        self._cache_var_names = []
        self._output_var_index = []
        self._output_var_names = []
        self._pred_var_names = []

        for subtask in self._subtasks:

            output_index = []
            output_var_names = subtask.output_var_names
            pred_var_names = subtask.pred_var_names

            if output_var_names is None:
                continue

            # set output_var_names
            if isinstance(output_var_names, list):
                self._output_var_names += output_var_names
            else:
                self._output_var_names.append(output_var_names)

            # set pred_var_names
            if pred_var_names is None:
                if isinstance(output_var_names, list):
                    self._pred_var_names += output_var_names
                else:
                    self._pred_var_names.append(output_var_names)
            else:
                if isinstance(pred_var_names, list):
                    self._pred_var_names += pred_var_names
                else:
                    self._pred_var_names.append(pred_var_names)

            # set cache_var_names
            if isinstance(output_var_names, str):
                output_var_names = (output_var_names, )

            for output_var_name in output_var_names:
                if output_var_name in self._cache_var_names:
                    logger.error(f'output_var_name: {output_var_name} is duplicated.')
                else:
                    self._cache_var_names.append(output_var_name)
                    output_index.append(self._cache_var_names.index(output_var_name))

            self._output_var_index.append(output_index)

    def set_input_var_index(self):
        """Set input_var_names and input_var_index."""
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
                        input_index.append(self.input_var_names.index(input_var_name))

                    elif input_var_name in self._cache_var_names:
                        index = self._cache_var_names.index(input_var_name)
                        index = (index + 1) * -1
                        input_index.append(index)

                    else:
                        self.input_var_names.append(input_var_name)
                        input_index.append(self.input_var_names.index(input_var_name))

            if isinstance(input_var_names, tuple):
                self._input_var_index.append(tuple(input_index))
            else:
                self._input_var_index.append(input_index)
        
        
        
        
    def _apply_variable_mapping(self, input_vars):
        """Convert variable name by given mapping."""
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
