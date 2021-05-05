from abc import ABCMeta, abstractmethod


class ConnectionModel(metaclass=ABCMeta):
    def __init__(self,
                 models=None,
                 input_var_index=None,
                 output_var_index=None,
                 *args,
                 **kwargs):
        """ Connecting ML models differentially.

        This ConnectionModel is usually build inside ``ModelConnectionTask`` to
        set proper indexes automatically. Inputs data for ``call()`` or
        ``forward()`` are assumed to be list format.

        Args:
            models (list): list of compiled models.
            input_var_index (list): list of index to ``input_var_names``.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._models = models
        self._input_var_index = input_var_index
        self._output_var_index = output_var_index

        self._num_outputs = 0

        for outputs in self._output_var_index:
            for output in outputs:
                if output >= self._num_outputs:
                    self._num_outputs = output + 1
