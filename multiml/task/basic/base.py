""" BaseTask module.
"""

from multiml import logger
from multiml.task import Task


class BaseTask(Task):
    """ Base task class for the default functions. 

    All subtasks defined by users, need to inherit this ``BaseTask``.
    In user defined class, super.__init__() must be called in __init__()
    method. A task class is assumed to call its methods by following sequence:
    ``set_hps()`` -> ``execute()`` -> ``finalize()``. If task class instance
    is registered to ``TaskScheduler`` as *subtask*, ``self._task_id`` and
    ``self._subtask_id`` are automatically set by ``TaskScheduler``.

    Examples:
        >>> task = BaseTask()
        >>> task.set_hps({'hp_layer': 5, 'hp_epoch': 256})
        >>> task.execute()
        >>> task.finalize()
    """
    def __init__(self,
                 saver=None,
                 input_saver_key='tmpkey',
                 output_saver_key='tmpkey',
                 storegate=None,
                 name=None):
        """ Initialize base task.

        Args:
            saver (Saver): ``Saver`` class instance to record metadata data.
            input_saver_key (int): unique saver key to retrieve metadata.
            output_saver_key (int): unique saver key to save metadata.
            storegate (Storegate): ``Storegate`` class instance to manage data.
            name (str): task's name. If None, ``classname`` is used alternatively.
        """
        self._storegate = storegate
        self._saver = saver
        self._input_saver_key = input_saver_key
        self._output_saver_key = output_saver_key
        self._task_type = 'base'
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name
        self._task_id = None
        self._subtask_id = None
        self._unique_id = None
        self._job_id = None
        self._pool_id = None

    def __repr__(self):
        result = f'{self.__class__.__name__}(task_type={self._task_type}, '\
                                           f'task_id={self._task_id}, '\
                                           f'subtask_id={self._subtask_id}, '\
                                           f'job_id={self._job_id})'
        return result

    @logger.logging
    def execute(self):
        """ Execute base task. Users implement their algorithms.
        """

    @logger.logging
    def finalize(self):
        """ Finalize base task. Users implement their algorithms.
        """

    def set_hps(self, params):
        """ Set hyperparameters to this task. 

        Class attributes (self._XXX) are automatically set based on keys and
        values of given dict. E.g. dict of {'key0': 0, 'key1': 1} is given,
        self._key0 = 0 and self._key1 = 1 are created.
        """
        for key, value in params.items():
            if not hasattr(self, '_' + key):
                raise AttributeError(f'{key} is not defined.')

            setattr(self, '_' + key, value)

        if self.saver is not None:
            self.saver[self.output_saver_key] = params

    @property
    def name(self):
        """ Return name of task.
        """
        return self._name

    @name.setter
    def name(self, name):
        """ Set name of task.
        """
        self._name = name

    @property
    def job_id(self):
        """ Return job_id of task.
        """
        return self._job_id

    @job_id.setter
    def job_id(self, job_id):
        """ Set job_id of task.
        """
        self._job_id = job_id

    @property
    def task_id(self):
        """ Return task_id of task.
        """
        return self._task_id

    @task_id.setter
    def task_id(self, task_id):
        """ Set task_id of task.
        """
        self._task_id = task_id

    @property
    def subtask_id(self):
        """ Return subtask_id of task.
        """
        return self._subtask_id

    @subtask_id.setter
    def subtask_id(self, subtask_id):
        """ Set subtask_id of task.
        """
        self._subtask_id = subtask_id

    @property
    def pool_id(self):
        """ Return pool_id of task.
        """
        return self._pool_id

    @pool_id.setter
    def pool_id(self, pool_id):
        """ Set pool_id of task.
        """
        self._pool_id = pool_id

    @property
    def storegate(self):
        """ Return storegate of task.
        """
        return self._storegate

    @storegate.setter
    def storegate(self, storegate):
        """ Set storegate.
        """
        self._storegate = storegate

    @property
    def saver(self):
        """ Return saver of task.
        """
        return self._saver

    @saver.setter
    def saver(self, saver):
        """ Set saver.
        """
        self._saver = saver

    @property
    def input_saver_key(self):
        """ Return input_saver_key.
        """
        return self._input_saver_key

    @input_saver_key.setter
    def input_saver_key(self, input_saver_key):
        """ Set input_saver_key.
        """
        self._input_saver_key = input_saver_key

    @property
    def output_saver_key(self):
        """ Return output_saver_key.
        """
        return self._output_saver_key

    @output_saver_key.setter
    def output_saver_key(self, output_saver_key):
        """ Set output_saver_key.
        """
        self._output_saver_key = output_saver_key

    def get_unique_id(self):
        """ Returns unique identifier of task.
        """
        if self._unique_id is None:
            unique_id = self._name

            if self._task_id is not None:
                unique_id += f'_{self._task_id}'

            if self._subtask_id is not None:
                unique_id += f'_{self._subtask_id}'

            if self._job_id is not None:
                unique_id += f'_{self._job_id}'

            return unique_id

        return self._unique_id
