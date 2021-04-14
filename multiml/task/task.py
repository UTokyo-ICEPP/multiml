""" Module to define Task abstraction.
"""

from abc import ABCMeta, abstractmethod


class Task(metaclass=ABCMeta):
    """ Tasks need be inherited this base class. Multi-ai agents assume that
        initialize, execute, finalize, set_hps, methods are available.
    """
    @abstractmethod
    def execute(self):
        """ Execute the task.
        """

    @abstractmethod
    def finalize(self):
        """ Finalize the task.
        """

    @abstractmethod
    def set_hps(self, params):
        """ Set hyperparameters of this task.
        """
