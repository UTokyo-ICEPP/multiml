""" Module to define agent abstraction
"""

from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):
    """ Base class of Agent
    """
    @abstractmethod
    def execute(self):
        """ Execute Agent
        """

    @abstractmethod
    def finalize(self):
        """ Finalize Agent
        """
