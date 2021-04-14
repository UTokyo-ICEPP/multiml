""" Module to define Database abstraction
"""

from abc import ABCMeta, abstractmethod


class Database(metaclass=ABCMeta):
    """ Base class of Database.
    """
    @abstractmethod
    def add_data(self, data_id, var_name, data, phase):
        """ Add data to database for given data_id, var_name and phase. If
            var_name already exists, data need to be appended
        """

    @abstractmethod
    def update_data(self, data_id, var_name, data, phase, index):
        """ Update (replace) data in database for given data_id, var_name,
            phase and index.
        """

    @abstractmethod
    def get_data(self, data_id, var_name, phase, index):
        """ Get data for given data_id, var_name, phase and index from database
        """

    @abstractmethod
    def delete_data(self, data_id, var_name, phase):
        """ Delete data for given data_id, var_name and phase from database
        """

    @abstractmethod
    def get_metadata(self, data_id, phase):
        """ Returns a dictionary of metadata for a given data_id and phase.
            The dict contains: {'var_name': {'type': type of variable,
                                             'total_events': number of samples}
        """
