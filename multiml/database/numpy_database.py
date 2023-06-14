"""NumpyDatabase module."""

import torch
import numpy as np

from multiml.database.database import Database

from .utils import get_slice


class NumpyDatabase(Database):
    """Base class of Numpy database."""
    def __init__(self):
        self._db = {}
        self._metadata = {}

    def __repr__(self):
        return 'NumpyDatabase()'

    def initialize(self, data_id):
        if data_id not in self._db.keys():
            self._db[data_id] = {'train': {}, 'valid': {}, 'test': {}}
            self._metadata[data_id] = {'train': {}, 'valid': {}, 'test': {}}

    def add_data(self, data_id, var_name, idata, phase, mode=None):
        if var_name in self._db[data_id][phase].keys():
            tmp_data = self._db[data_id][phase][var_name]
            self._db[data_id][phase][var_name] = np.concatenate((tmp_data, idata), axis=0)
            self._metadata[data_id][phase][var_name]['type'] = type(
                self._db[data_id][phase][var_name][0]).__name__
            self._metadata[data_id][phase][var_name]['shape'] = self._db[data_id][phase][var_name][
                0].shape[1:]
            self._metadata[data_id][phase][var_name]['total_events'] += len(idata)

        else:
            self._db[data_id][phase][var_name] = idata
            self._metadata[data_id][phase][var_name] = {
                'backend': 'numpy',
                'type': type(idata[0]).__name__,
                'shape': idata.shape[1:],
                'total_events': len(idata)
            }

    def update_data(self, data_id, var_name, idata, phase, index, mode=None):
        self._db[data_id][phase][var_name][get_slice(index)] = idata

    def get_data(self, data_id, var_name, phase, index):
        if isinstance(index,
                      (list, np.ndarray, torch.Tensor)):  # allow fancy index, experimental feature
            return np.take(self._db[data_id][phase][var_name], index, axis=0)
        else:
            return self._db[data_id][phase][var_name][get_slice(index)]

    def delete_data(self, data_id, var_name, phase):
        if var_name in self._db[data_id][phase].keys():
            del self._db[data_id][phase][var_name]
            del self._metadata[data_id][phase][var_name]

    def create_empty(self, data_id, var_name, phase, shape, dtype):
        empty = np.empty(shape, dtype=dtype)
        self._db[data_id][phase][var_name] = empty
        self._metadata[data_id][phase][var_name] = {
            'backend': 'numpy',
            'type': type(empty[0]).__name__,
            'shape': empty.shape[1:],
            'total_events': len(empty)
        }

    def get_metadata(self, data_id, phase, mode=None):
        if data_id not in self._metadata.keys():
            return {}
        return self._metadata[data_id][phase]

    def get_data_ids(self):
        return list(self._metadata.keys())
