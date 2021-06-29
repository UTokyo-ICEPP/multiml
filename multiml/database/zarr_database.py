""" ZarrDatabase module
"""
import tempfile
import zarr

from multiml import logger
from multiml.database.database import Database

from .utils import get_slice


class ZarrDatabase(Database):
    """ Base class of Zarr database
    """
    def __init__(self,
                 output_dir=None,
                 chunk=1000,
                 compressor='default',
                 mode='a'):

        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            logger.debug(f'creating tmpdir({output_dir}) for zarr')

        self._output_dir = output_dir
        self._chunk = chunk
        self._mode = mode
        self._db = zarr.open(self._output_dir, mode=mode)

        if compressor is not 'default':
            zarr.storage.default_compressor = compressor

    def __repr__(self):
        result = f'ZarrDatabase(output_dir={self._output_dir}, '\
                            f'chunk={self._chunk}, '\
                            f'mode={self._mode})'
        return result

    def initialize(self, data_id):
        if data_id not in list(self._db.group_keys()):
            db_data_id = self._db.create_group(data_id)
            db_data_id.create_group('train')
            db_data_id.create_group('valid')
            db_data_id.create_group('test')

    def add_data(self, data_id, var_name, idata, phase, mode=None):
        db = self._db[data_id][phase]

        if var_name in db.array_keys():
            db[var_name].append(idata)
        else:
            # db[var_name] = idata
            shape = idata.shape
            chunks = (self._chunk, ) + (None, ) * (len(shape) - 1)
            db.array(var_name, idata, chunks=chunks)

    def update_data(self, data_id, var_name, idata, phase, index, mode=None):
        self._db[data_id][phase][var_name][get_slice(index)] = idata

    def get_data(self, data_id, var_name, phase, index):
        return self._db[data_id][phase][var_name][get_slice(index)]

    def delete_data(self, data_id, var_name, phase):
        if var_name in self._db[data_id][phase].array_keys():
            del self._db[data_id][phase][var_name]

    def get_metadata(self, data_id, phase, mode=None):
        results = {}
        if data_id not in list(self._db.group_keys()):
            return results

        db = self._db[data_id][phase]

        for var_name in db.array_keys():
            results[var_name] = {
                'backend': 'zarr',
                'type': db[var_name].dtype.str,
                'shape': db[var_name].shape[1:],
                'total_events': db[var_name].shape[0]
            }
        return results

    def get_data_ids(self):
        return list(self._db.group_keys())
