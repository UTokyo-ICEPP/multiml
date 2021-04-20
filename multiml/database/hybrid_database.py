""" HybridDatabase module
"""
import tempfile

from multiml import logger
from multiml.database.database import Database
from multiml.database.zarr_database import ZarrDatabase
from multiml.database.numpy_database import NumpyDatabase


class HybridDatabase(Database):
    """ Base class of Hybrid database
    """
    def __init__(self, output_dir=None, chunk=1000, mode='a'):

        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            logger.debug(f'creating tmpdir({output_dir}) for zarr')

        self._output_dir = output_dir
        self._chunk = chunk
        self._mode = 'zarr'
        self._db = {}

        self._db['numpy'] = NumpyDatabase()
        self._db['zarr'] = ZarrDatabase(output_dir=output_dir,
                                        chunk=chunk,
                                        mode=mode)

    def __repr__(self):
        result = f'HybridDatabase(output_dir={self._output_dir}, '\
                            f'chunk={self._chunk}, '\
                            f'mode={self._mode})'
        return result

    def initialize(self, data_id):
        self._db['numpy'].initialize(data_id)
        self._db['zarr'].initialize(data_id)

    def add_data(self, data_id, var_name, idata, phase, mode=None):
        if mode is None:
            mode = self._mode

        self._check_backend(data_id, var_name, phase, mode)
        self._db[mode].add_data(data_id, var_name, idata, phase)

    def update_data(self, data_id, var_name, idata, phase, index):
        backend = self._get_backend(data_id, var_name, phase)
        self._db[backend].update_data(data_id, var_name, idata, phase, index)

    def get_data(self, data_id, var_name, phase, index):
        backend = self._get_backend(data_id, var_name, phase)
        return self._db[backend].get_data(data_id, var_name, phase, index)

    def delete_data(self, data_id, var_name, phase):
        backend = self._get_backend(data_id, var_name, phase)

        if data_id not in self._db[backend].get_data_ids():
            logger.info(
                f'data_id:{data_id} does not exist in backend:{backend}')
            return

        self._db[backend].delete_data(data_id, var_name, phase)

    def get_metadata(self, data_id, phase):
        metadata_zarr = self._db['zarr'].get_metadata(data_id, phase)
        metadata_numpy = self._db['numpy'].get_metadata(data_id, phase)

        metadata = dict(metadata_zarr, **metadata_numpy)
        return metadata

    def to_memory(self, data_id, var_name, phase):
        metadata_zarr = self._db['zarr'].get_metadata(data_id, phase)
        metadata_numpy = self._db['numpy'].get_metadata(data_id, phase)

        if var_name in metadata_numpy:
            logger.debug(f'{var_name} is already on memory (numpy)')

        elif var_name in metadata_zarr:
            tmp_data = self.get_data(data_id, var_name, phase, -1)
            self.delete_data(data_id, var_name, phase)
            self.add_data(data_id, var_name, tmp_data, phase, 'numpy')

        else:
            raise ValueError(f'{var_name} does not exist in hybrid database')

    def to_storage(self, data_id, var_name, phase):
        metadata_zarr = self._db['zarr'].get_metadata(data_id, phase)
        metadata_numpy = self._db['numpy'].get_metadata(data_id, phase)

        if var_name in metadata_zarr:
            logger.debug(f'{var_name} is already on storage (zarr)')

        elif var_name in metadata_numpy:
            tmp_data = self.get_data(data_id, var_name, phase, -1)
            self.delete_data(data_id, var_name, phase)
            self.add_data(data_id, var_name, tmp_data, phase, 'zarr')

        else:
            raise ValueError(f'{var_name} does not exist in hybrid database')

    def get_data_ids(self):
        return self._db['zarr'].get_data_ids()

    def _get_backend(self, data_id, var_name, phase):
        metadata_zarr = self._db['zarr'].get_metadata(data_id, phase)
        metadata_numpy = self._db['numpy'].get_metadata(data_id, phase)

        if var_name in metadata_zarr:
            return 'zarr'

        if var_name in metadata_numpy:
            return 'numpy'

        return self._mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    def _check_backend(self, data_id, var_name, phase, mode):
        metadata_zarr = self._db['zarr'].get_metadata(data_id, phase)
        metadata_numpy = self._db['numpy'].get_metadata(data_id, phase)

        if mode == 'numpy':
            if var_name in metadata_zarr:
                raise ValueError(f'{var_name} already exists in zarr backend')

        elif mode == 'zarr':
            if var_name in metadata_numpy:
                raise ValueError(f'{var_name} already exists in numpy backend')
