""" StoreGate module
"""

import numpy as np

from multiml import logger, const
from multiml.database import NumpyDatabase, ZarrDatabase, HybridDatabase


class StoreGate:
    """Data management class for multiml execution.

    StoreGate provides common interfaces to manage data between multiml
    *agents* and *tasks* with features of:
      * Different backends are supported (*numpy* or *zarr*, and *hybrid* of them),
      * Data are split into *train*, *valid* and *test* phases for ML,
      * Data are retrieved by ``var_names``, ``phase`` and ``index`` options.

    Each dataset in the storegate is keyed by unique ``data_id``. All data in
    the dataset are identified by ``var_names`` (column names). The number of
    samples in a phase is assumed to be the same for all variables in
    multiml agents and tasks. The ``compile()`` method ensures the validity of
    the dataset.

    Examples:
        >>> from multiml.storegate import StoreGate
        >>>
        >>> # User defined parameters
        >>> var_names = ['var0', 'var1', 'var2']
        >>> data = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        >>> phase = (0.5, 0.25, 0.25) # fraction of train, valid, test
        >>>
        >>> # Add data to storegate
        >>> storegate = StoreGate(backend = 'numpy', data_id='test_id')
        >>> storegate.add_data(var_names=var_names, data=data, phase=phase)
        >>>
        >>> # Get data from storegate
        >>> storegate.get_data(var_names=var_names, phase='train')
        >>> storegate['train'][var_names][0]
    """
    def __init__(self, backend='numpy', backend_args=None, data_id=None):
        """Initialize the storegate and the backend architecture.

        Initialize storegate and the backend architecture with its options.
        ``numpy`` backend manages data in memory, ``zarr`` backend reads and
        writes data to storage of given path. ``hybrid`` backend is combination
        of ``numpy`` and ``zarr`` backends, which allows to  move data between
        memory and storage.

        Args:
            backend (str): *numpy* (on memory), *zarr* (on storage), *hybrid*.
            backend_args (dict): backend options, e.g. path to zarr database.
                Please see ``ZarrDatabase`` and ``HybridDatabase`` classes for
                details.
            data_id (str): set default ``data_id`` if given.
        """
        if backend_args is None:
            backend_args = {}

        self._backend = backend
        self._backend_args = backend_args
        self._metadata = {}

        if self._backend == 'numpy':
            self._db = NumpyDatabase()
        elif self._backend == 'zarr':
            self._db = ZarrDatabase(**backend_args)
        elif self._backend == 'hybrid':
            self._db = HybridDatabase(**backend_args)
        else:
            raise NotImplementedError(
                f'Backend: {backend} is not supported in the storegate')

        self._data_id = None
        if data_id is not None:
            self.set_data_id(data_id)

        # setitem and getitem
        self._phase = None
        self._var_names = None

    def __repr__(self):
        result = f'StoreGate(backend={self._backend}, '\
                            f'backend_args={self._backend_args}, '\
                            f'data_id={self._data_id})'
        return result

    def __getitem__(self, item):
        """Retrieve data by python getitem syntax.

        Retrieve data by python getitem syntax, i.e.
        ``storegate[phase][var_names][index]``. ``data_id``, ``phase``,
        ``var_names`` and ``index`` need to be given to return selected data.
        If all parameters are set, selected data are returned. Otherwise, self
        instance class wit given parameters is returned.

        Args:
            item (str or list or int or slice): If item is str of *train* or
                *valid* or *test*, ``phase`` is set. If item is the other str
                or list of strs, ``var_names`` is set. If item is int or slice,
                data with index (slice) are returned.

        Returns:
            self or ndarray: please see description above.

        Example:
            >>> # get all train data
            >>> storegate['train']['var0'][:]
            >>> # slice train data by index
            >>> storegate['train']['var0'][0:2]
            >>> # loop by index
            >>> for data in storegate['train']['var0']:
            >>>     print(data)
        """
        self._check_valid_data_id()

        if (item in const.PHASES) or (item == 'all'):
            self._phase = item
            return self

        if isinstance(item, (str, list, tuple)):
            self._var_names = item
            return self

        if isinstance(item, (int, slice)):
            if self._phase == 'all' and item == slice(None, None, None):
                item = -1  # all index

            return self.get_data(var_names=self._var_names,
                                 phase=self._phase,
                                 index=item)

        raise NotImplementedError(f'item {item} is not supported')

    def __setitem__(self, item, data):
        """Update data by python setitem syntax.

        Update data by python setitem syntax, i.e.
        ``storegate[phase][var_names][index] = data``. ``data_id``, ``phase``,
        ``var_names`` and ``index`` need to be given to update data.

        Args:
            item (int or slice): Index of data to be updated.

        Example:
            >>> # update all train data
            >>> storegate['train']['var0'][:] = data
            >>> # update train data by index
            >>> storegate['train']['var0'][0:2] = data[0:2]
        """
        self._check_valid_data_id()

        if (self._phase is None) or (self._var_names is None):
            raise ValueError(f'Please provide phase and var_names')

        if not isinstance(item, (int, slice)):
            raise ValueError(f'item {item} must be int or slice')

        if self._phase == 'all' and item == slice(None, None, None):
            item = -1  # all index

        self.update_data(var_names=self._var_names,
                         data=data,
                         phase=self._phase,
                         index=item)

    def __delitem__(self, item):
        """Delete data by python delitem syntax.

        Delete data by python setitem syntax, i.e.
        ``del storegate[phase][var_names]``. ``data_id``, ``phase``,
        ``var_names`` need to be given to delete data.

        Args:
            item (str or list): ``var_names`` to be deleted.

        Example:
            >>> # delete var0 from train phase
            >>> del storegate['train']['var0']
        """
        self._check_valid_data_id()

        if self._phase is None:
            raise ValueError(f'Please provide phase (train, valid, test, all)')

        if not isinstance(item, (str, list)):
            raise ValueError(f'item {item} must be str or list')

        self.delete_data(item, phase=self._phase)

    def __len__(self):
        """Returns number of samples for given ``phase`` and ``data_id``.

        Returns:
           int: the number of samples in given conditions.

        Examples:
            >>> len(storegate['train'])
            >>> len(storegate['test'])
        """
        if not self._is_compiled():
            raise ValueError('len() is supported only after compile')

        if (self._data_id is None) or (self._phase is None):
            raise ValueError('data_id or phase is not provided')

        return self._metadata[self._data_id]['sizes'][self._phase]

    def __contains__(self, item):
        """Check if given ``var_name`` is available in storegate.

        Args:
            item (str): name of variables.

        Returns:
            bool: If ``item`` exists in given condisons or not.

        Examples:
            >>> 'var0' in storegate['train']
            >>> 'var1' in storegate['test']
        """
        if (self._data_id is None) or (self._phase is None):
            raise ValueError('data_id or phase is not provided')

        metadata = self._db.get_metadata(self._data_id, self._phase)
        return bool(item in metadata.keys())

    ##########################################################################
    # Public user APIs
    ##########################################################################
    @property
    def data_id(self):
        """Returns the current ``data_id``.

        Returns:
            str: the current ``data_id``.
        """
        return self._data_id

    def set_data_id(self, data_id):
        """ Set the default ``data_id`` and initialize the backend.

        If the default ``data_id`` is set, all methods defined in storegate,
        e.g. ``add_data()`` use the default ``data_id`` to manage data.

        Args:
            data_id (str): the default ``data_id``.

        """
        self._data_id = data_id
        self._check_valid_data_id()
        self._db.initialize(data_id)

    @property
    def backend(self):
        """Return the current backend of storegate.

        Returns:
            str: *numpy* or *zarr* or *hybrid*.
        """
        return self._backend

    def add_data(self,
                 var_names,
                 data,
                 phase='train',
                 shuffle=False,
                 do_compile=False):
        """Add data to the storegate with given options.

        If ``var_names`` already exists in given ``data_id`` and ``phase``,
        the data are appended, otherwise ``var_names`` are newly registered and
        the data are stored.

        Args:
            var_names (str or list): list of variable names, e.g.
                ['var0', 'var1', 'var2']. Single string, e.g. 'var0', is also
                allowed to add only one variable.
            data (list or ndarray): If ``var_names`` is single string, data
                shape must be (N,) where N is the number of samples.
                If ``var_names`` is a list, data shape must be (N, M, k),
                where M is the number of variables and k is an arbitrary shape
                of each data.
            phase (str or tuple): *all (auto)*, *train*, *valid*, *test* or
                tuple. *all* divides the data to *train*, *valid* and *test*
                automatically, but only after the ``compile``. If tuple
                (x, y, z) is given, the data are divided to *train*, *valid*
                and *test*. If contents of tuple is float and sum of the tuple
                is 1.0, the data are split to phases with fractions of
                (x, y, z) respectively. If contents of tuple is int, the data
                are split by given indexes.
            shuffle (bool or int): data are shuffled if True or int. If int is
                given, it is used as random seed of ``np.random``.
            do_compile (bool): do compile if True after adding data.

        Examples:
            >>> # add data to train phase
            >>> storegate.add_data(var_names='var0', data=[0, 1, 2], phase='train')
        """
        self._check_valid_data_id()

        var_names, data = self._view_to_list(var_names, data)
        self._check_valid_phase(phase)

        for var_name, idata in zip(var_names, data):
            idata = self._convert_to_np(idata)

            if shuffle:
                cur_seed = np.random.get_state()
                np.random.seed(int(shuffle))
                np.random.shuffle(idata)
                np.random.set_state(cur_seed)

            indices = self._get_phase_indices(phase, len(idata))

            for iphase, phase_data in zip(const.PHASES,
                                          np.split(idata, indices)):
                if len(phase_data) == 0:
                    continue

                self._db.add_data(self._data_id, var_name, phase_data, iphase)

        if self._data_id in self._metadata:
            self._metadata[self._data_id]['compiled'] = False

        if do_compile:
            self.compile(reset=False)

    def update_data(self,
                    var_names,
                    data,
                    phase='train',
                    index=-1,
                    do_compile=True):
        """Update data in storegate with given options.

        Update (replace) data in the storegate. If ``var_names`` does not exist
        in given ``data_id`` and ``phase``, data are newly added. Otherwise,
        selected data are replaced with given data.

        Args:
            var_names (str or list(srt)): see ``add_data()`` method.
            data (list or ndarray): see ``add_data()`` method.
            phase (str or tuple): see ``add_data()`` method.
            index (int or tuple): If ``index`` is -1 (default), all data are
                updated for given options. If ``index`` is int, only the data
                with ``index`` is updated. If index is (x, y), data in the
                range (x, y) are updated.
            do_compile (bool): do compile if True after updating data.

        Examples:
            >>> # update data of train phase
            >>> storegate.update_data(var_names='var0', data=[1], phase='train', index=1)
        """
        self._check_valid_data_id()

        var_names, data = self._view_to_list(var_names, data)
        self._check_valid_phase(phase)

        for var_name, idata in zip(var_names, data):
            idata = self._convert_to_np(idata)
            indices = self._get_phase_indices(phase, len(idata))

            for iphase, phase_data in zip(const.PHASES,
                                          np.split(idata, indices)):
                metadata = self._db.get_metadata(self._data_id, iphase)

                if len(phase_data) == 0:
                    continue

                if var_name not in metadata.keys():
                    logger.debug(
                        f'Adding {phase} : {var_name} to {self._data_id}')
                    self.add_data(var_name, phase_data, iphase, False)

                else:
                    self._db.update_data(self._data_id, var_name, phase_data,
                                         iphase, index)

        if self._data_id in self._metadata:
            self._metadata[self._data_id]['compiled'] = False

        if do_compile:
            self.compile(reset=False)

    def get_data(self, var_names, phase='train', index=-1):
        """Retrieve data from storegate with given options.

        Get data from the storegate. Python getitem sytax is also supprted,
        please see ``__getitem__`` method.

        Args:
            var_names (tuple or list or str): If tuple of variable names is
                given, e.g. ('var0', 'var1', 'var2'), data with ndarray format
                are returned. Please see the matrix below for shape of data.
                If list of variable names is given, e.g. ['var0', 'var1', 'var2'],
                list of data for each variable are returned. Single string,
                e.g. 'var0', is also allowed.
            phase (str or None): *all*, *train*, *valid*, *test* or *None*.
                If ``phase`` is *all* or *None*, data in all phases are
                returned, but it is allowed only after the ``compile``.
            index (int or tuple): see update_data method.

        Returns:
            ndarray or list: selected data by given options.

        Shape of returns:
            >>> # index   var_names  | single var | tuple vars
            >>> # ------------------------------------------------------------
            >>> # single index (>=0) | k          | (M, k)
            >>> # otherwise          | (N, k)     | (N, M, k)
            >>> # ------------------------------------------------------------
            >>> # k = arbitrary shape of data
            >>> # M = number of var_names
            >>> # N = number of samples

        Examples:
            >>> # get data by var_names, phase and index
            >>> storegate.get_data(var_names='var0', phase='train', index=1)
        """
        # recursive operation
        if isinstance(var_names, list):
            return [self.get_data(v, phase, index) for v in var_names]

        all_phase = bool(phase == 'all' or phase is None)

        if not self._is_compiled():
            raise ValueError('get_data is supported only after compile')

        if all_phase and (index != -1):
            raise ValueError(
                'phase=all is not supported together with index option')

        is_single_var = False
        if isinstance(var_names, str):
            is_single_var = True
            var_names = (var_names, )

        is_single_index = False
        if (isinstance(index, int)) and (index > -1):
            is_single_index = True

        # single variable and single index
        if is_single_var and is_single_index:
            return self._db.get_data(self._data_id, var_names[0], phase, index)

        results = []

        phases = [phase]
        if all_phase:
            phases = self._metadata[self._data_id]['valid_phases']

        for iphase in phases:
            phase_results = []
            metadata = self._db.get_metadata(self._data_id, iphase)

            for var_name in var_names:
                if var_name not in metadata.keys():
                    raise ValueError(
                        f'var_name {var_name} does not exist in storegate.')

                phase_results.append(
                    self._db.get_data(self._data_id, var_name, iphase, index))

            results.append(phase_results)

        if (not is_single_var) and is_single_index:
            np_results = np.array(results[0])

        elif is_single_var and (not is_single_index):
            np_results = np.concatenate(results, 1)
            np_results = np_results[0]

        else:
            np_results = np.concatenate(results, 1)
            shape = list(range(len(np_results.shape)))
            shape = [shape[1]] + [shape[0]] + shape[2:]
            np_results = np_results.transpose(tuple(shape))

        return np_results

    def delete_data(self, var_names, phase='train', do_compile=True):
        """Delete data associated with var_names.

        All data associated with ``var_names`` are deleted. Partial deletions
        with index is not supported for now.

        Args:
            var_names (str or list): see ``add_data()`` method.
            phase (str): see ``update_data()`` method.
            do_compile (bool): do compile if True after deletion.

        Examples:
            >>> # delete data associated with var_names
            >>> storegate.get_data(var_names='var0', phase='train')
        """
        self._check_valid_data_id()

        if isinstance(var_names, str):
            var_names = [var_names]

        if phase == 'all':
            phases = const.PHASES
        else:
            phases = [phase]

        for iphase in phases:
            for var_name in var_names:
                self._db.delete_data(self._data_id, var_name, iphase)

        if self._data_id in self._metadata:
            self._metadata[self._data_id]['compiled'] = False

        if do_compile:
            self.compile(reset=False)

    def get_data_ids(self):
        """Returns registered data_ids in the backend.

        Returns:
            list: list of registered ``data_id``.
        """
        return self._db.get_data_ids()

    def get_var_names(self, phase='train'):
        """Returns registered var_names for given phase.

        Args:
            phase (str): *train* or *valid* or *test*.

        Returns:
            list: list of variable names.
        """
        self._check_valid_data_id()

        metadata = self._db.get_metadata(self._data_id, phase)
        return list(metadata.keys())

    def get_var_shapes(self, var_names, phase='train'):
        """Returns shapes of variables for given phase.

        Args:
            var_names (str or list): variable names.
            phase (str): *train* or *valid* or *test*.

        Returns:
            ndarray.shape or list: shape of a variable, or list of shapes.
        """
        self._check_valid_data_id()

        metadata = self._db.get_metadata(self._data_id, phase)

        if isinstance(var_names, str):
            if var_names in metadata:
                return metadata[var_names]['shape']
            else:
                return None

        elif isinstance(var_names, list):
            shapes = []
            for var_name in var_names:
                shapes.append(self.get_var_shapes(var_name, phase))
            return shapes

        else:
            raise ValueError(f'invalid type of var_names: {var_names}.')

    def get_metadata(self):
        """Returns a dict of metadata.

        The metadata is available only after compile.

        Returns:
            dict: dict of metadata. Please see below for contents.

        Metadata contents:
            >>> {
            >>>   'compiled': 'compiled or not',
            >>>   'total_events': 'total events, sum of each phase',
            >>>   'sizes': {
            >>>       'train': 'total events of train phase',
            >>>       'valid': 'total events of valid phase',
            >>>       'test': 'total events of test phase',
            >>>       'all': 'total events',
            >>>   }
            >>>   'valid_phases': 'phases containing events'
            >>> }
        """
        self._check_valid_data_id()

        if not self._is_compiled():
            raise ValueError('get_metadata is supported only after compile')
        return self._metadata[self._data_id]

    def astype(self, var_names, dtype, phase='train'):
        """ Convert data type to given dtype (operation is limited by memory)

        Args:
            var_names (str or list): see ``add_data()`` method.
            dtype (numpy.dtype): dtypes of numpy. Please see numpy documents.
            phase (str): *all*, *train*, *valid*, *test*.
        """
        self._check_valid_data_id()

        tmp_data = self.get_data(var_names=var_names, phase=phase)
        tmp_data = tmp_data.astype(dtype)

        self.delete_data(var_names=var_names, phase=phase)

        self.update_data(var_names=var_names, data=tmp_data, phase=phase)

    def onehot(self, var_names, num_classes, phase='train'):
        """Convert data to onehot vectors (operation is limited by memory)

        Args:
            var_names (str or list): see ``add_data()`` method.
            num_classes (int): the number of classes.
            phase (str): *all*, *train*, *valid*, *test*.
        """
        self._check_valid_data_id()

        if isinstance(var_names, str):
            var_names = [var_names]

        for var_name in var_names:
            tmp_data = self.get_data(var_names=var_name, phase=phase)
            tmp_data = np.identity(num_classes)[tmp_data]

            self.delete_data(var_names=var_name, phase=phase)

            self.update_data(var_names=var_name, data=tmp_data, phase=phase)

    def argmax(self, var_names, axis, phase='train'):
        """Convert data to argmax (operation is limited by memory)

        Args:
            var_names (str or list): see ``add_data()`` method.
            axis (int): specifies axis.
            phase (str): *all*, *train*, *valid*, *test*.
        """
        self._check_valid_data_id()

        if isinstance(var_names, str):
            var_names = [var_names]

        for var_name in var_names:
            tmp_data = self.get_data(var_names=var_name, phase=phase)
            tmp_data = np.argmax(tmp_data, axis=axis)

            self.delete_data(var_names=var_name, phase=phase)

            self.update_data(var_names=var_name, data=tmp_data, phase=phase)

    def set_mode(self, mode):
        """Set backend mode of hybrid architecture.

        This method is valid for only *hybrid* database. If ``mode`` is
        *numpy*, basically data will be written in memory, and ``mode`` is
        *zarr*, dataw ill be written to storage.

        Args:
            mode (str): *numpy* or *zarr*.
        """
        if self._backend != 'hybrid':
            logger.warn(
                f'set_mode is valid for only hybrid database ({self._backend})'
            )

        else:
            self._db.mode = mode

    def to_memory(self, var_names, phase='train'):
        """Move data from storage to memory.

        This method is valid for only hybrid backend. This should be effective
        to reduce data I/O impacts.

        Args:
            var_names (str or list): see ``add_data()`` method.
            phase (str): *all*, *train*, *valid*, *test*.
        """
        if self._backend != 'hybrid':
            logger.warn(
                f'to_memory is valid for only hybrid database ({self._backend})'
            )

        self._check_valid_data_id()

        if isinstance(var_names, str):
            var_names = [var_names]

        if phase == 'all':
            phases = const.PHASES
        else:
            phases = [phase]

        for iphase in phases:
            for var_name in var_names:
                self._db.to_memory(self._data_id, var_name, iphase)

    def to_storage(self, var_names, phase='train'):
        """Move data from storage to memory.

        This method is valid for only hybrid backend. This is useful if data
        are large, then need to be escaped to storage.

        Args:
            var_names (str or list): see ``add_data()`` method.
            phase (str): *all*, *train*, *valid*, *test*.
        """
        if self._backend != 'hybrid':
            logger.warn('to_storage is valid for only hybrid database')

        self._check_valid_data_id()

        if isinstance(var_names, str):
            var_names = [var_names]

        if phase == 'all':
            phases = const.PHASES
        else:
            phases = [phase]

        for iphase in phases:
            for var_name in var_names:
                self._db.to_storage(self._data_id, var_name, iphase)

    def compile(self, reset=False, show_info=False):
        """Check if registered samples are valid.

        It is assumed that the ``compile`` is always called after
        ``add_data()`` or ``update_data()`` methods to validate registered,
        data.

        Args:
            reset (bool): special variable ``active`` is (re)set if True,
                ``active`` variable is used to indicate that samples should be
                used or not. e.g. in the metric calculation.
            show_info (bool): show information after compile.
        """
        self._check_valid_data_id()

        if self._is_compiled() and not reset:  # already complied
            return

        total_events = []
        phases = []

        for phase in const.PHASES:
            metadata = self._db.get_metadata(self._data_id, phase)

            phase_events = []
            for data in metadata.values():
                phase_events.append(data["total_events"])

            if len(set(phase_events)) > 1:
                raise ValueError(
                    f'Number of events are not consistent {metadata}')

            if phase_events:
                if reset:
                    self.update_data('active', [1] * phase_events[0], phase)
                total_events.append(phase_events[0])
                phases.append(phase)
            else:
                total_events.append(0)

        self._metadata[self._data_id] = {
            'compiled': True,
            'total_events': total_events,
            'sizes': {
                'train': total_events[0],
                'valid': total_events[1],
                'test': total_events[2],
                'all': sum(total_events)
            },
            'valid_phases': phases
        }

        if show_info:
            self.show_info()

    def show_info(self):
        """Show information currently registered in storegate.
        """
        self._check_valid_data_id()

        headers = dict(
            phase='phase'.ljust(6),
            backend='backend'.ljust(8),
            var_names='var_names'.ljust(15),
            var_types='var_types'.ljust(15),
            total_events='total_events'.ljust(15),
            var_shape='var_shape'.ljust(15),
        )

        is_compiled = self._is_compiled()
        logger.header3('')
        logger.info(f'data_id : {self._data_id}, compiled : {is_compiled}')

        for phase in ['train', 'valid', 'test']:
            metadata = self._db.get_metadata(self._data_id, phase)
            if not metadata.keys():
                continue

            logger.header2('')
            logger.info(' '.join(headers.values()))
            logger.header3('')
            for var_name, data in metadata.items():
                phase = phase.ljust(6)
                backend = data['backend'].ljust(8)
                var_name = var_name.ljust(15)
                dtype = data['type'].ljust(15)
                total_events = str(data["total_events"]).ljust(15)
                shape = data["shape"]

                logger.info(
                    f'{phase} {backend} {var_name} {dtype} {total_events} {shape}'
                )
        logger.header3('')

    ##########################################################################
    # Internal methods
    ##########################################################################
    @staticmethod
    def _view_to_list(var_names, data):
        """(private) utility method to convert var_names.

        ```var_names``` is converted to a list if it is string, and check shape
        of data.
        """
        if not isinstance(var_names, (tuple, list, str)):
            raise ValueError(
                f'{type(var_names)} is not supported for var_names')

        if isinstance(var_names, str):
            var_names = [var_names]
            data = [data]

        else:
            len_var_names = len(var_names)
            len_data = len(data[0])

            if len_var_names != len_data:
                raise ValueError(f'Shapes with bind_var are not consistent\
                        var_names:{len_var_names} data:{len_data}')

            bind_data = []
            for ivar in range(len(var_names)):
                bind_data.append([idata[ivar] for idata in data])
            data = bind_data

        len_var_names = len(var_names)
        len_data = len(data)
        if len_var_names != len_data:
            raise ValueError(f'Shapes are not consistent\
                    var_names:{len_var_names} data:{len_data}')

        return var_names, data

    @staticmethod
    def _convert_to_np(data):
        """(private) utility method to convert data.

        ```data``` is converted to numpy format if it is a list.
        """
        if isinstance(data, list):
            return np.array(data)

        return data

    def _check_valid_data_id(self):
        """(private) check if data_id is valid or not.
        """
        if self._data_id is None:
            raise ValueError('please set default data_id')

        if not isinstance(self._data_id, str):
            raise ValueError('data_id must be string')

    def _check_valid_phase(self, phase):
        """(private) check if given phase is valid or not.
        """
        if (phase in ('auto', 'all')) and (not self._is_compiled()):
            raise ValueError('Auto fraction is supported only after compile')

        if isinstance(phase, tuple) and (not isinstance(
                phase[0], int)) and (sum(phase) != 1.0):
            raise ValueError('Sum of fraction must be 1.0')

        if (phase not in ('auto', 'all')) and isinstance(
                phase, str) and (phase not in const.PHASES):
            raise ValueError(f'Phase {phase} is not supported')

    def _get_phase_indices(self, phase, ndata):
        """(private) returns a slice based on given phase.
        """
        if phase in ('auto', 'all'):
            total_events = self._metadata[self._data_id]['total_events']
            if ndata != sum(total_events):
                raise ValueError(
                    'Provided events are not consistent with metadata')
            indices = [total_events[0], total_events[0] + total_events[1]]

        elif isinstance(phase, tuple):
            if isinstance(phase[0], int):
                if sum(phase) != ndata:
                    raise ValueError(
                        f'Provided phases {phase} is not consistent with total # of data {ndata}'
                    )
                indices0 = phase[0]
                indices1 = indices0 + phase[1]
                indices = [int(indices0), int(indices1)]

            else:
                indices0 = ndata * phase[0]
                indices1 = indices0 + ndata * phase[1]
                indices = [int(indices0), int(indices1)]

        elif phase == 'train':
            indices = [ndata, ndata]

        elif phase == 'valid':
            indices = [0, ndata]

        else:  # test
            indices = [0, 0]

        return indices

    def _is_compiled(self):
        """(private) check if compiled or not.
        """
        if self._data_id not in self._metadata.keys():
            return False

        if not self._metadata[self._data_id]['compiled']:
            return False

        return True
