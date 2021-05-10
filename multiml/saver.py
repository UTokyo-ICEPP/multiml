""" Saver module.
"""

import os
import copy
import shelve
import tempfile

from multiml import logger


class Saver:
    """ Miscellaneous object management class.

    Dictionary to save miscellaneous objects, and provides utility methods to
    manage ML metadata. There are two type of backends, *shelve* and *dict*,
    to store objects. *shelve* mode stores persistent objects, and *dict* mode
    stores temporary objects.

    Examples:
        >>> from multiml import Saver
        >>> saver = Saver()
        >>> print(saver.save_dir)
        >>> saver['key0'] = obj0
        >>> saver['key1'] = obj1
    """
    def __init__(self,
                 save_dir=None,
                 serial_id=None,
                 mode='shelve',
                 recreate=False):
        """ Initialize Saver and create the base directory.

        Args:
            save_dir (str): directory path to save objects. If ``None`` is
                given, a temporary directory is created automatically by
                ``tempfile.mkdtemp()`` method.
            serial_id (int): suffix of ``save_dir``, i.e. *save_dir*.*serial_id*.
                If ``None`` is given, ``serial_id`` is incremented
                automatically based on existence of the directory.
            mode (str): *shelve* or *dict* for default metadata management.
            recreate (bool): recreate shelve database if True.
        """
        if save_dir is None:
            save_dir = tempfile.mkdtemp()

        if serial_id is None:
            serial_id = self._get_serial_id(save_dir)

        self._save_dir = f"{save_dir}.{serial_id}"
        os.makedirs(self._save_dir, exist_ok=True)

        self._serial_id = serial_id
        self._state = 'close'
        self._mode = mode
        self._dict = {}  # for memory db
        self._shelve = None  # for storage db
        self._shelve_name = 'results.slv'
        self._shelve_mode = 'c'

        if self._mode == 'shelve':
            self.init_shelve(recreate)

    def __repr__(self):
        result = f'Saver(save_dir={self._save_dir}, '\
                       f'serial_id={self._serial_id}, '\
                       f'mode={self._mode}, '\
                       f'state={self._state})'
        return result

    def __len__(self):
        """Returns the number of stored objects in shelve and dict.

        Returns:
            int: the total number of stored objects in shelve and dict.
        """
        total_objs = 0

        if self._shelve is not None:
            total_objs += len(self._shelve)

        if self._dict is not None:
            total_objs += len(self._dict)

        return total_objs

    def __setitem__(self, key, obj):
        """Set key and store object to the default backend.

        ``key`` and ``obj`` are stored to the default backend, shelve or dict.

        Args:
            key (str): unique identifier of given object.
            obj (obj): arbitrary object to be stored.
        """
        self.add(key, obj, self._mode)

    def __getitem__(self, key):
        """Returns object for given key.

        ``key`` is searched from the both *shelve* and *dict* backends
        regardless of the default backend mode.

        Args:
            key (str): unique identifier to retrieve object.

        Returns:
            obj: arbitrary object..
        """
        if key not in self.keys():
            raise ValueError(f'key {key} does not exist in saver')

        if key in self._dict:
            return self._dict[key]

        if self._state == 'open':
            return self._shelve[key]

        self.open('r')
        result = self._shelve[key]
        self.close()
        return result

    def __delitem__(self, key):
        """Delete key and object from backends.

        Args:
            key (str): unique identifier to be deleted.
        """
        self.delete(key)

    ##########################################################################
    # Public user APIs
    ##########################################################################
    def init_shelve(self, recreate=False):
        """Initialize shelve database and confirm connection.

        Args:
            recreate (bool): If ``recreate`` is True, existing database is
                overwritten by an empty database. 
        """
        if recreate:
            self.open(mode='n')
            self.close()

        else:
            self.open(mode='c')
            self.close()

    def set_mode(self, mode):
        """Set default database (backend) mode.

        Args:
            mode (str): *shelve* or *dict*. If *dict* is given, *shelve*
                database will be opened with read only mode.
        """
        if mode == 'dict':
            self._shelve_mode = 'r'
            self._mode = 'dict'

        elif mode == 'shelve':
            self.init_shelve()
            self._shelve_mode = 'c'
            self._mode = 'shelve'

        else:
            raise ValueError(f'mode is {mode}, which should be shelve or dict')

    def open(self, mode=None):
        """Open shelve database with given mode.

        Args:
            mode (str): 'r': reading only, 'w': reading and writing,
                'c' (default): reading and writing, creating it if it does not
                exist, 'n': always create a new empty database, reading and
                writing.
  
        Examples:
            >>> saver.open('r')
            >>> print(saver['key0'])
            >>> print(saver['key1'])
            >>> saver.close()
        """
        if mode is None:
            mode = self._shelve_mode

        if self._state == 'open':
            logger.debug('saver is already open')
        else:
            self._shelve = shelve.open(f'{self._save_dir}/{self._shelve_name}',
                                       flag=mode)
            self._state = 'open'

    def close(self):
        """Close the shelve database.
        """
        if self._state == 'close':
            logger.debug('saver is already close')
        else:
            self._shelve.close()
            self._state = 'close'

    def keys(self, mode=None):
        """Return registered keys in backends.

        Args:
            mode (str): If *shelve* is given, keys in shelve database are
                returned. If *dict* is given, keys in dict database are
                returned. If None (default) all keys stored in the both
                backends are returned.

        Returns:
            list: list of registered keys.
        """
        if self._state == 'open':
            shelve_keys = list(self._shelve.keys())

        else:
            self.open()
            shelve_keys = list(self._shelve.keys())
            self.close()

        dict_keys = list(self._dict.keys())

        if mode == 'shelve':
            return shelve_keys

        if mode == 'dict':
            return dict_keys

        return shelve_keys + dict_keys

    def add(self, key, obj, mode=None, check=True):
        """Add object to given backend by key.

        If given ``key`` already exists in the given backend, object is
        overwritten. If the ``key`` already exists in the other backend, raises
        error, which can be avoided by setting ``check`` = False. If ``mode``
        is None, the default backend is used to store object.

        Args:
            key (str): unique identifier of given object.
            obj (obj): arbitrary object to be stored.
            mode (str): *shelve* or *dict* to specify the backend database.
            check (bool): If True, consistency between the backends is checked.
        """
        if mode is None:
            mode = self._mode

        if check and (mode == 'shelve') and (key in self.keys('dict')):
            raise ValueError(
                f'mode is shelve, but {key} already exists in dict')

        if check and (mode == 'dict') and (key in self.keys('shelve')):
            raise ValueError(
                f'mode is dict, but {key} already exists in shelve')

        if mode == 'shelve':
            if self._state == 'open':
                self._shelve[key] = obj

            else:
                self.open()
                self._shelve[key] = obj
                self.close()

        elif mode == 'dict':
            self._dict[key] = obj

        else:
            raise ValueError(f'mode is {mode}, which should be shelve or dict')

    def delete(self, key):
        """Delete key and object from the backends.

        Args:
            key (str): unique identifier to be deleted.
        """
        if key in self._dict:
            del self._dict[key]

        else:
            if self._state == 'open':
                del self._shelve[key]

            else:
                self.open()
                del self._shelve[key]
                self.close()

    def save(self):
        """Save the objects registered in dict to shelve.
        """
        if self._mode == 'dict':
            self._mode = 'shelve'
            self._shelve_mode = 'c'

        for key, value in self._dict.items():
            ckey = copy.copy(key)
            cvalue = copy.copy(value)
            self.add(ckey, cvalue, 'shelve', check=False)

        self._dict.clear()

        if self._mode == 'dict':
            self._mode = 'dict'
            self._shelve_mode = 'r'

    def to_memory(self, key):
        """Move object from shelve to dict.

        Args:
            key (str): the unique identifier to be moved.
        """
        if key in self.keys('dict'):
            logger.debug(f'{key} already exist in dict (memory)')

        elif key in self.keys('shelve'):
            value = copy.copy(self[key])
            del self[key]
            self.add(key, value, 'dict')

        else:
            raise ValueError(f'{key} does not exist in shelve')

    def to_storage(self, key):
        """Move object from dict to storage.

        Args:
            key (str): the unique identifier to be moved.
        """
        if key in self.keys('shelve'):
            logger.debug(f'{key} already exist in shelve (storage)')

        elif key in self.keys('dict'):
            value = copy.copy(self[key])
            del self[key]
            self.add(key, value, 'shelve')

        else:
            raise ValueError(f'{key} does not exist in dict')

    @property
    def shelve_name(self):
        """Returns the name of shelve database file.
        
        Returns:
            str: the name of shelve database file.
        """
        return self._shelve_name

    @shelve_name.setter
    def shelve_name(self, name):
        """Set name of shelve database file. Default is 'results.slv'.
        """
        self._shelve_name = name

    @property
    def shelve_mode(self):
        """Returns the mode of shelve database file.
        
        Returns:
            str: the mode of shelve database file.
        """
        return self._shelve_mode

    @shelve_mode.setter
    def shelve_mode(self, mode):
        """Set mode of shelve database file. Default is 'c'.
        """
        self._shelve_mode = mode

    @property
    def save_dir(self):
        """Returns the name of base directory of Saver.

        Returns:
            str: the name of base directory.
        """
        return self._save_dir

    @save_dir.setter
    def save_dir(self, save_dir):
        """Set name of base directory of saver
        """
        self._save_dir = save_dir

    def dump_ml(self, key, suffix=None, ml_type=None, **kwargs):
        """Dump machine learning models and parameters.

        Args:
            key (str): the unique identifier to store metadata.
            suffix (str): arbitrary suffix to key (e.g. job_id, epoch) to avoid
                conflicts.
            ml_type (str): *keras* or *pytorch* or None. If it is ``None``,
                just ``kwargs`` are dumped, which means ML model is not dumped.
            kwargs: arbitrary arguments. Only standard types (int, float, str,
                list, dict) are dumped due to a limitation of *pickle*.
        """
        if suffix is not None:
            key = f'{key}__{suffix}'

        if ml_type is None:
            self.add(key, self._get_basic_kwargs(**kwargs))

        elif ml_type == 'pytorch':
            self._dump_pytorch(key, **kwargs)

        elif ml_type == 'keras':
            self._dump_keras(key, **kwargs)

        else:
            raise NotImplementedError(
                f'ml_type: {ml_type} is not supported in the saver')

    def load_ml(self, key, suffix=None):
        """Load machine learning models and parameters.

        Args:
            key (str): the unique identifier to load metadata.
            suffix (str): arbitrary suffix to key (e.g. job_id, epoch).

        Returns:
            obj: arbitrary object.
        """
        if suffix is not None:
            key = f'{key}__{suffix}'

        return self[key]

    ##########################################################################
    # Internal methods
    ##########################################################################
    def _dump_pytorch(self, key, model=None, model_path=None, **kwargs):
        """Dump pytorch model and parameters.
        """
        kwargs = self._get_basic_kwargs(**kwargs)
        kwargs['model_type'] = 'pytorch'
        kwargs['timestamp'] = logger.get_now()

        if model is not None:
            if model_path is None:
                model_path = f'{self._save_dir}/{key}'
            logger.info(f'pytorch model saved path = {model_path}')
            import torch
            torch.save(model.state_dict(), model_path)
            kwargs['model_path'] = model_path

        kwargs.pop('_dataloaders', None)
        self[key] = kwargs

    def _dump_keras(self, key, model=None, model_path=None, **kwargs):
        """Dump keras model and parameters.
        """
        kwargs = self._get_basic_kwargs(**kwargs)
        kwargs['model_type'] = 'keras'
        kwargs['timestamp'] = logger.get_now()

        if model is not None:
            if model_path is None:
                model_path = f'{self._save_dir}/{key}'
            logger.info(f'keras model saved path = {model_path}')
            model.save_weights(model_path, save_format='tf')
            # model.save(model_path, save_format='tf')  # not work if the model is complicated
            kwargs['model_path'] = model_path

        self[key] = kwargs

    @staticmethod
    def _get_serial_id(save_dir):
        serial_id = 0
        while os.path.exists(f"{save_dir}.{serial_id}"):
            serial_id += 1
        return serial_id

    @staticmethod
    def _get_basic_kwargs(**kwargs):
        new_kwargs = {}
        for args_key, args_value in kwargs.items():
            if isinstance(args_value, (int, float, str, list, dict)):
                new_kwargs[args_key] = args_value
        return new_kwargs
