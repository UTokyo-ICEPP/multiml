"""Hyperparameter module."""

import itertools

import numpy as np


class Hyperparameter:
    """Hyperparameter management class.

    This class describes hyperparameter names, type of parameters (*continuous* or *discrete*), and
    valid spaces. This class also retains the current value of hyperparameter, which is increased
    or decreased by ``step()`` method.
    """
    def __init__(self, name, data, is_continuous=False):
        """Initialize Hyperparameter class.

        Args:
            name (str): hyperparameter name.
            data (tuple or list): tuple must contain (``min``, ``max``, ``default_step``) for
                *continuous* mode, where ``min`` and ``max`` are the maximum and minimum value of
                the hyperparameter, and ``default_step`` indicates a interval of sampling (please
                see ``get_grid()`` method). list must contain valid parameter values for *discrete*
                mode.
            is_continuous (bool): hyperparameter is *continuous* or not.

        Examples:
            >>> # continuous hyperparameter
            >>> hp0 = Hyperparameter('hp0', (0., 1., 0.2), True)
            >>> # discrete hyperparameter
            >>> hp1 = Hyperparameter('hp1', [0., 0.2, 0.4, 0.6, 0.8. 1.0])
        """
        if not isinstance(is_continuous, bool):
            raise TypeError("is_continuous option should be bool")

        if not isinstance(data, (list, tuple)):
            raise TypeError("data option should be list or tuple")

        self._name = name
        self._data = data
        self._is_continuous = is_continuous
        self._value = data[0]

        if self._is_continuous:
            self._min = data[0]
            self._max = data[1]
            self._index = None
            self._default_step = data[2]
        else:
            self._min = 0
            self._max = len(data) - 1
            self._index = 0
            self._default_step = 1

    def __repr__(self):
        result = f'Hyperparameter(name={self._name}, '\
                                f'data={self._data}, '\
                                f'is_continuous={self._is_continuous})'
        return result

    def __len__(self):
        """Returns the number of possible hyperparameter values."""
        return len(self.get_grid())

    def __getitem__(self, item):
        """Returns the hyperparameter value by index.

        Args:
            item (int): index of ``get_grid()`` result.

        Returns:
            int or float: the hyperparameter value.
        """
        return self.get_grid()[item]

    @property
    def name(self):
        """Returns the name of hyperparameter.

        Returns:
            str: the name of hyperparameter.
        """
        return self._name

    @property
    def value(self):
        """Returns the current value of hyperparameter.

        Returns:
            int or float: the current value of hyperparameter.
        """
        return self._value

    @property
    def is_continuous(self):
        """Hyperparameter is *continuous* mode or *discrete* mode.

        Returns:
            bool: True if the hyperparameter is defined as *continuous*.
        """
        return self._is_continuous

    def step(self, step=None):
        """Increase or decrease the current hyperparameter value.

        Args:
            step (int or float): the hyperparameter value is increased or decreased according to
                sign of given ``step`` for *continuous* mode. Also the current index of
                hyperparameter is changed by ``step`` for *discrete* mode.

        Return:
            bool: if the value is changed or not.
        """
        if step is None:
            step = self._default_step

        if self._is_continuous:
            if not self._min <= (self._value + step) <= self._max:
                return False

            self._value += step
        else:
            if not isinstance(step, int):
                raise TypeError("step should be int for discrete hps")

            if not self._min <= (self._index + step) <= self._max:
                return False

            self._index += step
            self._value = self._data[self._index]

        return True

    def set_min_value(self):
        """Set the minimum value to the current value."""
        if self._is_continuous:
            self._value = self._data[0]
        else:
            self._index = 0
            self._value = self._data[0]

    def set_max_value(self):
        """Set the maximum value to the current value."""
        if self._is_continuous:
            self._value = self._data[1]
        else:
            self._index = len(self._data) - 1
            self._value = self._data[-1]

    def get_grid(self):
        """Returns available values of the hyperparameter.

        If the hyperparameter is *continuous* mode, values are sampled by
        step of ``default_step`` between ``min`` and ``max``.

        Returns:
            list: list of hyperparameter values.
        """
        if self._is_continuous:
            return np.arange(self._min, self._max, self._default_step)

        return self._data
    
    def make_layers(self, f) : 
        self._layers = [ f(v) for v in self._data ]
        self.set_active(0)
    
    def set_layers(self, layers):
        self._layers = layers
        self.set_active(0)
                
    ### TODO : following functionalities and properties are for list type hyperparameter    
    def set_active(self, idx):
        self.active_idx = idx
        self.active = self._layers[idx]
        self.active_data = self._data[idx]
        
    def active(self):
        return self.active
    
    def active_data(self):
        return self.active_data


class Hyperparameters:
    """Utility class to manage Hyperparameter classes.

    The *Hyperparameters* class provides interfances to manage *Hyperparameter* class instances.
    This *Hyperparameters* class instance should be passed to *TaskScheduler* together with
    corresponding *subtask*.

    Examples:
        >>> hps_dict = {
        >>>     'hp_layers': [5, 10, 15, 20], # discrete
        >>>     'hp_alpha': [1.0, 2.0, 3.0] # discrete
        >>> }
        >>> hps = Hyperparameters(hps_dict)
        >>> hps.set_min_hps()
        >>> hps.get_current_hps()
        >>>  -> {'hp_layers': 5, 'hp_alpha': 1.0}
        >>> hps.step('hp_layers', 2)
        >>> hps.get_current_hps()
        >>>  -> {'hp_layers': 15, 'hp_alpha': 1.0}
        >>> hps.step('hp_alpha', 1)
        >>> hps.get_current_hps()
        >>>  -> {'hp_layers': 15, 'hp_alpha': 2.0}
    """
    def __init__(self, hps=None):
        """Initialize Hyperparameters class.

        ``hps`` option provides a shortcut to register hyperparameters. This option works only for
        *discrete* hyperparameters for now.

        Args:
            hps (dict): a dictionary of hyperparameters. Please see ``add_hp_from_dict()`` method.
        """
        if hps is None:
            hps = {}

        self._hps = []
        self.add_hp_from_dict(hps)
        self._alias = False

    def __repr__(self):
        result = f'Hyperparameters(hps={self.get_hp_names()})'
        return result

    def __len__(self):
        """Returns the number of all possible combinations of hyperparameters."""
        if not self._hps:
            return 0

        result = 1
        for hp in self._hps:
            result *= len(hp)
        return result
    
    def _get_hp_by_index(self, index):
        if isinstance(index, int):
            grid_hps = self.get_grid_hps()
            return grid_hps[item]
        raise ValueError(f'this can be called by int(given: f{index})')

    def _get_hp_by_key(self, key):
        if isinstance(key, str):
            for hp in self._hps:
                if hp.name == key:
                    return hp


    def __getitem__(self, item):
        """Returns registered Hyperparameter class instance by index.

        If ``item`` is str, *Hyperparameter* is searched by its ``name``, and class instance is
        returned if it exists. If ``item`` is int, a dictionary of selected hyperparameters from
        ``get_grid_hps()`` method is returned.

        Args:
            item (str or int): the name of hyperparameter, or index of all possible combination
                from ``get_grid_hps()`` method.

        Returns:
            Hyperparameter or dict: please see the above description.
        """
        if isinstance(item, int):
            return self._get_hp_by_index(item)
        
        if isinstance(item, str):
            return self._get_hp_by_key(item)

        raise NotImplementedError(f'item {item} is not supported')

    def __contains__(self, item):
        """Check if Hyperparameter is registered or not.

        Args:
            item (str): the name of Hyperparameter.

        Returns:
            bool: True if Hyperparameter exists.
        """
        return item in self.get_hp_names()

    def step(self, hp_name=None, step=None):
        """Update the current hyperparameter values.

        Args:
            hp_name (str): name of hyperparameter. If given ``np_name`` is None, any updatable
                ``hp_name`` is selected arbitrarily.
            step (int or float): see *Hyperparameter* class.

        Return:
            bool: if the value is changed or not.
        """
        if hp_name is not None:
            for hp in self._hps:
                if hp.name == hp_name:
                    return hp.step(step)
            return False

        for hp in self._hps:
            result = hp.step(step)
            if result is True:
                return True
        return False

    def add_hp_from_dict(self, hps, is_alias = False ):
        """Add hyperparameters from dictionary.

        Values of the dictionary should be a list of allowed hyperparameter values. *Continuous*
        mode is not supported yet.

        Args:
            hps (dict): dict of hyperparameter values.

        Examples:
            >>> hp_dict = dict(hp0=[0, 1, 2], hp1=[3, 4, 5])
            >>> hps.add_hp_from_dict(hp_dict)
        """
        for _key, values in hps.items():
            if is_alias and _key not in self._alias.keys(): 
                continue
            
            key = self._alias[_key] if is_alias else _key
            
            if isinstance(values, list):
                self.add_hp(key, values, is_continuous=False)
            elif isinstance(values, tuple):
                self.add_hp(key, values, is_continuous=False)
            else:
                raise ValueError(
                    f'Only list of discrete values is supported for converting from dict : {_key}:{values}')
    
    def add_hp(self, name, values, is_continuous=False):
        """Add hyperparameter.

        Args:
            name (str): the name of hyperparameter.
            values (tuple or list): please see *Hyperparameter* class.
            is_continuous (bool): Hyperparameter is *continuous* or not.
        """
        self._hps.append(Hyperparameter(name, values, is_continuous))

    def set_min_hps(self):
        """Set the minimum value for each Hyperparameter."""
        for hp in self._hps:
            hp.set_min_value()

    def set_max_hps(self):
        """Set the maximum value for each Hyperparameter."""
        for hp in self._hps:
            hp.set_max_value()

    def get_current_hps(self):
        """Returns the current values of Hyperparameters.

        Returns:
            dict: dict of the current hyperparameter values.
        """
        results = {}
        for hp in self._hps:
            results[hp.name] = hp.value
        return results

    def get_grid_hps(self):
        """Returns all possible combination of hyperparameters values.

        If registered *Hyperparameter* class instance is *continuous* mode, values are sampled
        between ``min`` and ``max`` by dividing ``default step``.

        Returns:
            list: all possible combination of hyperparameters values.
        """
        hps_keys = []
        hps_values = []
        results = []

        if not self._hps:
            return results

        for hp in self._hps:
            hps_keys.append(hp.name)
            hps_values.append(hp.get_grid())

        for value in itertools.product(*hps_values):
            result = {}
            for index, key in enumerate(hps_keys):
                result[key] = value[index]
            results.append(result)
        return results
    
    def get_all_hps(self):
        results = {}
        for hp in self._hps:
            results[hp.name] = hp._data
        return results
        

    def get_hp_names(self):
        """Returns the names of hyperparameters.

        Returns:
            list: the names of hyperparameters.
        """
        return [hp.name for hp in self._hps]
    
    def set_alias(self, alias):
        self._alias = alias
        self._alias_inv = { v:k for k, v in alias.items() }
        
    def get_hps_parameters(self):
        results = {}
        for hp in self._hps : 
            if isinstance(hp._data, list) or isinstance(hp._data, tuple) :
                if len(hp._data) > 1 : 
                    results[hp.name] = len(hp._data)
        return results
    
    def set_active_hps(self, choice):
        '''
        choice should be dict, it must contains pair of like key : active index
        '''
        
        for hp in self._hps : 
            choice_name = self._alias_inv[hp.name]
            if choice_name in choice.keys() : 
                hp.set_active( choice[choice_name] )
            else : 
                hp.set_active( 0 )
            
        
        





