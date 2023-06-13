import torch
import torch.utils.data as tdata

import numpy as np


class StoreGateDataset(tdata.Dataset):
    """StoreGate dataset class."""
    def __init__(self,
                 storegate,
                 phase,
                 device=None,
                 preload=None,
                 input_var_names=None,
                 true_var_names=None,
                 callbacks=None):

        self._storegate = storegate
        self._true_var_names = true_var_names
        self._phase = phase
        self._device = device
        self._preload = preload
        self._input_var_names = input_var_names
        self._true_var_names = true_var_names
        self._callbacks = callbacks
        self._size = len(storegate[phase])

        if self._callbacks is None:
            self._callbacks = []

        self._data = None
        self._target = None

        if self._preload == 'cpu':
            self._data = storegate.get_data(input_var_names, phase)
            self._target = storegate.get_data(true_var_names, phase)

        elif self._preload == 'cuda':
            data = storegate.get_data(input_var_names, phase)
            target = storegate.get_data(true_var_names, phase)

            self._data = torch.from_numpy(data).to(device)
            self._target = torch.from_numpy(target).to(device)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self._preload:
            if isinstance(self._data, list):
                data = [idata[index] for idata in self._data]
            else:
                data = self._data[index]

            if isinstance(self._target, list):
                target = [idata[index] for idata in self._target]
            else:
                target = self._target[index]

        else:
            data = self._storegate.get_data(var_names=self._input_var_names,
                                            phase=self._phase,
                                            index=index)
            target = self._storegate.get_data(var_names=self._true_var_names,
                                              phase=self._phase,
                                              index=index)

        for callback in self._callbacks:
            data, target = callback(data, target)

        return data, target
