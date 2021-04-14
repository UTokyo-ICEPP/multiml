import torch.utils.data as tdata


class StoreGateDataset(tdata.Dataset):
    """ StoreGate dataset class
    """
    def __init__(self,
                 storegate,
                 phase,
                 input_var_names=None,
                 true_var_names=None):

        self._storegate = storegate
        self._true_var_names = true_var_names
        self._phase = phase
        self._input_var_names = input_var_names
        self._true_var_names = true_var_names
        self._size = len(storegate[phase])

        self._input_slice = 0
        self._true_slice = -1

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        data = self._storegate[self._phase][self._input_var_names][index]
        target = self._storegate[self._phase][self._true_var_names][index]

        return data, target

    @property
    def input_slice(self):
        return self._input_slice

    @property
    def true_slice(self):
        return self._true_slice

    @property
    def data_slice(self):
        return (self._input_slice, self._true_slice)
