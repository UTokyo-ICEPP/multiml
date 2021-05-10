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

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        data = self._storegate.get_data(var_names=self._input_var_names,
                                        phase=self._phase,
                                        index=index)
        target = self._storegate.get_data(var_names=self._true_var_names,
                                          phase=self._phase,
                                          index=index)

        return data, target
