import torch.utils.data as tdata


class NumpyDataset(tdata.Dataset):
    """StoreGate dataset class."""
    def __init__(self, inputs, targets):

        self._inputs = inputs
        self._targets = targets
        self._size = self.get_size(inputs)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        data = self.get_data(self._inputs, index)
        target = self.get_data(self._targets, index)

        return data, target

    def get_size(self, inputs):
        if isinstance(inputs, list):
            return self.get_size(inputs[0])
        else:
            return len(inputs)

    def get_data(self, data, index):
        if isinstance(data, list):
            return [self.get_data(idata, index) for idata in data]
        else:
            return data[index]
