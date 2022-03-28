import torch
from torch.utils.data import Sampler


class SimpleBatchSampler(Sampler):
    def __init__(self, num_samples, batch_size, shuffle):
        self.data = torch.arange(num_samples)
        self.index = 0
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.data = torch.randperm(self.num_samples)
        self.index = 0
        return self

    def __next__(self):
        index1 = self.batch_size * self.index
        index2 = min(self.batch_size * (self.index + 1), self.num_samples)

        if index1 >= self.num_samples:
            raise StopIteration()

        self.index += 1

        return self.data[index1:index2]

    def __len__(self):
        return -(-1 * self.num_samples // self.batch_size)
