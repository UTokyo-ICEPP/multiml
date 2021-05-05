import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from multiml import logger
from multiml.saver import Saver
from multiml.task.pytorch import PytorchBaseTask


class MyPytorchModel(nn.Module):
    def __init__(self):
        super(MyPytorchModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x, training):
        x = F.relu(self.fc1(x))
        return x


class MyPytorchTask(PytorchBaseTask):

    @logger.logging
    def execute(self):
        inputs = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        datasets = TensorDataset(inputs, targets)
        dataloader = DataLoader(datasets, batch_size=2, shuffle=True)

        dataloaders = dict(train=dataloader,
                           valid=dataloader,
                           test=dataloader)

        self.compile()
        history = self.fit(dataloaders=dataloaders)

        assert len(history['train']) == self._num_epochs

        result = self.predict(dataloader=dataloaders['test'])
        assert result.shape == (2, 2)

        self._saver.dump_ml(key='mypytorch',
                            ml_type='pytorch',
                            model=self.ml.model,
                            **self.__dict__)

def test_mypytorchtask():
    logger.set_level(logger.DEBUG)
    saver = Saver()
    task = MyPytorchTask(saver=saver,
                         model=MyPytorchModel,
                         optimizer='SGD',
                         optimizer_args=dict(lr=0.1),
                         loss='CrossEntropyLoss',
                         metrics=['acc', 'lrs'])
    task.set_hps({'num_epochs': 5})

    task.execute()
    task.finalize()

    assert saver.load_ml(key='mypytorch')['_num_epochs'] == 5
    
if __name__ == '__main__':
    test_mypytorchtask()
