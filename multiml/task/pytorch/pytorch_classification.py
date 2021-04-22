""" PytorchClassificationTask module
"""
from multiml.task.pytorch import PytorchBaseTask


class PytorchClassificationTask(PytorchBaseTask):
    """ Pytorch task for classification
    """
    def predict(self, **kwargs):
        kwargs['argmax'] = 1
        return super().predict(**kwargs)
