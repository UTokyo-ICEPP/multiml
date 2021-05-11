""" Pytorch ModelConnectionTask module.
"""
from . import PytorchBaseTask
from ..basic import ModelConnectionTask
from .modules import ConnectionModel
from multiml.task.pytorch.datasets import StoreGateDataset


class ModelConnectionTask(ModelConnectionTask, PytorchBaseTask):
    """ Pytorch implementation of ModelConnectionTask.
    """
    def build_model(self):
        models = [subtask.ml.model for subtask in self._subtasks]

        self._model = ConnectionModel(models=models,
                                      input_var_index=self._input_var_index,
                                      output_var_index=self._output_var_index)
