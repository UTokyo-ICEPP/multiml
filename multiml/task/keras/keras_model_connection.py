"""Keras ModelConnectionTask module."""
from . import KerasBaseTask
from ..basic import ModelConnectionTask
from .modules import ConnectionModel


class ModelConnectionTask(ModelConnectionTask, KerasBaseTask):
    """Keras implementation of ModelConnectionTask."""
    def build_model(self):
        models = [subtask.ml.model for subtask in self._subtasks]

        self._model = ConnectionModel(models=models,
                                      input_var_index=self._input_var_index,
                                      output_var_index=self._output_var_index)
