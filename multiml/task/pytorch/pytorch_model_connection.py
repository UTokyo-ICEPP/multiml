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

    def get_storegate_dataset(self, phase):
        """ Returns storegate dataset.
        """
        class ModelConnectionDataset(StoreGateDataset):
            def __getitem__(self, index):
                input_ret = []
                true_ret = []

                for var_name in self._input_var_names:
                    input_data = self._storegate.get_data(var_names=var_name,
                                                          phase=self._phase,
                                                          index=index)
                    input_ret.append(input_data)

                for var_name in self._true_var_names:
                    true_data = self._storegate.get_data(var_names=var_name,
                                                         phase=self._phase,
                                                         index=index)
                    true_ret.append(true_data)

                return input_ret, true_ret

        true_var_names = [subtask.true_var_names for subtask in self._subtasks]
        dataset = ModelConnectionDataset(self.storegate,
                                         phase,
                                         input_var_names=self.input_var_names,
                                         true_var_names=true_var_names)
        return dataset
