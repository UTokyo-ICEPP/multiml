import inspect

import torch
from torch.nn import Module, ModuleList
from multiml.task.basic.modules import ConnectionModel


class ConnectionModel(ConnectionModel, Module):
    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._sub_models = ModuleList([])

        for i_subtask in self._models:
            self._sub_models.append(i_subtask)

    def forward(self, inputs):
        outputs = []
        caches = [None] * self._num_outputs

        for index, sub_model in enumerate(self._sub_models):
            # Create input tensor
            input_indexes = self._input_var_index[index]
            tensor_inputs = [None] * len(input_indexes)

            for ii, input_index in enumerate(input_indexes):
                if input_index >= 0:  # inputs
                    tensor_inputs[ii] = inputs[input_index]
                else:  # caches
                    input_index = (input_index + 1) * -1
                    tensor_inputs[ii] = caches[input_index]

            # only one variable, no need to wrap with list
            if len(tensor_inputs) == 1:
                tensor_inputs = tensor_inputs[0]

            # If index is tuple, convert from list to tensor
            elif isinstance(input_indexes, tuple):
                tensor_inputs = [
                    torch.unsqueeze(tensor_input, 1)
                    for tensor_input in tensor_inputs
                ]
                tensor_inputs = torch.cat(tensor_inputs, dim=1)

            tensor_outputs = sub_model(tensor_inputs)
            output_indexes = self._output_var_index[index]

            # TODO: If outputs is list, special treatment
            if isinstance(tensor_outputs, list):
                outputs += tensor_outputs

                # only the first output is passed to next
                tensor_outputs = tensor_outputs[0]
            else:
                outputs.append(tensor_outputs)

            # model output is not list
            tensor_outputs = self.squeeze_without_batch(tensor_outputs)
            if len(output_indexes) == 1:
                caches[output_indexes[0]] = tensor_outputs

            else:
                for ii, output_index in enumerate(output_indexes):
                    caches[output_index] = tensor_outputs[:, ii]

        return outputs

    @staticmethod
    def squeeze_without_batch(tensor_outputs):
        offset = 0
        for index, shape in enumerate(tensor_outputs.shape):
            if shape == 1 and index != 0:
                dim = index - offset
                tensor_outputs = torch.squeeze(tensor_outputs, dim=dim)
                offset += 1
        return tensor_outputs
