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

    def forward(self, inputs, training=True):
        outputs = []
        caches = [None] * self._num_outputs

        for index, sub_model in enumerate(self._sub_models):
            # Create input tensor
            tensor_inputs = [None] * len(self._input_var_index[index])
            for ii, input_index in enumerate(self._input_var_index[index]):
                if input_index >= 0:  # inputs
                    tensor_inputs[ii] = inputs[input_index]
                else:  # caches
                    input_index = (input_index + 1) * -1
                    tensor_inputs[ii] = caches[input_index]

            # only one variable, no need to wrap with list
            if len(tensor_inputs) == 1:
                tensor_inputs = tensor_inputs[0]

            # check the shapes of each variable and set a flag
            elif self._binds[index] is None:
                shapes = [tensor_input.shape for tensor_input in tensor_inputs]
                self._binds[index] = len(set(shapes)) == 1

            # If shape of each variable is same, convert from list to tensor
            if self._binds[index] is True:
                tensor_inputs = [
                    torch.unsqueeze(tensor_input, 1)
                    for tensor_input in tensor_inputs
                ]
                tensor_inputs = torch.cat(tensor_inputs, dim=1)

            # Apply model in subtask
            if 'training' in inspect.getargspec(sub_model.forward)[0]:
                tensor_outputs = sub_model(tensor_inputs, training=training)
            else:
                tensor_outputs = sub_model(tensor_inputs)

            # TODO: If outputs is list, special treatment
            if isinstance(tensor_outputs, list):
                for tensor_output in tensor_outputs:
                    outputs.append(tensor_output)

                # only the first output is passed to next
                tensor_outputs = tensor_outputs[0]
            else:
                outputs.append(tensor_outputs)

            output_indexes = self._output_var_index[index]

            # model output is not list
            tensor_outputs = self.squeeze_without_batch(tensor_outputs)
            if len(output_indexes) == 1:
                caches[output_indexes[0]] = tensor_outputs

            elif len(output_indexes) == tensor_outputs.shape[1]:
                for ii, output_index in enumerate(output_indexes):
                    caches[output_index] = tensor_outputs[:, ii]
            else:
                raise ValueError(
                    'length of model outoputs and indeses are not consistent.')

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
