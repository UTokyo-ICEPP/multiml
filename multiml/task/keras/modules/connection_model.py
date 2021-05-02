import tensorflow as tf
from tensorflow.keras import Model
from multiml.task.basic.modules import ConnectionModel


class ConnectionModel(ConnectionModel, Model):
    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        # Defining as member variables results in registering as trainable variables
        self._submodel_trainable_variables = [
            u for model in self._models for u in model.layers if u.trainable
        ]

    def call(self, inputs):
        outputs = []
        caches = [None] * self._num_outputs

        for index, model in enumerate(self._models):
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
                    tf.expand_dims(tensor_input, 1)
                    for tensor_input in tensor_inputs
                ]
                tensor_inputs = tf.concat(tensor_inputs, axis=1)

            # Apply model in subtask
            tensor_outputs = model(tensor_inputs)

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

    def _get_variables(self):
        # Caching a variable list as a member variable results in setting these variables to be trainable.
        # So, we should not keep it as a member variable to avoid a mis-leading non-trainable behavior.
        return [u for v in self._models for u in v.variables]

    @staticmethod
    def squeeze_without_batch(tensor_outputs):
        offset = 0
        for index, shape in enumerate(tensor_outputs.shape):
            if shape == 1 and index != 0:
                dim = index - offset
                tensor_outputs = tf.squeeze(tensor_outputs, axis=dim)
                offset += 1
        return tensor_outputs
