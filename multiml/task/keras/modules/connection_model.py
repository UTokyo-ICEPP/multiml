import tensorflow as tf
from multiml.task.basic.modules import ConnectionModel
from multiml.task.keras.modules import BaseModel


class ConnectionModel(ConnectionModel, BaseModel):
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
            output_indexes = self._output_var_index[index]

            # TODO: If outputs is list, special treatment
            if isinstance(tensor_outputs, list):
                outputs += tensor_outputs
                for ii, output_index in enumerate(output_indexes):
                    caches[output_index] = tensor_outputs[ii]
            else:
                outputs.append(tensor_outputs)
                if len(output_indexes) == 1:
                    caches[output_indexes[0]] = tensor_outputs
                else:
                    for ii, output_index in enumerate(output_indexes):
                        caches[output_index] = tensor_outputs[:, ii]

        return outputs

    def _get_variables(self):
        # Caching a variable list as a member variable results in setting these variables to be trainable.
        # So, we should not keep it as a member variable to avoid a mis-leading non-trainable behavior.
        return [u for v in self._models for u in v.variables]
