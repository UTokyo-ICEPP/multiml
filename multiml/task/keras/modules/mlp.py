from tensorflow.keras import Model


class MLPBlock(Model):
    def __init__(self,
                 layers=None,
                 activation=None,
                 activation_last=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 batch_norm=False,
                 *args,
                 **kwargs):
        """ Constructor

        Args:
            layers (list): list of hidden layers
            activation (str): activation function for MLP
            activation_last (str): activation function for the MLP last layer
            batch_norm (bool): use batch normalization
            kernel_regularizer (str): kernel regularizer
            bias_regularizer (str): bias regularizer
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._layers = []

        from tensorflow.keras.layers import (Activation, BatchNormalization,
                                             Dense)

        for i, node in enumerate(layers):
            self._layers.append(
                Dense(node,
                      activation=None,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer))

            if batch_norm:
                self._layers.append(BatchNormalization())

            if i == len(layers) - 1:
                self._layers.append(Activation(activation_last))
            else:
                self._layers.append(Activation(activation))

    def call(self, input_tensor, training=False):
        x = input_tensor
        for op in self._layers:
            x = op(x, training=training)
        return x
