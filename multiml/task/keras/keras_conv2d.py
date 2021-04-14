from . import MLPTask


class Conv2DTask(MLPTask):
    """ Keras MLP task
    """
    def __init__(self, conv2d_layers=None, **kwargs):
        """

        Args:
            conv2d_layers list(tuple(str, dict)): list of conv2d layer config(op_name, op_args).
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._conv2d_layers = conv2d_layers

    def build_model(self):
        """ Build a Keras MLP model
        """
        from tensorflow.keras import backend as K
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.models import Model

        input = self.get_inputs()[0]
        x = input
        x = K.reshape(input, (-1, ) + tuple(self._input_shapes[-3:]))

        from multiml.task.keras.modules import Conv2DBlock, MLPBlock
        conv2d = Conv2DBlock(layers_conv2d=self._conv2d_layers)
        x = conv2d(x)

        x = Flatten()(x)

        mlp = MLPBlock(layers=self._layers,
                       activation=self._activation,
                       activation_last=self._activation_last,
                       batch_norm=self._batch_norm)
        x = mlp(x)

        x = K.reshape(x, (-1, len(self._output_var_names), self._layers[-1]))

        self._model = Model(inputs=input, outputs=x)
