from multiml import logger

from . import KerasBaseTask


class MLPTask(KerasBaseTask):
    """ Keras MLP task
    """
    def __init__(self,
                 input_shapes=None,
                 layers=None,
                 activation=None,
                 activation_last=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 batch_norm=False,
                 **kwargs):
        """

        Args:
            input_shapes (tuple): shape for Keras.Inputs
            layers (list): list of hidden layers
            activation (str): activation function for MLP
            activation_last (str): activation function in last layer
            kernel_regularizer (str): kernel regularizer
            bias_regularizer (str): bias regularizer
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._input_shapes = input_shapes
        self._layers = layers
        self._activation = activation
        self._activation_last = activation_last
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._batch_norm = batch_norm

        self._format_member_variables()

    def set_hps(self, hps):
        """ Set thresholds. Cut-names are given by get_hyperparameters methods.

        Args:
            hps (dict): (hyperparameter name => hyperparameter value)
        """
        super().set_hps(hps)

        self._format_member_variables()

    def _format_member_variables(self):
        if self._input_shapes is None:
            self._input_shapes = [len(self.input_var_names)]

    def build_model(self):
        """ Build a Keras MLP model
        """
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Model

        input = self.get_inputs()[0]
        x = input
        # MEMO: Flatten() causes shape error when using model saving...
        x = K.reshape(input, (-1, ) + tuple(self._input_shapes))

        from .modules import MLPBlock
        mlp = MLPBlock(layers=self._layers,
                       activation=self._activation,
                       activation_last=self._activation_last,
                       kernel_regularizer=self._kernel_regularizer,
                       bias_regularizer=self._bias_regularizer,
                       batch_norm=self._batch_norm)
        x = mlp(x)

        self._model = Model(inputs=input, outputs=x)

    def get_inputs(self):
        from tensorflow.keras.layers import Input
        return [Input(shape=self._input_shapes)]
