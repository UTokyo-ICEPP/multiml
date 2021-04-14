from torch.nn import BatchNorm1d, Identity, Linear, Module, Sequential, init
from torch.nn.modules import activation as act


class MLPBlock(Module):
    def __init__(self,
                 layers,
                 activation,
                 activation_last=None,
                 batch_norm=False,
                 initialize=True,
                 input_shape=None,
                 output_shape=None,
                 *args,
                 **kwargs):
        """

        Args:
            layers (list): list of hidden layers
            activation (str): activation function for MLP
            activation_last (str): activation function for the MLP last layer
            batch_norm (bool): use batch normalization
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super(MLPBlock, self).__init__(*args, **kwargs)

        if input_shape is not None:
            layers = [input_shape] + layers

        if output_shape is not None:
            layers = layers + [output_layers]

        _layers = []
        for i, node in enumerate(layers):
            if i == len(layers) - 1:
                break
            else:
                _layers.append(Linear(layers[i], layers[i + 1]))

            if batch_norm:
                _layers.append(BatchNorm1d(layers[i + 1]))

            if i == len(layers) - 2:
                if activation_last is None or activation_last == 'Identity':
                    _layers.append(Identity())
                else:
                    _layers.append(getattr(act, activation_last)())
            else:
                if activation == 'Identity':
                    _layers.append(Identity())
                else:
                    _layers.append(getattr(act, activation)())

        self._layers = Sequential(*_layers)
        if initialize:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == Linear:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        return self._layers(x)
