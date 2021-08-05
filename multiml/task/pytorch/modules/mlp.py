from torch.nn import BatchNorm1d, Identity, Linear, Module, Sequential, init
from torch.nn.modules import activation as act


class MLPBlock(Module):
    def __init__(
            self,
            hps,
            #  layers,
            #  activation,
            #  activation_last=None,
            #  batch_norm=False,
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
        self._hps = hps
        layers = self._hps['layers']._data
        activation = self._hps['activation']._data
        activation_last = self._hps['activation_last']._data
        batch_norm = self._hps['batch_norm']._data

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


class MLPBlock_HPS(Module):
    def __init__(
            self,
            hps,
            #  layers,
            #  activation,
            #  activation_last=None,
            #  batch_norm=False,
            initialize=True,
            input_shape=None,
            output_shape=None,
            *args,
            **kwargs):
        """

        Args:
            hps : Hyperparameters, it contains following args : 
                layers (list of list): list of hidden layers
                activation (list of str): activation function for MLP
                activation_last (list of str): activation function for the MLP last layer
                batch_norm (list of bool): use batch normalization
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)
        self._hps = hps
        from torch.nn import ModuleList

        self._layers = ModuleList([])
        self._activation = ModuleList([])
        self._activation_last = ModuleList([])

        #
        for layer in self._hps['layers']._data:
            self._layers.append(ModuleList(self._make_layer(layer, input_shape, output_shape)))
        self._hps['layers'].set_layers(self._layers)

        #
        for activation in self._hps['activation']._data:
            if 'Softmax' in activation:
                self._activation.append(getattr(act, activation)(dim=1))

            elif 'Identity' in activation:
                self._activation.append(Identity())

            else:
                self._activation.append(getattr(act, activation)())

        self._hps['activation'].set_layers(self._activation)

        #
        for activation_last in self._hps['activation_last']._data:
            if 'Softmax' in activation_last:
                self._activation_last.append(getattr(act, activation_last)(dim=1))

            elif 'Identity' in activation_last:
                self._activation_last.append(Identity())

            else:
                self._activation_last.append(getattr(act, activation_last)())
        self._hps['activation_last'].set_layers(self._activation_last)

        #
        self._batch_norm = ModuleList()
        for bn in self._hps['batch_norm']._data:

            _batch_norms = ModuleList()

            for layer in self._hps['layers']._data:
                _batch_norm = ModuleList([BatchNorm1d(l) if bn else Identity() for l in layer[1:]])
                _batch_norms.append(_batch_norm)

            self._batch_norm.append(_batch_norms)

        self._hps['batch_norm'].set_layers(self._batch_norm)

        if initialize:
            self.apply(self._init_weights)

        self._hps_layers = self._hps['layers']
        self._hps_activation = self._hps['activation']
        self._hps_activation_last = self._hps['activation_last']
        self._hps_batch_norm = self._hps['batch_norm']

    def set_active_hps(self, choice):
        self._hps.set_active_hps(choice)

    def _make_layer(self, layers, input_shape, output_shape):

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

        return _layers

    def forward(self, x):

        batch_norm_idx = self._hps_layers.active_idx
        last = len(self._hps_layers.active) - 1
        for i, (layer, batch_norm) in enumerate(
                zip(self._hps_layers.active, self._hps_batch_norm.active[batch_norm_idx])):
            x = layer(x)
            x = batch_norm(x)
            if i != last:
                x = self._hps_activation.active(x)

        x = self._hps_activation_last.active(x)

        return x

    @staticmethod
    def _init_weights(m):
        if type(m) == Linear:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
