from torch.nn import Conv2d, Identity, MaxPool2d, BatchNorm2d, Module, Sequential, init, AdaptiveAvgPool2d
from torch.nn.modules import activation as act


class Conv2DBlock(Module):
    def __init__( self, hps, initialize=True, *args, **kwargs):
        """
            Args:
                layers_conv2d (list(tuple(str, dict))): configs of conv2d layer. list of tuple(op_name, op_args).
                *args: Variable length argument list
                **kwargs: Arbitrary keyword arguments
        """
        super(Conv2DBlock, self).__init__(*args, **kwargs)
        from copy import copy
        self._hps = hps
        layers_conv2d = self._hps['layers_conv2d']._data

        _layers = []
        conv2d_args = {"stride": 1, "padding": 0}
        maxpooling2d_args = {"kernel_size": 2, "stride": 2}

        for layer, args in layers_conv2d:
            if layer == 'conv2d':
                layer_args = copy(conv2d_args)
                layer_args.update(args)
                _layers.append(Conv2d(**layer_args))
            elif layer == 'maxpooling2d':
                layer_args = copy(maxpooling2d_args)
                layer_args.update(args)
                _layers.append(MaxPool2d(**layer_args))
            elif layer == 'AdaptiveAvgPool2d':
                _layers.append(AdaptiveAvgPool2d(**layer_args))
            elif layer == 'batch_norm2d':
                layer_args = copy(args)
                _layers.append(BatchNorm2d(**layer_args))
            elif layer == 'activation':
                if 'Softmax' in args:
                    _layers.append(getattr(act, args)(dim=1))
                elif 'Identity' in args:
                    _layers.append(Identity())
                else:
                    _layers.append(getattr(act, args)())
            else:
                raise ValueError(f"{layer}:{args} is not implemented")

        self._layers = Sequential(*_layers)
        if initialize:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == Conv2d:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        return self._layers(x)


class Conv2DBlock_HPS(Module):
    def __init__(self, hps, initialize=True, *args, **kwargs):
        """
            Args:
                hps : Hyperparameters, it must contain following 
                    layers (list(list(tuple(str, dict)))): configs of conv2d layer. list of list of tuple(op_name, op_args).
                *args: Variable length argument list
                **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._hps = hps

        from torch.nn import ModuleList
        self._layers = ModuleList()
        for layer in self._hps['layers_conv2d']._data:
            self._layers.append(self._make_layer(layer, initialize=initialize))

        self._hps['layers_conv2d'].set_layers(self._layers)

        self._hps_layers_conv2d = self._hps['layers_conv2d'].active

    def set_active_hps(self, choice):
        self._hps.set_active_hps(choice)

    def _make_layer(self, layers_conv2d=None, initialize=True):
        from copy import copy
        _layers = []
        conv2d_args = {"stride": 1, "padding": 0}
        maxpooling2d_args = {"kernel_size": 2, "stride": 2}

        for layer, args in layers_conv2d:

            if layer == 'conv2d':
                layer_args = copy(conv2d_args)
                layer_args.update(args)
                _layers.append(Conv2d(**layer_args))
            elif layer == 'maxpooling2d':
                layer_args = copy(maxpooling2d_args)
                layer_args.update(args)
                _layers.append(MaxPool2d(**layer_args))
            elif layer == 'AdaptiveAvgPool2d':
                _layers.append(AdaptiveAvgPool2d(**args))
            elif layer == 'batch_norm2d':
                layer_args = copy(args)
                _layers.append(BatchNorm2d(**layer_args))
            elif layer == 'activation':
                if 'Softmax' in args:
                    _layers.append(getattr(act, args)(dim=1))
                elif 'Identity' in args:
                    _layers.append(Identity())
                else:
                    _layers.append(getattr(act, args)())

            else:
                raise ValueError(f"{layer} is not implemented")

        _layers = Sequential(*_layers)

        if initialize:
            self.apply(self._init_weights)

        return _layers

    @staticmethod
    def _init_weights(m):
        if type(m) == Conv2d:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        #x = self._hps_layers_conv2d.active(x)
        x = self._hps_layers_conv2d(x)
        return x
