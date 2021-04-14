from torch.nn import Conv2d, Identity, MaxPool2d, Module, Sequential, init
from torch.nn.modules import activation as act


class Conv2DBlock(Module):
    def __init__(self, layers_conv2d=None, initialize=True, *args, **kwargs):
        """

            Args:
                layers_conv2d (list(tuple(str, dict))): configs of conv2d layer. list of tuple(op_name, op_args).
                *args: Variable length argument list
                **kwargs: Arbitrary keyword arguments
        """
        super(Conv2DBlock, self).__init__(*args, **kwargs)
        from copy import copy
        _layers = []
        conv2d_args = {"stride": 1, "padding": 0, "activation": 'ReLU'}
        maxpooling2d_args = {"kernel_size": 2, "stride": 2}

        for layer, args in layers_conv2d:
            if layer == 'conv2d':
                layer_args = copy(conv2d_args)
                layer_args.update(args)
                activation = layer_args.pop('activation')
                _layers.append(Conv2d(**layer_args))
                if activation == 'Identity':
                    _layers.append(Identity())
                else:
                    _layers.append(getattr(act, activation)())
            elif layer == 'maxpooling2d':
                layer_args = copy(maxpooling2d_args)
                layer_args.update(args)
                _layers.append(MaxPool2d(**layer_args))
            else:
                raise ValueError(f"{layer} is not implemented")

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
