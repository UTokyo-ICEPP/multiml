from torch.nn import LSTM, BatchNorm1d, Identity, Module, Sequential, init
from torch.nn.modules import activation as act


class LSTMBlock(Module):
    def __init__(self,
                 layers,
                 activation=None,
                 batch_norm=False,
                 initialize=True,
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
        super(LSTMBlock, self).__init__(*args, **kwargs)
        from collections import OrderedDict

        _layers = OrderedDict()
        for i, node in enumerate(layers):
            if i == len(layers) - 1:
                break
            else:
                _layers[f'LSTM{i}'] = LSTM(layers[i], layers[i + 1])

        if batch_norm:
            _layers['batchnorm1d'] = BatchNorm1d(layers[-1])

        if activation is not None:
            if activation == 'Identity':
                _layers[activation] = Identity()
            else:
                _layers[activation] = getattr(act, activation)()

        self._layers = Sequential(_layers)
        if initialize:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        for layer in self._layers:
            if type(layer) == LSTM:
                x, _ = layer(x)
            else:
                x = layer(x)
        return x
