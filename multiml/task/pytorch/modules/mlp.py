from torch.nn import BatchNorm1d, Identity, Linear, Module, Sequential, init
from torch.nn.modules import activation as act


class MLPBlock(Module):
    def __init__(self,
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
    def __init__(self,
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
        self._hps['layers'].make_layers( lambda x : self._make_layer(x, input_shape, output_shape) )
        self._hps['activation'].make_layers( lambda x : getattr(act, x)() )
        self._hps['activation_last'].make_layers( lambda x : getattr(act, x)() )
        
        _batch_norm = []
        for bn in self._hps['batch_norm']._data : 
            if bn : 
                _batch_norm.append( [ [ BatchNorm1d( l[-1] ) for l in layer] for layer in layers ] ) # all layers should have same #of output
            else : 
                _batch_norm.append( [ [ lambda x : x for l in layer] for layer in layers ] )
        self._hps['batch_norm'].set_layers( _batch_norm )
        
        if initialize:
            self.apply(self._init_weights)
    
    def _make_layer(self, layers, input_shape, output_shape ) : 
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
        
        for layer, batch_norm in zip(self._hps['layers'].active(), self._hps['batch_norm'].active() ): 
            x = layer(x)
            x = batch_norm(x)
            x = self._hps['activation'].active()(x)
        x = self._hps['activation_last'].active()(x)
        return x

    @staticmethod
    def _init_weights(m):
        if type(m) == Linear:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
