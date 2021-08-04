from torch.nn import LSTM, BatchNorm1d, Identity, Module, Sequential, init
from torch.nn.modules import activation as act


class LSTMBlock(Module):
    def __init__(self,
                 hps, 
                #  layers,
                #  activation=None,
                #  batch_norm=False,
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
        self._hps = hps
        layers = self._hps['layers']._data
        activation = self._hps['activation']._data
        batch_norm = self._hps['batch_norm']._data
        
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

class LSTMBlock_HPS(Module):
    def __init__(self, hps, initialize = True, *args, **kwargs):
        """

        Args:
            hps : Hyperparamets, it must contain following:
                layers (list of list): list of list of hidden layers
                activation (list of str): list of activation function for MLP
                batch_norm (list of bool): use batch normalization or not
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._hps = hps
        
        from torch.nn import ModuleList
        self._layers = ModuleList([])
        self._activation = ModuleList([])
        
        #
        for layer in self._hps['layers']._data : 
            self._layers.append( ModuleList( self._make_layer(layer) ) )
        self._hps['layers'].set_layers( self._layers )
        
        #
        for activation in self._hps['activation']._data : 
            if 'Softmax' in activation : 
                self._activation.append( getattr(act, activation)( dim = 1 ) )
                
            elif 'Identity' in activation : 
                self._activation.append( Identity() )
                
            else :
                self._activation.append( getattr(act, activation)( ) )

        self._hps['activation'].set_layers(self._activation)
        
        # 
        last_layer = self._hps['layers']._data[0][-1]
        self._batch_norm = ModuleList([])
        for bn in self._hps['batch_norm']._data : 
            if bn : 
                self._batch_norm.append( BatchNorm1d( last_layer ) ) # all layers should have same #of output
            else : 
                self._batch_norm.append( Identity() )
        
        self._hps['batch_norm'].set_layers(self._batch_norm)
        
        if initialize:
            self.apply(self._init_weights)
        
        self._hps_layers = self._hps['layers']
        self._hps_activation = self._hps['activation']
        self._hps_batch_norm = self._hps['batch_norm'] 
        
    def set_active_hps(self, choice):
        self._hps.set_active_hps( choice )
        
    def _make_layer(self, layers) : 
        _layers = []
        for i, layer in enumerate( layers ) : 
            if i == len(layers) - 1 : 
                break
            else : 
                _layers.append( LSTM(layers[i], layers[i + 1]) )
        return _layers 
        
    def forward(self, x):
        
        for layer in self._hps_layers.active:
            x, _ = layer(x)
        
        x = self._hps_batch_norm.active(x)
        x = self._hps_activation.active(x)
        
        
        
        return x 
    
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
    
    
    
    
    
    
