from torch.nn import BatchNorm1d, Identity, Linear, Module, Sequential, init, Dropout, Linear, ModuleList
from torch.nn.modules import activation as act
from torch import nn


class FiLMed_MLPBlock(Module):
    def __init__(self, n_input, n_output, activation, dropout_rate = None):
        '''
        n_input : number of input for first Linear layer
        n_output : number of outpu for first Linear layer
        activation : list(tuple) [0] is string for activation type('ReLU' or other), [1] should be kwargs for activation function
        dropout_rate : if this is None, will not apply dropout
        '''
        super().__init__()
        
        self.n_layers_to_yoto = n_output 
        
        self.linear = Linear( n_input, n_output)
        self.batch_norm = BatchNorm1d( n_output, affine = False )
        
        if hasattr(act, activation['key']) : 
            self.activation = getattr(act, activation['key'])( **activation['args'] )
        elif hasattr(nn, activation['key']) : 
            self.activation = getattr(nn, activation['key'])( **activation['args'] )
        else : 
            raise ValueError(f'There is no {activation}!!')
        
        
        if dropout_rate is None or dropout_rate == 'None': 
            self.dropout = Identity()
        else : 
            self.dropout = Dropout(dropout_rate)
    
    def get_n_layers_to_yoto(self):
        return self.n_layers_to_yoto
    
    def set_yoto_layer(self, gamma, beta):
        self.gamma = gamma
        self.beta  = beta
    
        
    def forward(self, x, x_gamma, x_beta): 
        gamma = self.gamma(x_gamma)
        beta = self.beta(x_beta)
        
        h = self.linear(x)
        h = self.batch_norm(h)
        h = self.activation(h)
        h = h * gamma.view( gamma.size()[0], -1) + beta.view(beta.size()[0], -1)
        h = self.dropout(h)
        return h 

class MLPResBlock(Module):
    def __init__(self, module ): 
        super().__init__()
        self.layers = ModuleList(module)
        
        self.n_layers_to_yoto = []
        self.has_yoto = []
        n = 0
        for layer in self.layers : 
            if hasattr( layer, 'get_n_layers_to_yoto') : 
                self.n_layers_to_yoto.append(layer.get_n_layers_to_yoto())
                self.has_yoto.append(n)
                n += 1
                
            else : 
                self.has_yoto.append(None)

    def set_yoto_layer(self, gamma, beta) : 
        for layer, yoto_idx in zip(self.layers, self.has_yoto):
            if yoto_idx is not None:
                layer.set_yoto_layer(gamma[yoto_idx], beta[yoto_idx])
        
    
    def forward(self, x, x_gamma, x_beta):
        h = x
        
        for layer, yoto_idx in zip(self.layers, self.has_yoto):
            if yoto_idx is not None:
                h = layer(h, x_gamma, x_beta)
            else : 
                h = layer(h)
            
        return x + h 

class MLPBlock_Yoto(Module):
    def __init__(self, hps, initialize = False, *args, **kwargs):
        """

        Args:
            hps: Hyperparameters, it contains following args : 
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)
        self._hps = hps
        
        self.n_layers_to_yoto = []
        self.yoto_idx         = []
        
        from copy import copy
        from collections import OrderedDict
        _layers = OrderedDict()
        
        n = 0 
        for hp in self._hps : 
            layer, hp_args = hp['key'], hp['args']
            
            if 'ResBlock' in layer : 
                res_block = []
                yoto_idx = []
                for _block_args in hp_args : 
                    block_args = copy(_block_args)
                    block = FiLMed_MLPBlock(**block_args) 
                    res_block.append( block )
                    self.n_layers_to_yoto.append( block.get_n_layers_to_yoto() )
                    yoto_idx.append(n)
                    n += 1
                
                _layers[layer] = MLPResBlock( res_block )
                
                self.yoto_idx.append( yoto_idx )
                
            elif 'FiLMed_MLPBlock' in layer :
                layer_args = copy( hp_args )
                block = FiLMed_MLPBlock( **layer_args ) 
                _layers[layer] = block 
                self.n_layers_to_yoto.append( block.get_n_layers_to_yoto() )
                self.yoto_idx.append( n )
                n += 1
            
            elif 'Linear' in layer : 
                layer_args = copy( hp_args )
                _layers[layer] = Linear( **layer_args )
                self.yoto_idx.append(None)
                
            elif 'activation' in layer: 
                layer_args = copy( hp_args )
                activation = layer_args.pop('name')
                
                if hasattr(act, activation):
                    _layers[layer] = getattr(act, activation)( **layer_args )
                elif hasattr(nn, activation ) : 
                    _layers[layer] = getattr(nn, activation)( **layer_args )
                else : 
                    raise ValueError(f'There is no {activation}!!')
                
                self.yoto_idx.append(None)
            
            else : 
                raise ValueError(f'There is no choice for {layer} : {hp_args}!!')
        
        self._layers = Sequential(_layers)

        #
        if initialize:
            self.apply(self._init_weights)
    
    def get_n_layers_to_yoto(self):
        return self.n_layers_to_yoto 
    
    def set_yoto_layer(self, gamma, beta):
        
        for layer, yoto_idx in zip(self._layers, self.yoto_idx):
            if yoto_idx is not None : 
                if isinstance(yoto_idx, list) or isinstance(yoto_idx, tuple) :
                    _gamma = [gamma[i] for i in yoto_idx] 
                    _beta = [beta[i] for i in yoto_idx] 
                else : 
                    _gamma = gamma[yoto_idx]
                    _beta = beta[yoto_idx]
                
                layer.set_yoto_layer(_gamma, _beta)
        
    def forward(self, x, x_gamma, x_beta):
        
        for layer, yoto_idx in zip(self._layers, self.yoto_idx):
            if yoto_idx is not None : 
                x = layer(x, x_gamma, x_beta)
            else : 
                x = layer(x)
        return x 
    
    @staticmethod
    def _init_weights(m):
        if type(m) == Linear:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
