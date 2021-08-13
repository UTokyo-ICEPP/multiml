from torch.nn import Conv2d, Identity, MaxPool2d, BatchNorm2d, Module, Sequential, init, ModuleList, Linear
from torch.nn.modules import activation as act
from torch import nn



class FiLMed_Conv2dResBlock(Module):
    def __init__(self, in_plane, out_plane, conv_args, activation):
        super().__init__()
        
        self.n_layers_to_yoto = out_plane 
        
        self.conv2d1 = Conv2d( in_plane, out_plane, **conv_args[0])
        self.batch_norm1 = BatchNorm2d( out_plane )
        self.activation1 = getattr(act, activation['key'])( **activation['args'] )
        
        self.conv2d2 = Conv2d( out_plane, out_plane, **conv_args[1])
        self.batch_norm2 = BatchNorm2d( out_plane, affine = False )
        
        if hasattr(act, activation['key']) : 
            self.activation2 = getattr(act, activation['key'])( **activation['args'] )
        elif hasattr(nn, activation['key']) : 
            self.activation2 = getattr(nn, activation['key'])( **activation['args'] )
        else : 
            raise ValueError(f'There is no {activation}!!')
            
        self.shortcut = Sequential(
            Conv2d(in_plane, out_plane, kernel_size = 1, stride = conv_args[0]['stride'], bias = False),
            BatchNorm2d(out_plane),
        )
        
    def get_n_layers_to_yoto(self):
        return self.n_layers_to_yoto
        
    def set_yoto_layer(self, gamma, beta):
        self.gamma = gamma
        self.beta  = beta

    def forward(self, x, x_gamma, x_beta):
        gamma = self.gamma(x_gamma)
        beta = self.beta(x_beta)

        h = self.conv2d1(x)
        h = self.batch_norm1(h)
        h = self.activation1(h)
        
        h = self.conv2d2(h)
        h = self.batch_norm2(h)
        
        h = h * gamma.view( gamma.size()[0], -1, 1, 1) + beta.view( beta.size()[0], -1, 1, 1)

        h += self.shortcut(x)
        h = self.activation2(h)
        return h
    

class Conv2DBlock_Yoto(Module):
    def __init__(self, hps, initialize=False, *args, **kwargs):
        """
            Args:
                hps : Hyperparameters, it must contain following 
                *args: Variable length argument list
                **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._hps = hps
        from copy import copy

        from collections import OrderedDict
        _layers = OrderedDict()
        
        self.n_layers_to_yoto = []
        self.yoto_idx = []
        n = 0
        
        for hp in self._hps : 
            layer, hp_args = hp['key'], hp['args']
            
            if 'ConvResBlock' in layer : 
                layer_args = copy( hp_args )
                _layers[layer] = FiLMed_Conv2dResBlock( **layer_args )
                self.n_layers_to_yoto += [ _layers[layer].get_n_layers_to_yoto() ]
                self.yoto_idx.append( n )
                n += 1
                
            elif 'Linear' in layer : 
                layer_args = copy( hp_args )
                _layers[layer] = Linear( **layer_args )
                self.yoto_idx.append(None)

            elif 'activation' in layer : 
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

        if initialize:
            self.apply(self._init_weights)
            
            
    def get_n_layers_to_yoto(self):
        return self.n_layers_to_yoto

    @staticmethod
    def _init_weights(m):
        if type(m) == Conv2d:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
    
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
