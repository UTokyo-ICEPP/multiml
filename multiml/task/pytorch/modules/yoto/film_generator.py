
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LambdaSampler(nn.Module):
    def __init__(self, device, ranges=((0.1, 2.0),), init = (1.0,), dist = 'uniform'):
        """
        Args:
            ranges: ((param1 low lim, param1 high lim), (param2 low lim, param2 high lim), ...)
            init: initial value, but this is actually not used in sampling phase
            dist: 'log1m_uniform', 'log_uniform', 'uniform'
        """
        super().__init__()
        self.dist = dist
        
        self.n_param = len(ranges)
        self.device = device 
        ranges = torch.from_numpy(np.array(ranges)[None,:,:].astype('float32')).to(self.device)
        init   = torch.from_numpy(np.array(init)[None,:].astype('float32')).to(self.device)
        
        self.ranges = nn.Parameter( ranges, requires_grad = False)
        self.init   = nn.Parameter( init,   requires_grad = False)
        self.range_size_1 = self.ranges.size()[1]

    def sampling(self, batch_size = 1):
        lambda_range = self._convert_func(self.ranges)

        #prms = np.random.rand(batch_size, self.ranges.size()[1])
        lambdas = torch.FloatTensor(batch_size, self.range_size_1).uniform_().to(self.device) 
        
        # is below really need or not ?
        # prms = torch.from_numpy(prms.astype('float32')).clone().cuda() 
        lambdas = lambdas * (lambda_range[:,:,1] - lambda_range[:,:,0]) + lambda_range[:,:,0]
        return self._inverse_func(lambdas)

    def enhance(self, lambdas, batch_size = 1):
        lambda_range = self._convert_func(self.ranges)
        
        #prms = np.random.rand(batch_size, self.ranges.size()[1])
        ones = torch.ones(batch_size, self.range_size_1, dtype=torch.float, device=self.device)
        l = torch.FloatTensor(lambdas).to(self.device)
        lambdas = ones * l
        
        
        # is below really need or not ?
        # prms = torch.from_numpy(prms.astype('float32')).clone().cuda() 
        lambdas = lambdas * (lambda_range[:,:,1] - lambda_range[:,:,0]) + lambda_range[:,:,0]
        return self._inverse_func(lambdas)

    def normalize_lambdas(self, lambdas):
        lambda_range = self._convert_func(self.ranges)
        lambdas = self._convert_func(lambdas)
        normed_lambdas = (lambdas - lambda_range[:,:,0]) / (lambda_range[:,:,1] - lambda_range[:,:,0]) * 2.0 - 1.0
        return normed_lambdas
    
    
    def _convert_func(self, x):
        if self.dist == 'log_uniform':
            return torch.log(x).to(self.device)
        elif self.dist == 'log1m_uniform':
            return torch.log(1 - x).to(self.device)
        elif self.dist == 'uniform':
            return x
        else:
            return

    def _inverse_func(self, x):
        if self.dist == 'log_uniform':
            return torch.exp(x).to(self.device)
        elif self.dist == 'log1m_uniform':
            return 1 - torch.exp(x).to(self.device)
        elif self.dist == 'uniform':
            return x
        else:
            return

class FiLM_MLP(nn.Module):
    def __init__(self, n_in, n_out, hiddens = (512,), dropout=0.1):
        super().__init__()

        n_neurons = (n_in,) + hiddens + (n_out,)

        self.layers = nn.ModuleList()
        for i in range(len(n_neurons) - 1):
            self.layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        
        self.activation = nn.ReLU(inplace = True) 
        self.dropout = nn.Dropout( dropout )

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

class FiLM_MultiheadMLP(nn.Module):
    def __init__(self, 
                n_in, 
                n_out = [16, 32], 
                common_hiddens = [64,], 
                head_hiddens = [[128, 16], [128, 32]],
                dropout = 0.1):
        super().__init__()
        
        n_head = len(n_out)
        
        
        # common layer
        if common_hiddens is not None:
            common_neurons = [n_in] + common_hiddens
            self.common_layers = []
            for i in range(len(common_neurons) - 1):
                self.common_layers.append(nn.Linear(common_neurons[i], common_neurons[i+1]))
                self.common_layers.append(nn.ReLU(inplace = True) )
                self.common_layers.append(nn.Dropout(dropout))
            self.common_layers = nn.Sequential(*self.common_layers)
        else:
            common_neurons = (n_in,)
            self.common_layers = nn.Identity()

        # multi head layer
        self.head_layers = nn.ModuleList()
        for ih in range(n_head):
            if head_hiddens is not None and head_hiddens[ih] is not None:
                h_neurons = [common_neurons[-1]] + head_hiddens[ih] + [n_out[ih]]
            else:
                h_neurons = [common_neurons[-1]] + [n_out[ih]]

            h_layers = []
            for i in range(len(h_neurons) - 1):
                h_layers.append(nn.Linear(h_neurons[i], h_neurons[i+1]))
                if i < len(h_neurons) - 2:
                    h_layers.append(nn.ReLU(inplace = True))
                    h_layers.append(nn.Dropout(dropout))
            self.head_layers.append(nn.Sequential(*h_layers))
            
    def get_head_layers(self):
        return self.head_layers
        
    def forward(self, x):
        x = self.common_layers(x)
        return x

class FiLMGenerator(nn.Module):
    def __init__(self, layer_channels, dropout, common_hiddens, layer_hiddens, sampler_args, device ):
        super().__init__()
        
        self.sampler = LambdaSampler(**sampler_args, device = device)
        self.layer_channels = layer_channels
        
        self.gamma = FiLM_MultiheadMLP( self.sampler.n_param, n_out = self.layer_channels, 
                                        common_hiddens = common_hiddens,
                                        head_hiddens = layer_hiddens, 
                                        dropout = dropout ) 
        self.beta   = FiLM_MultiheadMLP( self.sampler.n_param, n_out = self.layer_channels, 
                                        common_hiddens = common_hiddens,
                                        head_hiddens = layer_hiddens, 
                                        dropout = dropout ) 
        
    
    
    
    
    def forward(self, batch_size, lambdas = None ) : 
        if lambdas is None : 
            lambdas = self.sampler.sampling( batch_size )
        else : 
            lambdas = self.sampler.enhance( lambdas, batch_size )
        
        return self.forward_impl( lambdas )
        
    def forward_impl(self, lambdas ) :
        
        normed_lambdas =  self.sampler.normalize_lambdas(lambdas)
        gamma = self.gamma( normed_lambdas )
        beta  = self.beta( normed_lambdas )
        
        return gamma, beta, lambdas
        








