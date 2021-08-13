import inspect

import torch
from torch.nn import Module, ModuleList
from multiml.task.basic.modules import ConnectionModel
import numpy as np
from multiml import logger


class YotoConnectionModel(ConnectionModel, Module):
    def __init__(self, common_hiddens, layer_hiddens, dropout, sampler_args, lambda_to_weight, device, batch_size, *args, **kwargs):
        """
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._sub_models = ModuleList([])
        self.loss_weights = []
        self.yoto_layer_channels = []
        self.yoto_layer_index = []
        self._device = device 
        self.batch_size = batch_size
        self.n_yoto = 0
        self.lambda_to_weight = lambda_to_weight
        
        
        for subtask in self._models:
            self._sub_models.append(subtask)
            self.loss_weights.append(None)
            
            yoto_layers = subtask.get_n_layers_to_yoto() 
            
            self.yoto_layer_channels += yoto_layers 
            for yoto_layer in yoto_layers : 
                self.yoto_layer_index.append( self.n_yoto )
            
            self.n_yoto += 1
        
        print(self.yoto_layer_channels)
        print(self.yoto_layer_index)
        
        yoto_hidden_layers = []
        for yoto_layer in self.yoto_layer_channels : 
            yoto_hidden_layers.append( [ int(yoto_layer * rate) for rate in layer_hiddens ] )
        
        from multiml.task.pytorch.modules.yoto import FiLMGenerator
        self._film = FiLMGenerator( layer_channels = self.yoto_layer_channels, dropout = dropout, 
                                   common_hiddens = common_hiddens, 
                                   layer_hiddens = yoto_hidden_layers,
                                   sampler_args =  sampler_args, 
                                   device = self._device)
        
        _gamma_layers = self._film.gamma.get_head_layers()
        _beta_layers  = self._film.beta.get_head_layers()
        
        gamma_layers = [ [] for i in range(self.n_yoto)]
        beta_layers  = [ [] for i in range(self.n_yoto)]
        for i, idx in enumerate(self.yoto_layer_index) : 
            gamma_layers[idx].append(_gamma_layers[i])
            beta_layers[idx].append(_beta_layers[i])
        
        for idx, subtask in enumerate(self._models):
            subtask.set_yoto_layer(gamma_layers[idx], beta_layers[idx])


        self.lambdas = None
            
    def get_loss_weight(self):
        return self.loss_weights 

    def set_lambdas(self, lambdas):
        self.lambdas = lambdas
        
    def forward(self, inputs) : 
        
        idx = self._input_var_index[0][0]
        batch_size = inputs[idx].size()[0] 
        
        x_gamma, x_beta, lambdas = self._film(batch_size, self.lambdas ) 
        
        loss_weights = self.lambda_to_weight(lambdas)
        self.loss_weights = [ torch.flatten(lw) for lw in loss_weights ]
        
        return self.forward_impl(inputs, x_gamma, x_beta)
        
    def forward_impl(self, inputs, x_gamma, x_beta):
        outputs = []
        caches = [None] * self._num_outputs
        
        for index, sub_model in enumerate(self._sub_models):
            # Create input tensor
            input_indexes = self._input_var_index[index]
            tensor_inputs = [None] * len(input_indexes)

            for ii, input_index in enumerate(input_indexes):
                if input_index >= 0:  # inputs
                    tensor_inputs[ii] = inputs[input_index]
                else:  # caches
                    input_index = (input_index + 1) * -1
                    tensor_inputs[ii] = caches[input_index]

            # only one variable, no need to wrap with list
            if len(tensor_inputs) == 1:
                tensor_inputs = tensor_inputs[0]

            # If index is tuple, convert from list to tensor
            elif isinstance(input_indexes, tuple):
                tensor_inputs = [torch.unsqueeze(tensor_input, 1) for tensor_input in tensor_inputs]
                tensor_inputs = torch.cat(tensor_inputs, dim=1)

            # Apply model in subtask
            tensor_outputs = sub_model(tensor_inputs, x_gamma, x_beta)
            output_indexes = self._output_var_index[index]

            # TODO: If outputs is list, special treatment
            if isinstance(tensor_outputs, list):
                outputs += tensor_outputs
                for ii, output_index in enumerate(output_indexes):
                    caches[output_index] = tensor_outputs[ii]
            else:
                outputs.append(tensor_outputs)
                if len(output_indexes) == 1:
                    caches[output_indexes[0]] = tensor_outputs

                else:
                    for ii, output_index in enumerate(output_indexes):
                        caches[output_index] = tensor_outputs[:, ii]

        return outputs
        
