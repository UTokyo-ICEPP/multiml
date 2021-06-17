import inspect

import torch
from torch.nn import Module, ModuleList
from multiml.task.basic.modules import ConnectionModel
import numpy as np

class ASNGModel(ConnectionModel, Module):
    def __init__(self, lam, delta_init_factor, *args, **kwargs):
        """
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)
        
        self._sub_models = ModuleList([])
        
        categories = []
        
        for subtask in self._models:
            self._sub_models.append(subtask)
            categories += [subtask.n_subtask()]
        
        categories = np.array( categories )
        from multiml.task.pytorch.modules import AdaptiveSNG
        
        n = np.sum(categories - 1) + 0 # 0 is integer part
        
        self.asng = AdaptiveSNG( categories, lam = lam, delta_init = 1.0/(n**delta_init_factor), delta_max = np.inf )
        self.is_fix = False
        
    def set_most_likely(self):
        self.c_cat, self.c_int = self.asng.most_likely_value()
        self.is_fix = True
        
        
    def get_most_likely(self):
        return self.c_cat, self.c_int 
    
    def update_theta(self, losses, range_restriction=True) : 

        self.asng.update_theta(self.c_cats, self.c_ints, losses, range_restriction )
        
        
    def get_thetas(self):
        return self.asng.get_thetas()
        
    def best_models(self):
        return self.best_task_ids, self.best_subtask_ids 
    
    def forward(self, inputs):
        outputs = []
        
        if self.is_fix :
            outputs = self._forward( inputs, self.c_cat, self.c_int )
        else :
            self.c_cats, self.c_ints = self.asng.sampling()
            # print(f'forward')
            # print(f'c_cats is --> {self.c_cats.argmax(axis = 2)[0]}, {self.c_cats.argmax(axis = 2)[1]}')
            for c_cat, c_int in zip( self.c_cats, self.c_ints ) : 
                o = self._forward( inputs, c_cat, c_int )
                outputs.append(o)
        
        return outputs
    
    def _forward(self, inputs, c_cat, c_int):
        outputs = []
        caches = [None] * self._num_outputs
        
        for index, sub_model in enumerate(self._sub_models):
            sub_model.set_prob(c_cat[index], None ) # FIXME : c_int is not implemented
            
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
                tensor_inputs = [
                    torch.unsqueeze(tensor_input, 1)
                    for tensor_input in tensor_inputs
                ]
                tensor_inputs = torch.cat(tensor_inputs, dim=1)

            # Apply model in subtask
            tensor_outputs = sub_model(tensor_inputs)
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
