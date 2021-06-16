import inspect

import torch
from torch.nn import Module, ModuleList
from multiml.task.basic.modules import ConnectionModel
import numpy as np

class ASNGModel(ConnectionModel, Module):
    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._sub_models = ModuleList([])
        
        categories = []
        
        for subtask in self._models:
            print(subtask._name)
            self._sub_models.append(subtask)
            categories += [subtask.n_subtask()]
            
        from multiml.task.pytorch.modules import AdaptiveSNG
        self.asng = AdaptiveSNG( np.array(categories) )
        
    def update_theta(self, losses, range_restriction=True) : 
        self.asng.update_theta(self.c_cats, self.c_ints, losses, range_restriction )
        
    def best_models(self):
        self.c_cat, self.c_int = self.asng.get_most_likely()
        cat_idx = self.c_cat.argmax( axis = 1 )
        
        best_task_ids = []        
        best_subtask_ids = []
        for idx, model in zip(cat_idx, self._models) : 
            subtask_id = model._subtasks[idx].subtask_id
            subtask_id = model._subtasks[idx].task_id
            best_task_ids.append( task_id )
            best_subtask_ids.append( subtask_id )
            
        return best_task_ids, best_subtask_ids
    
    def forward(self, inputs):
        c_cats, c_ints = self.asng.sampling()
        outputs = []
        print('-')
        print(len(self._sub_models))
        print(len(inputs))
        print(self._input_var_index)
        
        for c_cat, c_int in zip( c_cats, c_ints ) : 
            o = self._forward( inputs, c_cat, c_int )
            outputs.append(o)
        return outputs
    
    def _forward(self, inputs, c_cat, c_int):
        outputs = []
        caches = [None] * self._num_outputs
        
        
        
        for index, sub_model in enumerate(self._sub_models):
            sub_model.set_prob(c_cat, c_int)
            print(sub_model._name)
            print(self._input_var_index[index])
            # Create input tensor
            input_indexes = self._input_var_index[index]
            tensor_inputs = [None] * len(input_indexes)
            
            print(len(tensor_inputs), len(inputs))
            
            
            for ii, input_index in enumerate(input_indexes):
                print(ii, input_index)
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
