import inspect

import torch
from torch.nn import Module, ModuleList
from multiml.task.basic.modules import ConnectionModel
import numpy as np
from multiml import logger


class ASNGModel(ConnectionModel, Module):
    def __init__(self,
                 task_ids,
                 lam,
                 delta_init_factor,
                 alpha=1.5,
                 range_restriction=True,
                 *args,
                 **kwargs):
        """
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(*args, **kwargs)

        self._sub_models = ModuleList([])

        categories = []
        integers = []

        key_to_idx = {}
        n = 0

        for task, task_id in zip(self._models, task_ids):
            self._sub_models.append(task)

            key_to_idx[task_id] = {}
            hps_params = task.get_hps_parameters(
            )  # this should like {'choice_block_model_name':N} or {'hps0':N0, 'hps1':N1,,,}

            for key, param in hps_params.items():  #
                key_to_idx[task_id][key] = n
                if isinstance(param, int):
                    categories += [param]
                else:
                    raise ValueError(f'param is not valid type {key}:{param}')
                n += 1
        #
        self.idx_to_key = {}
        self.task_id_keys = {}
        for task_id, val in key_to_idx.items():
            self.task_id_keys[task_id] = []
            for key, v in val.items():
                self.idx_to_key[v] = (task_id, key)
                self.task_id_keys[task_id].append(key)

        #
        categories = np.array(categories)

        self.task_ids = task_ids

        logger.info(f'key_to_idx is {key_to_idx}')
        logger.info(f'idx_to_key is {self.idx_to_key}')
        logger.info(f'categories is{categories}')

        n = np.sum(categories - 1) + 0  # 0 is integer part

        if len(categories) > 0 and len(integers) > 0:
            from multiml.task.pytorch.modules import AdaptiveSNG
            self.asng = AdaptiveSNG(categories,
                                    lam=lam,
                                    delta_init=1.0 / (n**delta_init_factor),
                                    delta_max=np.inf,
                                    alpha=alpha,
                                    range_restriction=range_restriction)

        elif len(categories) > 0 and len(integers) == 0:
            from multiml.task.pytorch.modules import AdaptiveSNG_cat
            self.asng = AdaptiveSNG_cat(categories,
                                        lam=lam,
                                        delta_init=1.0 / (n**delta_init_factor),
                                        delta_max=np.inf,
                                        alpha=alpha,
                                        range_restriction=range_restriction)

        elif len(categories) == 0 and len(integers) > 0:
            from multiml.task.pytorch.modules import AdaptiveSNG_int
            self.asng = AdaptiveSNG_int(categories,
                                        lam=lam,
                                        delta_init=1.0 / (n**delta_init_factor),
                                        delta_max=np.inf,
                                        alpha=alpha,
                                        range_restriction=range_restriction)

        else:
            raise ValueError('ASNGModel : Both categories and integers is not active...')

        self.set_fix(False)

    def convert_cat_to_hps_choice(self, c_cat):
        choice = {
            task_id: {key: None
                      for key in keys}
            for task_id, keys in self.task_id_keys.items()
        }
        cat_idx = c_cat.argmax(axis=1)
        for i, idx in enumerate(cat_idx):
            task_id, key = self.idx_to_key[i]
            choice[task_id][key] = idx
        return choice

    def get_task_idx(self, task_id):
        for idx, _task_id in enumerate(self.task_id_keys.keys()):
            if task_id == _task_id:
                return idx
        raise ValueError(f'There is no task with {task_id}')

    def set_most_likely(self):

        self.c_cat, self.c_int = self.asng.most_likely_value()
        print(self.c_cat)
        self.best_choice = self.convert_cat_to_hps_choice(self.c_cat)
        print(self.best_choice)
        self.set_fix(True)

        for task_id, value in self.best_choice.items():
            task_idx = self.get_task_idx(task_id)
            task = self._models[task_idx]
            task.choice(value)

    def set_fix(self, fix):
        self.is_fix = fix
        if self.is_fix:
            self.forward = self.forward_fix
        else:
            self.forward = self.forward_sampling

    def get_most_likely(self):
        return self.best_choice

    def update_theta(self, losses):
        self.asng.update_theta(self.c_cats, self.c_ints, losses)

    def get_thetas(self):
        thetas = self.asng.get_thetas()
        return thetas

    def set_thetas(self, theta_cat, theta_int):
        self.asng.set_thetas(theta_cat, theta_int)

    def best_choice(self):
        return self.best_choice

    def forward_fix(self, inputs):
        outputs = self._forward(inputs, self.best_choice)
        return outputs

    def forward_sampling(self, inputs):
        outputs = []
        self.c_cats, self.c_ints = self.asng.sampling()

        for c_cat in self.c_cats:
            choice = self.convert_cat_to_hps_choice(c_cat)
            o = self._forward(inputs, choice)
            outputs.append(o)
        return outputs

    def _forward(self, inputs, choice):
        outputs = []
        caches = [None] * self._num_outputs

        for index, (sub_model, task_id) in enumerate(zip(self._sub_models, self.task_ids)):
            sub_model.choice(choice[task_id])
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
                    torch.unsqueeze(tensor_input, 1) for tensor_input in tensor_inputs
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
