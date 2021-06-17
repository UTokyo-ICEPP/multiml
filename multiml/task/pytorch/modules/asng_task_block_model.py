from torch.nn import Module


class ASNGTaskBlockModel(Module):
    def __init__(self, name, models, *args, **kwargs):
        '''

        Args:
            models (list(torch.nn.Module)): list of pytorch models for choiceblock
        '''
        super().__init__(*args, **kwargs)
        self._name = name
        from torch.nn import ModuleList

        self._models = models
        self._asng_task_block = ModuleList([])
        for model in self._models:
            self._asng_task_block.append(model)

        self._cat_idx = None
        self._n_subtask = len(self._asng_task_block)

    def n_subtask(self):
        return self._n_subtask

    def set_prob(self, c_cat, c_int):
        self._cat_idx = c_cat.argmax(axis=0)

    def forward(self, x):
        output = self._asng_task_block[self._cat_idx](x)

        # o0 = self._asng_task_block[0](x)
        # o1 = self._asng_task_block[1](x)
        # o2 = self._asng_task_block[2](x)

        # print(f'o0,o1,o2 = {o0[0][0]:.2e}, {o1[0][0]:.2e}, {o2[0][0]:.2e}')

        return output
