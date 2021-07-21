from torch.nn import Module


class SPOSChoiceBlockModel(Module):
    def __init__(self, models, *args, **kwargs):
        '''

        Args:
            models (list(torch.nn.Module)): list of pytorch models for choiceblock
        '''
        super().__init__(*args, **kwargs)
        self._name = 'ChoiceBlock'
        from torch.nn import ModuleList

        self._models = models
        self._len_task_candidate = len(models)
        self._choice_block = ModuleList([])
        for model in self._models:
            self._choice_block.append(model)
        self._choice = None

    @property
    def choice(self):
        return self._choice

    @choice.setter
    def choice(self, value):
        if value is not None:
            self._name = ('ChoiceBlock' + self._choice_block[value].__class__.__name__)
        else:
            self._name = 'ChoiceBlock'
        self._choice = value

    @staticmethod
    def _random_index(len_task_candidate):
        from numpy.random import randint
        return randint(len_task_candidate)

    def forward(self, x):
        if self._choice is None:
            choice = self._random_index(self._len_task_candidate)
        else:
            choice = self._choice
        output = self._choice_block[choice](x)
        return output
    
class ASNGChoiceBlockModel(Module):
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
        return output
