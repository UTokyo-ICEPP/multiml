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
            this module doesn't support hps optimization... : TODO
        '''
        super().__init__(*args, **kwargs)
        self._name = name
        from torch.nn import ModuleList

        self._models = models
        self._asng_task_block = ModuleList([])
        self._hps_params = {}
        
        for model in self._models:
            self._asng_task_block.append(model)
        self._hps_params[self._name] = len(self._models)
        self._cat_idx = None

        
    def get_hps_parameters(self):
        return self._hps_params
    
    def choice(self):
        return self._choice
    
    def choice(self, choice):
        # choice should be int
        self._choice = choice
    
    def forward(self, x):
        output = self._asng_task_block[self._choice[self._name]](x)
        return output
