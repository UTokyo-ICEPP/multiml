from torch.nn import Module


class ChoiceBlockModel(Module):
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
