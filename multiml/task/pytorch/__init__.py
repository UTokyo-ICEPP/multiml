from multiml.task.pytorch.pytorch_base import PytorchBaseTask
from multiml.task.pytorch.pytorch_ddp import PytorchDDPTask
from multiml.task.pytorch.pytorch_classification import PytorchClassificationTask
from .pytorch_model_connection import ModelConnectionTask
from .pytorch_choice_block import PytorchChoiceBlockTask

__all__ = [
    'ModelConnectionTask',
    'PytorchChoiceBlockTask',
    'PytorchBaseTask',
    'PytorchDDPTask',
    'PytorchClassificationTask',
]
