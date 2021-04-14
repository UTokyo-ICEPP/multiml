from ..basic import ConnectionGridSearchAgent

from . import PytorchConnectionRandomSearchAgent


class PytorchConnectionGridSearchAgent(PytorchConnectionRandomSearchAgent,
                                       ConnectionGridSearchAgent):
    """ Pytorch implementation for ConnectionGridSearchAgent
    """
