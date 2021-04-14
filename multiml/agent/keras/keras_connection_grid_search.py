from ..basic import ConnectionGridSearchAgent

from . import KerasConnectionRandomSearchAgent


class KerasConnectionGridSearchAgent(KerasConnectionRandomSearchAgent,
                                     ConnectionGridSearchAgent):
    """ Keras implementation for ConnectionGridSearchAgent
    """
