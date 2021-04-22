class MLEnv:
    """ Data class to store compiled ML objects.
    """
    def __init__(
        self,
        model=None,
        optimizer=None,
        loss=None,
        loss_weights=None,
    ):
        """ Initialize MLEnv
        """
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._loss_weights = loss_weights

    def clear(self):
        self._model = None
        self._optimizer = None
        self._loss = None
        self._loss_weights = None

    @property
    def model(self):
        """ Returns model.
        """
        return self._model

    @model.setter
    def model(self, model):
        """ Set model.
        """
        self._model = model

    @property
    def optimizer(self):
        """ Returns optimizer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """ Set optimizer.
        """
        self._optimizer = optimizer

    @property
    def loss(self):
        """ Returns loss.
        """
        return self._loss

    @loss.setter
    def loss(self, loss):
        """ Set loss.
        """
        self._loss = loss

    @property
    def loss_weights(self):
        """ Returns loss_weights.
        """
        return self._loss_weights

    @loss_weights.setter
    def loss_weights(self, loss_weights):
        """ Set loss.
        """
        self._loss_weights = loss_weights
