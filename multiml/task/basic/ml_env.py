from multiml import logger


class MLEnv:
    """ Data class to store compiled ML objects.
    """
    def __init__(
        self,
        model=None,
        optimizer=None,
        loss=None,
        loss_weights=None,
        multi_inputs=None,
        multi_outputs=None,
        multi_loss=None,
    ):
        """ Initialize MLEnv
        """
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._loss_weights = loss_weights
        self._multi_inputs = multi_inputs
        self._multi_outputs = multi_outputs
        self._multi_loss = multi_loss

    def clear(self):
        self._model = None
        self._optimizer = None
        self._loss = None
        self._loss_weights = None
        self._multi_inputs = None
        self._multi_outputs = None
        self._multi_loss = None

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

    @property
    def multi_inputs(self):
        """ Returns multi_inputs.
        """
        return self._multi_inputs

    @multi_inputs.setter
    def multi_inputs(self, multi_inputs):
        """ Set multi_inputs.
        """
        self._multi_inputs = multi_inputs

    @property
    def multi_outputs(self):
        """ Returns multi_outputs.
        """
        return self._multi_outputs

    @multi_outputs.setter
    def multi_outputs(self, multi_outputs):
        """ Set multi_outputs.
        """
        self._multi_outputs = multi_outputs

    @property
    def multi_loss(self):
        """ Returns multi_loss.
        """
        return self._multi_loss

    @multi_loss.setter
    def multi_loss(self, multi_loss):
        """ Set multi_loss.
        """
        self._multi_loss = multi_loss

    def show_info(self):
        """ Print information.
        """
        logger.debug(f'model = {self._model.__class__.__name__}')
        logger.debug(f'optimizer = {self._optimizer.__class__.__name__}')
        logger.debug(f'loss = {self._loss}')
        logger.debug(f'loss_weights = {self._loss_weights}')

    def validate(self, phase):
        """ Validate environment for given phase.
        """
        if phase in ('train', 'valid'):
            if self.model is None:
                raise AttributeError('model is not defined')

            if self.optimizer is None:
                raise AttributeError('optimizer is not defined')

            if self.loss is None:
                raise AttributeError('loss is not defined')

        elif phase == 'test':
            if self.model is None:
                raise AttributeError('model is not defined')
