import numpy as np
import tensorflow as tf

from multiml import logger


class AlphaDumperCallback(tf.keras.callbacks.Callback):
    """Dump alpha values in DARTS training.

    Attributes:
        alpha_history: list of alpha values on each epoch
        model: instance of `keras.models.Model`.
               Reference of the model being trained.
               (member variable of Callback class)
    """
    def __init__(self):
        super().__init__()

        self._alpha_history = None

    @staticmethod
    def formatting(var):
        """Format tensor for display.

        Args:
            var (Tensor): Tensor

        Returns:
            str: formatted alpha values
        """
        return str(list(var.numpy().reshape(-1)))

    def on_train_begin(self, logs=None):
        for var in self.model.alpha_vars:
            logger.debug(f"Initial alpha {var.name} = {self.formatting(var)}")

        self._alpha_history = [[] for _ in range(len(self.model.alpha_vars))]

    def on_epoch_end(self, epoch, logs=None):
        logger.debug('')
        for i, var in enumerate(self.model.alpha_vars):
            logger.debug(f"epoch = {epoch}: alpha {var.name} = {self.formatting(var)}")
            self._alpha_history[i].append(var.numpy().reshape(-1))
        logger.debug('')
        logger.debug('')

    # def on_train_batch_begin(self, batch, logs=None):
    #     logger.debug('')
    #     for i, var in enumerate(self.model.alpha_vars):
    #         logger.debug(
    #             f"batch = {batch} begin: alpha {var.name} = {self.formatting(var)}")
    #         self._alpha_history[i].append(var.numpy().reshape(-1))
    #     logger.debug('')
    #     logger.debug('')

    # def on_train_batch_end(self, batch, logs=None):
    #     logger.debug('')
    #     for i, var in enumerate(self.model.alpha_vars):
    #         logger.debug(
    #             f"batch = {batch} end: alpha {var.name} = {self.formatting(var)}")
    #         self._alpha_history[i].append(var.numpy().reshape(-1))
    #     logger.debug('')
    #     logger.debug('')

    def on_train_end(self, logs=None):
        for var in self.model.alpha_vars:
            logger.debug(f"DARTS final alpha {var.name} = {self.formatting(var)}")

    def get_alpha_history(self):
        return self._alpha_history


class EachLossDumperCallback(tf.keras.callbacks.Callback):
    """Dump each loss values in DARTS training.

    Attributes:
        loss_history: list of loss values on each epoch
        model: instance of `keras.models.Model`.
               Reference of the model being trained.
               (member variable of Callback class)
    """
    def __init__(self):
        super().__init__()

        self._loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        loss = []
        for metric in self.model.compiled_loss.metrics:
            loss.append(metric.result().numpy().reshape(-1))
        self._loss_history.append(loss)

    def get_loss_history(self):
        values = np.array(self._loss_history)  # batch, losses, dim of loss
        return values


class NaNKillerCallback(tf.keras.callbacks.Callback):
    """Stop training when nan is found in alphas."""
    def on_epoch_end(self, epoch, logs=None):
        if self._isnan(self.model.alpha_vars):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    @staticmethod
    def _isnan(alphas):
        for v in alphas:
            if not isinstance(v, np.ndarray):
                v = tf.keras.backend.eval(v)

            if any(np.isnan(v)):
                return True

        return False
