import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.engine import data_adapter

from multiml import logger


class FunctionalModel(Model):
    def __init__(self, *args, **kwargs):
        """ Base model to overwrite train_step().

        TODO: this class is to avoid mix of functional API and subclass.
        """
        super().__init__(*args, **kwargs)

        self._pred_index = None

        logger.warn('FunctionalModel assumes that Functional API is used to '\
                    'create a model. Pease consider to use Subclassing API '\
                    'instead to avoid a mix of different APIs.')

    def set_pred_index(self, pred_index):
        self._pred_index = pred_index

    def train_step(self, data):
        # original implementation:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py#L768

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # select pred_var_names
            if self._pred_index is not None:
                y_pred = self.select_pred_data(y_pred)

            loss = self.compiled_loss(y,
                                      y_pred,
                                      sample_weight,
                                      regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def select_pred_data(self, y_pred):
        if len(self._pred_index) == 1:
            return y_pred[self._pred_index[0]]
        else:
            return [y_pred[index] for index in self._pred_index]
