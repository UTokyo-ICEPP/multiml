import tensorflow as tf

from multiml import logger

from tensorflow.keras import Model


class BaseModel(Model):
    def __init__(self, *args, **kwargs):
        """ Base model to overwrite train_step().
        """
        super().__init__(*args, **kwargs)

        self._output_var_names = None
        self._pred_var_names = None

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # select pred_var_names
            if self._pred_var_names is not None:
                y_pred = self.select_pred_data(y_pred)

            loss = self.compiled_loss(y,
                                      y_pred,
                                      regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def select_pred_data(self, y_pred):
        # do selection
        return
