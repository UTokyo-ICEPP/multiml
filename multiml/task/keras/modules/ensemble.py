from tensorflow.keras import Model


class EnsembleModel(Model):
    def __init__(self,
                 models,
                 prefix,
                 ensemble_type,
                 dropout_rate=None,
                 individual_loss=False,
                 *args,
                 **kwargs):
        """Constructor.

        Args:
            models (list(tf.keras.Model)): list of keras models for ensembling
            prefix (str): prefix for a layer's name
            ensemble_type (str): type of ensemble way (linear or softmax)
            dropout_rate (float): dropout rate. Valid only for ensemble_type = softmax
            individual_loss (bool): use multiple outputs
        """
        super().__init__(*args, **kwargs)

        self._models = models
        self._individual_loss = individual_loss

        if ensemble_type == 'linear':
            from tensorflow.keras.layers import Dense
            self.ensemble_layer = Dense(1, activation=None, name=f"{prefix}_ensemble_weights")
        elif ensemble_type == 'softmax':
            from . import SoftMaxDenseLayer
            self.ensemble_layer = SoftMaxDenseLayer(kernel_initializer='zeros',
                                                    dropout_rate=dropout_rate,
                                                    name=f"{prefix}_ensemble_weights")
        else:
            raise ValueError(
                f'ensemble_type should be linear or softmax. {ensemble_type} is given.')

    def call(self, inputs, training=False):
        from tensorflow import expand_dims, squeeze
        from tensorflow.keras.layers import Concatenate

        outputs = []
        for model in self._models:
            x = model(inputs)
            x = expand_dims(x, axis=-1)
            outputs.append(x)

        x = Concatenate(axis=-1)(outputs)
        x = self.ensemble_layer(x)

        x = squeeze(x, -1)

        if self._individual_loss:
            outputs = [squeeze(v, -1) for v in outputs]
            return [x] + outputs
        else:
            return x
