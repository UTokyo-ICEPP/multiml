from tensorflow.keras.layers import Layer


class SoftMaxDenseLayer(Layer):
    def __init__(self,
                 kernel_initializer='zeros',
                 kernel_regularizer=None,
                 dropout_rate=None,
                 **kwargs):
        """ Constructor

        Args:
            kernel_initializer (str): initializer for softmax weights
            kernel_regularizer (str): regularizer for softmax weights
            dropout_rate (float): dropout rate
        """
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.units = 1
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

    def build(self, input_shape):
        from tensorflow.python.framework import tensor_shape

        self._last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(shape=(self._last_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='kernel',
                                      trainable=True)

        super().build(input_shape)

    def call(self, inputs, training=None):
        from tensorflow.python.keras import backend as K
        if training is None:
            training = K.learning_phase()

        w = K.softmax(self.kernel, axis=0)  # n_models, 1

        if training and (self.dropout_rate is not None):
            from tensorflow import matmul, ones, reshape, shape, stack, tile
            from tensorflow.math import divide_no_nan, reduce_sum
            from tensorflow.nn import dropout
            batch_size = shape(inputs)[0]
            dropout_layer_shape = stack(
                [batch_size, self._last_dim, self.units])
            dropout_layer = dropout(ones(dropout_layer_shape),
                                    rate=self.dropout_rate)
            dropout_sum = reduce_sum(dropout_layer, axis=1, keepdims=True)
            dropout_layer = divide_no_nan(dropout_layer, dropout_sum)
            dropout_layer *= self._last_dim
            w = reshape(tile(w, [batch_size, 1]), dropout_layer_shape)
            w = dropout_layer * w  # Batch, n_models, 1
            x = matmul(inputs, w)  # Batch, n_vars, 1
            return x
        else:
            return K.dot(inputs, w)

    def get_config(self):
        return {'kernel_initializer': self.kernel_initializer}
