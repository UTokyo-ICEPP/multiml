import os

import numpy as np

from multiml import logger
from multiml.task.keras.modules import SoftMaxDenseLayer


def test_keras_module_softmax_dense_layer():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    layer1 = SoftMaxDenseLayer()
    layer2 = SoftMaxDenseLayer(dropout_rate=0.3)

    layer1.get_config()

    input_tensor = np.random.normal(size=(3, 2))
    layer1(input_tensor, training=None)
    layer2(input_tensor, training=True)


if __name__ == '__main__':
    test_keras_module_softmax_dense_layer()
