import os

import numpy as np
import pytest

from multiml import logger
from multiml.task.keras.modules import Conv2DBlock


def test_keras_module_conv2d():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    block = Conv2DBlock(
        layers_conv2d=[
            ('conv2d', {
                'filters': 4,
                'kernel_size': (2, 2)
            }),
            ('maxpooling2d', {
                'pool_size': (2, 2)
            }),
            ('upsampling2d', {
                'size': (2, 2)
            }),
        ],
        conv2d_padding='valid',
    )

    input_tensor = np.random.normal(size=(3, 3, 3, 2))
    block(input_tensor)


def test_keras_module_conv2d_invalid_layer():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    with pytest.raises(ValueError):
        Conv2DBlock(layers_conv2d=[('dummy_layers', {})])


if __name__ == '__main__':
    test_keras_module_conv2d()

    test_keras_module_conv2d_invalid_layer()
