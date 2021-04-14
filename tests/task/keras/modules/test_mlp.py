import os

import numpy as np

from multiml import logger
from multiml.task.keras.modules import MLPBlock


def test_keras_module_mlp():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    block = MLPBlock(
        layers=[2, 1],
        batch_norm=True,
    )

    input_tensor = np.random.normal(size=(3, 2))
    block(input_tensor)


if __name__ == '__main__':
    test_keras_module_mlp()
