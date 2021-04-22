import os

import numpy as np
import pytest

from multiml import logger
from multiml.saver import Saver
from multiml.task.keras.modules import MLPBlock


def test_keras_util():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    saver = Saver()

    model = MLPBlock(
        layers=[2, 1],
        batch_norm=True,
    )
    model.compile(optimizer='adam', loss='binary_crossentropy')

    x_train = np.random.normal(size=(100, 2))
    x_valid = np.random.normal(size=(100, 2))
    y_train = np.random.binomial(n=1, p=0.5, size=(100, ))
    y_valid = np.random.binomial(n=1, p=0.5, size=(100, ))
    chpt_path = f"{saver.save_dir}/test_keras_util"
    
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    my_cb = ReduceLROnPlateau(patience=1)
    from multiml.task.keras.keras_util import training_keras_model
    training_keras_model(model=model,
                         num_epochs=2,
                         batch_size=10,
                         max_patience=2,
                         x_train=x_train,
                         y_train=y_train,
                         x_valid=x_valid,
                         y_valid=y_valid,
                         chpt_path=chpt_path,
                         callbacks=['EarlyStopping', 'ModelCheckpoint', 'TensorBoard', my_cb],
                         tensorboard_path=f'{saver.save_dir}/test_keras_util')

    from multiml.task.keras.keras_util import get_optimizer
    get_optimizer('adam', dict(learning_rate=None))
    get_optimizer('adam', dict(learning_rate=0.001))
    get_optimizer('sgd', dict(learning_rate=None))
    get_optimizer('sgd', dict(learning_rate=0.1))
    from tensorflow.keras import optimizers
    get_optimizer(optimizers.Adam())

    with pytest.raises(ValueError):
        get_optimizer(None)

    with pytest.raises(NotImplementedError):
        get_optimizer('dummyoptimizer')


if __name__ == '__main__':
    test_keras_util()
