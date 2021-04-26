import os

import numpy as np
import pytest

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import KerasBaseTask


def build_storegate():
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='test_keras_mlp')
    data0 = np.random.normal(size=(100, 2))
    label = np.random.binomial(n=1, p=0.5, size=(100, ))
    phase = (0.8, 0.1, 0.1)
    storegate.add_data(var_names=['var0', 'var1'], data=data0, phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_keras_base():
    KerasBaseTask.__abstractmethods__ = set()

    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    saver = Saver()

    from tensorflow.keras.callbacks import ReduceLROnPlateau
    my_cb = ReduceLROnPlateau(patience=1)
    task = KerasBaseTask(saver=saver, callbacks=['EarlyStopping', 'ModelCheckpoint', my_cb])
    task.load_metadata()  # Do nothing


if __name__ == '__main__':
    test_keras_base()
