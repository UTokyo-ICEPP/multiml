import os

import numpy as np
import pytest

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import MLPTask


def build_storegate():
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='test_keras_mlp')
    data0 = np.random.normal(size=(100, 2))
    label = np.random.binomial(n=1, p=0.5, size=(100, ))
    phase = (0.8, 0.1, 0.1)
    storegate.add_data(var_names=('var0', 'var1'), data=data0, phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_keras_mlp():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    storegate = build_storegate()

    saver = Saver()

    args_task = {
        # BaseTask
        'saver': saver,
        'storegate': storegate,
        # MLBaseTask
        'phases': None,
        'save_weights': True,
        # KerasBaseTask
        'input_var_names': ('var0', 'var1'),
        'output_var_names': 'output0',
        'true_var_names': 'label',
        'optimizer': 'adam',
        'num_epochs': 2,
        'max_patience': 1,
        'loss': 'binary_crossentropy',
        'run_eagerly': True,
        # MLPTask
        'layers': [8, 1],
        'activation': 'relu',
        'activation_last': 'sigmoid',
        'batch_norm': True,
    }
    task = MLPTask(**args_task)
    assert task._layers == [8, 1]

    task.set_hps({
        'layers': [4, 1],
        'input_shapes': None,
        'output_var_names': 'output0'
    })
    assert task._layers == [4, 1]
    assert task._input_shapes == [len(['var0', 'var1'])]
    assert task.get_inputs()[0].shape[1] == len(['var0', 'var1'])

    task.execute()
    task.finalize()

    # Validate model save/load
    args_task['phases'] = ['test']
    args_task['save_weights'] = False
    args_task['load_weights'] = saver.load_ml(task._name)['model_path']
    args_task['layers'] = [4, 1]
    task2 = MLPTask(**args_task)

    # Fail due to call models before defining the model
    with pytest.raises(ValueError):
        y_pred_load  = task2.predict(phase='test')

    task2.execute()
    task2.finalize()

    y_pred = task.predict(phase='test')
    y_pred_load  = task2.predict(phase='test')
    # assert (np.sum(np.square(y_pred - y_pred_load)) < 1e-10)
    assert (np.array_equal(y_pred, y_pred_load))


if __name__ == '__main__':
    test_keras_mlp()
