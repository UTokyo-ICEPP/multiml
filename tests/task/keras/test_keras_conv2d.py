import os

import numpy as np

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate


def build_storegate():
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='test_keras_conv2d')
    data0 = np.random.normal(size=(100, 3, 3, 1))
    label = np.random.binomial(n=1, p=0.5, size=(100, ))
    phase = (0.8, 0.1, 0.1)
    storegate.add_data(var_names='var0', data=data0, phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_keras_conv2d():
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
        'input_var_names': ('var0',),
        'output_var_names': ('output0',),
        'optimizer': 'adam',
        'num_epochs': 2,
        'max_patience': 1,
        'loss': 'binary_crossentropy',
        'run_eagerly': True,
        # MLPTask
        'true_var_names': ('label',),
        'layers': [4, 1],
        'input_shapes': (1, 3, 3, 1),
        'activation': 'relu',
        'activation_last': 'sigmoid',
        # Conv2DTask
        'conv2d_layers': [
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
    }
    from multiml.task.keras import Conv2DTask
    task = Conv2DTask(**args_task)

    task.execute()
    task.finalize()

    # Validate model save/load
    args_task['phases'] = ['test']
    args_task['save_weights'] = False
    args_task['load_weights'] = saver.load_ml(task._name)['model_path']
    task2 = Conv2DTask(**args_task)
    task2.execute()
    task2.finalize()

    y_pred = task.predict(phase='test')
    y_pred_load = task2.predict(phase='test')
    assert (np.array_equal(y_pred, y_pred_load))


if __name__ == '__main__':
    test_keras_conv2d()
