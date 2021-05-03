import os

import numpy as np
import pytest

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import MLPTask
from multiml.task.keras.modules import ConnectionModel
from multiml.task_scheduler import TaskScheduler


def build_storegate():
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='test_keras_mlp')
    data = np.random.normal(size=(100, 6))
    label = np.random.binomial(n=1, p=0.5, size=(100, ))
    phase = (0.8, 0.1, 0.1)
    storegate.add_data(var_names=('var0', 'var1', 'var2', 'var3', 'var4', 'var5'),
                       data=data,
                       phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_keras_module_connection_model():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    saver = Saver()
    storegate = build_storegate()

    args_mlptask = {
        # BaseTask
        'saver': saver,
        'storegate': storegate,
        # KerasBaseTask
        'optimizer': 'adam',
        'num_epochs': 2,
        'max_patience': 1,
        'loss': 'mse',
        'run_eagerly': True,
        # MLPTask
        'activation': 'relu',
        'activation_last': 'sigmoid',
    }

    task0 = MLPTask(input_var_names=('var0', 'var1'),
                    output_var_names=('output0', 'output1'),
                    true_var_names=('var2', 'var3'),
                    layers=[4, 2],
                    **args_mlptask)
    task1 = MLPTask(input_var_names=('output0', 'output1'),
                    output_var_names=('output2'),
                    true_var_names=('var4',),
                    layers=[4, 1],
                    **args_mlptask)
    task2 = MLPTask(input_var_names=('output2',),
                    output_var_names=('output3',),
                    true_var_names=('label',),
                    layers=[4, 1],
                    **args_mlptask)
    task0.execute()
    task1.execute()
    task2.execute()

    model = ConnectionModel(models=[task0.ml.model, task1.ml.model, task2.ml.model],
                            input_var_index=[(0, 1), (-1, -2), (-3,)],
                            output_var_index=[(0, 1), (2,), (3,)])

    trainalbe_variables = model._get_variables()
    assert len(trainalbe_variables) > 0

    input_tensor = np.random.normal(size=(3, 2))
    model(input_tensor)

if __name__ == '__main__':
    test_keras_module_connection_model()
