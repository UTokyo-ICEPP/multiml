import os

import numpy as np

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import EnsembleTask, MLPTask
from multiml.task_scheduler import TaskScheduler


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


def test_keras_ensemble():
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

    subtask0 = MLPTask(name='subtask0',
                       input_var_names=('var0', 'var1'),
                       output_var_names=('output0'),
                       true_var_names=('label'),
                       layers=[4, 1],
                       **args_mlptask)
    subtask1 = MLPTask(name='subtask1',
                       input_var_names=('var0', 'var1'),
                       output_var_names=('output0'),
                       true_var_names=('label'),
                       layers=[4, 1],
                       **args_mlptask)

    subtask0.execute()
    subtask1.execute()

    task = EnsembleTask(
        subtasks=[subtask0, subtask1],
        dropout_rate=None,
        individual_loss=True,
        individual_loss_weights=1.0,
        saver=saver,
        storegate=storegate,
        save_weights=True,
        # do_training=False,
        phases=["train", "valid", "test"],
    )

    assert task.get_inputs()[0].shape[1] == len(['var0', 'var1'])

    assert task.input_var_names == ('var0', 'var1')
    assert task.output_var_names == ('output0')

    task.execute()
    task.finalize()

    task2 = EnsembleTask(
        name='EnsembleTask2',
        subtasks=[subtask0, subtask1],
        dropout_rate=None,
        individual_loss=True,
        individual_loss_weights=1.0,
        saver=saver,
        storegate=storegate,
        phases=["test"],
        load_weights=saver.load_ml(task.get_unique_id())['model_path'],
    )
    task2.execute()
    task2.finalize()

    y_pred = task.predict(phase='test')
    y_pred_load = task2.predict(phase='test')
    assert (np.array_equal(y_pred, y_pred_load))


if __name__ == '__main__':
    test_keras_ensemble()
