import os

import numpy as np

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import MLPTask, ModelConnectionTask
from multiml.task_scheduler import TaskScheduler


def build_storegate():
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='test_keras')
    data = np.random.normal(size=(100, 5))
    label = np.random.binomial(n=1, p=0.5, size=(100, ))
    phase = (0.8, 0.1, 0.1)
    storegate.add_data(var_names=['var0', 'var1', 'var2', 'var3', 'var4'],
                       data=data,
                       phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_keras_model_connection():
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
                       input_var_names=['var0', 'var1'],
                       output_var_names=['output0', 'output1'],
                       true_var_names=['var2', 'var3'],
                       layers=[4, 2],
                       **args_mlptask)
    subtask1 = MLPTask(name='subtask1',
                       input_var_names=['var2', 'var3'],
                       output_var_names=['output2'],
                       true_var_names=['label'],
                       layers=[4, 1],
                       **args_mlptask)

    subtask0.execute()
    subtask1.execute()

    chpt_path = f"{saver.save_dir}/test_keras_model_connection"
    task = ModelConnectionTask(
        # ConnectionModel
        subtasks=[subtask0, subtask1],
        use_multi_loss=True,
        loss_weights=[0.5, 0.5],
        variable_mapping=[('var2', 'output0'), ('var3', 'output1')],
        # KerasBaseTask
        saver=saver,
        storegate=storegate,
        optimizer='adam',
        num_epochs=2,
        max_patience=1,
        loss='binary_crossentropy',
        run_eagerly=True,
        load_weights=False,
        save_weights=chpt_path,
        phases=['train', 'valid', 'test'],
    )

    task.execute()
    task.finalize()

    # Load model weights
    task2 = ModelConnectionTask(
        # ConnectionModel
        subtasks=[subtask0, subtask1],
        use_multi_loss=True,
        loss_weights=[0.5, 0.5],
        variable_mapping=[('var2', 'output0'), ('var3', 'output1')],
        # KerasBaseTask
        saver=saver,
        storegate=storegate,
        optimizer='adam',
        num_epochs=2,
        max_patience=1,
        loss='binary_crossentropy',
        run_eagerly=True,
        save_weights=False,
        load_weights=chpt_path,
        phases=['test'],
    )

    task2.execute()
    task2.finalize()

    y_pred = task.predict(phase='test')
    y_pred_load = task2.predict(phase='test')
    assert (np.array_equal(y_pred[0], y_pred_load[0]))
    assert (np.array_equal(y_pred[1], y_pred_load[1]))

    # Connect as series dag
    task3 = ModelConnectionTask(
        # ConnectionModel
        subtasks=[subtask0, subtask1],
        use_multi_loss=True,
        loss_weights=[0.5, 0.5],
        variable_mapping=[('var2', 'output0'), ('var3', 'output1')],
        # KerasBaseTask
        saver=Saver(),
        storegate=storegate,
        optimizer='adam',
        num_epochs=2,
        max_patience=1,
        loss='binary_crossentropy',
        run_eagerly=True,
        load_weights=False,
        save_weights=chpt_path,
        phases=['train', 'valid', 'test'],
    )
    task3.execute()
    task3.finalize()


if __name__ == '__main__':
    test_keras_model_connection()
