import os

import numpy as np
import pytest
from tensorflow.keras.optimizers import Adam

from multiml import logger
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import EnsembleTask, MLPTask
from multiml.task.keras.modules import DARTSModel, SumTensor
from multiml.task_scheduler import TaskScheduler


def build_storegate():
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='test_keras')
    data = np.random.normal(size=(100, 5))
    label = np.random.binomial(n=1, p=0.5, size=(100, ))
    phase = (0.8, 0.1, 0.1)
    storegate.add_data(var_names=('var0', 'var1', 'var2', 'var3', 'var4'),
                       data=data,
                       phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_keras_module_darts_model_sum_tensor():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    sum_tensor = SumTensor(name='test_sum')

    with pytest.raises(ValueError):
        sum_tensor.result()

    sum_tensor(0.1)
    sum_tensor(0.2)
    sum_tensor.result()


def test_keras_module_darts_model():
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

    subtask0 = MLPTask(input_var_names=('var0', 'var1'),
                       output_var_names=('output0', 'output1'),
                       true_var_names=('var2', 'var3'),
                       layers=[4, 2],
                       **args_mlptask)
    subtask1 = MLPTask(input_var_names=('var0', 'var1'),
                       output_var_names=('output0', 'output1'),
                       true_var_names=('var2', 'var3'),
                       layers=[4, 2],
                       **args_mlptask)
    subtask2 = MLPTask(input_var_names=('output0', 'output1'),
                       output_var_names=('output2'),
                       true_var_names=('label'),
                       layers=[4, 1],
                       **args_mlptask)
    subtask3 = MLPTask(input_var_names=('output0', 'output1'),
                       output_var_names=('output2'),
                       true_var_names=('label'),
                       layers=[4, 1],
                       **args_mlptask)
    subtask0.execute()
    subtask1.execute()
    subtask2.execute()
    subtask3.execute()

    task0 = EnsembleTask(
        subtasks=[subtask0, subtask1],
        dropout_rate=None,
        saver=saver,
        storegate=storegate,
        phases=['test'],
    )
    task1 = EnsembleTask(
        subtasks=[subtask2, subtask3],
        dropout_rate=None,
        saver=saver,
        storegate=storegate,
        phases=['test'],
    )
    task0.execute()
    task1.execute()

    model = DARTSModel(
        # DARTSModel
        optimizer_alpha='adam',
        optimizer_weight='adam',
        learning_rate_alpha=1e-3,
        learning_rate_weight=1e-3,
        zeta=1e-3,
        # ConnectionModel
        models=[task0.ml.model, task1.ml.model],
        input_var_index=[[0, 1], [-1, -2]],
        output_var_index=[[0, 1], [2]])

    # Directly use objest for optimizer instead of str
    DARTSModel(
        # DARTSModel
        optimizer_alpha=Adam(learning_rate=0.001),
        optimizer_weight=Adam(learning_rate=0.001),
        learning_rate_alpha=-1,  # dummy
        learning_rate_weight=-1,  # dummy
        zeta=1e-3,
        # ConnectionModel
        models=[task0.ml.model, task1.ml.model],
        input_var_index=[[0, 1], [-1, -2]],
        output_var_index=[[0, 1], [2]])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        run_eagerly=True,
    )

    input_tensor_train = [np.random.normal(size=(10, 1)), np.random.normal(size=(10, 1))]
    input_tensor_test = [np.random.normal(size=(10, 1)), np.random.normal(size=(10, 1))]
    output_tensor_train = [
        np.random.normal(size=(10, 2)),
        np.random.normal(size=(10, 1))
    ]
    output_tensor_test = [
        np.random.normal(size=(10, 2)),
        np.random.normal(size=(10, 1))
    ]

    model._batch_size_train.assign(2)
    model.fit(
        x=input_tensor_train,
        y=output_tensor_train,
        validation_data=(input_tensor_test, output_tensor_test),
        epochs=2,
    )

    alphas = model._get_variable_numpy(model.alpha_vars)
    assert alphas[0].shape == (2, 1)  # alphas of step0
    assert alphas[1].shape == (2, 1)  # alphas of step1

    best_submodel_index = model.get_index_of_best_submodels()
    assert len(best_submodel_index) == 2


if __name__ == '__main__':
    test_keras_module_darts_model_sum_tensor()
    test_keras_module_darts_model()
