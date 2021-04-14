import os

import numpy as np
import pytest

from multiml import logger
from multiml.agent.keras import KerasConnectionRandomSearchAgent
from multiml.agent.metric import RandomMetric
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.keras import MLPTask
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


def test_agent_keras_connection_simple():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logger.set_level(logger.DEBUG)

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
                       batch_norm=True,
                       **args_mlptask)
    subtask1 = MLPTask(name='subtask1',
                       input_var_names=['var2', 'var3'],
                       output_var_names=['output2'],
                       true_var_names=['label'],
                       layers=[4, 1],
                       **args_mlptask)

    task_scheduler = TaskScheduler()
    task_scheduler.add_task('step0')
    task_scheduler.add_task('step1', parents=['step0'])
    task_scheduler.add_subtask('step0', 'subtask0', env=subtask0)
    task_scheduler.add_subtask('step1', 'subtask1', env=subtask1)
    task_scheduler.show_info()

    import copy
    task_scheduler0 = copy.deepcopy(task_scheduler)
    task_scheduler1 = copy.deepcopy(task_scheduler)

    metric = RandomMetric()

    agent_args = {
        # BaseAgent
        'saver': saver,
        'storegate': storegate,
        'task_scheduler': task_scheduler0,
        'metric': metric,
        # KerasConnectionSimpleAgent
        'freeze_model_weights': False,
        'do_pretraining': False,
        'connectiontask_name': 'connectiontask',
        'connectiontask_args': {
            'num_epochs': 2,
            'max_patience': 1,
            'batch_size': 1200,
            'save_weights': True,
            'phases': None,
            'use_multi_loss': True,
            'loss_weights': [0.5, 0.5],
            'optimizer': 'adam',
            'optimizer_args': dict(learning_rate=1e-3),
            'variable_mapping': [('var2', 'output0'), ('var3', 'output1')],
            'run_eagerly': True,
        },
    }

    agent = KerasConnectionRandomSearchAgent(**agent_args)
    agent.execute()
    agent.finalize()

    # Get the 1st task 1st model
    subtask = agent._task_prod[0][0]

    # Apply pretraining
    agent._execute_subtask(subtask, is_pretraining=True)

    # Set all variables in keras model to trainable
    KerasConnectionRandomSearchAgent._set_trainable_flags(subtask.env._model,
                                                    do_trainable=True)

    # Build connect-model
    agent._build_connected_models(
        subtasks=[agent._task_prod[0][0].env, agent._task_prod[1][0].env],
        job_id='dummy',
        use_task_scheduler=False)

    # Load saved model
    for (task_id, subtask_id) in [
        ('step0', 'subtask0'),
        ('step1', 'subtask1'),
    ]:
        task = task_scheduler1.get_subtask(task_id, subtask_id).env
        task._save_weights = False
        task._load_weights = False
        task._phases = ['test']
    agent_args['connectiontask_args']['save_weights'] = False
    agent_args['connectiontask_args']['load_weights'] = True
    #agent_args['saver'] = Saver()
    agent_args['saver'] = saver
    agent_args['task_scheduler'] = task_scheduler1
    agent_args['connectiontask_args']['phases'] = ['test']
    agent2 = KerasConnectionRandomSearchAgent(**agent_args)
    agent2.execute()
    agent2.finalize()

    task_scheduler0.show_info()
    task_scheduler1.show_info()
    task0 = task_scheduler0.get_subtask('connection', 'connectiontask').env
    task1 = task_scheduler1.get_subtask('connection', 'connectiontask').env
    y_pred = task0.predict(phase='test')
    y_pred_load = task1.predict(phase='test')
    assert (np.array_equal(y_pred[0], y_pred_load[0]))
    assert (np.array_equal(y_pred[1], y_pred_load[1]))


if __name__ == '__main__':
    test_agent_keras_connection_simple()
