import os

import numpy as np

from multiml import logger
from multiml.agent.keras import KerasConnectionGridSearchAgent
from multiml.agent.metric import MSEMetric
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
    storegate.add_data(var_names=('var0', 'var1', 'var2', 'var3', 'var4'),
                       data=data,
                       phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def test_agent_keras_connection_grid_scan():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # logger.set_level(logger.DEBUG)

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
        'save_weights': True,
        # MLPTask
        'activation': 'relu',
        'activation_last': 'sigmoid',
    }

    subtask0 = MLPTask(name='subtask0',
                       input_var_names=('var0', 'var1'),
                       output_var_names=('output0', 'output1'),
                       true_var_names=('var2', 'var3'),
                       layers=[4, 2],
                       batch_norm=True,
                       **args_mlptask)
    subtask1 = MLPTask(name='subtask1',
                       input_var_names=('var0', 'var1'),
                       output_var_names=('output0', 'output1'),
                       true_var_names=('var2', 'var3'),
                       layers=[4, 2],
                       **args_mlptask)
    subtask2 = MLPTask(name='subtask2',
                       input_var_names=('var2', 'var3'),
                       output_var_names=('output2'),
                       true_var_names=('label'),
                       layers=[4, 1],
                       **args_mlptask)
    subtask3 = MLPTask(name='subtask3',
                       input_var_names=('var2', 'var3'),
                       output_var_names=('output2'),
                       true_var_names=('label'),
                       layers=[4, 1],
                       **args_mlptask)

    task_scheduler = TaskScheduler()
    task_scheduler.add_task('step0')
    task_scheduler.add_task('step1', parents=['step0'])
    task_scheduler.add_subtask('step0', 'subtask0', env=subtask0)
    task_scheduler.add_subtask('step0', 'subtask1', env=subtask1)
    task_scheduler.add_subtask('step1', 'subtask2', env=subtask2)
    task_scheduler.add_subtask('step1', 'subtask3', env=subtask3)
    task_scheduler.show_info()

    import copy
    task_scheduler0 = copy.deepcopy(task_scheduler)
    task_scheduler1 = copy.deepcopy(task_scheduler)
    task_scheduler2 = copy.deepcopy(task_scheduler)

    metric = MSEMetric(
        storegate=storegate,
        pred_var_name="output2",
        true_var_name="label",
    )

    agent_args = {
        # BaseAgent
        'saver': saver,
        'storegate': storegate,
        'task_scheduler': task_scheduler0,
        'metric': metric,
        # KerasConnectionSimpleAgent
        'freeze_model_weights': False,
        'do_pretraining': True,
        'connectiontask_name': 'connectiontask',
        'connectiontask_args': {
            'num_epochs': 2,
            'max_patience': 1,
            'batch_size': 1200,
            'save_weights': True,
            'phases': None,
            'loss_weights': [0.5, 0.5],
            'optimizer': 'adam',
            'optimizer_args': dict(learning_rate=1e-3),
            'variable_mapping': [('var2', 'output0'), ('var3', 'output1')],
            'run_eagerly': True,
        },
        # KerasConnectionGridScanAgent
        'reuse_pretraining': True,
    }

    agent = KerasConnectionGridSearchAgent(**agent_args)
    agent.execute()
    agent.finalize()

    agent.get_best_result()

    # Test reuse_pretraining = False
    agent_args2 = {**agent_args}
    agent_args2['reuse_pretraining'] = False
    agent_args2['saver'] = saver
    agent_args2['task_scheduler'] = task_scheduler1
    agent2 = KerasConnectionGridSearchAgent(**agent_args2)
    agent2.execute()
    agent2.finalize()

    # Load saved model
    for (task_id, subtask_id) in [
        ('step0', 'subtask0'),
        ('step0', 'subtask1'),
        ('step1', 'subtask2'),
        ('step1', 'subtask3'),
    ]:
        # break
        task = task_scheduler2.get_subtask(task_id, subtask_id).env
        task._save_weights = False
        task._load_weights = False
        task._phases = ['test']
    agent_args['connectiontask_args']['save_weights'] = False
    agent_args['connectiontask_args']['load_weights'] = True
    agent_args['saver'] = saver
    agent_args['task_scheduler'] = task_scheduler2
    agent_args['connectiontask_args']['phases'] = ['test']
    agent2 = KerasConnectionGridSearchAgent(**agent_args)
    agent2.execute()
    agent2.finalize()

    task_scheduler0.show_info()
    task_scheduler2.show_info()

    for (task_id, subtask_id) in [
        ('step0', 'subtask0'),
        ('step0', 'subtask1'),
        ('step1', 'subtask2'),
        ('step1', 'subtask3'),
        ('connection', 'connectiontask.0'),
    ]:
        task0 = task_scheduler0.get_subtask(task_id, subtask_id).env
        task1 = task_scheduler2.get_subtask(task_id, subtask_id).env
        y_pred = task0.predict(phase='test')
        y_pred_load = task1.predict(phase='test')
        for y, y_load in zip(y_pred, y_pred_load):
            assert (np.array_equal(y, y_load))


if __name__ == '__main__':
    test_agent_keras_connection_grid_scan()
