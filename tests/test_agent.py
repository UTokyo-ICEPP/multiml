from test_task import MyTask

from multiml import logger
from multiml.agent.basic import GridSearchAgent
from multiml.agent.basic import RandomSearchAgent
from multiml.agent.metric import RandomMetric
from multiml.hyperparameter import Hyperparameters
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task_scheduler import TaskScheduler


def test_agent():
    env0 = MyTask()
    env1 = MyTask()
    env2 = MyTask()
    env3 = MyTask()

    hps_dict = {'hp_layer': [5, 10, 15], 'hp_epoch': [128, 256, 512]}
    hps = Hyperparameters(hps_dict)

    task_scheduler = TaskScheduler(['step0', 'step1'])

    task_scheduler.add_subtask('step0', 'task0', env=env0, hps=hps)
    task_scheduler.add_subtask('step0', 'task1', env=env1)

    task_scheduler.add_subtask('step1', 'task2', env=env2, hps=hps)
    task_scheduler.add_subtask('step1', 'task3', env=env3)

    task_scheduler.show_info()
    task_scheduler.get_sorted_task_ids()

    assert task_scheduler.get_sorted_task_ids() == ['step0', 'step1']
    assert task_scheduler.get_subtask_ids('step0') == ['task0', 'task1']
    assert task_scheduler.get_children_task_ids('step0') == ['step1']
    assert task_scheduler.get_parents_task_ids('step1') == ['step0']
    assert task_scheduler.get_subtask('step0', 'task0').env == env0

    storegate = StoreGate(backend='numpy', data_id='test_agent')
    saver = Saver()
    metric = RandomMetric()

    logger.set_level(logger.DEBUG)
    agent = GridSearchAgent(saver=saver,
                            storegate=storegate,
                            task_scheduler=task_scheduler,
                            metric=metric,
                            dump_all_results=True)

    assert agent._storegate is storegate
    assert agent._saver is saver
    assert agent._task_scheduler is task_scheduler
    assert agent._metric is metric

    agent.storegate = storegate
    agent.saver = saver
    agent.task_scheduler = task_scheduler
    agent.metric = metric

    assert agent.storegate is storegate
    assert agent.saver is saver
    assert agent.task_scheduler is task_scheduler
    assert agent.metric is metric

    agent.execute()
    agent.finalize()

    best_result = agent.result
    assert best_result['metric_value'] > 0


if __name__ == '__main__':
    test_agent()
