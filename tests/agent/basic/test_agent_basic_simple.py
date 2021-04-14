from multiml.agent.basic import RandomSearchAgent
from multiml.agent.metric import RandomMetric
from multiml.hyperparameter import Hyperparameters
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task.basic import BaseTask
from multiml.task_scheduler import TaskScheduler


def test_agent_basic_simple():

    saver = Saver()
    storegate = StoreGate(backend='numpy', data_id='test_agent')
    task_scheduler = TaskScheduler()
    metric = RandomMetric()

    subtask0 = BaseTask()
    subtask1 = BaseTask()

    task_scheduler.add_task('step0')
    task_scheduler.add_subtask('step0', 'task0', env=subtask0)
    task_scheduler.add_task('step1', parents=['step0'])
    task_scheduler.add_subtask('step1',
                               'task1',
                               env=subtask1,
                               hps=Hyperparameters({'job_id': [0, 1]}))

    agent = RandomSearchAgent(saver=saver,
                              storegate=storegate,
                              task_scheduler=task_scheduler,
                              metric=metric)
    agent.execute()
    agent.finalize()


if __name__ == '__main__':
    test_agent_basic_simple()
