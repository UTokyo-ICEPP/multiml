from multiml.agent.basic import BaseAgent
from multiml.agent.metric import RandomMetric
from multiml.saver import Saver
from multiml.storegate import StoreGate
from multiml.task_scheduler import TaskScheduler


def test_agent_basic_base():

    saver = Saver()
    storegate = StoreGate(backend='numpy', data_id='test_agent')
    task_scheduler = TaskScheduler()
    metric = RandomMetric()

    agent = BaseAgent()
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

    print(agent)


if __name__ == '__main__':
    test_agent_basic_base()
