from multiml import logger
from multiml.task.basic import BaseTask


class MyTask(BaseTask):
    def __init__(self):
        super().__init__()
        self._hp_layer = None
        self._hp_epoch = None

    @logger.logging
    def execute(self):
        for key, value in self.__dict__.items():
            logger.debug(f'{key} : {value}')


def test_mytask():
    logger.set_level(logger.DEBUG)
    task = MyTask()
    task.set_hps({'hp_layer': 5, 'hp_epoch': 256})
    assert task._hp_layer == 5
    assert task._hp_epoch == 256

    task.execute()
    task.finalize()


if __name__ == '__main__':
    test_mytask()
