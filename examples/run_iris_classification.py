from multiml import StoreGate
from multiml.agent import GridScanAgent
from multiml.task import SkleanPipelineTask
from multiml.task.pytorch import PytorchClassificationTask

def run_iris_classification():
    # load data
    import multiml.data.data_util as dutil
    sg = StoreGate(data_id='iris')
    dutil.load_iris(sg, var_names='data true', phase=(0.8, 0.1, 0.1))
    sg.show_info()

    # user defined tasks
    from sklearn.feature_selection import SelectKBest, chi2
    step0 = SkleanPipelineTask(var_names='data data_k true', 
                               model=SelectKBest,
                               model_args=dict(score_func=chi2))

    step1 = PytorchClassificationTask(var_names='data_k pred true',
                               model='MLPBlock', 
                               model_args=dict(layers=[3, 3], activation='ReLU'),
                               optimizer='SGD', 
                               loss='CrossEntropyLoss')

    hps_step0 = dict(model__k=[1, 2, 3, 4])
    hps_step1 = dict(model__input_shape=['saver__tmpkey__model\_\_k'], 
                     optimizer__lr=[1e-2, 1e-3, 1e-4, 1e-5])

    steps = [[(step0, hps_step0)], [(step1, hps_step1)]]

    # agent optimization
    agent = GridScanAgent(storegate=sg, task_scheduler=steps, metric='ACC')
    agent.execute_finalize()


if __name__ == '__main__':
    run_iris_classification()
