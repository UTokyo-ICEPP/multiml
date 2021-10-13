import glob
import numpy as np

from multiml import StoreGate, logger


def load_iris(storegate,
              data_var_name='data',
              target_var_name='true',
              var_names=None,
              phase='train',
              shuffle=True):
    from sklearn.datasets import load_iris

    if var_names is not None:
        data_var_name, target_var_name = var_names.split()

    logger.info(f'Load iris dataset: {data_var_name}, {target_var_name}')

    iris_dataset = load_iris()
    data = iris_dataset.data.astype(np.float32)
    target = iris_dataset.target

    storegate.add_data(data_var_name, data, phase=phase, shuffle=shuffle)
    storegate.add_data(target_var_name, target, phase=phase, shuffle=shuffle)
    storegate.compile()


def load_boston(storegate, target_var_name='true', phase='train', shuffle=True):
    from sklearn.datasets import load_boston

    logger.info(f'Load boston dataset: {target_var_name}')

    boston_dataset = load_boston()

    feature_names = boston_dataset.feature_names.tolist()

    data = boston_dataset.data.astype(np.float32)
    target = boston_dataset.target.astype(np.float32)
    storegate.add_data(feature_names, data, phase=phase, shuffle=shuffle)
    storegate.add_data(target_var_name, target, phase=phase, shuffle=shuffle)
    storegate.compile()


def load_numpy(storegate, file_path, var_names=None, phase='train', dtype=None, shuffle=False):
    if isinstance(storegate, dict):
        storegate = StoreGate(**storegate)

    input_files = glob.glob(file_path)

    for input_file in input_files:
        logger.info(f'open {input_file}')

        data = np.load(input_file)

        if isinstance(phase, (tuple, list)):
            if sum(phase) > 1.:
                data = data[:sum(phase)]

        if dtype is not None:
            data = data.astype(dtype)

        if var_names is None:
            tmp_var_names = input_file.split('/')[-1]
        else:
            tmp_var_names = var_names

        storegate.add_data(tmp_var_names, data, phase)

    if shuffle:
        storegate.compile()
        storegate.shuffle()

    storegate.show_info()
