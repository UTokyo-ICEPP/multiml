import os

import numpy as np
import pytest

from multiml import logger
from multiml.task.keras.modules import EnsembleModel, MLPBlock


def test_keras_module_ensemble():
    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    subblock1 = MLPBlock(layers=[2, 1])
    subblock2 = MLPBlock(layers=[2, 1])

    model1 = EnsembleModel(models=[subblock1, subblock2],
                           prefix='test',
                           ensemble_type='linear',
                           individual_loss=True)
    model2 = EnsembleModel(
        models=[subblock1, subblock2],
        prefix='test',
        ensemble_type='softmax',
    )

    input_tensor = np.random.normal(size=(3, 2))
    model1(input_tensor)
    model2(input_tensor)

    with pytest.raises(ValueError):
        EnsembleModel(
            models=[subblock1, subblock2],
            prefix='test',
            ensemble_type='dummy_type',
        )


if __name__ == '__main__':
    test_keras_module_ensemble()
