""" Collection of pre-defined Metric classes.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from multiml import logger


class Metric(metaclass=ABCMeta):
    """ Abstraction class of metric calculation.
    """
    @abstractmethod
    def calculate(self):
        """ Return calculated metric.
        """


class BaseMetric(Metric):
    """ Base class of metric calculation.

    All Metric class need to inherit this ``BaseMetric`` class. Metric class is
    usually passed to *agent* class to calculate metric.

    Examples:
        >>> metric = BaseMetric(storegate=storegate,
        >>>                     pred_var_name='pred',
        >>>                     true_var_name='true')
        >>> metric.calculate()
    """
    def __init__(self,
                 storegate=None,
                 pred_var_name='pred',
                 true_var_name='true',
                 var_names=None,
                 phase='test',
                 data_id=None):
        """ Initialize the base class.

        Args:
            storegate (Storegate): ``Storegate`` class instance.
            pred_var_name (str): name of variable for predicted values.
            true_var_name (str): name of variable for true values.
            var_names (str): 'pred true' variable names for shortcut.
            phase (str): 'train' or 'valid' or 'test' phase to calculate metric.
            data_id (str): specify ``data_id`` of storegate.
        """
        if var_names is not None:
            pred_var_name, true_var_name = var_names.split()

        self._storegate = storegate
        self._pred_var_name = pred_var_name
        self._true_var_name = true_var_name
        self._phase = phase
        self._data_id = data_id
        self._name = 'base'
        self._type = 'min'

    def __repr__(self):
        result = f'{self.__class__.__name__}(pred_var_name={self._pred_var_name}, '\
                                           f'true_var_name={self._true_var_name}, '\
                                           f'phase={self._phase}, '\
                                           f'data_id={self._data_id})'
        return result

    @property
    def storegate(self):
        """ Returns storegate of the base metric.
        """
        return self._storegate

    @storegate.setter
    def storegate(self, storegate):
        """ Set storegate to the base metric.
        """
        self._storegate = storegate

    def calculate(self):
        """ Returns calculated metric. Users need to implement algorithms.
        """

    @property
    def name(self):
        """ Returns name of metric.
        """
        return self._name

    @property
    def pred_var_name(self):
        """ Returns pred_var_name of metric,
        """
        return self._pred_var_name

    @property
    def true_var_name(self):
        """ Returns true_var_name of metric.
        """
        return self._true_var_name

    @property
    def phase(self):
        """ Returns phase of metric.
        """
        return self._phase

    @property
    def data_id(self):
        """ Returns data_id of metric.
        """
        return self._data_id

    @property
    def type(self):
        """ Return type of metric (e.f. min).
        """
        return self._type

    def get_true_pred_data(self):
        """ Return true and pred data.

        If special variable *active* is available, only samples with active is
        True are selected.

        Returns:
            ndarray, ndarray: true and pred values.
        """
        if self._data_id is not None:
            self._storegate.set_data_id(self._data_id)

        y_true = self._storegate.get_data(phase=self._phase,
                                          var_names=self._true_var_name)

        y_pred = self._storegate.get_data(phase=self._phase,
                                          var_names=self._pred_var_name)

        if 'active' in self._storegate.get_var_names():
            metadata = self._storegate.get_data(phase=self._phase,
                                                var_names='active')

            y_true = y_true[metadata == True]
            y_pred = y_pred[metadata == True]

        return y_true, y_pred


class ZeroMetric(BaseMetric):
    """ A dummy metric class to return always zero.
    """
    def __init__(self, **kwargs):
        """ Initialize ZeroMetric
        """
        super().__init__(**kwargs)
        self._name = 'zero'
        self._type = 'min'

    def calculate(self):
        """ Returns zero
        """
        return 0


class RandomMetric(BaseMetric):
    """ A dummy metric class to return random value.
    """
    def __init__(self, **kwargs):
        """ Initialize RandomMetric
        """
        super().__init__(**kwargs)
        self._name = 'random'
        self._type = 'min'

    def calculate(self):
        """ Return random value.
        """
        import random
        return random.uniform(0, 1)


class ValueMetric(BaseMetric):
    """ A metric class to return a single value.
    """
    def __init__(self, **kwargs):
        """ Initialize ValueMetric
        """
        super().__init__(**kwargs)
        self._name = 'value'
        self._type = 'min'

    def calculate(self):
        """ Returns value.
        """
        if self._data_id is not None:
            self._storegate.set_data_id(self._data_id)

        value = self._storegate.get_data(phase=self._phase,
                                         var_name=self._pred_var_name)
        return value


class MSEMetric(BaseMetric):
    """ A metric class to return Mean Square Error.
    """
    def __init__(self, **kwargs):
        """ Initialize MSEMetric
        """
        super().__init__(**kwargs)
        self._name = 'mse'
        self._type = 'min'

    def calculate(self):
        """ Calculate MSE.
        """
        y_true, y_pred = self.get_true_pred_data()

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)

        return mse


class ACCMetric(BaseMetric):
    """ A metric class to return ACC.
    """
    def __init__(self, **kwargs):
        """ Initialize ACCMetric
        """
        super().__init__(**kwargs)
        self._name = 'acc'
        self._type = 'max'

    def calculate(self):
        """ Calculate ACC.
        """
        y_true, y_pred = self.get_true_pred_data()

        if len(y_true.shape) != 1:
            y_true = np.argmax(y_true, axis=1)

        if len(y_pred.shape) != 1:
            y_pred = np.argmax(y_pred, axis=1)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)

        return acc


class AUCMetric(BaseMetric):
    """ A metric class to return AUC.
    """
    def __init__(self, **kwargs):
        """ Initialize AUCMetric
        """
        super().__init__(**kwargs)
        self._name = 'auc'
        self._type = 'max'

    def calculate(self):
        """ Calculate AUC.
        """
        y_true, y_pred = self.get_true_pred_data()

        if len(y_pred.shape) != 1:
            y_pred = y_pred[:, 1]

        if any(np.isnan(y_pred)):
            logger.warn(
                "There is nan in prediction values for auc. Replace nan with zero"
            )
            np.nan_to_num(y_pred, copy=False, nan=0.0)

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)

        return roc_auc
