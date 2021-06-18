""" MLBaseTask module.
"""
from multiml import logger, const
from multiml.task.basic import BaseTask, MLEnv


class MLBaseTask(BaseTask):
    """ Base task class for (deep) machine learning tasks.
    """
    def __init__(self,
                 phases=None,
                 input_var_names=None,
                 output_var_names=None,
                 pred_var_names=None,
                 true_var_names=None,
                 var_names=None,
                 model=None,
                 model_args=None,
                 optimizer=None,
                 optimizer_args=None,
                 loss=None,
                 loss_args=None,
                 max_patience=None,
                 loss_weights=None,
                 load_weights=False,
                 save_weights=False,
                 metrics=None,
                 num_epochs=10,
                 batch_size=64,
                 verbose=None,
                 **kwargs):
        """ Initialize ML base task.

        This base class will be inherited by deep learning task classes, ``KerasBaseTask()``
        and ``PytorchBaseTask()``. ``input_var_names`` and ``output_var_names`` specify data
        for model inputs and outputs. If ``input_var_names`` is list, e.g. ['var0', 'var1'],
        model will receive data with format of [(batch size, k), (batch size, k)], where k
        is arbitrary shape of each variable. If ``input_var_names`` is tuple,
        e.g. ('var0', 'var1'), model will receive data with (batch size, M, k), where M is
        the number of variables. If `output_var_names`` is list, model must returns list of
        tensor data for each variable. If `output_var_names`` is tuple, model must returns
        a tensor data. ``pred_var_names`` and ``true_var_names`` specify data for loss
        calculations. If ``pred_var_names`` is given, only variables indicated by
        ``pred_var_names`` are selected from model outputs before being passed to loss calculation.
        Please see ``KerasBaseTask()` or ``PytorchBaseTask()`` for actual examples.

        Args:
            phases (list): list to indicates ML phases, e.g. ['train', 'test'].
                If None is given, ['train', 'valid', 'test'] is set.
            input_var_names (str or list or tuple): input variable names in StoreGate.
            output_var_names (str or list or tuple): output variable names of model.
            pred_var_names (str or list): prediction variable names passed to loss.
            true_var_names (str or list or tuple): true variable names.
            var_names (str): str of "input output true" variable names for
                shortcut. This is not valid to specify multiple variables.
            model (str or obj): name of model, or class object of model.
            model_args (dict): args of model, e.g. dict(param0=0, param1=1).
            optimizer (str or obj): name of optimizer, or class object of optimizer
            optimizer_args (dict): args of optimizer.
            loss (str or obj): name of loss, or class object of loss
            loss_args (dict): args of loss.
            max_patience (int): max number of patience for early stopping.
                ``early_stopping`` is enabled if ```max_patience` is given.
            loss_weights (list): scalar coefficients to weight the loss.
            load_weights (bool or str): user defined algorithms should assume
                the following behavior. If False, not load model weights. If
                True, load model weights from default location. If str, load
                weights from given path.
            save_weights (bool or str): user defined algorithms should assume
                the following behavior. If False, not save model weights. If
                True, save model weights to default location. If str, save
                weights to given path.
            metrics (list): metrics of evaluation.
            num_epochs (int): number of epochs.
            batch_size (int or dict): size of mini batch, you can set different batch_size for test, train, valid.
            verbose (int): verbose option for fitting step. If None, it's set
                based on logger.MIN_LEVEL
        """
        super().__init__(**kwargs)

        if phases is None:
            phases = ['train', 'valid', 'test']

        if model_args is None:
            model_args = {}

        if optimizer_args is None:
            optimizer_args = {}

        if loss_args is None:
            loss_args = {}

        if var_names is not None:
            input_var_names, output_var_names, true_var_names = var_names.split(
            )

        self._ml = MLEnv()

        self._phases = phases
        self._input_var_names = input_var_names
        self._output_var_names = output_var_names
        self._pred_var_names = pred_var_names
        self._true_var_names = true_var_names
        self._var_names = var_names

        self._model = model
        self._model_args = model_args
        self._optimizer = optimizer
        self._optimizer_args = optimizer_args
        self._loss = loss
        self._loss_args = loss_args
        self._loss_weights = loss_weights
        self._max_patience = max_patience
        self._load_weights = load_weights
        self._save_weights = save_weights
        self._metrics = metrics
        self._num_epochs = num_epochs

        self._verbose = verbose
        self._batch_size = batch_size
        self._task_type = 'ml'

    def __repr__(self):
        result = f'{self.__class__.__name__}(task_type={self._task_type}, '\
                                           f'job_id={self._job_id}, '\
                                           f'phases={self._phases}, '\
                                           f'input_var_names={self._input_var_names}, '\
                                           f'output_var_names={self._output_var_names}, '\
                                           f'true_var_names={self._true_var_names})'
        return result

    def set_hps(self, params):
        """ Set hyperparameters to this task. 

        Class attributes (self._XXX) are automatically set based on keys and
        values of given dict. Hyperparameters start with 'model__',
        'optimizer__' and 'loss__' are considred as args of model, optimizer,
        loss, respectively. If value of hyperparameters is str and starts with
        'saver__', value is retrieved from ```Saver``` instance, please see
        exampels below.

        Args:
            params (dict): key and value of hyperparameters.

        Example:
            >>> hps_dict = {
            >>>    'num_epochs': 10, # normal hyperparameter    
            >>>    'optimizer__lr': 0.01 # hyperparameter of optimizer
            >>>    'saver_hp': 'saver__key__value' # hyperparamer from saver
            >>> }
            >>> task.set_hps(hps_dict)
        """
        for key, value in params.items():

            if isinstance(value, str) and value.startswith('saver__'):
                _, *params = value.split('__')

                for index, param in enumerate(params):
                    param = param.replace('\\', '')

                    if index == 0:
                        value = self._saver[param]
                    else:
                        value = value[param]

            if key.startswith('model__'):
                self._model_args[key.replace('model__', '')] = value

            elif key.startswith('optimizer__'):
                self._optimizer_args[key.replace('optimizer__', '')] = value

            elif key.startswith('loss__'):
                self._loss_args[key.replace('loss__', '')] = value

            else:
                if not hasattr(self, '_' + key):
                    raise AttributeError(f'{key} is not defined.')

                setattr(self, '_' + key, value)

        if self.saver is not None:
            self.saver[self.output_saver_key] = params

    @logger.logging
    def execute(self):
        """ Execute a task.
        """

        self.compile()

        result = None

        if const.TRAIN in self.phases:
            result = self.fit()

        if self._save_weights:
            self.dump_model(dict(result=result))

        if const.TEST in self.phases:
            self.predict_update()

    def fit(self, train_data=None, valid_data=None):
        """ Fit model.

        Args:
            train_data (ndarray): training data.
            valid_data (ndarray): validation data.
        """
        pass

    def predict(self, data=None):
        """ Predict model.

        Args:
            data (ndarray): prediction data.
        """
        pass

    def fit_predict(self, fit_args=None, predict_args=None):
        """ Fit and predict model.

        Args:
            fit_args (dict): arbitrary dict passed to ``fit()``.
            predict_args (dict): arbitrary dict passed to ``predict()``.

        Returns:
            ndarray or list: results of prediction.
        """
        if fit_args is None:
            fit_args = {}

        if predict_args is None:
            predict_args = {}

        self.fit(**fit_args)

        return self.predict(**predict_args)

    def predict_update(self, data=None):
        """ Predict and update data in StoreGate.

        Args:
            data (ndarray): data passed to ``predict()`` method.
        """
        self.storegate.update_data(data=self.predict(data=data),
                                   var_names=self._output_var_names,
                                   phase='auto')

    @property
    def phases(self):
        """ Returns ML phases.
        """
        return self._phases

    @phases.setter
    def phases(self, phases):
        """ Set ML phases.

        Args:
            phases (list): a list contains 'train' or 'valid' or 'test'.
        """
        self._phases = phases

    @property
    def input_var_names(self):
        """ Returns input_var_names.
        """
        return self._input_var_names

    @input_var_names.setter
    def input_var_names(self, input_var_names):
        """ Set input_var_names.
        """
        self._input_var_names = input_var_names

    @property
    def output_var_names(self):
        """ Returns output_var_names.
        """
        return self._output_var_names

    @output_var_names.setter
    def output_var_names(self, output_var_names):
        """ Set output_var_names.
        """
        self._output_var_names = output_var_names

    @property
    def pred_var_names(self):
        """ Returns pred_var_names.
        """
        return self._pred_var_names

    @pred_var_names.setter
    def pred_var_names(self, pred_var_names):
        """ Set pred_var_names.
        """
        self._pred_var_names = pred_var_names

    @property
    def true_var_names(self):
        """ Returns true_var_names.
        """
        return self._true_var_names

    @true_var_names.setter
    def true_var_names(self, true_var_names):
        """ Set true_var_names.
        """
        self._true_var_names = true_var_names

    @property
    def ml(self):
        """ Returns ML data class.
        """
        return self._ml

    @ml.setter
    def ml(self, ml):
        """ Set ML data class.

        Args:
            ml (MLEnv): ML data class.
        """
        self._ml = ml

    def compile(self):
        """ Compile model, optimizer and loss.

        Compiled objects will be avaialble via ``self.ml.model``, ``self.ml.optimizer``
        and ``self.ml.loss``.

        Examples:
            >>> # compile all together,
            >>> self.compile()
            >>> # which is equivalent to:
            >>> self.build_model() # set self._model
            >>> self.compile_model() # set self.ml.model
            >>> self.compile_optimizer() # set self.ml.optimizer
            >>> self.compile_loss() # set self.ml.loss
        """

        self.ml.clear()
        self.compile_var_names()

        if self._model is None:
            self.build_model()

        self.compile_loss()
        self.compile_model()
        self.compile_optimizer()

        # self.show_info()

    def build_model(self):
        """ Build model.
        """
        pass

    def compile_var_names(self):
        """ Compile var_names.
        """
        if isinstance(self.input_var_names, list):
            self.ml.multi_inputs = True
        else:
            self.ml.multi_inputs = False

        if isinstance(self.output_var_names, list):
            self.ml.multi_outputs = True
        else:
            self.ml.multi_outputs = False

        if isinstance(self.true_var_names, list):
            self.ml.multi_loss = True
        else:
            self.ml.multi_loss = False

    def compile_model(self):
        """ Compile model.
        """
        pass

    def compile_optimizer(self):
        """ Compile optimizer.
        """
        pass

    def compile_loss(self):
        """ Compile loss.
        """
        pass

    def load_model(self):
        """ Load pre-trained model path from ``Saver``.

        Returns:
            str: model path.
        """
        if isinstance(self._load_weights, str):
            model_path = self._load_weights

        elif self._load_weights is True:
            model_path = self.get_metadata('model_path')

        else:
            raise ValueError('Unsupported load_weights type')

        return model_path

    def dump_model(self, extra_args=None):
        """ Dump current model to ``saver``.

        Args:
            extra_args (dict): extra metadata to be stored together with model.
        """
        args_dump_ml = dict(key=self.get_unique_id())

        if isinstance(self._save_weights, str):
            args_dump_ml['model'] = self.ml.model
            args_dump_ml['model_path'] = self._save_weights

        elif self._save_weights is True:
            args_dump_ml['model'] = self.ml.model

        if extra_args is not None:
            args_dump_ml.update(extra_args)

        # to avoid overwrite
        if args_dump_ml['key'] in self.saver.keys():
            serial_id = 0
            while f'{args_dump_ml["key"]}-{serial_id}' in self.saver.keys():
                serial_id += 1
            args_dump_ml['key'] = f'{args_dump_ml["key"]}-{serial_id}'

        self._saver.dump_ml(**args_dump_ml)

    def load_metadata(self):
        """ Load metadata.
        """
        pass

    def get_input_true_data(self, phase):
        """ Get input and true data.

        Args:
            phase (str): data type (train, valid, test or None).

        Returns:
            tuple: (input, true) data for model.
        """
        input_data, true_data = None, None

        if self._input_var_names is not None:
            input_data = self._storegate.get_data(
                var_names=self._input_var_names, phase=phase)
        if self._true_var_names is not None:
            true_data = self._storegate.get_data(
                var_names=self._true_var_names, phase=phase)
        return input_data, true_data

    def get_input_var_shapes(self, phase='train'):
        """ Get shape of input_var_names.

        Args:
            phase (str): train, valid, test or None.

        Returns:
            ndarray,shape of lit: shape of a variable, or list of shapes
        """
        return self.storegate.get_var_shapes(self.input_var_names, phase=phase)

    def get_metadata(self, metadata_key):
        """ Returns metadata.

        Args:
            metadata_key (str): key of ``Saver()``.

        Returns:
            Obj: arbitrary object stored in ``Saver``.
        """
        unique_id = self.get_unique_id()
        return self.saver.load_ml(unique_id)[metadata_key]

    def get_pred_index(self):
        """ Returns prediction index passed to loss calculation.

        Returns:
            list: list of prediction index.
        """
        pred_index = []
        output_var_names = self.output_var_names
        pred_var_names = self.pred_var_names

        if not isinstance(output_var_names, list):
            raise ValueError(
                f'output_var_names: {output_var_names} is not list.')

        if not isinstance(pred_var_names, list):
            pred_var_names = [pred_var_names]

        for pred_var_name in pred_var_names:
            if pred_var_name in output_var_names:
                pred_index.append(output_var_names.index(pred_var_name))

        if not pred_index:
            raise ValueError(f'Not valid pred_var_names: {pred_var_names}.')

        return pred_index

    def show_info(self):
        """ Print information.
        """
        logger.header2(f'{self.name} information', level=logger.debug)
        logger.debug(f'input_var_names = {self.input_var_names}')
        logger.debug(f'output_var_names = {self.output_var_names}')
        logger.debug(f'pred_var_names = {self.pred_var_names}')
        logger.debug(f'true_var_names = {self.true_var_names}')
        self.ml.show_info()
        logger.header2('')
