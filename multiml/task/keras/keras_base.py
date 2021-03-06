"""KerasBaseTask module."""
from multiml import logger, const
from multiml.task.keras import modules
from .keras_util import training_keras_model, compile
from ..basic import MLBaseTask


class KerasBaseTask(MLBaseTask):
    """Base task for Keras model.

    Examples:
        >>> # your keras model
        >>> class MyKerasModel(Model):
        >>>     def __init__(self, units=1):
        >>>         super(MyKerasModel, self).__init__()
        >>>
        >>>         self.dense = Dense(units)
        >>>         self.relu = ReLU()
        >>>
        >>>     def call(self, x):
        >>>         return self.relu(self.dense(x))
        >>>
        >>> # create task instance
        >>> task = KerasBaseTask(storegate=storegate,
        >>>                      model=MyKerasModel,
        >>>                      input_var_names=('x0', 'x1'),
        >>>                      output_var_names='outputs-keras',
        >>>                      true_var_names='labels',
        >>>                      optimizer='adam',
        >>>                      optimizer_args=dict(lr=0.1),
        >>>                      loss='binary_crossentropy')
        >>> task.set_hps({'num_epochs': 5})
        >>> task.execute()
        >>> task.finalize()
    """
    def __init__(self,
                 run_eagerly=None,
                 callbacks=['EarlyStopping', 'ModelCheckpoint'],
                 save_tensorboard=False,
                 **kwargs):
        """

        Args:
            run_eagerly (bool): Run on eager execution mode (not graph mode).
            callbacks (list(str or keras.Callback)): callback for keras model training.
                Predefined callbacks (EarlyStopping, ModelCheckpoint, and TensorBoard) can be selected by str.
                Other user-defined callbacks should be given as keras.Callback object.
            save_tensorboard (bool): use tensorboard callback in training.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)

        self._run_eagerly = run_eagerly
        self._callbacks = callbacks
        self._save_tensorboard = save_tensorboard

        if save_tensorboard and ('TensorBoard' not in self._callbacks):
            self._callbacks += ['TensorBoard']

        if self._metrics is None:
            self._metrics = ['accuracy']

        self._trainable_model = True

    def compile_model(self):
        """Compile keras model."""
        self.ml.model = compile(self._model, self._model_args, modules)

        if self._pred_var_names is not None:
            self.ml.model.set_pred_index(self.get_pred_index())

        from .keras_util import get_optimizer
        self.ml.optimizer = get_optimizer(self._optimizer, self._optimizer_args)

        self.ml.model.compile(optimizer=self.ml.optimizer,
                              loss=self.ml.loss,
                              loss_weights=self.ml.loss_weights,
                              run_eagerly=self._run_eagerly,
                              steps_per_execution=None,
                              metrics=self._metrics)

        if self._load_weights:
            self.load_model()
            self.load_metadata()

        if self.ml.model.built and logger.MIN_LEVEL <= logger.DEBUG:
            self.ml.model.summary()

    def compile_loss(self):
        """Compile keras model."""
        if isinstance(self._loss, str):
            import tensorflow as tf
            self.ml.loss = tf.keras.losses.get(self._loss)
        else:
            self.ml.loss = self._loss

        self.ml.loss_weights = self._loss_weights

    def load_model(self):
        """Load pre-trained keras model weights."""
        model_path = super().load_model()
        logger.info(f'load {model_path}')
        self.ml.model.load_weights(model_path).expect_partial()

    def dump_model(self, extra_args=None):
        """Dump current keras model."""
        args_dump_ml = dict(ml_type='keras')

        if extra_args is not None:
            args_dump_ml.update(extra_args)

        super().dump_model(args_dump_ml)

    def fit(self, train_data=None, valid_data=None):
        """Training model.

        Returns:
            dict: training results.
        """
        if train_data is None:
            x_train, y_train = self.get_input_true_data("train")
        else:
            x_train, y_train = train_data

        if valid_data is None:
            x_valid, y_valid = self.get_input_true_data("valid")
        else:
            x_valid, y_valid = valid_data

        if self._save_tensorboard:
            tensorboard_path = f'{self._saver.save_dir}/{self._name}'
        else:
            tensorboard_path = None

        result = training_keras_model(self.ml.model,
                                      num_epochs=self._num_epochs,
                                      batch_size=self._batch_size,
                                      max_patience=self._max_patience,
                                      x_train=x_train,
                                      y_train=y_train,
                                      x_valid=x_valid,
                                      y_valid=y_valid,
                                      chpt_path=None,
                                      callbacks=self._callbacks,
                                      tensorboard_path=tensorboard_path,
                                      verbose=self._verbose)

        return result

    def predict(self, data=None, phase=None):
        """Evaluate model prediction.

        Args:
            phase (str): data type (train, valid, test or None)

        Returns:
            ndarray: prediction by the model
            ndarray: target
        """
        if self.ml.model is None:
            raise ValueError('model is not defined. Need build_model() or execute().')

        if data is None:
            x_data, y_data = self.get_input_true_data(phase)
        else:
            x_data, y_data = data

        return self.ml.model.predict(x_data)

    def get_inputs(self):
        """Returns keras Input from input_var_names."""
        from tensorflow.keras.layers import Input
        shapes = self.get_input_var_shapes()

        if shapes is None:
            return

        if not isinstance(shapes, list):
            shapes = [shapes]

        return [Input(shape=shape) for shape in shapes]
