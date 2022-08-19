import numpy as np

from multiml import logger

from . import ModelConnectionTask


class DARTSTask(ModelConnectionTask):
    def __init__(self,
                 optimizer_alpha,
                 optimizer_weight,
                 learning_rate_alpha,
                 learning_rate_weight,
                 zeta=0.01,
                 **kwargs):
        """
        Args:
            optimizer_darts_alpha (str): optimizer for alpha in DARTS optimization
            optimizer_darts_weight (str): optimizer for weight in DARTS optimization
            learning_rate_darts_alpha (float): learning rate (epsilon) for alpha in DARTS optimization
            learning_rate_darts_weight (float): learning rate (epsilon) for weight in DARTS optimization
            zeta (float): zeta parameter in DARTS optimization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._optimizer = None
        self._learning_rate = None

        self._optimizer_alpha = optimizer_alpha
        self._optimizer_weight = optimizer_weight
        self._learning_rate_alpha = learning_rate_alpha
        self._learning_rate_weight = learning_rate_weight
        self._zeta = zeta

    def fit(self, train_data=None, valid_data=None):
        x = {}
        y = {}
        x['train'], y['train'] = self.get_input_true_data("train")
        x['valid'], y['valid'] = self.get_input_true_data("valid")
        x['test'], y['test'] = self.get_input_true_data("test")

        result = self._training_darts(x, y)

        return result

    def _training_darts(self, x, y):
        result = {}

        n_train = len(x['train'][0])
        n_valid = len(x['valid'][0])
        logger.debug(f"num of training samples = {n_train}")
        logger.debug(f"num of validation samples = {n_valid}")

        ###################################
        # Check consistency of batch size #
        ###################################
        import math
        v_gcd = math.gcd(n_train, n_valid)
        frac_train = n_train // v_gcd
        frac_sum = (n_train + n_valid) // v_gcd

        if n_train < self._batch_size:
            self._batch_size = n_train

        if self._batch_size % frac_train > 0:
            raise ValueError(
                f"batch_size of darts training should be divisible by training/valid ratio. bsize_darts_train = {self._batch_size}, frac_train = {frac_train}"
            )

        batch_size_total = self._batch_size * frac_sum // frac_train
        logger.debug(f"total batch size (train + valid) in DARTS training = {batch_size_total}")

        alpha_model_names = [v.name for v in self._model.alpha_vars]
        result['alpha_model_names'] = alpha_model_names

        # Validate
        for var in self._model.weight_vars:
            if 'batch_normalization' in var.name:
                logger.warn('DARTS should not have batch normalization layer.')

        #######################################
        # Merging training/validation samples #
        #######################################
        x_train_valid = []
        y_train_valid = []
        bsize_valid = batch_size_total - self._batch_size
        logger.debug(f"validation batch size in DARTS training = {bsize_valid}")
        for v1, v2 in zip(x['train'], x['valid']):
            v1 = v1.reshape((self._batch_size, -1) + v1.shape[1:])
            v2 = v2.reshape((bsize_valid, -1) + v2.shape[1:])
            v = np.concatenate([v1, v2], axis=0)
            v = v.reshape((-1, ) + v.shape[2:])
            x_train_valid.append(v)

        for v1, v2 in zip(y['train'], y['valid']):
            v1 = v1.reshape((self._batch_size, -1) + v1.shape[1:])
            v2 = v2.reshape((bsize_valid, -1) + v2.shape[1:])
            v = np.concatenate([v1, v2], axis=0)
            v = v.reshape((-1, ) + v.shape[2:])
            y_train_valid.append(v)

        ##################
        # DARTS training #
        ##################
        self.ml.model._batch_size_train.assign(self._batch_size)

        import tempfile
        chpt_path = f'{tempfile.mkdtemp()}/tf_chpt'

        cbs = []

        from tensorflow.keras.callbacks import EarlyStopping
        es_cb = EarlyStopping(monitor='valid_loss',
                              patience=self._max_patience,
                              verbose=0,
                              mode='min',
                              restore_best_weights=True)
        cbs.append(es_cb)

        from tensorflow.keras.callbacks import ModelCheckpoint
        cp_cb = ModelCheckpoint(filepath=chpt_path,
                                monitor='valid_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='min')
        cbs.append(cp_cb)

        from tensorflow.keras.callbacks import TerminateOnNaN
        nan_cb = TerminateOnNaN()
        cbs.append(nan_cb)

        if self._save_tensorboard:
            from tensorflow.keras.callbacks import TensorBoard
            tb_cb = TensorBoard(log_dir=f'{self._saver.save_dir}/{self._name}',
                                histogram_freq=1,
                                profile_batch=5)
            cbs.append(tb_cb)

        from multiml.agent.keras.callback import (AlphaDumperCallback, EachLossDumperCallback)
        alpha_cb = AlphaDumperCallback()
        loss_cb = EachLossDumperCallback()
        cbs.append(alpha_cb)
        cbs.append(loss_cb)

        training_verbose_mode = 0
        if logger.MIN_LEVEL <= logger.DEBUG:
            training_verbose_mode = 1

        history = self.ml.model.fit(x=x_train_valid,
                                    y=y_train_valid,
                                    batch_size=batch_size_total,
                                    epochs=self._num_epochs,
                                    callbacks=cbs,
                                    validation_data=(x['test'], y['test']),
                                    shuffle=False,
                                    verbose=training_verbose_mode)

        history0 = history.history
        result['darts_loss_train'] = history0['train_loss']
        result['darts_loss_valid'] = history0['valid_loss']
        result['darts_loss_test'] = history0['val_test_loss']
        result['darts_alpha_history'] = alpha_cb.get_alpha_history()
        result['darts_loss_history'] = loss_cb.get_loss_history()
        result['darts_lambda_history'] = history0['lambda']
        result['darts_alpha_gradients_sum'] = np.array(history0['alpha_gradients_sum']).tolist()
        result['darts_alpha_gradients_sq_sum'] = np.array(
            history0['alpha_gradients_sq_sum']).tolist()
        result['darts_alpha_gradients_n'] = history0['alpha_gradients_n']

        # Check nan in alpha parameters
        # self._has_nan_in_alpha = nan_cb._isnan(self._model.alpha_vars)

        ##################
        # Save meta data #
        ##################
        self._index_of_best_submodels = self.ml.model.get_index_of_best_submodels()

        return result

    def load_metadata(self):
        #self._io_index = self.get_metadata('io_index')
        self._index_of_best_submodels = self.get_metadata('index_of_best_submodels')

    def get_best_submodels(self):
        """Returns indices of the best submodels determined by the alpha.

        Returns:
            list (int): list of index of the selected submodels
        """
        subtask_ids_best = []
        for subtask_env, i_model in zip(self._subtasks, self._index_of_best_submodels):
            subtask_ids_best.append(subtask_env.get_submodel(i_model))

            logger.info(
                f"Submodels used in DARTS = {[m.subtask_id for m in subtask_env._subtasks]}")
            logger.info(f"  Selected = {subtask_ids_best[-1].subtask_id}")

        return subtask_ids_best

    def build_model(self):
        from .modules import DARTSModel
        models = [subtask.ml.model for subtask in self._subtasks]

        self._model = DARTSModel(optimizer_alpha=self._optimizer_alpha,
                                 optimizer_weight=self._optimizer_weight,
                                 learning_rate_alpha=self._learning_rate_alpha,
                                 learning_rate_weight=self._learning_rate_weight,
                                 zeta=self._zeta,
                                 models=models,
                                 input_var_index=self._input_var_index,
                                 output_var_index=self._output_var_index)

        self._optimizer = 'SGD'  # This is dummy

    def dump_model(self, extra_args=None):
        """Dump current DARTS model."""
        args_dump_ml = dict(index_of_best_submodels=self._index_of_best_submodels)

        if extra_args is not None:
            args_dump_ml.update(extra_args)

        super().dump_model(args_dump_ml)
