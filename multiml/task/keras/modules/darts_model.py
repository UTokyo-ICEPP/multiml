import tensorflow as tf

from multiml import logger

from . import ConnectionModel


class SumTensor(tf.keras.metrics.MeanTensor):
    # pytest seems to not able to parse 'result' method for coverage check...
    # If 'result' is chaned to '_result' or 'result2', pytest works well.
    def result(self):  # pragma: no cover
        if not self._built:
            raise ValueError(
                'MeanTensor does not have any result yet. Please call the MeanTensor '
                'instance or use `.update_state(value)` before retrieving the result.')
        return self.total


class DARTSModel(ConnectionModel):
    def __init__(self, optimizer_alpha, optimizer_weight, learning_rate_alpha,
                 learning_rate_weight, zeta, *args, **kwargs):
        """Constructor.

        Args:
            optimizer_alpha (str): optimizer for alpha in DARTS optimization
            optimizer_weight (str): optimizer for weight in DARTS optimization
            learning_rate_alpha (float): learning rate (epsilon) for alpha in DARTS optimization
            learning_rate_weight (float): learning rate (epsilon) for weight in DARTS optimization
            zeta (float): zeta parameter in DARTS optimization
        """
        super().__init__(*args, **kwargs)

        self.weight_vars = self._get_weight_variables()
        self.alpha_vars = self._get_alpha_variables()

        self._batch_size_train = tf.Variable(1)  # dummy value

        self._epsilon_L2 = 0.01

        self._zeta = tf.constant(zeta)

        from tensorflow.keras.optimizers import deserialize as opt_deserialize
        if not isinstance(optimizer_alpha, str):
            self.optimizer_alpha = optimizer_alpha
        else:
            self.optimizer_alpha = opt_deserialize({
                'class_name': optimizer_alpha,
                'config': {
                    'learning_rate': learning_rate_alpha
                }
            })
        if not isinstance(optimizer_alpha, str):
            self.optimizer_weight = optimizer_weight
        else:
            self.optimizer_weight = opt_deserialize({
                'class_name': optimizer_weight,
                'config': {
                    'learning_rate': learning_rate_weight
                }
            })

        self._train_loss = tf.keras.metrics.Mean(name='train_loss')
        self._valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self._test_loss = tf.keras.metrics.Mean(name='test_loss')
        self._lambda = tf.keras.metrics.Mean(name='lambda')
        self._alpha_gradients_sum = SumTensor(name='alpha_gradients_sum')
        self._alpha_gradients_sq_sum = SumTensor(name='alpha_gradients_sq_sum')
        self._alpha_gradients_n = tf.keras.metrics.Sum(name='alpha_gradients_n')

    def train_step(self, data):
        x, y = data
        # FIXME: This strategy is very dangerous. Risk of valid leaks.
        x_train = tuple([v[:self._batch_size_train] for v in x])
        x_valid = tuple([v[self._batch_size_train:] for v in x])
        y_train = tuple([v[:self._batch_size_train] for v in y])
        y_valid = tuple([v[self._batch_size_train:] for v in y])

        ##################
        # DARTS training #
        ##################

        # Update alpha
        grad_alpha = self._get_alpha_grad(x_train, y_train, x_valid, y_valid, training=True)
        grad_alpha, _ = tf.clip_by_global_norm(grad_alpha, clip_norm=0.1)
        self.optimizer_alpha.apply_gradients(zip(grad_alpha, self.alpha_vars))
        self._alpha_gradients_sum(grad_alpha)
        self._alpha_gradients_sq_sum(tf.square(grad_alpha))
        self._alpha_gradients_n(1)

        # Update weights
        grad_w = self._get_grad_weights(x_train, y_train, training=True)
        grad_w, _ = tf.clip_by_global_norm(grad_w, clip_norm=0.1)
        self.optimizer_weight.apply_gradients(zip(grad_w, self.weight_vars))

        # Evaluate losses
        train_loss = self._eval_loss(x_train, y_train, training=True)
        valid_loss = self._eval_loss(x_valid, y_valid, training=True)
        self._train_loss(train_loss)
        self._valid_loss(valid_loss)

        # Evaluate lambda (see arXiv: 1909.09656)
        if False:
            # This operation block causes memory leak...
            with tf.device("CPU"):
                hessian = self._get_hessian(x_valid, y_valid, self.alpha_vars)
                eigenvalues, _ = tf.linalg.eigh(hessian)
                lambda_max_alpha = tf.reduce_max(eigenvalues)
        else:
            lambda_max_alpha = -1.
        self._lambda(lambda_max_alpha)

        return {
            'train_loss': self._train_loss.result(),
            'valid_loss': self._valid_loss.result(),
            'lambda': self._lambda.result(),
            'alpha_gradients_sum': self._alpha_gradients_sum.result(),
            'alpha_gradients_sq_sum': self._alpha_gradients_sq_sum.result(),
            'alpha_gradients_n': self._alpha_gradients_n.result(),
        }

    def test_step(self, data):
        # Evaluate "test" loss (not valid loss)
        x, y = data
        loss = self._eval_loss(x, y, training=False)
        self._test_loss(loss)
        return {
            'test_loss': self._test_loss.result(),
        }

    @staticmethod
    def _get_variable_numpy(vars):
        return [v.numpy() for v in vars]

    def _get_weight_variables(self):
        variables = []
        for var in self._get_variables():
            if not var.trainable:
                continue
            if "_ensemble_weights/" not in var.name:
                variables.append(var)
        return variables

    def _get_alpha_variables(self):
        variables = []
        for var in self._get_variables():
            if not var.trainable:
                continue
            if "_ensemble_weights/" in var.name:
                variables.append(var)
        return variables

    def _eval_loss(self, x, y, training):
        y_pred = self(x, training=training)
        return self.compiled_loss(y, y_pred)

    def _get_grad(self, x, y, variables, training):
        with tf.GradientTape() as tape:
            loss_value = self._eval_loss(x, y, training=training)
        return tape.gradient(loss_value, variables)

    def _get_grad_alpha(self, x, y, training):
        return self._get_grad(x, y, self.alpha_vars, training=training)

    def _get_grad_weights(self, x, y, training):
        return self._get_grad(x, y, self.weight_vars, training=training)

    def _get_alpha_grad(self, x_train, y_train, x_valid, y_valid, training):
        # Keep original w
        w_original = [var for var in self.weight_vars]

        # Get delta_w L_train
        grad_w = self._get_grad_weights(x_train, y_train, training=training)

        # Get the first term: delta_a L_val(w', a)
        for w, dw in zip(
                self.weight_vars,
                grad_w,
        ):
            if dw is not None:
                w.assign_sub(self._zeta * dw)
        first_term = self._get_grad_alpha(x_valid, y_valid, training=training)

        # Get delta_wdash L_val for w^+-
        grad_wd = self._get_grad_weights(x_valid, y_valid, training=training)
        epsilon_norm = sum(
            [tf.math.reduce_sum(tf.math.square(tf.keras.backend.flatten(v))) for v in grad_wd])
        epsilon = self._epsilon_L2 / (tf.keras.backend.epsilon() + epsilon_norm)

        # Get delta_alpha L_train
        # w^+
        for var, w, dw in zip(self.weight_vars, w_original, grad_wd):
            if dw is not None:
                wplus = w + epsilon * dw
            else:
                wplus = w
            var.assign(wplus)
        alphaplus = self._get_grad_alpha(x_train, y_train, training=training)

        # w^-
        for var, w, dw in zip(self.weight_vars, w_original, grad_wd):
            if dw is not None:
                wminus = w - epsilon * dw
            else:
                wminus = w
            var.assign(wminus)
        alphaminus = self._get_grad_alpha(x_train, y_train, training=training)

        second_term = []
        for v1, v2 in zip(alphaplus, alphaminus):
            second_term.append((v1 - v2) / (2 * epsilon))

        # Set original w
        for var, w in zip(self.weight_vars, w_original):
            var.assign(w)

        total_grad = [v1 - self._zeta * v2 for v1, v2 in zip(first_term, second_term)]
        return total_grad

    def _get_hessian(self, x, y, variables, training):  # pragma: no cover
        with tf.GradientTape(persistent=True) as tape:
            loss_value = self._eval_loss(x, y, training=training)
            grads = tape.gradient(loss_value, variables)
            grads = [tf.reshape(grad, [-1]) for grad in grads]
            flattened_grads = tf.concat(grads, axis=0)
        hessians = tape.jacobian(flattened_grads, variables)
        hessians = [tf.reshape(hess, [hess.shape[0], -1]) for hess in hessians]
        flattened_hessians = tf.concat(hessians, 1)
        return flattened_hessians

    def get_index_of_best_submodels(self):
        return self._get_best_submodels(self.alpha_vars)

    @staticmethod
    def _get_best_submodels(alphas):
        import tensorflow as tf

        subtask_index = []
        for var in alphas:
            if tf.executing_eagerly():
                values = var.numpy().reshape(-1)
            else:
                values = tf.keras.backend.eval(var).reshape(-1)

            idx_max = values.argmax()
            subtask_index.append(idx_max)

            logger.info(f'darts_final_alpha_index_{var.name}: {idx_max}')
        return subtask_index
