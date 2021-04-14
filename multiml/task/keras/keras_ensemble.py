from multiml import logger

from . import KerasBaseTask


class EnsembleTask(KerasBaseTask):
    def __init__(self,
                 subtasks,
                 dropout_rate=None,
                 individual_loss=False,
                 individual_loss_weights=0.0,
                 **kwargs):
        """

        Args:
            subtasks (list): list of task instances.
            dropout_rate (float): dropout_rate for ensemble weights. If None, no dropout.
            individual_loss (bool): use multiple outputs
            individual_loss_weights (float): coefficient for multiple outputs
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._subtasks = subtasks
        # TODO: get task information instead of submodels[0]
        self._proxy_model = self._subtasks[0]

        self._ensemble_type = 'softmax'
        self._dropout_rate = dropout_rate
        self._individual_loss = individual_loss
        self._individual_loss_weights = individual_loss_weights

        self._input_var_names = self._proxy_model._input_var_names
        self._true_var_names = self._proxy_model._true_var_names

        if self._output_var_names is None:
            self._output_var_names = self._proxy_model.output_var_names

        if self._optimizer is None:
            self._optimizer = self._proxy_model._optimizer

        #self._num_epochs = self._proxy_model._num_epochs
        #self._batch_size = self._proxy_model._batch_size
        #self._max_patience = self._proxy_model._max_patience

    def compile(self):
        for subtask in self._subtasks:
            subtask.compile()

        super().compile()

    def compile_loss(self):
        if self._loss is None:
            self.ml.loss = self._proxy_model.ml.loss
        else:
            super().compile_loss()

        if self._individual_loss:
            self.ml.loss = [self.ml.loss]
            self.ml.loss_weights = [1.0]

            for subtask in self._subtasks:
                self.ml.loss.append(subtask.ml.loss)
                self.ml.loss_weights.append(self._individual_loss_weights)

    def build_model(self):
        from .modules import EnsembleModel

        if self._proxy_model.task_id is None:
            prefix = 'prefix'
        else:
            prefix = self._proxy_model.task_id

        self._model = EnsembleModel(
            models=[v.ml.model for v in self._subtasks],
            prefix=prefix,
            ensemble_type=self._ensemble_type,
            dropout_rate=self._dropout_rate,
            individual_loss=self._individual_loss,
        )

    def get_input_true_data(self, phase):
        return self._proxy_model.get_input_true_data(phase)

    def predict_update(self, data=None):
        """ Update storegate.
        """
        y_preds = self.predict(data)
        output_var_names = self._output_var_names

        if self._individual_loss:
            for index, y_pred in enumerate(y_preds):
                if index > 0:
                    output_var_names = self._subtasks[index -
                                                      1].output_var_names

                self._storegate.update_data(data=y_pred,
                                            var_names=output_var_names,
                                            phase='auto')
        else:
            self._storegate.update_data(data=y_preds,
                                        var_names=output_var_names,
                                        phase='auto')

    def get_inputs(self):
        return self._proxy_model.get_inputs()

    def get_submodel_names(self):
        """ Returns subtask_id used in ensembling.

        Returns:
            list (str): list of subtask_id
        """
        return [v.subtask_id for v in self._subtasks]

    def get_submodel(self, i_models):
        """ Get a submodel by model index

        Args:
            i_models (int): submodel index

        Returns:
            subtasktuple: submodel for the input index
        """
        return self._subtasks[i_models]

    @staticmethod
    def get_ensemble_weights(model):
        """ Collect ensemble_weights in the keras model

        Args:
            model (keras.Model):

        Returns:
            list (tf.Variable): list of ensemble weights
        """
        ensemble_weights = []
        for var in model._get_variables():
            if not var.trainable:
                continue

            if "_ensemble_weights/" in var.name:
                ensemble_weights.append(var)

        return ensemble_weights
