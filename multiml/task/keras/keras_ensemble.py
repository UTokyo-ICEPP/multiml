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
            output_var_names = self._proxy_model.output_var_names
            if self._individual_loss:
                self._output_var_names = [output_var_names]

                for index in range(len(subtasks)):
                    sub_var_names = self._get_sub_var_names(
                        output_var_names, index)
                    self._output_var_names.append(sub_var_names)

            else:
                self._output_var_names = output_var_names

        if self._optimizer is None:
            self._optimizer = self._proxy_model._optimizer

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

    def _get_sub_var_names(self, output_var_names, index):
        if isinstance(output_var_names, str):
            return f'{output_var_names}.{index}'

        elif isinstance(output_var_names, list):
            results = [
                self._get_sub_var_names(v, index) for v in output_var_names
            ]
            return results

        elif isinstance(output_var_names, tuple):
            results = [
                self._get_sub_var_names(v, index) for v in output_var_names
            ]
            return tuple(results)
