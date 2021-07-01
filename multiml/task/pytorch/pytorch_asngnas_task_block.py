from multiml import logger

from . import PytorchBaseTask


class PytorchASNGNASBlockTask(PytorchBaseTask):
    def __init__(self, subtasks, job_id, **kwargs):
        """

        Args:
            subtasks (list): list of task instances.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)

        self._subtasks = subtasks
        self.n_subtaks = len(subtasks)
        # TODO: get task information instead of submodels[0]
        self._proxy_model = self._subtasks[0]

        self._task_id = self._subtasks[0].task_id
        self._name = self._subtasks[0].task_id
        self._job_id = job_id
        self._input_var_names = self._proxy_model._input_var_names
        self._output_var_names = self._proxy_model._output_var_names
        self._true_var_names = self._proxy_model._true_var_names

        self._loss = self._proxy_model._loss
        self._optimizer = self._proxy_model._optimizer
        self._optimizer_args = self._proxy_model._optimizer_args
        self._num_epochs = self._proxy_model._num_epochs
        self._batch_size = self._proxy_model._batch_size
        self.subtask_ids = []
        for subtask in self._subtasks:
            self.subtask_ids.append(subtask.subtask_id)

    def build_model(self):
        from .modules import ASNGTaskBlockModel
        self._model = ASNGTaskBlockModel(
            name=self._name, models=[v._model for v in self._subtasks])

    def get_input_true_data(self, phase):
        return self._proxy_model.get_input_true_data(phase)

    def get_storegate_dataset(self, phase):
        return self._proxy_model.get_storegate_dataset(phase)

    def get_submodel_names(self):
        return [v.subtask_id for v in self._subtasks]

    def get_inputs(self):
        return self._proxy_model.get_inputs()

    def get_submodel(self, i_models):
        return self._subtasks[i_models]
