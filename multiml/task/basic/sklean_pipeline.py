"""SkleanPipelineTask module."""

from multiml.task.basic import MLBaseTask
from multiml import logger


class SkleanPipelineTask(MLBaseTask):
    """Wrapper task to process sklean object."""
    @logger.logging
    def execute(self):
        """Execute fit."""
        x_train, y_train = self.get_input_true_data('train')
        x_all, y_all = self.get_input_true_data('all')

        self._model_fit = self._model(**self._model_args).fit(x_train, y_train)

        results = self._model_fit.transform(x_all)

        self.storegate.delete_data(self.output_var_names, phase='all')
        self.storegate.update_data(self.output_var_names, results, 'auto')
