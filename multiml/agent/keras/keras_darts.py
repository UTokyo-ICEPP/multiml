from multiml.task.keras import DARTSTask, ModelConnectionTask

from . import KerasEnsembleAgent


class KerasDartsAgent(KerasEnsembleAgent):
    """
    Model selection agent inspired by Differential Architecture Search (DARTS)
    """
    def __init__(self,
                 select_one_models=True,
                 use_original_darts_optimization=True,
                 dartstask_args={},
                 **kwargs):
        """
        Args:
            select_one_models (bool): (Expert option) If false, all models are kept for the final prediction
            use_original_darts_optimization (bool): (Expert option) If false, both model parameter and alpha parameters are simultaneously trained.
            dartstask_args (dict): args for DARTSTask
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._select_one_models = select_one_models
        self._use_original_darts_optimization = use_original_darts_optimization
        self._dartstask_args = dartstask_args

        if self._freeze_model_weights is True:
            raise ValueError(
                "DARTS model sets all model variables to be trainable. Currect freeze_model_weights(=True) might be misleading."
            )

    def _build_connected_models(self, subtasks, job_id=None):

        ######################
        # DARTS optimization #
        ######################
        task_id = 'connection'
        subtask_id = 'darts'
        if self._use_original_darts_optimization:
            subtask = DARTSTask(name=subtask_id,
                                subtasks=subtasks,
                                saver=self._saver,
                                **self._dartstask_args)
        else:
            subtask = ModelConnectionTask(name=subtask_id,
                                          subtasks=subtasks,
                                          saver=self._saver,
                                          **self._dartstask_args)

            if job_id is not None:
                subtask.set_hps({"job_id": job_id})
        subtask.task_id = task_id
        subtask.subtask_id = subtask_id

        self._task_scheduler.add_task(task_id=task_id, add_to_dag=False)
        self._task_scheduler.add_subtask(task_id, subtask_id, env=subtask)
        task = self._task_scheduler.get_subtask(task_id, subtask_id)

        if not self._select_one_models:
            return task

        self._execute_subtask(task, is_pretraining=False)

        # Get best submodels in the last state of DARTS
        if self._use_original_darts_optimization:
            best_subtasks = task.env.get_best_submodels()
        else:
            # Extract alpha
            from multiml.task.keras.keras_ensemble import EnsembleTask
            alpha_vars = EnsembleTask.get_ensemble_weights(task.env._model)

            alpha_model_names = [v.name for v in alpha_vars]
            self._saver.add('alpha_model_names', alpha_model_names)

            # Get best models by alpha values
            from multiml.task.keras.modules import DARTSModel
            index_of_best_submodels = DARTSModel._get_best_submodels(
                alpha_vars)

            best_subtasks = []
            for subtask_env, i_model in zip(task.env._subtasks,
                                            index_of_best_submodels):
                best_subtasks.append(subtask_env.get_submodel(i_model))

            # Save ensemble weights
            self._save_ensemble_weights(task, "darts")

        ######################
        # Final optimization #
        ######################
        subtask_id = self.connectiontask_name_with_jobid(job_id)
        subtask = ModelConnectionTask(name=subtask_id,
                                      subtasks=best_subtasks,
                                      saver=self._saver,
                                      **self._connectiontask_args)
        subtask.task_id = task_id
        subtask.subtask_id = subtask_id

        self._task_scheduler.add_task(task_id=task_id, add_to_dag=False)
        self._task_scheduler.add_subtask(task_id, subtask_id, subtask)
        task = self._task_scheduler.get_subtask(task_id, subtask_id)
        return task
