from multiml import logger
from multiml.task.pytorch import PytorchChoiceBlockTask

from . import PytorchConnectionRandomSearchAgent


class PytorchSPOSNASAgent(PytorchConnectionRandomSearchAgent):
    """ Agent packing subtasks using Pytorch SPOS-NAS Model
    """
    def __init__(self, training_choiceblock_model=True, **kwargs):
        """

        Args:
            training_choiceblock_model (bool): Training choiceblock model after connecting submodels
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._training_choiceblock_model = training_choiceblock_model

    @logger.logging
    def execute(self):
        """ Execute
        """
        choiceblock_list = []  # list of choiceblockd subtasks

        tasks_list = self._task_scheduler.get_sorted_task_ids()
        for subtasktuples, task_id in zip(self._task_prod, tasks_list):
            # Execute sub-models
            for subtasktuple in subtasktuples:
                subtask_env = subtasktuple.env
                subtask_hps = subtasktuple.hps

                subtask_env.set_hps(subtask_hps)
                self._execute_subtask(subtasktuple, is_pretraining=True)

                if '_model_fit' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env._model_fit,
                                                  False)

            # Save model hyperparameters
            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'choiceblock_{task_id}_submodel_params',
                            params_list)

            # Not apply choiceblock for single submodel set
            if len(subtasktuples) == 1:
                choiceblock_list.append(subtasktuples[0].env)
                continue

            subtasks = [subtasktuple.env for subtasktuple in subtasktuples]
            choiceblock = self._build_choiceblock_task(subtasks, task_id)
            choiceblock_list.append(choiceblock.env)

        # Connecting each choiceblock model
        subtask = self._build_connected_models(choiceblock_list,
                                               job_id='SPOS-NAS')
        self._execute_subtask(subtask, is_pretraining=False)

        # evaluate each choice block
        # TODO: Implement Evolutionary Algorithm
        results = []
        comb_range = []
        for subtask_env in subtask.env._subtasks:
            comb_range.append(range(subtask_env.ml.model._len_task_candidate))
        from itertools import product
        for choices in product(*comb_range):
            model_name = []
            for i, choice in enumerate(choices):
                subtask.env._subtasks[i].choice = choice
                model_name.append(subtask.env._subtasks[i].ml.model.
                                  _choice_block[choice].__class__.__name__[1:])
            all_data = subtask.env.get_input_true_data('all')
            subtask.env.predict_update(all_data)
            self._metric.storegate = self._storegate
            metric = self._metric.calculate()
            result = dict(model_name='_'.join(model_name),
                          choices=choices,
                          metric=metric)
            self._saver.add(f"results.{'_'.join(model_name)}", result)
            results.append(result)

        from numpy import argmax
        model_name, choices, metric = results[argmax(
            [i['metric'] for i in results])].values()
        for i, choice in enumerate(choices):
            subtask.env._subtasks[i].ml.model._choice = choice
        self._saver.add('SPOS-NAS_train.selected_model', model_name)
        self._saver.add('SPOS-NAS_train.metric', metric)

        # re-train
        subtask = self._build_connected_models(subtasks=[
            self._task_prod[num][choice].env
            for num, choice in enumerate(choices)
        ],
                                               job_id='SPOS-NAS.final')
        self._execute_subtask(subtask, is_pretraining=False)
        self._metric.storegate = self._storegate
        metric = self._metric.calculate()

        self.result = dict(task_ids=['connection'],
                           subtask_ids=[f"choiceblock_{model_name}"],
                           subtask_hps=[None],
                           metric_value=metric)

    def _build_choiceblock_task(self, subtasks, task_id=None):
        # Merge sub-tasks with choiceblock
        choiceblock = PytorchChoiceBlockTask(
            subtasks=subtasks,
            saver=self._saver,
            phases=self._training_choiceblock_model,
            load_weights=self._connectiontask_args["load_weights"],
        )

        self._task_scheduler.add_task(task_id=task_id, add_to_dag=False)
        self._task_scheduler.add_subtask(task_id,
                                         task_id + '_choiceblock',
                                         env=choiceblock)
        subtask = self._task_scheduler.get_subtask(task_id,
                                                   task_id + '_choiceblock')

        self._execute_subtask(subtask, is_pretraining=True)

        if not self._connectiontask_args["load_weights"]:
            unique_id = choiceblock.get_unique_id()
            self.saver.dump_ml(unique_id,
                               ml_type='pytorch',
                               model=choiceblock.ml.model)

        # Save model ordering (model index)
        submodel_names = subtask.env.get_submodel_names()
        self._saver.add(f'choiceblock_{task_id}_submodel_names',
                        submodel_names)

        return subtask
