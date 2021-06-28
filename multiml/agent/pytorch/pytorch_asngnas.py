from multiml import logger
from multiml.task.pytorch import PytorchASNGNASTask
from multiml.task.pytorch import PytorchASNGNASBlockTask

from . import PytorchConnectionRandomSearchAgent
from multiml.task.pytorch.datasets import StoreGateDataset, NumpyDataset
import numpy as np


class PytorchASNGNASAgent(PytorchConnectionRandomSearchAgent):
    """ Agent packing subtasks using Pytorch ASNG-NAS Model
    """
    def __init__(self,
                 verbose=1,
                 num_epochs=100,
                 batch_size={
                     'type': 'equal_length',
                     'length': 500,
                     'test': 100
                 },
                 lam=2,
                 delta_init_factor=1,
                 **kwargs):
        """

        Args:
            training_choiceblock_model (bool): Training choiceblock model after connecting submodels
            **kwargs: Arbitrary keyword arguments
        """
        self.do_pretraining = kwargs['do_pretraining']
        self._verbose = verbose
        self._num_epochs = num_epochs
        self.lam = int(lam)
        self.delta_init_factor = delta_init_factor
        self.batch_size = batch_size
        super().__init__(**kwargs)

    @logger.logging
    def execute(self):
        """ Execute
        Currently, only categorical ASNG NAS is implemented.
        """

        categories = []
        task_ids = []
        asng_block_list = []

        for task_idx, task_id in enumerate(
                self._task_scheduler.get_sorted_task_ids()):
            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id)
            n_cat = 0

            for subtask_idx, subtask in enumerate(subtasktuples):
                n_cat += 1
                subtask_env = subtask.env
                subtask_hps = subtask.hps
                subtask_env.set_hps(subtask_hps)
                subtask_env._verbose = self._verbose
                subtask_env._num_epochs = self._num_epochs

                if self.do_pretraining:
                    self._execute_subtask(subtask, is_pretraining=True)
                else:
                    subtask.env.storegate = self._storegate
                    subtask.env.saver = self._saver
                    subtask.env.compile()

                if '_model_fit' in dir(subtask_env):
                    if self._freeze_model_weights:
                        self._set_trainable_flags(subtask_env._model_fit,
                                                  False)

            params_list = [v.hps for v in subtasktuples]
            self._saver.add(f'asng_block_{task_id}_submodel_params',
                            params_list)

            # build asng task block
            subtasks = [v.env for v in subtasktuples]
            asng_block = PytorchASNGNASBlockTask(
                subtasks=subtasks,
                job_id=f'ASNG-NAS-Block-{task_id}',
                saver=self._saver,
                load_weights=self._connectiontask_args['load_weights'],
            )

            parents = []
            if len(task_ids) > 0:
                parents = [task_ids[-1]]

            asng_task_id = 'ASNG-NAS-' + task_id
            self._task_scheduler.add_task(task_id=asng_task_id,
                                          parents=parents)
            self._task_scheduler.add_subtask(asng_task_id,
                                             'BlockTask',
                                             env=asng_block)

            asng_block_subtask = self._task_scheduler.get_subtask(
                asng_task_id, 'BlockTask')

            # if self.do_pretraining :
            #     self._execute_subtask(asng_block_subtask, is_pretraining=True)
            # else :
            asng_block_subtask.env.storegate = self._storegate
            asng_block_subtask.env.saver = self._saver
            asng_block_subtask.env.compile()

            if not self._connectiontask_args['load_weights']:
                unique_id = asng_block.get_unique_id()
                self.saver.dump_ml(unique_id,
                                   ml_type='pytorch',
                                   model=asng_block.ml.model)

            submodel_names = asng_block_subtask.env.get_submodel_names()
            self._saver.add(f'asng_block_{task_id}_submodel_names',
                            submodel_names)

            asng_block_list.append(asng_block_subtask.env)

            categories += [n_cat]
            task_ids.append(task_id)

        asng_task = PytorchASNGNASTask(
            lam=self.lam,
            delta_init_factor=self.delta_init_factor,
            subtasks=asng_block_list,
            saver=self._saver,
            device=self._connectiontask_args['device'],
            gpu_ids=None,
            amp=False,  # expert option
            verbose=self._verbose,
            num_epochs=self._num_epochs,
            optimizer=self._connectiontask_args['optimizer'],
            optimizer_args=self._connectiontask_args['optimizer_args'],
            batch_size=self.batch_size,
            max_patience=self._connectiontask_args['max_patience'],
        )

        self._task_scheduler.add_task(task_id='ASNG-NAS', add_to_dag=False)
        self._task_scheduler.add_subtask('ASNG-NAS',
                                         'main-task',
                                         env=asng_task)
        asng_subtask = self._task_scheduler.get_subtask(
            'ASNG-NAS', 'main-task')

        if not self._connectiontask_args["load_weights"]:
            unique_id = asng_task.get_unique_id()
            self.saver.dump_ml(unique_id,
                               ml_type='pytorch',
                               model=asng_task.ml.model)

        # Save model ordering (model index)
        submodel_names = asng_subtask.env.get_submodel_names()
        self._saver.add(f'ASNG-NAS_{submodel_names}', submodel_names)
        asng_subtask.env.verbose = self._verbose

        self._execute_subtask(asng_subtask, is_pretraining=False)

        # check best model
        asng_task.set_most_likely()

        c_cat, c_int = asng_task.get_most_likely()
        theta_cat, theta_int = asng_task.get_thetas()
        cat_idx = c_cat.argmax(axis=1)

        # re-train
        best_task_ids, best_subtask_ids = asng_task.best_model()
        best_subtasks = [
            self._task_scheduler.get_subtask(task_id, subtask_id)
            for task_id, subtask_id in zip(task_ids, best_subtask_ids)
        ]
        best_combination_task = self._build_connected_models(
            subtasks=[t.env for t in best_subtasks],
            job_id='ASNG-NAS-Final',
            use_task_scheduler=True)
        best_comb = '+'.join(s for s in best_subtask_ids)

        self._execute_subtask(best_combination_task, is_pretraining=False)
        self._metric.storegate = self._storegate
        metric = self._metric.calculate()

        self.result = dict(task_ids=['ASNG-NAS-Final'],
                           subtask_ids=best_subtask_ids,
                           subtask_hps=[None],
                           metric_value=metric)

        ### evaluate
        # make results for json output
        # seed, nevents, walltime will be set at outside
        results_json = {'agent': 'ASNG-NAS', 'tasks': {}}

        # best_combination_task.env._unpack_inputs = True
        pred_result = best_combination_task.env.predict(label=True)
        best_combination_task.env._storegate.update_data(
            data=pred_result['pred'],
            var_names=best_combination_task.env._output_var_names,
            phase='auto')
        self._metric._storegate = best_combination_task.env._storegate
        test_metric = self._metric.calculate()

        test_result = dict(model_name='ASNG-NAS-Final',
                           cat_idx=cat_idx,
                           metric=test_metric)
        self._saver.add(f"results.ASNG-NAS-Final", test_result)

        results_json['loss_test'] = pred_result['loss']
        results_json['subloss_test'] = pred_result['subloss']
        results_json['metric_test'] = test_metric

        for task_idx, task_id in enumerate(task_ids):
            results_json['tasks'][task_id] = {}
            results_json['tasks'][task_id][
                'weight'] = best_combination_task.env.ml.loss_weights[task_idx]
            results_json['tasks'][task_id]['models'] = []
            results_json['tasks'][task_id]['theta_cat'] = []

            subtasktuples = self._task_scheduler.get_subtasks_with_hps(task_id)
            for subtask_idx, subtask in enumerate(subtasktuples):
                this_id = subtask.subtask_id.split('-')[
                    -1]  # FIXME : hard coded
                theta = theta_cat[task_idx, subtask_idx]
                results_json['tasks'][task_id]['models'].append(this_id)
                results_json['tasks'][task_id]['theta_cat'].append(theta)

                if subtask_idx == cat_idx[task_idx]:
                    results_json['tasks'][task_id]['model_selected'] = this_id

                if theta_cat is not None:
                    logger.info(f'  theta_cat is {this_id: >20} : {theta:.3e}')
                else:
                    logger.info(f'theta_cat is None')

        if theta_int is not None:
            for theta, job_id in zip(theta_int.tolist(), ):
                for t, j in zip(theta, job_id):
                    logger.info(f'  theta_cat is {j: >20} : {t:.3e}')
        else:
            logger.info(f'theta_int is None')

        logger.info(f'best cat_idx is {cat_idx}')

        logger.info(f'best combination is {best_comb}')

        self.results_json = results_json

    #def finalize(self) :
    #    super().finalize()
    #    return None
