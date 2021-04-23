Machine learning task examples
==============================

Preparing toy data
------------------
.. code-block:: python
   
   import numpy as np

   from multiml import StoreGate

   storegate = StoreGate(backend='numpy', data_id='toy-dataset')

   phase = (0.8, 0.1, 0.1) # fraction of train, valid, test phase
   storegate.add_data('x0', np.random.rand(10), phase)
   storegate.add_data('x1', np.random.rand(10), phase)
   storegate.add_data('x2', np.random.rand(10), phase)
   storegate.add_data('x3', np.random.rand(10), phase)
   storegate.add_data('labels', np.random.randint(0, 2, 10), phase)
   storegate.compile()

   storegate.astype('labels', dtype=np.int8)
   storegate.show_info()
   
Executing single task
---------------------
Base arguments:
  * ``input_var_names`` (str or list): name of input variables in StoreGate
  * ``true_var_names`` (str or list): name of true variables (e.g. labels, ground truth, etc) in StoreGate
  * ``output_var_names`` (str or list): name of output variables to be registered to StoreGate
  * ``model`` (str or obj): name of model, model class, model instance
  * ``model_args`` (dict): args passed to model
  * ``oprimizer`` (str or obj): name of optimizer, optimizer class, optimizer instance
  * ``optimizer_args`` (dict): args passed to optimizer, e.g. learning rate  
  * ``loss`` (str or obj): name of loss function, loss function class, loss function instance
  * ``loss_args`` (dict): args passed to loss function  

Pytorch implementaion
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   import torch.nn as nn

   from multiml.task.pytorch import PytorchBaseTask

   # your pytorch model
   class MyPytorchModel(nn.Module):
       def __init__(self, inputs=2, outputs=2):
           super(MyPytorchModel, self).__init__()

           self.fc1 = nn.Linear(inputs, outputs)
           self.relu = nn.ReLU()

       def forward(self, x):
           return self.relu(self.fc1(x))

   # create task instance
   subtask = PytorchBaseTask(storegate=storegate,
                             model=MyPytorchModel,
                             input_var_names=('x0', 'x1'),
                             output_var_names='outputs-pytorch',
                             true_var_names='labels',
                             optimizer='SGD',
                             optimizer_args=dict(lr=0.1),
                             loss='CrossEntropyLoss')

   # set hyperparameters
   subtask.set_hps({'num_epochs': 5})

   # execute
   subtask.execute()

Keras implementaion
^^^^^^^^^^^^^^^^^^^
.. code-block:: python

   from tensorflow.keras import Model
   from tensorflow.keras.layers import Dense, ReLU

   from multiml.task.keras import KerasBaseTask
   
   # your keras model
   class MyKerasModel(Model):
       def __init__(self, units=1):
           super(MyKerasModel, self).__init__()

           self.dense = Dense(units)
           self.relu = ReLU()

       def call(self, x):
           return self.relu(self.dense(x))
           
   # create task instance
   subtask = KerasBaseTask(storegate=storegate,
                           model=MyKerasModel,
                           input_var_names=('x0', 'x1'),
                           output_var_names='outputs-keras',
                           true_var_names='labels',
                           optimizer='adam',
                           optimizer_args=dict(lr=0.1),
                           loss='binary_crossentropy')
   # set hyperparameters
   subtask.set_hps({'num_epochs': 5})

   # execute
   subtask.execute()

Sklean implementaion
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python


   from sklearn.feature_selection import SelectKBest, chi2

   from multiml.task import SkleanPipelineTask

   # set sklean pipeline model and create task instance
   subtask = SkleanPipelineTask(storegate=storegate,
                                model=SelectKBest,
                                model_args=dict(score_func=chi2, k=2),
                                input_var_names=('x0', 'x1', 'x2', 'x3'),
                                output_var_names=('k0', 'k1'),
                                true_var_names='labels')

   # execute
   subtask.execute()

Connecting differentiable deep learning models 
----------------------------------------------

Connecting multi-step tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following example connects two-step tasks and creates a differentiable deep learning model.

.. image:: _static/connection_model.png

.. code-block:: python

   from multiml.task.pytorch import ModelConnectionTask
   #from multiml.task.keras import ModelConnectionTask

   # create subtask instances to be connected, Pytorch can be replaced with Keras
   common_args = dict(storegate=storegate,
                      optimizer='SGD',
                      optimizer_args=dict(lr=0.1))

   subtask0 = PytorchBaseTask(model=MyPytorchModel(2, 2),
                              input_var_names=('x0', 'x1'),
                              output_var_names=('output0-pytorch', 'output1-pytorch'),
                              true_var_names=('x2', 'x3'),
                              loss='MSELoss',
                              **common_args)

   subtask1 = PytorchBaseTask(model=MyPytorchModel(3, 2),
                              input_var_names=('output0-pytorch', 'output1-pytorch', 'x2'),
                              output_var_names='output2-pytorch',
                              true_var_names='labels',
                              loss='CrossEntropyLoss',
                              **common_args)

   # connect subtasks 
   subtask = ModelConnectionTask(subtasks=[subtask0, subtask1],
                                 **common_args)

   # set hyperparameters
   subtask.set_hps({'num_epochs': 5})

   # execute
   subtask.execute()

Ensemble of multiple tasks
^^^^^^^^^^^^^^^^^^^^^^^
The following example connects multiple tasks in parallel and creates a differentiable deep learning model. For now, only Keras models are supported.

.. image:: _static/ensemble_model.png

.. code-block:: python

   from multiml.task.keras import EnsembleTask

   # create subtask instances to be connected
   common_args = dict(storegate=storegate,
                      optimizer='SGD',
                      optimizer_args=dict(lr=0.1),
                      loss='binary_crossentropy')

   subtask0 = KerasBaseTask(model=MyKerasModel,
                            input_var_names=('x0', 'x1', 'x2'),
                            output_var_names=('output0-keras'),
                            true_var_names='labels',
                            **common_args)

   subtask1 = KerasBaseTask(model=MyKerasModel,
                            input_var_names=('x0', 'x1', 'x2'),
                            output_var_names=('output1-keras',),
                            true_var_names='labels',
                            **common_args)

   subtask2 = KerasBaseTask(model=MyKerasModel,
                            input_var_names=('x0', 'x1', 'x2'),
                            output_var_names=('output2-keras',),
                            true_var_names='labels',
                            **common_args)

   # connect subtasks
   subtask = EnsembleTask(subtasks=[subtask0, subtask1, subtask2],
                          output_var_names=('outputs',),
                          **common_args)

   # set hyperparameters
   subtask.set_hps({'num_epochs': 5})

   # execute
   subtask.execute()
