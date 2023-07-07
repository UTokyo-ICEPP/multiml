.. multiml documentation master file, created by
   sphinx-quickstart on Wed Mar 24 17:21:28 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to multiml's documentation!
===================================
multiml is a prototype framework for developing multi-step machine learnings.

.. image:: _static/classes.png

Quick start
===========
This section runs through the APIs to demonstrate Grid Search optimization. 

Installation
-------
Requirements:
  * CentosOS 7.6+
  * Python 3.8+

.. code-block:: bash

   $ git clone https://github.com/UTokyo-ICEPP/multiml.git
   $ cd multiml
   $ pip install -e .
   
Preparing data (``StoreGate``)  
--------------------------
.. code-block:: python

   import numpy as np
   from multiml import StoreGate
   
   storegate = StoreGate(data_id='dataset0')
   phase = (0.8, 0.1, 0.1) # fraction of train, valid, test
   storegate.add_data(var_names='data', data=np.arange(0, 10), phase=phase)
   storegate.add_data(var_names='true', data=np.arange(0, 10), phase=phase)
   storegate.compile()
   storegate.show_info()

Out:

.. code-block:: bash

   ================================================================================
   data_id : dataset0, compiled : True
   --------------------------------------------------------------------------------
   phase  backend  var_names       var_types       total_events    var_shape      
   ================================================================================
   train  numpy    data            int64           8               (8,)
   train  numpy    true            int64           8               (8,)
   --------------------------------------------------------------------------------
   phase  backend  var_names       var_types       total_events    var_shape      
   ================================================================================
   valid  numpy    data            int64           1               (1,)
   valid  numpy    true            int64           1               (1,)
   --------------------------------------------------------------------------------
   phase  backend  var_names       var_types       total_events    var_shape      
   ================================================================================
   test   numpy    data            int64           1               (1,)
   test   numpy    true            int64           1               (1,)
   ================================================================================
   
Please see :doc:`StoreGate tutorial <storegate>` for more details.

Impementing algorithms (``Task``)  
------------------------------
.. code-block:: python

   from multiml import logger
   from multiml.task import BaseTask
   
   class MyTask(BaseTask):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self._weight = None

       def execute(self):
           if self._weight is None:
               return # no hyperparameter

           data = self.storegate['test']['data'][:]
           pred = data * self._weight

           logger.info(f'pred value = {pred}')
           self.storegate['test']['pred'][:] = pred
                                              
   task = MyTask(storegate=storegate)
   task.set_hps(dict(weight=0.5)) # set hyperparameter 
   task.execute()

Out:

.. code-block:: bash

   pred value = [4.5]

Please see :doc:`machine learning task examples <mltask>`.
 
Registering tasks (``TaskScheduler``)  
---------------------------------
.. code-block:: python

   from multiml import TaskScheduler
   
   task0 = MyTask()
   task1 = MyTask()
   hps0 = dict(weight=[0.5, 1.0, 1.5])
   
   steps = [[(task0, hps0)], [(task1, None)]]
   task_scheduler = TaskScheduler(steps)
   task_scheduler.show_info()

Out:

.. code-block:: bash

   --------------------------------------------------------------------------------
   task_id: step0, DAG: True (parents: [], children: ['step1']):
   subtask_id: MyTask, hps: ['weight']
   --------------------------------------------------------------------------------
   task_id: step1, DAG: True (parents: ['step0'], children: []):
   subtask_id: MyTask, hps: []
   --------------------------------------------------------------------------------

Optimization (``Agent``)  
--------------------
.. code-block:: python

   from multiml.agent import GridSearchAgent
   
   # minimize Mean Squared Error
   agent = GridSearchAgent(storegate=storegate, 
                           task_scheduler=task_scheduler, 
                           metric='MSE')
   agent.execute_finalize()

Out:

.. code-block:: bash
   
   (1/3) events processed (metric=20.25)
   (2/3) events processed (metric=0.0)
   (3/3) events processed (metric=20.25)
   ------------------------------------ Result ------------------------------------
   task_id step0 and subtask_id MyTask with:
     weight = 1.0
     job_id = 1
   task_id step1 and subtask_id MyTask with:
     job_id = 1
   Metric (mse) is 0.0

weight = 1.0 shows the best performance as expected.

API references
==============

.. toctree::
   :maxdepth: 4

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
