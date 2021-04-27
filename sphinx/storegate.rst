StoreGate tutorial
==============================

Adding data
-----------
Base arguments:
  * ``backend`` (str): ``numpy``, ``zarr`` or ``hybrid``. ``numpy`` is suitable for small dataset, and ``zarr`` is suitable for large dataset that exceds capacity of memory 
  * ``data_id``` (str): unique identifer of dataset
  * ``backend_args`` (dict): args passed to backend database, e.g. directory path to zarr
  
.. code-block:: python

   >>> import numpy as np
   >>> from multiml import StoreGate

   >>> storegate = StoreGate(backend='numpy', data_id='toy-dataset')

   >>> phase = (0.8, 0.1, 0.1) # fraction of train, valid, test phase
   >>> storegate.add_data('x0', np.array(range(5)), phase)
   >>> storegate.add_data('x1', np.array(range(5)), phase)
   >>> storegate.compile()
   >>> storegate.show_info()
   [I] ================================================================================
   [I] data_id : toy-dataset, compiled : True
   [I] --------------------------------------------------------------------------------
   [I] phase  backend  var_names       var_types       total_events    var_shape      
   [I] ================================================================================
   [I] train  numpy    x0              int64           3               ()
   [I] train  numpy    x1              int64           3               ()
   [I] --------------------------------------------------------------------------------
   [I] phase  backend  var_names       var_types       total_events    var_shape      
   [I] ================================================================================
   [I] valid  numpy    x0              int64           1               ()
   [I] valid  numpy    x1              int64           1               ()
   [I] --------------------------------------------------------------------------------
   [I] phase  backend  var_names       var_types       total_events    var_shape      
   [I] ================================================================================
   [I] test   numpy    x0              int64           1               ()
   [I] test   numpy    x1              int64           1               () 
   [I] ================================================================================  
   
   
Retrieving data
---------------
Base arguments:
  * ``var_names`` (str or tuple or list): name of variables, please see the examples below
  * ``phase`` (str): ``train``, ``valid`` or ``test``
  
.. code-block:: python

   >>> # retrive all x0 data from train phase
   >>> storegate['train']['x0'][:]
   [0 1 2]
   
   >>> # indedexing 
   >>> storegate['train']['x0'][0]
   0
   >>> storegate['train']['x0'][0:2]
   [0 1]
   
   >>> # retrive all x0 and x1 data from train phase, 
   >>> # if tuple is given, numpy.ndarray are returned,
   >>> data = storegate['train'][('x0', 'x1')][:]
   [[0 0]
    [1 1]
    [2 2]]
   >>> # if list is given, list of numpy.ndarray for each variable are returned 
   >>> data = storegate['train'][['x0', 'x1']][:]
   [array([0, 1, 2]), array([0, 1, 2])]

   >>> # loop the StoreGate
   >>> for index, data in enumerate(storegate['train'][('x0', 'x1')]):
   >>>     print (index, data)
   0 [0 0]
   1 [1 1]
   2 [2 2]

   
