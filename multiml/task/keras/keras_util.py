import copy
import inspect

from multiml import logger


def compile(obj, obj_args, modules):
    # str object
    if isinstance(obj, str):
        return getattr(modules, obj)(**obj_args)

    # class object
    elif inspect.isclass(obj):
        return obj(**obj_args)

    # instance object
    else:
        if obj_args != {}:
            logger.warn('instance object is given but args is also provided')
        return copy.copy(obj)


def training_keras_model(model,
                         num_epochs,
                         batch_size,
                         max_patience,
                         x_train,
                         y_train,
                         x_valid,
                         y_valid,
                         chpt_path=None,
                         callbacks=['EarlyStopping', 'ModelCheckpoint', 'TensorBoard'],
                         tensorboard_path=None):
    """ Training keras model

    Args:
        num_epochs (int): maximum number of epochs
        batch_size (int): mini-batch size
        max_patience (int): maximum patience for early stopping
        x_train (np.darray): input array for training
        y_train (np.darray): output array for training
        x_valid (np.darray): input array for validation
        y_valid (np.darray): output array for validation
        chpt_path (str): path for Keras check-point saving. If None, temporary directory will be used.
        callbacks (str or keras.Callback): callback for keras model training. Predefined callbacks (EarlyStopping, ModelCheckpoint, and TensorBoard) can be selected by str. Other user-defined callbacks should be given as keras.Callback object.
        tensorboard_path (str): Path for tensorboard callbacks. If None, tensorboard callback is not used.

    Returns:
        dict: training results, which contains loss histories.
    """

    if chpt_path is None:
        import tempfile
        tmpdir = tempfile.TemporaryDirectory()
        chpt_path = f'{tmpdir.name}/tf_chpt'

    logger.info(f'chpt_path = {chpt_path}')

    cbs = []
    from tensorflow.keras.callbacks import Callback
    for callback in callbacks:
        if callback == 'EarlyStopping':
            if max_patience is not None:
                from tensorflow.keras.callbacks import EarlyStopping
                es_cb = EarlyStopping(monitor='val_loss',
                                      patience=max_patience,
                                      verbose=0,
                                      mode='min',
                                      restore_best_weights=True)
                cbs.append(es_cb)

        elif callback == 'ModelCheckpoint':
            from tensorflow.keras.callbacks import ModelCheckpoint
            cp_cb = ModelCheckpoint(filepath=chpt_path,
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='min')
            cbs.append(cp_cb)

        elif callback == 'TensorBoard':
            if tensorboard_path is not None:
                from tensorflow.keras.callbacks import TensorBoard
                tb_cb = TensorBoard(log_dir=tensorboard_path,
                                    histogram_freq=1,
                                    profile_batch=5)
                cbs.append(tb_cb)

        elif issubclass(callback.__class__, Callback):
            cbs.append(callback)

        else:
            raise ValueError(f"{callback} is not supported.")

    training_verbose_mode = 0
    if logger.MIN_LEVEL <= logger.DEBUG:
        training_verbose_mode = 1

    history = model.fit(x_train,
                        y_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=(x_valid, y_valid),
                        callbacks=cbs,
                        verbose=training_verbose_mode)

    # Save loss history
    loss_train = history.history.get('loss', [-1])
    loss_valid = history.history.get('val_loss', [-1])
    return {
        'loss_train': loss_train,
        'loss_valid': loss_valid,
    }


def get_optimizer(optimizer, optimizer_args=None):
    """ Get Keras optimizer

    Args:
        optimizer (str or obj): If str, the corresponding optimizer is searched for in the keras class.
        learning_rate (float): learning rate for the optimizer
    """
    if optimizer_args is None:
        optimizer_args = {}

    from tensorflow.keras import optimizers
    if optimizer is None:
        raise ValueError("optimizer is None. Please use a valid optimizer")

    elif not isinstance(optimizer, str):
        return optimizer
    elif optimizer == "sgd" or optimizer == "SGD":
        return optimizers.SGD(**optimizer_args)
    elif optimizer == "adam" or optimizer == "ADAM":
        return optimizers.Adam(**optimizer_args)

    else:
        raise NotImplementedError(f"{optimizer} is not defined")
