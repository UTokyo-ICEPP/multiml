import inspect
import copy

import torch

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


def inputs_size(inputs):
    if isinstance(inputs, torch.Tensor):
        result = inputs.size(0)
    else:
        result = inputs[0].size(0)
    return result


class EarlyStopping:
    def __init__(self, patience=7, save=False, path=None):
        self.patience = patience
        self.save = save
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model = None
        self.theta_cat = None
        self.theta_int = None
        logger.info(f'EarlyStopping with {self.patience}')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = self.save_checkpoint(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            logger.debug(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = self.save_checkpoint(val_loss, model)

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        from copy import deepcopy

        from torch import save
        logger.debug(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
            + 'updating model ...')
        if self.save:
            from os.path import isdir, join
            if isdir(self.path):
                save_path = join(self.path, 'checkpoint.pt')
            else:
                save_path = self.path
            save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        return deepcopy(model)


class ASNG_EarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        from copy import deepcopy

        from torch import save
        logger.debug(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
            + 'updating model ...')
        if self.save:
            from os.path import isdir, join
            if isdir(self.path):
                save_path = join(self.path, 'checkpoint.pt')
            else:
                save_path = self.path
            save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        self.theta_cat, self.theta_int = model.get_thetas()
        return deepcopy(model)

    def get_thetas(self):
        return self.theta_cat, self.theta_int
