"""Constant module.

Attributes:
    TRAIN (str): constant str to indicate *train* phase.
    VALID (str): constant str to indicate *valid* phase.
    TEST (str): constant str to indicate *test* phase.
    PHASES (list): constant list of TRAIN, VALID and TEST.
    INVALID (int): integer to indicate invalid value.
    PBAR_FORMAT (str): format of tqdm progress bar.
    PBAR_ARGS (str): args of tqdm progress bar.

Examples:
    >>> from multiml import const
    >>>
    >>> phase = 'train'
    >>> if phase == const.TRAIN:
    >>>     pass
    >>> if phase in const.PHASES:
    >>>     pass
"""

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
PHASES = (TRAIN, VALID, TEST)
INVALID = -9999.

PBAR_FORMAT = '{desc}: {percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]'
PBAR_ARGS = dict(unit=' batch', ncols=150, bar_format=PBAR_FORMAT)
