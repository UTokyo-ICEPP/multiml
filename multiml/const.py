"""Constant module.

Attributes:
    TRAIN (str): constant str to indicate *train* phase.
    VALID (str): constant str to indicate *valid* phase.
    TEST (str): constant str to indicate *test* phase.
    PHASES (list): constant list of TRAIN, VALID and TEST.

Examples:
    >>> from multiml import const
    >>> 
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
