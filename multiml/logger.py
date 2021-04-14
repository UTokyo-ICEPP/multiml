"""Logger module.

In principle, modules in multiml library use this logger.
There are some print levels: ``DEBUG``, ``INFO``, ``WARN``, ``ERROR`` and
``DISABLED``.

Examples:
    >>> from multiml import logger
    >>> logger.set_level(logger.INFO)  # Set to DEBUG
    >>> logger.debug("This message is not printed at INFO level.")
"""

import datetime
import functools

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

MIN_LEVEL = 20


def set_level(level):
    """ Set log level.

    Args:
        level (int): ``DEBUG``=10, ``INFO``=20, ``WARN``=30, ``ERROR``=40,
            ``DISABLED``=50.
    """
    global MIN_LEVEL
    MIN_LEVEL = level


def get_now():
    """ Get current time with ``%Y-%m-%d %H:%M:%S`` formant.

    Returns:
        str: the current timestamp. 
    """
    dt_now = datetime.datetime.now()

    return dt_now.strftime('%Y-%m-%d %H:%M:%S')


def debug(msg, *args):
    """ Show debug [D] message.
    """
    if MIN_LEVEL <= DEBUG:
        print(f'{get_now()} [D] {msg % args}')


def info(msg, *args):
    """ Show information [I] message.
    """
    if MIN_LEVEL <= INFO:
        print(f'{get_now()} [I] {msg % args}')


def warn(msg, *args):
    """ Show warning [W] message.
    """
    if MIN_LEVEL <= WARN:
        print(f'{get_now()} [W] {msg % args}')


def error(msg, *args):
    """ Show error [E] message.
    """
    if MIN_LEVEL <= ERROR:
        print(f'{get_now()} [E] {msg % args}')


def counter(count, max_counts, divide=1, message=None):
    """ Show process counter as information.

    >>> ({count}/{max_counts}) events processed (message)
    """
    if count == 0:
        return

    if (count % divide == 0) or (count == max_counts):
        if message is None:
            info(f'({count}/{max_counts}) events processed')
        else:
            info(f'({count}/{max_counts}) events processed ({message})')


def header1(message, level=info):
    """ Show the following header.

    >>> =================================
    >>> ============ message ============
    >>> =================================
    """
    if len(message) % 2 == 1:
        message += ' '
    len1 = 80
    len2 = (len1 - len(message) - 2) // 2
    level("=" * len1)
    if message == '':
        level(("=" * len2) + '==' + ("=" * len2))
    else:
        level(("=" * len2) + ' ' + message + ' ' + ("=" * len2))
    level("=" * len1)


def header2(message, level=info):
    """ Show the following header.

    >>> ------------ message ------------
    """
    if len(message) % 2 == 1:
        message += ' '
    len1 = 80
    len2 = (len1 - len(message) - 2) // 2
    if message == '':
        level(("-" * len2) + '--' + ("-" * len2))
    else:
        level(("-" * len2) + ' ' + message + ' ' + ("-" * len2))


def header3(message, level=info):
    """ Show the following header.

    >>> ============ message ============
    """
    if len(message) % 2 == 1:
        message += ' '
    len1 = 80
    len2 = (len1 - len(message) - 2) // 2
    if message == '':
        level(("=" * len2) + '==' + ("=" * len2))
    else:
        level(("=" * len2) + ' ' + message + ' ' + ("=" * len2))


def logging(func):
    """ Show the header and footer indicating start and end algorithm.

    Examples:
        >>> @logger.logging
        >>> def your_func(arg0, arg1):
        >>>     pass
    """
    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        header2(f'{obj.__class__.__name__} {func.__qualname__} START', debug)
        debug(f'args={args} kwargs={kwargs}')
        rtn = func(obj, *args, **kwargs)
        debug(f'return={rtn}')
        header2(f'{obj.__class__.__name__} {func.__qualname__} END', debug)
        return rtn

    return wrapper
