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

def convert(level):
    """ convert str to int
    
    """
    ret = INFO
    if level.upper() == 'DEBUG' : 
        ret = DEBUG
    elif level.upper() == 'INFO' : 
        ret = INFO
    elif level.upper() == 'WARN' : 
        ret = WARN
    elif level.upper() == 'ERROR' : 
        ret = ERROR
    elif level.upper() == 'DISABLED' : 
        ret = DISABLED
    else : 
        print(f'Your choice({level}) is not valid, set to INFO')
        ret = INFO
    return ret
        
def set_level(level):
    """ Set log level.

    Args:
        level (int): ``DEBUG``=10, ``INFO``=20, ``WARN``=30, ``ERROR``=40,
            ``DISABLED``=50.
    """
    if type(level) == str: 
        level = convert(level)
    
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


def table(names, data, header=None, footer=None, max_length=30):
    """ Show table. All data must be str.
   
    >>> names = ['var0', 'var1']
    >>> data = [['hoge0', 'hoge1'], ['hoge2', 'hoge3']]
    >>> header = 'header message'
    >>> footer = 'footer message'
    >>> 
    >>> ==============
    >>> header message
    >>> ==============
    >>> var0   var1
    >>> --------------
    >>> hoge0  hoge1
    >>> hoge2  hoge3
    >>> --------------
    >>> footer message
    >>> ==============
    """
    lengths = [5] * len(names)

    for index, name in enumerate(names):
        if len(name) > lengths[index]:
            lengths[index] = len(name)
        if len(name) > max_length:
            lengths[index] = max_length

    for idata in data:
        for index, var in enumerate(idata):
            if len(var) > lengths[index]:
                lengths[index] = len(var)

            if len(var) > max_length:
                lengths[index] = max_length

    total_length = sum(lengths) + len(names) * 2

    # header
    if header is not None:
        info('=' * total_length)
        info(header)

    # names
    info('=' * total_length)
    message = ''
    for index, name in enumerate(names):
        name = name.ljust(lengths[index])
        message += f'{name[:max_length]}  '
    info(message)
    info('-' * total_length)

    # data
    for idata in data:
        if idata == '-':
            info('-' * total_length)
            continue

        message = ''
        for index, var in enumerate(idata):
            var = var.ljust(lengths[index])
            message += f'{var[:max_length]}  '
        info(message)
    info('=' * total_length)

    # footer
    if footer is not None:
        info(footer)
        info('=' * total_length)


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
