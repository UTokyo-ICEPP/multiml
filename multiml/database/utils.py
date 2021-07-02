"""Utility module for Database."""


def get_slice(index):
    """Returns a slice converted from index."""

    if index == -1:
        return slice(0, None)

    if isinstance(index, int):
        # return slice(index, index+1)
        return index

    if isinstance(index, tuple):
        return slice(index[0], index[1])

    if isinstance(index, slice):
        return index

    raise ValueError(f'Un supported index: {index}')
