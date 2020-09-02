from collections.abc import Iterable

from ._base import List


class MultiIndexList(List):
    """Allow indexing with tuples.
    """

    def __prsitem__(self, keys):
        if not isinstance(keys, tuple):
            return super().__prsitem__(keys)

        indices = []

        for key in keys:
            i = self.__prsitem__(key)
            can_iter = isinstance(i, Iterable)
            if can_iter:
                indices.extend(i)
            else:
                indices.append(i)

        # TODO: how to handle duplicates?

        return indices

    def __setitem__(self, key, value):
        indices = self.__prsitem__(key)
        can_iter = isinstance(indices, Iterable)

        if can_iter:
            if hasattr(value, '__getitem__') and hasattr(value, '__len__'):
                # value is a vector
                if len(value) != len(indices):
                    raise ValueError(
                        f'expected {len(indices)} values; ' f'got {len(value)}'
                    )
                for o, i in enumerate(indices):
                    super().__setitem__(i, value[o])
            else:
                # value is a scalar
                for i in indices:
                    super().__setitem__(i, value)
        else:
            super().__setitem__(indices, value)

    def __delitem__(self, key):
        indices = self.__prsitem__(key)
        can_iter = isinstance(indices, Iterable)

        if can_iter:
            for i in indices:
                super().__delitem__(i)
        else:
            super().__delitem__(indices)
