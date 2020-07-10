from collections.abc import Iterable


class List(list):
    """Inheritable list that better connects related/dependent functions,
    allowing for an easier time making modifications with reusable components.

    It has the following new methods:
    `__locitem__(key)` : transform a key into the index of its corresponding item  # noqa
    `__prsitem__(key)` : parse a key such as `0:1` into indices
    `__newlike__(iterable)` : create a new instance given an iterable

    TODO: handle operators (e.g. +, *, etc.)
    """

    def __contains__(self, key):
        try:
            self.index(key)
            return True
        except ValueError:
            return False

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        indices = self.__prsitem__(key)
        can_iter = isinstance(indices, Iterable)

        if can_iter:
            return self.__newlike__(
                super(List, self).__getitem__(i) for i in indices
            )

        return super().__getitem__(indices)

    def __setitem__(self, key, value):
        super().__setitem__(self.__locitem__(key), value)

    def __delitem__(self, key):
        super().remove(key)

    def __prsitem__(self, key):
        """Parse a key into list indices.

        Default implementation handles slices

        Parameters
        ----------
        key : any
            Key to parse.

        Returns
        -------
        indices : int or iterable of int
            Key parsed into indices.
        """
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step

            if start is not None:
                try:
                    start = self.__locitem__(start)
                except IndexError:
                    if start != len(self):
                        raise

            if stop is not None:
                try:
                    stop = self.__locitem__(stop)
                except IndexError:
                    if stop != len(self):
                        raise

            return range(*slice(start, stop, step).indices(len(self)))
        else:
            return self.__locitem__(key)

    def __locitem__(self, key):
        """Parse a key into a list index.

        Default implementation handles integers.

        Parameters
        ----------
        key : any
            Key to parse.

        Returns
        -------
        index : int
            Location of the object ``key`` is referencing.

        Raises
        ------
        IndexError
            When the index is out of bounds.
        KeyError
            When the key is otherwise invalid.
        """
        if not isinstance(key, int):
            raise TypeError(f'expected int; got {type(key)}')

        if key < 0:
            key += len(self)

        if not (0 <= key < len(self)):
            raise IndexError(
                f'expected index to be in [0, {len(self)}); got {key}'
            )

        return key

    def __newlike__(self, iterable):
        """Create a new instance from an iterable with the same properties
        as this one.

        Parameters
        ----------
        iterable : iterable
            Elements to make the new list from.

        Returns
        -------
        new_list : List
            New list created from the iterable with the same properties.
        """
        cls = type(self)
        return cls(iterable)

    def copy(self):
        return self.__newlike__(self)

    def count(self, key):
        super().count(self.__locitem__(key))

    def extend(self, iterable):
        for e in iterable:
            self.append(e)

    def insert(self, index, item):
        super().insert(self.__locitem__(index), item)

    def pop(self, index):
        return super().pop(self.__locitem__(index))

    def remove(self, item):
        self.pop(self.__locitem__(item))
