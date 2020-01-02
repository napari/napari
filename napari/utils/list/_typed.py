from ._base import List


def cpprint(obj):
    return obj.__name__


class TypedList(List):
    """Enforce list elements to be of a specific type and allow indexing with
    their unique properties.

    Parameters
    ----------
    basetype : type
        Type of the elements in the list.
    iterable : iterable, optional
        Elements to initialize the list with.
    lookup : dict of type : function(object, ``basetype``) -> bool
        Functions that determine if an object is a reference to an
        element of the list.
    """

    def __init__(self, basetype, iterable=(), lookup=None):
        if lookup is None:
            lookup = {}
        self._basetype = basetype
        self._lookup = lookup
        super().__init__(self._check(e) for e in iterable)

    def __setitem__(self, key, value):
        self._check(value)
        super().__setitem__(key, value)

    def __locitem__(self, key):
        if not isinstance(key, int):
            key = self.index(key)
        return super().__locitem__(key)

    def __newlike__(self, iterable):
        cls = type(self)
        return cls(self._basetype, iterable, self._lookup)

    def _check(self, e):
        if not isinstance(e, self._basetype):
            raise TypeError(
                f'expected {cpprint(self._basetype)}; '
                f'got {cpprint(type(e))}'
            )
        return e

    def insert(self, key, object):
        self._check(object)
        super().insert(key, object)

    def append(self, object):
        self._check(object)
        super().append(object)

    def index(self, value, start=None, stop=None):
        q = value
        basetype = self._basetype

        if not isinstance(q, basetype):
            lookup = self._lookup

            for t in lookup:
                if isinstance(q, t):
                    break
            else:
                raise TypeError(
                    f'expected object of type {cpprint(basetype)} '
                    f'or one of {set(cpprint(t) for t in lookup)}; '
                    f'got {cpprint(type(q))}'
                )

            ref = lookup[t]

            for e in self[start:stop]:
                if ref(q, e):
                    break
            else:
                raise KeyError(f'could not find element {q} was referencing')

            q = e

        return super().index(q)
