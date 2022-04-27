class PropertyView:
    """
    Proxy object that wraps the return value of a property so that any changes
    to it are redirected to the property setter (therefore changing the parent object)
    """

    def __init__(self, viewed, key, parent, prop=None):
        self._viewed = viewed
        self._key = key
        self._parent = parent
        self._prop = prop

    def _call_setter(self):
        # this won't fire for nested views, so only the top level does something
        if self._prop is not None:
            self._prop.fset(self._parent, self._viewed)

    def __getattribute__(self, name):
        # proxy as much as possible
        if name in ('_viewed', '_key', '_prop', '_parent', '_call_setter'):
            return super().__getattribute__(name)
        return getattr(self._viewed, name)

    def __getitem__(self, k):
        # recursively return views so you can index as deep as you want
        return PropertyView(self._viewed[k], k, self)

    def __setitem__(self, k, v):
        self._viewed[k] = v
        if self._prop is None:
            # if nested, update the *parent*. (directly the viewed object to avoid recursion)
            self._parent._viewed[self._key][k] = v
            self._parent._call_setter()
        else:
            # call directly setter
            self._call_setter()

    def __iter__(self, k, v):
        yield from self._viewed

    def __repr__(self):
        return f'View({repr(self._viewed)})'

    # more proxy methods


class property_view(property):
    def __get__(self, obj, objtype=None):
        return PropertyView(
            super().__get__(obj, objtype), key=None, parent=obj, prop=self
        )
