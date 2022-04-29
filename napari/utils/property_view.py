from wrapt import ObjectProxy


class _BlackHole:
    """
    Dummy object that accepts setattr and setitem and does nothing
    """

    def __setattr__(self, name, value):
        pass

    def __setitem__(self, key, value):
        pass


class PropertyView(ObjectProxy):
    """
    Proxy object that wraps the return value of a property so that any changes
    to it are redirected to the property setter (therefore changing the parent object)
    """

    def __init__(self, wrapped, parent=None, key=None, attr=None, setter=None):
        super().__init__(wrapped)
        self._self_parent = parent
        self._self_key = key
        self._self_attr = attr
        self._self_setter = setter

    def _self_call_setter(self):
        """
        Call the first ancestor's setter
        """
        if self._self_setter is None:
            self._self_parent._self_call_setter()
        else:
            self._self_setter(self.__wrapped__)

    def _self_ascend(self):
        """
        Return the same as self, but as an element/attribute of the parent's wrapped object.
        For example:
            self == PropertyView(parent.__wrapped__)[1]
            self._self_ascend() == parent.__wrapped__[1]
        This allows to cascade setting items/attributes up the chain without going recursive.
        """
        if self._self_key is not None:
            one_up = self._self_parent.__wrapped__[self._self_key]
        elif self._self_attr is not None:
            one_up = getattr(self._self_parent.__wrapped__, self._self_attr)
        else:
            one_up = _BlackHole()
        return one_up

    def __getattribute__(self, name):
        # _self_ methods are special for ObjectProxy, and we never really want views on dunder methods
        # anything else should be safe to proxy
        if name.startswith('_self_') or (
            name.startswith('__') and name.endswith('__')
        ):
            return super().__getattribute__(name)
        return PropertyView(
            getattr(self.__wrapped__, name), attr=name, parent=self
        )

    def __setattr__(self, name, value):
        if name.startswith('_self_') or (
            name.startswith('__') and name.endswith('__')
        ):
            super().__setattr__(name, value)
            return

        setattr(self.__wrapped__, name, value)
        setattr(self._self_ascend(), name, value)

        self._self_call_setter()

    def __getitem__(self, key):
        # recursively return views so you can index as deep as you want
        return PropertyView(self.__wrapped__[key], key=key, parent=self)

    def __setitem__(self, key, value):
        self.__wrapped__[key] = value
        self._self_ascend()[key] = value
        self._self_call_setter()

    def __repr__(self):
        return f'PropertyView({repr(self.__wrapped__)})'


class property_view(property):
    def __get__(self, obj, objtype=None):
        def bound_setter(x):
            self.fset(obj, x)

        return PropertyView(super().__get__(obj, objtype), setter=bound_setter)
