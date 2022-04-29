from wrapt import ObjectProxy


class _BlackHole:
    """
    Dummy object that accepts setattr and setitem and does nothing
    """

    def __setattr__(self, name, value):
        pass

    def __setitem__(self, key, value):
        pass


class View(ObjectProxy):
    """
    Proxy object that wraps a mutable object so that any changes
    to it or its items/attributes are recursively redirected to the ancestor
    """

    def __init__(self, wrapped, parent=None, key=None, attr=None):
        if isinstance(wrapped, View):
            wrapped = wrapped.__wrapped__
        super().__init__(wrapped)
        self._self_parent = parent
        self._self_key = key
        self._self_attr = attr

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
            self == View(parent.__wrapped__)[1]
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
        return View(getattr(self.__wrapped__, name), attr=name, parent=self)

    def __setattr__(self, name, value):
        if name.startswith('_self_') or (
            name.startswith('__') and name.endswith('__')
        ):
            super().__setattr__(name, value)
            return

        old = getattr(self.__wrapped__, name)
        setattr(self.__wrapped__, name, value)
        setattr(self._self_ascend(), name, value)

        try:
            self._self_call_setter()
        except Exception:
            setattr(self.__wrapped__, name, old)
            setattr(self._self_ascend(), name, old)
            raise

    def __getitem__(self, key):
        # recursively return views so you can index as deep as you want
        return View(self.__wrapped__[key], key=key, parent=self)

    def __setitem__(self, key, value):
        old = self.__wrapped__[key]
        self.__wrapped__[key] = value
        self._self_ascend()[key] = value
        try:
            self._self_call_setter()
        except Exception:
            self.__wrapped__[key] = old
            self._self_ascend()[key] = old
            raise

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    def __repr__(self):
        return f'View({repr(self.__wrapped__)})'

    def __str__(self):
        return repr(self)


class property_view(property):
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        # use setattr in case anything else needs to happen other than just the property
        # setter (for example events in EventedModel)
        def setter(value):
            setattr(obj, self._name, value)

        return View(super().__get__(obj, objtype), setter=setter)


def field_view(model, field, value):
    def setter(value):
        setattr(model, field, value)

    return View(value, setter=setter)
