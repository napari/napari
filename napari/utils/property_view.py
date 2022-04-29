from wrapt import ObjectProxy


class View(ObjectProxy):
    """
    Proxy object that wraps a mutable object so that any changes
    to it or its items/attributes are recursively redirected to the ancestor
    """

    def __init__(self, wrapped, parent, key=None, attr=None):
        if isinstance(wrapped, View):
            wrapped = wrapped.__wrapped__
        if not (key is None) ^ (attr is None):
            raise ValueError('exactly one of key or attr must be set')
        super().__init__(wrapped)
        self._self_parent = parent
        self._self_key = key
        self._self_attr = attr

    def __getattribute__(self, name):
        # _self_ methods are special for ObjectProxy, and we never really want views on dunder methods
        # anything else should be safe to proxy
        if name.startswith('_self_') or (
            name.startswith('__') and name.endswith('__')
        ):
            return super().__getattribute__(name)
        return View(getattr(self.__wrapped__, name), attr=name, parent=self)

    def __getitem__(self, key):
        return View(self.__wrapped__[key], key=key, parent=self)

    def __setattr__(self, name, value):
        if name.startswith('_self_') or (
            name.startswith('__') and name.endswith('__')
        ):
            super().__setattr__(name, value)
            return

        old = getattr(self.__wrapped__, name)
        setattr(self.__wrapped__, name, value)
        try:
            self.__supersetter__()
        except Exception:
            # something went wrong along the recursion; undo it
            # TODO use psygnal pause events to avoid triggering if unnecessary?
            setattr(self.__wrapped__, name, old)
            self.__supersetter__()
            raise

    def __setitem__(self, key, value):
        old = self.__wrapped__[key]
        self.__wrapped__[key] = value
        try:
            self.__supersetter__()
        except Exception:
            # something went wrong along the recursion; undo it
            # TODO use psygnal pause events to avoid triggering if unnecessary?
            self.__wrapped__[key] = old
            self.__supersetter__()
            raise

    def __supersetter__(self):
        """
        Recursively update all the parents
        """
        if isinstance(self._self_parent, View):
            target = self._self_parent.__wrapped__
        else:
            target = self._self_parent

        if self._self_key is not None:
            target[self._self_key] = self.__wrapped__
        elif self._self_attr is not None:
            setattr(target, self._self_attr, self.__wrapped__)
        else:
            raise Exception('???')

        # keep going
        if isinstance(self._self_parent, View):
            self._self_parent.__supersetter__()

    def __repr__(self):
        return f'View({repr(self.__wrapped__)})'

    def __str__(self):
        return repr(self)

    # need __call__ to be defined for callables to work
    # TODO: how to programmatically add this only for callable wrapped
    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)
