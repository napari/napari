from ..layers import Layer


# dict of LayerView base classes (direct subclass of LayerView)
# -> dict of Layer -> LayerView base class subclasses
_baseclass_layer_to_subclass = {}


class LayerView:
    """Non-instantiable mix-in class for views corresponding to
    an aspect of a layer.

    Direct subclasses of ``LayerView`` (ones that explicitly list it as a base)
    are considered **base classes**.

    Classes that inherit from **base classes** can specify the ``layer``
    class definition keyword argument
    to associate themselves with a specific layer type.

    When a **base class** is initialized with a layer of that type, it will
    return an instance of the class associated with that layer type.

    Parameters
    ----------
    layer : Layer
        Layer to initialize the view with.

    Notes
    -----
    Subclasses **must** have the same signature as ``LayerView``.

    Examples
    --------
    >>> class Foo(LayerView): ...
    >>> class ImageFoo(Foo, layer=Image): ...
    >>> foo = Foo(Image(...))
    >>> type(foo).__name__
    'ImageFoo'
    """

    def __new__(cls, layer):
        if cls is LayerView:
            raise TypeError(f'cannot instantiate LayerView class')

        try:
            # if this is a base class and it has a layer type
            # associated with one of its subclasses, switch to that class
            cls = _baseclass_layer_to_subclass[cls][type(layer)]
        except KeyError:
            pass

        return super().__new__(cls)

    def __init_subclass__(cls, layer=None, **kwargs):
        maps = _baseclass_layer_to_subclass

        for baseclass in maps:
            if issubclass(cls, baseclass):
                if layer is not None and issubclass(layer, Layer):
                    # associate a layer type with this class
                    maps[baseclass][layer] = cls
                break
        else:  # this is not a subclass of any base classes
            # so this is a new base class
            maps[cls] = {}

        super().__init_subclass__(**kwargs)
