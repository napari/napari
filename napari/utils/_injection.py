import sys
from concurrent.futures import Future
from contextlib import nullcontext, suppress
from functools import partial, wraps
from inspect import isgeneratorfunction, signature
from typing import Any, Callable, Dict, Optional, Set, Type, TypeVar

from typing_extensions import get_type_hints

from .. import components, layers, types, viewer
from ..layers._source import layer_source
from ..utils.misc import ensure_list_of_layer_data_tuple

T = TypeVar("T")
_NULL = object()


def _access_viewer() -> Optional[viewer.Viewer]:
    return get_accessor(viewer.Viewer)()


def _get_active_layer() -> Optional[layers.Layer]:
    return v.layers.selection.active if (v := _access_viewer()) else None


def _get_active_layer_list() -> Optional[components.LayerList]:
    return v.layers if (v := _access_viewer()) else None


# registry of Type -> "accessor function"
# where each value is a function that is capable
# of retrieving an instance of it's corresponding key type.
_ACCESSORS: Dict[Type, Callable[..., Optional[object]]] = {
    layers.Layer: _get_active_layer,
    viewer.Viewer: viewer.current_viewer,
    components.LayerList: _get_active_layer_list,
}


def get_accessor(type_: Type[T]) -> Optional[Callable[..., Optional[T]]]:
    """Return object accessor function given a type.

    An object accessor is a function that returns an instance of a
    particular object type. For example, given type `napari.Viewer`, we return
    a function that can be called to get the current viewer.

    This is a form of dependency injection, and, along with
    `inject_napari_dependencies`, allows us to inject current napari objects
    into functions based on type hints.
    """
    if type_ in _ACCESSORS:
        return _ACCESSORS[type_]

    if isinstance(type_, type):
        for key, val in _ACCESSORS.items():
            if issubclass(type_, key):
                return val  # type: ignore [return-type]
    return None


def _add_layer_data_tuples_to_viewer(
    data: Any, return_type=None, viewer=None, source: Optional[dict] = None
):
    """_summary_

    Parameters
    ----------
    data : Any
        _description_
    viewer : _type_, optional
        _description_, by default None
    return_type : _type_, optional
        _description_, by default None

    Raises
    ------
    TypeError
        If `data` is not a valid [list of] layer data tuple.

    """
    if viewer is None:
        viewer = _access_viewer()
    if viewer and data is not None:
        data = data if isinstance(data, list) else [data]
        for datum in ensure_list_of_layer_data_tuple(data):
            # then try to update a viewer layer with the same name.
            if len(datum) > 1 and (name := datum[1].get("name")):
                with suppress(KeyError):
                    layer = viewer.layers[name]
                    layer.data = datum[0]
                    for k, v in datum[1].items():
                        setattr(layer, k, v)
                    continue
            with layer_source(**source) if source else nullcontext():
                # otherwise create a new layer from the layer data
                viewer._add_layer_from_data(*datum)


def _add_layer_data_to_viewer(
    data: Any,
    return_type: Any,
    viewer: Optional[viewer.Viewer] = None,
    layer_name: Optional[str] = None,
    source: Optional[dict] = None,
):
    """Show a result in the viewer.

    Parameters
    ----------
    data : Any
        The result of the function call. For this function, this should be
        *just* the data part of the corresponding layer type.
    return_type : Any
        The return annotation that was used in the decorated function.
    viewer : Optional[Viewer]
        an optional viewer to use.  otherwise use current viewer.

    Examples
    --------
    This allows the user to do this, and add the result as a viewer Image.

    >>> def make_layer() -> napari.types.ImageData:
    ...     return np.random.rand(256, 256)

    """
    if data is not None and (viewer := viewer or _access_viewer()):
        if layer_name:
            with suppress(KeyError):
                viewer.layers[layer_name].data = data
                return
        layer_type = return_type.__name__.replace("Data", "").lower()
        with layer_source(**source) if source else nullcontext():
            getattr(viewer, f'add_{layer_type}')(data=data, name=layer_name)


def _add_layer_to_viewer(
    layer: layers.Layer,
    viewer: Optional[viewer.Viewer] = None,
    source: Optional[dict] = None,
):
    if layer is not None and (viewer := viewer or _access_viewer()):
        layer._source = layer.source.copy(update=source or {})
        viewer.add_layer(layer)


_FUTURES: Set[Future] = set()


def _add_future_data(
    future: Future,
    return_type: Any,
    _from_tuple=True,
    viewer: Optional[viewer.Viewer] = None,
    source: dict = None,
):
    """Process a Future object.

    This function will be called to process function that has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in `result` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    future : Future
        An instance of `concurrent.futures.Future` (or any third-party) object
        with the same interface, that provides `add_done_callback` and `result`
        methods.  When the future is `done()`, the `result()` will be added
        to the viewer.
    return_type : type
        The return annotation that was used in the decorated function.
    _from_tuple : bool, optional
        (only for internal use). True if the future returns `LayerDataTuple`,
        False if it returns one of the `LayerData` types.
    """

    # when the future is done, add layer data to viewer, dispatching
    # to the appropriate method based on the Future data type.
    adder = (
        _add_layer_data_tuples_to_viewer
        if _from_tuple
        else _add_layer_data_to_viewer
    )

    def _on_future_ready(f: Future):
        adder(
            f.result(),
            return_type=return_type,
            viewer=viewer,
            source=source,
        )
        _FUTURES.discard(future)

    # We need the callback to happen in the main thread...
    # This still works (no-op) in a headless environment, but
    # we could be even more granular with it, with a function
    # that checks if we're actually in a QApp before wrapping.
    # with suppress(ImportError):
    #     from superqt.utils import ensure_main_thread

    #     _on_future_ready = ensure_main_thread(_on_future_ready)

    future.add_done_callback(_on_future_ready)
    _FUTURES.add(future)


# add default processors
_PROCESSORS = {
    layers.Layer: _add_layer_to_viewer,
    types.LayerDataTuple: _add_layer_data_tuples_to_viewer,
}


for t in types._LayerData.__args__:
    _PROCESSORS[t] = partial(_add_layer_data_to_viewer, return_type=t)

    if sys.version_info >= (3, 9):
        _PROCESSORS[Future[t]] = partial(
            _add_future_data, return_type=t, _from_tuple=False
        )


def get_processor(type_: Type[T]) -> Optional[Callable[[], Optional[T]]]:
    """Return processor function for a given type.

    A processor is a function that can "process" a given return type.  The term
    process here leaves a lot of ambiguity, it mostly means the function "can
    do something" with a single input of the given type. For example, given
    type `napari.types.LayerData`, we return a function that can be called to
    add a layer data tuple to the currently active viewer.
    """
    if type_ in _PROCESSORS:
        return _PROCESSORS[type_]

    if isinstance(type_, type):
        for key, val in _PROCESSORS.items():
            if isinstance(key, type) and issubclass(type_, key):
                return val  # type: ignore [return-type]


class set_processor:
    """Set processor(s) for given type(s).

    "Processors" are functions that can "do something" with an instance of the
    type that they support.  For example, a processor that supports
    `napari.types.ImageData` might take the data and add an Image layer to the
    current viewer.

    This is a class that behaves as a function or a context manager, that
    allows one to set a processor function for a given type.

    Parameters
    ----------
    mapping : Dict[Type[T], Callable[..., Optional[T]]]
        a map of type -> processor function, where each value is a function
        that is capable of retrieving an instance of the associated key/type.
    clobber : bool, optional
        Whether to override any existing processor function, by default False.

    Raises
    ------
    ValueError
        if clobber is `True` and one of the keys in `mapping` is already
        registered.
    """

    def __init__(
        self, mapping: Dict[Type[T], Callable[..., Optional[T]]], clobber=False
    ):
        self._before = {}
        for k in mapping:
            if k in _PROCESSORS and not clobber:
                raise ValueError(
                    f"Class {k} already has a processor and clobber is False"
                )
            self._before[k] = _PROCESSORS.get(k, _NULL)
        _PROCESSORS.update(mapping)

    def __enter__(self):
        return None

    def __exit__(self, *_):
        for key, val in self._before.items():
            if val is _NULL:
                del _PROCESSORS[key]
            else:
                _PROCESSORS[key] = val


class set_accessor:
    """Set accessor(s) for given type(s).

    "Accessors" are functions that can retrieve an instance of a given type.
    For instance, `napari.viewer.current_viewer` is a function that can
    retrieve an instance of `napari.Viewer`.

    This is a class that behaves as a function or a context manager, that
    allows one to set an accessor function for a given type.

    Parameters
    ----------
    mapping : Dict[Type[T], Callable[..., Optional[T]]]
        a map of type -> accessor function, where each value is a function
        that is capable of retrieving an instance of the associated key/type.
    clobber : bool, optional
        Whether to override any existing accessor function, by default False.

    Raises
    ------
    ValueError
        if clobber is `True` and one of the keys in `mapping` is already
        registered.
    """

    def __init__(
        self, mapping: Dict[Type[T], Callable[..., Optional[T]]], clobber=False
    ):
        self._before = {}
        for k in mapping:
            if k in _ACCESSORS and not clobber:
                raise ValueError(
                    f"Class {k} already has an accessor and clobber is False"
                )
            self._before[k] = _ACCESSORS.get(k, _NULL)
        _ACCESSORS.update(mapping)

    def __enter__(self):
        return None

    def __exit__(self, *_):
        for key, val in self._before.items():
            if val is _NULL:
                del _ACCESSORS[key]
            else:
                _ACCESSORS[key] = val


def napari_type_hints(obj: Any) -> Dict[str, Any]:
    """variant of get_type_hints with napari namespace awareness."""
    import napari

    return get_type_hints(
        obj,
        {
            'napari': napari,
            **viewer.__dict__,
            **layers.__dict__,
            **components.__dict__,
        },
    )


def inject_napari_dependencies(func: Callable[..., T]) -> Callable[..., T]:
    """Create callable that can access napari objects based on type hints.

    This is form of dependency injection.  If a function includes a parameter
    that has a recognized napari type (e.g. `Viewer`, or `Layer`), then this
    function will return a new version of the input function that can be called
    *without* that particular parameter.

    Examples
    --------

    >>> def f(viewer: Viewer): ...
    >>> inspect.signature(f)
    <Signature (x: 'Viewer')>
    >>> f2 = inject_napari_dependencies(f)
    >>> inspect.signature(f2)
    <Signature (x: typing.Optional[napari.Viewer] = None)>
    # if f2 is called without x, the current_viewer will be provided for x


    Parameters
    ----------
    func : Callable
        A function with napari type hints.

    Returns
    -------
    Callable
        A function with napari dependencies injected
    """
    if not func.__code__.co_argcount and 'return' not in getattr(
        func, '__annotations__', {}
    ):
        return func

    sig = signature(func)
    # get type hints for the object, with forward refs of napari hints resolved
    hints = napari_type_hints(func)
    # get accessor functions for each required parameter
    required = {}
    return_hint = None
    for name, hint in hints.items():
        if name == 'return':
            return_hint = hint
            continue
        if sig.parameters[name].default is sig.empty:
            required[name] = hint

    @wraps(func)
    def _exec(*args, **kwargs):
        # when we call the function, we call the accessor functions to get
        # the current napari objects
        _kwargs = {}
        for n, hint in required.items():
            if accessor := get_accessor(hint):
                _kwargs[n] = accessor()

        # but we use bind_partial to allow the caller to still provide
        # their own objects if desired.
        # (i.e. the injected deps are only used if needed)
        _kwargs.update(**sig.bind_partial(*args, **kwargs).arguments)
        result = func(**_kwargs)
        if return_hint and (processor := get_processor(return_hint)):
            processor(result)
        return result

    out = _exec

    # if it came in as a generatorfunction, it needs to go out as one.
    if isgeneratorfunction(func):

        @wraps(func)
        def _gexec(*args, **kwargs):
            yield from _exec(*args, **kwargs)

        out = _gexec

    # update the signature
    p = [
        p.replace(default=None, annotation=Optional[hints[p.name]])
        if p.name in required
        else p
        for p in sig.parameters.values()
    ]
    out.__signature__ = sig.replace(parameters=p)
    return out
