"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

"""
from __future__ import annotations

import weakref
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Type

from typing_extensions import get_args

from ..utils._proxies import PublicOnlyProxy

if TYPE_CHECKING:
    from concurrent.futures import Future

    from magicgui.widgets._bases import CategoricalWidget

    from .._qt.qthreading import FunctionWorker
    from ..layers import Layer
    from ..viewer import Viewer


def add_layer_data_to_viewer(gui, result, return_type):
    """Show a magicgui result in the viewer.

    This function will be called when a magicgui-decorated function has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in ``result`` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call. For this function, this should be
        *just* the data part of the corresponding layer type.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the result as a viewer Image.

    >>> @magicgui
    ... def make_layer() -> napari.types.ImageData:
    ...     return np.random.rand(256, 256)

    """
    from ..layers._source import layer_source

    if result is None:
        return

    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

    with layer_source(widget=gui):
        try:
            viewer.layers[gui.result_name].data = result
        except KeyError:
            layer_type = return_type.__name__.replace("Data", "").lower()
            adder = getattr(viewer, f'add_{layer_type}')
            adder(data=result, name=gui.result_name)


def add_layer_data_tuples_to_viewer(gui, result, return_type):
    """Show a magicgui result in the viewer.

    This function will be called when a magicgui-decorated function has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in ``result`` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call. For this function, this should be
        *just* the data part of the corresponding layer type.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the result to the viewer

    >>> @magicgui
    ... def make_layer() -> napari.types.LayerDataTuple:
    ...     return (np.ones((10,10)), {'name': 'hi'})

    >>> @magicgui
    ... def make_layer() -> List[napari.types.LayerDataTuple]:
    ...     return [(np.ones((10,10)), {'name': 'hi'})]

    """
    if result is None:
        return
    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

    from ..layers._source import layer_source
    from ..utils.misc import ensure_list_of_layer_data_tuple
    from ..utils.translations import trans

    result = result if isinstance(result, list) else [result]
    try:
        result = ensure_list_of_layer_data_tuple(result)
    except TypeError:
        raise TypeError(
            trans._(
                'magicgui function {gui} annotated with a return type of napari.types.LayerDataTuple did not return LayerData tuple(s)',
                deferred=True,
                gui=gui,
            )
        )

    with layer_source(widget=gui):
        for layer_datum in result:
            # if the layer data has a meta dict with a 'name' key in it...
            if (
                len(layer_datum) > 1
                and isinstance(layer_datum[1], dict)
                and layer_datum[1].get("name")
            ):
                # then try to update the viewer layer with that name.
                try:
                    layer = viewer.layers[layer_datum[1].get('name')]
                    layer.data = layer_datum[0]
                    for k, v in layer_datum[1].items():
                        setattr(layer, k, v)
                    continue
                except KeyError:  # layer not in the viewer
                    pass
            # otherwise create a new layer from the layer data
            viewer._add_layer_from_data(*layer_datum)


_FUTURES: Set[Future] = set()


def add_worker_data(
    widget, worker: FunctionWorker, return_type, _from_tuple=True
):
    """Handle a thread_worker object returned from a magicgui widget.

    This allows someone annotate their magicgui with a return type of
    `FunctionWorker[...]`, create a napari thread worker (e.g. with the
    ``@thread_worker`` decorator), then simply return the worker.  We will hook up
    the `returned` signal to the machinery to add the result of the
    long-running function to the viewer.

    Parameters
    ----------
    widget : MagicGui
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    worker : WorkerBase
        An instance of `napari._qt.qthreading.WorkerBase`, on worker.returned,
        the result will be added to the viewer.
    return_type : type
        The return annotation that was used in the decorated function.
    _from_tuple : bool, optional
        (only for internal use). True if the worker returns `LayerDataTuple`,
        False if it returns one of the `LayerData` types.

    Examples
    --------

    .. code-block:: python

        @magicgui
        def my_widget(...) -> FunctionWorker[ImageData]:

            @thread_worker
            def do_something_slowly(...) -> ImageData:
                ...

            return do_something_slowly(...)

    """

    cb = (
        add_layer_data_tuples_to_viewer
        if _from_tuple
        else add_layer_data_to_viewer
    )
    _return_type = get_args(return_type)[0]
    worker.signals.returned.connect(
        partial(cb, widget, return_type=_return_type)
    )


def add_future_data(gui, future, return_type, _from_tuple=True):
    """Process a Future object from a magicgui widget.

    This function will be called when a magicgui-decorated function has a
    return annotation of one of the `napari.types.<layer_name>Data` ... and
    will add the data in ``result`` to the current viewer as the corresponding
    layer type.

    Parameters
    ----------
    gui : FunctionGui
        The instantiated magicgui widget.  May or may not be docked in a
        dock widget.
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
    from .._qt.utils import Sentry

    # get the actual return type from the Future type annotation
    _return_type = get_args(return_type)[0]

    if _from_tuple:
        # when the future is done, add layer data to viewer, dispatching
        # to the appropriate method based on the Future data type.

        def _on_future_ready():
            add_layer_data_tuples_to_viewer(gui, future.result(), return_type)
            _FUTURES.discard(future)

    else:

        def _on_future_ready():
            add_layer_data_to_viewer(gui, future.result(), _return_type)
            _FUTURES.discard(future)

    # some future types (such as a dask Future) will call the callback in
    # another thread, which wont always work here.  So we create a very small
    # QObject that can signal back to the main thread to call `_on_done`.
    sentry = Sentry()
    sentry.alerted.connect(_on_future_ready)
    future.add_done_callback(sentry.alert)
    _FUTURES.add(future)


def find_viewer_ancestor(widget) -> Optional[Viewer]:
    """Return the Viewer object if it is an ancestor of ``widget``, else None.

    Parameters
    ----------
    widget : QWidget
        A widget

    Returns
    -------
    viewer : napari.Viewer or None
        Viewer instance if one exists, else None.
    """
    from .._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget

    # magicgui v0.2.0 widgets are no longer QWidget subclasses, but the native
    # widget is available at widget.native
    if hasattr(widget, 'native') and hasattr(widget.native, 'parent'):
        parent = widget.native.parent()
    else:
        parent = widget.parent()
    while parent:
        if hasattr(parent, '_qt_viewer'):  # QMainWindow
            return parent._qt_viewer.viewer
        if isinstance(parent, QtViewerDockWidget):  # DockWidget
            qt_viewer = parent._ref_qt_viewer()
            if qt_viewer is not None:
                return qt_viewer.viewer
            return None
        parent = parent.parent()
    return None


def proxy_viewer_ancestor(widget) -> Optional[PublicOnlyProxy[Viewer]]:
    viewer = find_viewer_ancestor(widget)
    if viewer:
        return PublicOnlyProxy(viewer)
    return None


def get_layers(gui: CategoricalWidget) -> List[Layer]:
    """Retrieve layers matching gui.annotation, from the Viewer the gui is in.

    Parameters
    ----------
    gui : magicgui.widgets.Widget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.

    Returns
    -------
    tuple
        Tuple of layers of type ``gui.annotation``

    Examples
    --------
    This allows the user to do this, and get a dropdown box in their GUI
    that shows the available image layers.

    >>> @magicgui
    ... def get_layer_mean(layer: napari.layers.Image) -> float:
    ...     return layer.data.mean()

    """
    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return []
    return [x for x in viewer.layers if isinstance(x, gui.annotation)]


def get_layers_data(gui: CategoricalWidget) -> List[Tuple[str, Any]]:
    """Retrieve layers matching gui.annotation, from the Viewer the gui is in.

    As opposed to `get_layers`, this function returns just `layer.data` rather
    than the full layer object.

    Parameters
    ----------
    gui : magicgui.widgets.Widget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.

    Returns
    -------
    tuple
        Tuple of layer.data from layers of type ``gui.annotation``

    Examples
    --------
    This allows the user to do this, and get a dropdown box in their GUI
    that shows the available image layers, but just get the data from the image
    as function input

    >>> @magicgui
    ... def get_layer_mean(data: napari.types.ImageData) -> float:
    ...     return data.mean()

    """
    from .. import layers

    viewer = find_viewer_ancestor(gui.native)
    if not viewer:
        return ()

    layer_type_name = gui.annotation.__name__.replace("Data", "").title()
    layer_type = getattr(layers, layer_type_name)
    choices = []
    for layer in [x for x in viewer.layers if isinstance(x, layer_type)]:
        choice_key = f'{layer.name} (data)'
        choices.append((choice_key, layer.data))
        layer.events.data.connect(_make_choice_data_setter(gui, choice_key))

    return choices


@lru_cache(maxsize=None)
def _make_choice_data_setter(gui: CategoricalWidget, choice_name: str):
    """Return a function that sets the ``data`` for ``choice_name`` in ``gui``.

    Note, using lru_cache here so that the **same** function object is returned
    if you call this twice for the same widget/choice_name combination. This is
    so that when we connect it above in `layer.events.data.connect()`, it will
    only get connected once (because ``.connect()`` will not add a specific callback
    more than once)
    """
    gui_ref = weakref.ref(gui)

    def setter(event):
        _gui = gui_ref()
        if _gui is not None:
            _gui.set_choice(choice_name, event.value)

    return setter


def add_layer_to_viewer(gui, result: Any, return_type: Type[Layer]) -> None:
    """Show a magicgui result in the viewer.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    result : Any
        The result of the function call.
    return_type : type
        The return annotation that was used in the decorated function.

    Examples
    --------
    This allows the user to do this, and add the resulting layer to the viewer.

    >>> @magicgui
    ... def make_layer() -> napari.layers.Image:
    ...     return napari.layers.Image(np.random.rand(64, 64))

    """
    if result is None:
        return

    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

    result._source = result.source.copy(update={'widget': gui})
    viewer.add_layer(result)
