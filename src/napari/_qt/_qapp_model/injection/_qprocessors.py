"""Qt processors.

Non-Qt processors can be found in `napari/_app_model/injection/_processors.py`.
"""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import nullcontext, suppress
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
    get_origin,
)

from magicgui.widgets import FunctionGui, Widget
from qtpy.QtWidgets import QWidget
from superqt import ensure_main_thread

from napari import layers, types, viewer
from napari._qt._qapp_model.injection._qproviders import (
    _provide_viewer,
    _provide_viewer_or_raise,
)
from napari.layers._source import layer_source
from napari.utils.notifications import notification_manager

if TYPE_CHECKING:
    from collections.abc import Callable


def _add_plugin_dock_widget(
    widget_name_tuple: tuple[FunctionGui | QWidget | Widget, str, str, bool],
    viewer: viewer.Viewer | None = None,
) -> None:
    if viewer is None:
        viewer = _provide_viewer_or_raise(
            msg='Widgets cannot be opened in headless mode.',
        )
    widget, full_name, default_dock_area, tabify = widget_name_tuple
    area = viewer.window._resolve_dock_area(
        name=full_name, default_area=default_dock_area
    )
    viewer.window.add_dock_widget(
        widget, name=full_name, area=area, tabify=tabify
    )


def _add_layer_data_tuples_to_viewer(
    data: tuple | list[tuple],
    return_type: Any | None = None,
    viewer: viewer.Viewer | None = None,
    source: dict | None = None,
) -> None:
    from napari.utils.misc import ensure_list_of_layer_data_tuple

    if viewer is None:
        viewer = _provide_viewer()
    if viewer and data is not None:
        data = data if isinstance(data, list) else [data]
        for datum in ensure_list_of_layer_data_tuple(data):
            # then try to update a viewer layer with the same name.
            if len(datum) > 1 and (name := datum[1].get('name')):
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
    viewer: viewer.Viewer | None = None,
    layer_name: str | None = None,
    source: dict | None = None,
    meta: dict | None = None,
) -> None:
    """Show a result in the viewer.

    Parameters
    ----------
    data : Any
        The result of the function call. For this function, this should be
        *just* the data part of the corresponding layer type.
    return_type : Any
        The return annotation that was used in the decorated function.
    viewer : Viewer or None
        An optional viewer to use. Otherwise use current viewer.
    layer_name : str or None
        An optional layer name to use. If a layer with this name exists, it will
        be updated.
    source : dict or None
        An optional layer source to use.
    meta: dict or None
        An optional dict pased as keyword arguments to the layer constructor.
        Currently used to pass information like a scale, units, etc.

    Examples
    --------
    This allows the user to do this, and add the result as a viewer Image.

    >>> def make_layer() -> napari.types.ImageData:
    ...     return np.random.rand(256, 256)

    """
    if meta is None:
        meta = {}
    if data is not None and (viewer := viewer or _provide_viewer()):
        if layer_name:
            with suppress(KeyError):
                # layerlist also allow lookup by name
                viewer.layers[layer_name].data = data
                return
        if get_origin(return_type) is Union:
            if len(return_type.__args__) != 2 or return_type.__args__[
                1
            ] is not type(None):
                # this case should be impossible, but we'll check anyway.
                raise TypeError(
                    f'napari supports only Optional[<layer_data_type>], not {return_type}'
                )
            return_type = return_type.__args__[0]
        layer_type = return_type.__name__.replace('Data', '').lower()

        with layer_source(**source) if source else nullcontext():
            getattr(viewer, f'add_{layer_type}')(
                data=data, name=layer_name, **meta
            )


def _add_layer_to_viewer(
    layer: layers.Layer,
    viewer: viewer.Viewer | None = None,
    source: dict | None = None,
) -> None:
    if layer is not None and (viewer := viewer or _provide_viewer()):
        layer._source = layer.source.model_copy(update=source or {})
        viewer.add_layer(layer)


# here to prevent garbage collection of the future object while processing.
_FUTURES: set[Future] = set()


def _add_future_data(
    future: Future,
    return_type: Any,
    _from_tuple: bool = True,
    viewer: viewer.Viewer | None = None,
    source: dict | None = None,
) -> None:
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

    add_kwargs = {
        'return_type': return_type,
        'viewer': viewer,
        'source': source,
    }

    # The future may finish in a background thread, but GUI objects
    # must be created on the main thread.
    @ensure_main_thread
    def _add_result(f: Future) -> None:
        try:
            if f.cancelled():
                return
            # Show the error to the user here. If we let it raise, it
            # would only be printed to the log, or could crash the app.
            if (exc := f.exception()) is not None:
                notification_manager.receive_error(
                    type(exc), exc, exc.__traceback__
                )
                return
            if _from_tuple:
                _add_layer_data_tuples_to_viewer(f.result(), **add_kwargs)
            else:
                _add_layer_data_to_viewer(f.result(), **add_kwargs)
        finally:
            _FUTURES.discard(future)

    # Keep the future first, then add the callback. If the future is
    # already finished, the callback runs right away and removes it.
    _FUTURES.add(future)
    future.add_done_callback(_add_result)


QPROCESSORS: dict[object, Callable] = {
    Optional[
        tuple[FunctionGui | QWidget | Widget, str, str, bool]
    ]: _add_plugin_dock_widget,
    types.LayerDataTuple: _add_layer_data_tuples_to_viewer,
    list[types.LayerDataTuple]: _add_layer_data_tuples_to_viewer,
    layers.Layer: _add_layer_to_viewer,
}

# Add future and LayerData processors for each layer type.
for t in types._LayerData.__args__:  # pyrefly: ignore [missing-attribute]
    QPROCESSORS[t] = partial(_add_layer_data_to_viewer, return_type=t)

    QPROCESSORS[Future[t]] = partial(
        _add_future_data, return_type=t, _from_tuple=False
    )
