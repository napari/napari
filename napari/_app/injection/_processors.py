import sys
from concurrent.futures import Future
from contextlib import nullcontext, suppress
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ... import layers, types, viewer
from ...layers._source import layer_source
from ._providers import _provide_viewer


def _add_layer_data_tuples_to_viewer(
    data: Union[types.LayerDataTuple, List[types.LayerDataTuple]],
    return_type=None,
    viewer=None,
    source: Optional[dict] = None,
):
    from ...utils.misc import ensure_list_of_layer_data_tuple

    if viewer is None:
        viewer = _provide_viewer()
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
        An optional viewer to use. Otherwise use current viewer.
    layer_name : Optional[str]
        An optional layer name to use. If a layer with this name exists, it will
        be updated.
    source : Optional[dict]
        An optional layer source to use.

    Examples
    --------
    This allows the user to do this, and add the result as a viewer Image.

    >>> def make_layer() -> napari.types.ImageData:
    ...     return np.random.rand(256, 256)

    """
    if data is not None and (viewer := viewer or _provide_viewer()):
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
    if layer is not None and (viewer := viewer or _provide_viewer()):
        layer._source = layer.source.copy(update=source or {})
        viewer.add_layer(layer)


# here to prevent garbace collection of the future object while processing.
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


# Add future and LayerData processors for each layer type.
PROCESSORS: Dict[object, Callable] = {
    types.LayerDataTuple: _add_layer_data_tuples_to_viewer,
    List[types.LayerDataTuple]: _add_layer_data_tuples_to_viewer,
    layers.Layer: _add_layer_to_viewer,
}
for t in types._LayerData.__args__:  # type: ignore [attr-defined]
    PROCESSORS[t] = partial(_add_layer_data_to_viewer, return_type=t)

    if sys.version_info >= (3, 9):
        PROCESSORS[Future[t]] = partial(
            _add_future_data, return_type=t, _from_tuple=False
        )
