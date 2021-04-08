"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

"""
import weakref
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

from .. import layers, types
from ..layers._source import Source, layer_source
from ..utils.misc import ensure_list_of_layer_data_tuple
from ..utils.translations import trans
from ..viewer import Viewer

try:
    from magicgui import register_type
except ImportError:

    def register_type(*args, **kwargs):
        pass


if TYPE_CHECKING:
    from magicgui.widgets._bases import CategoricalWidget


def register_types_with_magicgui():
    """Register napari types with magicgui.

    Parameter Annotations -> Widgets:
        napari.layers.Layer, will be rendered as a ComboBox.
            if a parameter is annotated as a subclass Layer type, then the
            combobox options will be limited to that layer type.
        napari.Viewer, will be rendered as a ComboBox, with the current viewer
            as the only choice.

    Return Annotations -> Widgets:
        napari.layers.Layer will add a new layer to the Viewer.
            if a return is annotated as a subclass of Layer, then the
            corresponding layer type will be added.  As of 0.4.3, the user
            must return an actual layer instance
            see `add_layer_to_viewer` for detail
        napari.types.<layer_type>Data will add a new layer to the Viewer.
            using a bare data array (e.g. numpy array) as a return value.
        napari.types.LayerDataTuple will add a new layer to the Viewer.
            and expects the user to return a single layer data tuple
        List[napari.types.LayerDataTuple] will add multiple new layer to the
            Viewer. And expects the user to return a list of layer data tuples.

    """
    from magicgui.widgets import FunctionGui

    # the widget field in `_source.py` was defined with a forward reference
    # to avoid having to import magicgui when we define the layer `Source` obj.
    # Now that we know we have imported magicgui, we update that forward ref
    # https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
    Source.update_forward_refs(FunctionGui=FunctionGui)

    register_type(
        layers.Layer, choices=get_layers, return_callback=add_layer_to_viewer
    )
    register_type(Viewer, bind=find_viewer_ancestor)
    register_type(
        types.LayerDataTuple,
        return_callback=add_layer_data_tuples_to_viewer,
    )
    register_type(
        List[types.LayerDataTuple],
        return_callback=add_layer_data_tuples_to_viewer,
    )
    for layer_name in layers.NAMES:
        data_type = getattr(types, f'{layer_name.title()}Data')
        register_type(
            data_type,
            choices=get_layers_data,
            return_callback=add_layer_data_to_viewer,
        )


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
    # magicgui v0.2.0 widgets are no longer QWidget subclasses, but the native
    # widget is available at widget.native
    if hasattr(widget, 'native') and hasattr(widget.native, 'parent'):
        parent = widget.native.parent()
    else:
        parent = widget.parent()
    while parent:
        if hasattr(parent, 'qt_viewer'):
            return parent.qt_viewer.viewer
        parent = parent.parent()
    return None


def get_layers(gui: 'CategoricalWidget') -> List[layers.Layer]:
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
        return ()
    return [x for x in viewer.layers if isinstance(x, gui.annotation)]


def get_layers_data(gui: 'CategoricalWidget') -> List[Tuple[str, Any]]:
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
def _make_choice_data_setter(gui: 'CategoricalWidget', choice_name: str):
    """Return a function that sets the ``data`` for ``choice_name`` in ``gui``.

    Note, using lru_cache here so that the **same** function object is returned
    if you call this twice for the same widget/choice_name combination. This is
    so that when we connect it above in `layer.events.data.connect()`, it will
    only get connected once (because .connect() will not add a specific callback
    more than once)
    """
    gui_ref = weakref.ref(gui)

    def setter(event):
        _gui = gui_ref()
        if _gui is not None:
            _gui.set_choice(choice_name, event.value)

    return setter


def add_layer_to_viewer(
    gui, result: Any, return_type: Type[layers.Layer]
) -> None:
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
