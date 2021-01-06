"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

"""
import warnings
from typing import Any, List, Optional, Tuple, Type

from .. import types
from ..layers import NAMES, Layer
from ..utils.misc import ensure_list_of_layer_data_tuple
from ..viewer import Viewer

try:
    from magicgui import register_type
except ImportError:

    def register_type(*args, **kwargs):
        pass


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
    register_type(
        Layer, choices=get_layers, return_callback=add_layer_to_viewer
    )
    register_type(Viewer, choices=get_viewers)
    register_type(
        types.LayerDataTuple,
        return_callback=add_layer_data_tuples_to_viewer,
    )
    register_type(
        List[types.LayerDataTuple],
        return_callback=add_layer_data_tuples_to_viewer,
    )
    for layer_name in NAMES:
        data_type = getattr(types, f'{layer_name.title()}Data')
        register_type(data_type, return_callback=add_layer_data_to_viewer)


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
    """

    if result is None:
        return

    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

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
            f'magicgui function {gui} annotated with a return type of '
            'napari.types.LayerDataTuple did not return LayerData tuple(s)'
        )

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
        layer = viewer._add_layer_from_data(*layer_datum)
        layer._source = gui


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


def get_viewers(gui, *args) -> Tuple[Viewer, ...]:
    """Return the viewer that the magicwidget is in, or a list of all Viewers."""
    viewer = find_viewer_ancestor(gui)
    if viewer:
        return (viewer,)
    else:
        # until we maintain a list of instantiated viewers, this might be the
        # only option
        return tuple(v for v in globals().values() if isinstance(v, Viewer))


def get_layers(gui) -> Tuple[Layer, ...]:
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
    return tuple(x for x in viewer.layers if isinstance(x, gui.annotation))


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
    """
    if result is None:
        return

    # This is the pre 0.4.3 API, warn user and pass to the correct function.
    if not isinstance(result, Layer):
        import textwrap

        if return_type == Layer:
            msg = (
                'Annotating a magicgui function with a return type of '
                '`napari.layers.Layer` is deprecated.  To indicate that your '
                'function returns a layer data tuple, please use a return '
                'annotation of `napari.types.LayerDataTuple` or '
                '`List[napari.types.LayerDataTuple]`\n'
                'This will raise an exception in napari v0.4.5'
            )
            msg = "\n" + "\n".join(textwrap.wrap(msg, width=70))
            warnings.warn(msg)
            return add_layer_data_tuples_to_viewer(
                gui, result, types.LayerDataTuple
            )

        # it's a layer subclass
        msg = (
            'As of napari 0.4.3 magicgui functions with a return annotation of '
            "a napari layer type (such as 'napari.layers.Image') must now "
            f"return an actual layer instance, rather than {type(result)}. To "
            "have a plain array object converted to a napari layer, please "
            "use a return annotation of napari.types.<layer_name>Data, (with "
            "the corresponding layer name.  For example, the following "
            "would add an image layer to the viewer:"
        )
        msg = "\n" + "\n".join(textwrap.wrap(msg, width=70))
        msg += (
            "\n\n@magicgui\n"
            "def func(nx: int, ny: int) -> napari.types.ImageData:\n"
            "    return np.random.rand(ny, nx)\n\n"
            "This will raise an exception in napari v0.4.5"
        )
        warnings.warn(msg)
        data_type = getattr(types, f'{return_type.__name__.title()}Data')
        return add_layer_data_to_viewer(gui, result, data_type)

    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

    # After 0.4.3 a return type of a Layer subclass should return a layer.
    viewer.add_layer(result)
