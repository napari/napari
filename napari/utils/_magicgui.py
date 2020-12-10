"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

"""
from typing import Any, Optional, Tuple, Type

from qtpy.QtWidgets import QWidget

from ..layers import Layer
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

    Return Annotations -> Widgets:
        napari.layers.Layer will add a new layer to the Viewer.
            if a return is annotated as a subclass of Layer, then the
            corresponding layer type will be added.
            see `show_layer_result` for detail
    """
    register_type(Layer, choices=get_layers, return_callback=show_layer_result)
    register_type(Viewer, choices=get_viewers)


def find_viewer_ancestor(widget: QWidget) -> Optional[Viewer]:
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
    if hasattr(widget, 'native') and isinstance(widget.native, QWidget):
        parent = widget.native.parent()
    else:
        parent = widget.parent()
    while parent:
        if hasattr(parent, 'qt_viewer'):
            return parent.qt_viewer.viewer
        parent = parent.parent()
    return None


def get_viewers(gui, *args) -> Tuple[Viewer, ...]:
    """Return the viewer that the magicwidget is in, or a list of all Viewers.
    """
    viewer = find_viewer_ancestor(gui)
    if viewer:
        return (viewer,)
    else:
        # until we maintain a list of instantiated viewers, this might be the
        # only option
        return tuple(v for v in globals().values() if isinstance(v, Viewer))


def get_layers(gui, layer_type: Type[Layer] = None) -> Tuple[Layer, ...]:
    """Retrieve layers of type `layer_type`, from the Viewer the gui is in.

    Parameters
    ----------
    gui : MagicGui or QWidget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    layer_type : type
        This is the exact type used in the type hint of the user's
        function. It may be a subclass of napari.layers.Layer.
        REMOVED in magicgui v0.2.0!

    Returns
    -------
    tuple
        Tuple of layers of type ``layer_type``

    Examples
    --------
    This allows the user to do this, and get a dropdown box in their GUI
    that shows the available image layers.

    >>> @magicgui
    ... def get_layer_mean(layer: napari.layers.Image) -> float:
    ...     return layer.data.mean()

    """
    # in magicgui v0.2.0 `gui` will be the magicgui parameter widget itself,
    # which contains all of the necessary information.
    # the layer type (which was just the annotation), is at gui.annotation
    if layer_type is None:
        gui, layer_type = gui.native, gui.annotation
    # else:
    #     import warnings
    #     from magicgui import __version__ as magicgui_version
    #
    #     warnings.warn(
    #         f"A future version of napari will no longer support magicgui "
    #         f"<0.2.0. (You have v{magicgui_version}). "
    #         "Please use `pip install -U magicgui` to update to >=0.2.0",
    #         FutureWarning,
    #     )

    viewer = find_viewer_ancestor(gui)
    if viewer:
        return tuple(
            layer for layer in viewer.layers if isinstance(layer, layer_type)
        )
    return ()


def show_layer_result(gui, result: Any, return_type: Type[Layer]) -> None:
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

    viewer = find_viewer_ancestor(gui)
    if not viewer:
        return

    # if they have annotated the return type as a base layer (layers.Layer),
    # NOT a subclass of it, then the function MUST return a list of
    # LayerData tuples where:
    # LayerData = Union[Tuple[data], Tuple[data, Dict], Tuple[data, Dict, str]]
    # where `data` is the data for a given layer
    # and `Dict` is a dict of keywords args that could be passed to that layer
    # type's add_* method
    if return_type == Layer:
        if (
            isinstance(result, list)
            and (len(result))
            and all(isinstance(i, tuple) and (0 < len(i) <= 3) for i in result)
        ):
            for layer_datum in result:
                # if the layer data has a meta dict with a 'name' key in it...
                if (
                    len(layer_datum) > 1
                    and isinstance(layer_datum[1], dict)
                    and layer_datum[1].get('name')
                ):
                    # then try to update the viewer layer with that name.
                    try:
                        name = layer_datum[1].get('name')
                        viewer.layers[name].data = layer_datum[0]
                        continue
                    except KeyError:
                        pass
                # otherwise create a new layer from the layer data
                viewer._add_layer_from_data(*layer_datum)
            return
        raise TypeError(
            'Functions annotated with a return type of '
            'napari.layers.Layer MUST return a list of LayerData tuples'
        )

    # Otherwise they annotated it as a subclass of layers.Layer, and we allow
    # the simpler behavior where they only return the layer data.
    try:
        viewer.layers[gui.result_name].data = result
    except KeyError:
        adder = getattr(viewer, f'add_{return_type.__name__.lower()}')
        adder(data=result, name=gui.result_name)
