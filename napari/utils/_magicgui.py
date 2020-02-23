"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

"""
from typing import Tuple, Type, Any
from ..layers import Layer

try:
    from magicgui import register_type

    def get_layers(gui, layer_type: Type[Layer]) -> Tuple[Layer, ...]:
        """Retrieve layers of type `layer_type`, from the Viewer the gui is in.

        Parameters
        ----------
        gui : MagicGui or QWidget
            The instantiated MagicGui widget.  May or may not be docked in a
            dock widget.
        layer_type : type
            This is the exact type used in the type hint of the user's
            function. It may be a subclass of napari.layers.Layer

        Returns
        -------
        tuple
            Tuple of layers of type ``layer_type``

        Example
        -------
        This allows the user to do this, and get a dropdown box in their GUI
        that shows the available image layers.

        >>> @magicgui
        ... def get_layer_mean(layer: napari.layers.Image) -> float:
        ...     return layer.data.mean()

        """
        try:
            # look for the parent Viewer based on where the magicgui is docked.
            # if the magicgui widget does not have a parent, it is unattached
            # to any viewers, and therefore we cannot return a list of layers
            viewer = gui.parent().qt_viewer.viewer
            return tuple(l for l in viewer.layers if isinstance(l, layer_type))
        except AttributeError:
            return ()

    def show_result(gui, result: Any, return_type: Type[Layer]) -> None:
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

        try:
            viewer = gui.parent().qt_viewer.viewer
        except AttributeError:
            return

        result_name = gui.name() + " result"
        try:
            viewer.layers[result_name].data = result
        except KeyError:
            adder = getattr(viewer, f'add_{return_type.__name__.lower()}')
            adder(data=result, name=result_name)

    # tell magicgui that whenever an argument is annotated as a
    # napari.layers.Layer, then it should be rendered as a ComboBox,
    # where the items are the current Viewer.layers of a certain type.
    register_type(Layer, choices=get_layers, return_callback=show_result)

except ImportError:
    pass
