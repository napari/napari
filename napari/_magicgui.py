"""This module installs some napari-specific types in magicgui, if present.

magicgui is a package that allows users to create GUIs from python functions
https://magicgui.readthedocs.io/en/latest/

It offers a function ``register_type`` that allows developers to specify how
their custom classes or types should be converted into GUIs.  Then, when the
end-user annotates one of their function arguments with a type hint using one
of those custom classes, magicgui will know what to do with it.

"""
from .layers import Layer

try:
    from magicgui import register_type

    def get_layers(magicgui, layer_type):
        """Retrieve layers of type `layer_type`, from the Viewer the gui is in.

        Parameters
        ----------
        magicgui : object
            The instantiated MagicGui widget.  May or may not be docked in a
            dock widget.
        layer_type : type
            This is the exact type used in the type hint of the user's
            function. It may be a subclass of napari.layers.Layer

        Returns
        -------
        tuple
            Tuple of layers of type ``layer_type``
        """
        try:
            # look for the parent Viewer based on where the magicgui is docked.
            viewer = magicgui.parent().qt_viewer.viewer
            return tuple(l for l in viewer.layers if isinstance(l, layer_type))
        except AttributeError:
            return ()

    # tell magicgui that whenever an argument is annotated as a
    # napari.layers.Layer, then it should be rendered as a ComboBox,
    # where the items are the current Viewer.layers of a certain type.
    register_type(Layer, choices=get_layers)

except Exception:
    pass
