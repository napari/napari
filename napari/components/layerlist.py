from typing import Optional, List
from ..layers import Layer
from ..utils.naming import force_unique_name
from ..utils.list import ListModel
from ..utils.event import Event


class LayerList(ListModel):
    """List-like layer collection with built-in reordering and callback hooks.

    Parameters
    ----------
    iterable : iterable
        Iterable of napari.layer.Layer

    Attributes
    ----------
    events : vispy.util.event.EmitterGroup
        Event hooks:
            * added((item, index)): whenever an item is added
            * removed((item, index)): whenever an item is removed
            * reordered((indices, new_indices)): whenever the list is reordered
            * changed(None): after the list is changed in any way
    """

    def __init__(self, iterable=()):
        super().__init__(
            basetype=Layer,
            iterable=iterable,
            lookup={str: lambda q, e: q == e.name},
        )
        self.events.add(selected_layers=Event,)

    def __newlike__(self, iterable):
        return ListModel(self._basetype, iterable, self._lookup)

    def _on_added_change(self, value):
        """When a layer is added, set its name.

        Parmeters
        ---------
        value : 2-tuple
            Tuple of layer and index where layer is being added.
        """
        super()._on_added_change(value)
        layer = value[0]
        # Coerce name into being unique in layer list
        layer.name = force_unique_name(layer.name, [l.name for l in self[:-1]])
        # Register layer event handler
        layer.event_handler.register_listener(self)
        # Unselect all other layers
        self.selected = [len(self) - 1]

    def _on_name_unique_change(self, names):
        """Receive layer name tuple and update the name if is already in list.

        As the layer list can be indexed by name each layer must have a unique
        name to ensure that there is only one layer corresponding to every
        name. This method updates the name to be unique if it is not already.

        The name is made unique by appending an index to it, for example
        'Image' -> 'Image [1]' or adding to the index, say 'Image [2]'.

        Parameters
        ----------
        names : 2-tuple of str
            Tuple of old name and new name.
        """
        old_name, new_name = names
        unique_name = force_unique_name(new_name, [l.name for l in self])
        # Re-emit unique name
        self[old_name].events.name(unique_name)

    @property
    def selected(self):
        """List of indices of selected layers."""
        return [i for i, layer in enumerate(self) if layer.selected]

    @selected.setter
    def selected(self, selected):
        """List: Indices of layers to select layers."""
        self.events.selected_layers(selected)

    def unselect_all(self, ignore=None):
        """Unselects all layers expect any specified in ignore.

        Parameters
        ----------
        ignore : Layer | None
            Layer that should not be unselected if specified.
        """
        if ignore is not None:
            index = self.index(ignore)
            self.selected = [index]
        else:
            self.selected = []

    def select_all(self):
        """Select all layers."""
        self.selected = list(range(len(self)))

    def _on_selected_layers_change(self, selected_layers):
        """When selected layers are changed update the actual layers.

        Parmeters
        ---------
        selected_layers : list
            List of selected indices.
        """
        for i, layer in enumerate(self):
            layer.selected = i in selected_layers

    def _on_selected_change(self, selected):
        """When selected state of any layer changes emit selected layers event.

        Parmeters
        ---------
        selected : bool
            Whether layer is selected or not.
        """
        # The fact that we are emitting a new event inside the callback of
        # this event is probably bad, as we don't actually know that the
        # selected state has been update when this event gets emitted
        self.events.selected_layers(self.selected)

    def remove_selected(self):
        """Removes selected items from list."""
        to_delete = []
        for i in range(len(self)):
            if self[i].selected:
                to_delete.append(i)
        to_delete.reverse()
        for i in to_delete:
            self.pop(i)
        if len(to_delete) > 0:
            first_to_delete = to_delete[-1]
            if first_to_delete == 0 and len(self) > 0:
                self[0].selected = True
            elif first_to_delete > 0:
                self[first_to_delete - 1].selected = True

    def toggle_selected_visibility(self):
        """Toggle visibility of selected layers"""
        for layer in self:
            if layer.selected:
                layer.visible = not layer.visible

    def save(
        self,
        path: str,
        *,
        selected: bool = False,
        plugin: Optional[str] = None,
    ) -> List[str]:
        """Save all or only selected layers to a path using writer plugins.

        If ``plugin`` is not provided and only one layer is targeted, then we
        directly call the corresponding``napari_write_<layer_type>`` hook (see
        :ref:`single layer writer hookspecs <write-single-layer-hookspecs>`)
        which will loop through implementations and stop when the first one
        returns a non-``None`` result. The order in which implementations are
        called can be changed with the Plugin sorter in the GUI or with the
        corresponding hook's
        :meth:`~napari.plugins._hook_callers._HookCaller.bring_to_front`
        method.

        If ``plugin`` is not provided and multiple layers are targeted,
        then we call
        :meth:`~napari.plugins.hook_specifications.napari_get_writer` which
        loops through plugins to find the first one that knows how to handle
        the combination of layers and is able to write the file. If no plugins
        offer :meth:`~napari.plugins.hook_specifications.napari_get_writer` for
        that combination of layers then the default
        :meth:`~napari.plugins.hook_specifications.napari_get_writer` will
        create a folder and call ``napari_write_<layer_type>`` for each layer
        using the ``Layer.name`` variable to modify the path such that the
        layers are written to unique files in the folder.

        If ``plugin`` is provided and a single layer is targeted, then we
        call the ``napari_write_<layer_type>`` for that plugin, and if it fails
        we error.

        If ``plugin`` is provided and multiple layers are targeted, then
        we call we call
        :meth:`~napari.plugins.hook_specifications.napari_get_writer` for
        that plugin, and if it doesn’t return a ``WriterFunction`` we error,
        otherwise we call it and if that fails if it we error.

        Parameters
        ----------
        path : str
            A filepath, directory, or URL to open.  Extensions may be used to
            specify output format (provided a plugin is avaiable for the
            requested format).
        selected : bool
            Optional flag to only save selected layers. False by default.
        plugin : str, optional
            Name of the plugin to use for saving. If None then all plugins
            corresponding to appropriate hook specification will be looped
            through to find the first one that can save the data.

        Returns
        -------
        list of str
            File paths of any files that were written.
        """
        from ..plugins.io import save_layers

        layers = (
            [layer for layer in self if layer.selected]
            if selected
            else list(self)
        )

        if not layers:
            import warnings

            warnings.warn(f"No layers {'selected' if selected else 'to save'}")
            return []

        return save_layers(path, layers, plugin=plugin)
