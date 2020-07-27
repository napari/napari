from typing import Optional, List
from ..layers import Layer
from ..utils.naming import inc_name_count
from ..utils.events import TypedEventedList


class LayerList(TypedEventedList):
    """List-like layer collection with built-in reordering and callback hooks.

    Parameters
    ----------
    data : iterable
        Iterable of napari.layer.Layer
    """

    def __init__(self, data=()):
        super().__init__(
            data=data, basetype=Layer, lookup={str: lambda e: e.name},
        )

        self.events.inserted.connect(self._inserted)
        self.events.removed.connect(self._removed)

    def _inserted(self, event):
        layer = event.value
        layer.name = self._coerce_name(layer.name, layer)
        layer.events.name.connect(lambda e: self._update_name(e))
        layer.selected = True
        self.unselect_all(ignore=layer)

    def _removed(self, event):
        # FIXME: Currently need complex logic to manage event connections
        # when a layer gets disconnected, in case it gets connected again.
        layer = event.value
        layer.events.disconnect()
        for em in layer.events.emitters.values():
            em.disconnect()
        layer.dims.events.disconnect()
        for em in layer.dims.events.emitters.values():
            em.disconnect()
        # But then need to reconnect any events internal to the layer
        layer.events.data.connect(lambda e: layer._set_editable())
        layer.dims.events.ndisplay.connect(lambda e: layer._set_editable())
        layer.dims.events.order.connect(layer.refresh)
        layer.dims.events.ndisplay.connect(layer._update_dims)
        layer.dims.events.order.connect(layer._update_dims)
        layer.dims.events.axis.connect(layer.refresh)

    def _coerce_name(self, name, layer=None):
        """Coerce a name into a unique equivalent.

        Parameters
        ----------
        name : str
            Original name.
        layer : napari.layers.Layer, optional
            Layer for which name is generated.

        Returns
        -------
        new_name : str
            Coerced, unique name.
        """
        for l in self:
            if l is layer:
                continue
            if l.name == name:
                name = inc_name_count(name)

        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    @property
    def selected(self):
        """List of selected layers."""
        return [layer for layer in self if layer.selected]

    def move_selected(self, index, insert):
        """Reorder list by moving the item at index and inserting it
        at the insert index. If additional items are selected these will
        get inserted at the insert index too. This allows for rearranging
        the list based on dragging and dropping a selection of items, where
        index is the index of the primary item being dragged, and insert is
        the index of the drop location, and the selection indicates if
        multiple items are being dragged. If the moved layer is not selected
        select it.

        Parameters
        ----------
        index : int
            Index of primary item to be moved
        insert : int
            Index that item(s) will be inserted at
        """
        if not self[index].selected:
            self.unselect_all()
            self[index].selected = True
            moving = (index,)
        else:
            moving = [i for i, item in enumerate(self) if item.selected]
        offset = insert >= index
        self.move_multiple(moving, insert + offset)

    def unselect_all(self, ignore=None):
        """Unselects all layers expect any specified in ignore.

        Parameters
        ----------
        ignore : Layer | None
            Layer that should not be unselected if specified.
        """
        for layer in self:
            if layer.selected and layer != ignore:
                layer.selected = False

    def select_all(self):
        """Selects all layers."""
        for layer in self:
            if not layer.selected:
                layer.selected = True

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

    def select_next(self, shift=False):
        """Selects next item from list.
        """
        selected = []
        for i in range(len(self)):
            if self[i].selected:
                selected.append(i)
        if len(selected) > 0:
            if selected[-1] == len(self) - 1:
                if shift is False:
                    self.unselect_all(ignore=self[selected[-1]])
            elif selected[-1] < len(self) - 1:
                if shift is False:
                    self.unselect_all(ignore=self[selected[-1] + 1])
                self[selected[-1] + 1].selected = True
        elif len(self) > 0:
            self[-1].selected = True

    def select_previous(self, shift=False):
        """Selects previous item from list.
        """
        selected = []
        for i in range(len(self)):
            if self[i].selected:
                selected.append(i)
        if len(selected) > 0:
            if selected[0] == 0:
                if shift is False:
                    self.unselect_all(ignore=self[0])
            elif selected[0] > 0:
                if shift is False:
                    self.unselect_all(ignore=self[selected[0] - 1])
                self[selected[0] - 1].selected = True
        elif len(self) > 0:
            self[0].selected = True

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
            specify output format (provided a plugin is available for the
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

        layers = self.selected if selected else list(self)

        if not layers:
            import warnings

            warnings.warn(f"No layers {'selected' if selected else 'to save'}")
            return []

        return save_layers(path, layers, plugin=plugin)
