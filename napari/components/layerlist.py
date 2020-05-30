from typing import Optional, List
import numpy as np
import itertools
from .dims import Dims
from ..layers import Layer
from ..utils.naming import inc_name_count
from ..utils.list import ListModel


def _add(event):
    """When a layer is added, set its name."""
    layers = event.source
    layer = event.item
    layer.name = layers._coerce_name(layer.name, layer)
    layer.events.name.connect(lambda e: layers._update_name(e))
    layers.unselect_all(ignore=layer)
    layers._update_layer_dims(layer)


def _change(event):
    """When layer or layer list is changed update dimesnions."""
    layers = event.source
    layers._update_dims()


class LayerList(ListModel):
    """List-like layer collection with built-in reordering and callback hooks.

    Parameters
    ----------
    iterable : iterable
        Iterable of napari.layer.Layer
    dims : Dimensions
        Contains axes, indices, dimensions and sliders.

    Attributes
    ----------
    events : vispy.util.event.EmitterGroup
        Event hooks:
            * added(item, index): whenever an item is added
            * removed(item): whenever an item is removed
            * reordered(): whenever the list is reordered
    """

    def __init__(
        self, iterable=(), *, ndisplay=2, order=None, axis_labels=None
    ):
        super().__init__(
            basetype=Layer,
            iterable=iterable,
            lookup={str: lambda q, e: q == e.name},
        )

        self.dims = Dims(
            ndim=None, ndisplay=ndisplay, order=order, axis_labels=axis_labels
        )

        self.events.added.connect(_add)
        self.events.changed.connect(_change)

    def __newlike__(self, iterable):
        return ListModel(self._basetype, iterable, self._lookup)

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
        total = len(self)
        indices = list(range(total))
        if not self[index].selected:
            self.unselect_all()
            self[index].selected = True
        selected = [i for i in range(total) if self[i].selected]

        # remove all indices to be moved
        for i in selected:
            indices.remove(i)
        # adjust offset based on selected indices to move
        offset = sum([i < insert and i != index for i in selected])
        # insert indices to be moved at correct start
        for insert_idx, elem_idx in enumerate(selected, start=insert - offset):
            indices.insert(insert_idx, elem_idx)
        # reorder list
        self[:] = self[tuple(indices)]

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

        layers = self.selected if selected else list(self)

        if not layers:
            import warnings

            warnings.warn(f"No layers {'selected' if selected else 'to save'}")
            return []

        return save_layers(path, layers, plugin=plugin)

    @property
    def _world_range(self):
        """(2, D) array: Range of layers in world coordinates.

        Default to 2D with (0, 512) min/ max values if no data is present.
        """
        if len(self) == 0:
            min_v = [np.nan, np.nan]
            max_v = [np.nan, np.nan]
        else:
            extrema = [l._world_range for l in self]
            mins = [e[0] for e in extrema]
            maxs = [e[1] for e in extrema]

            min_v = np.nanmin(
                list(itertools.zip_longest(*mins, fillvalue=np.nan)), axis=1
            )
            max_v = np.nanmax(
                list(itertools.zip_longest(*maxs, fillvalue=np.nan)), axis=1
            )

        min_vals = np.nan_to_num(min_v, nan=0)
        max_vals = np.nan_to_num(max_v, nan=512)

        return np.vstack([min_vals, max_vals])

    @property
    def ndim(self):
        """int: Maximum dimensionality of layers.

        Defaults to 2 if no data is present.
        """
        ndims = [l.ndim for l in self]
        return max(ndims, default=2)

    @property
    def _increments(self):
        """(D, ) array: Maximum increment for layers in world coordinates.

        Computes the maxiumum step size that allows all data planes to be
        sampled if moving through the data full range of world coordinates.
        This step size is given by the greatest common divisor.
        """
        if len(self) == 0:
            return np.ones(self.ndim)
        else:
            scales = [l.scale for l in self]
            full_scales = np.array(
                list(itertools.zip_longest(*scales, fillvalue=np.nan))
            ).T
            min_scales = np.nanmin(full_scales, axis=0)
            adj_scales = np.round(full_scales / min_scales).astype(int)
            adj_scales = np.nan_to_num(adj_scales, nan=1)
            raw_increments = np.ones(self.ndim, dtype=int)
            for a in adj_scales:
                raw_increments = np.gcd(raw_increments, a)
            return raw_increments * min_scales

    @property
    def _scene_shape(self):
        """tuple: Size and corner of viewed dimensions in world coordinates.

        Returns
        ----------
        size : list
            List of size of scene, length 2 or 3 if displayed view is 2D or 3D.
        corner : list
            List of coordinates of top left corner of scene, length 2 or 3 if
            displayed view is 2D or 3D.
        """
        # Scale the camera to the contents in the scene
        wr = self._world_range
        size = np.subtract(wr[1], wr[0])
        size = [size[i] for i in self.dims.displayed]
        corner = [wr[0, i] for i in self.dims.displayed]
        return size, corner

    def _update_layer_dims(self, layer):
        """Updates the dims of a particular layer."""
        # adjust the order of the global dims based on the number of
        # dimensions that a layer has - for example a global order of
        # [2, 1, 0, 3] -> [0, 1] for a layer that only has two dimesnions
        # or -> [1, 0, 2] for a layer with three as that corresponds to
        # the relative order of the last two and three dimensions
        # respectively
        offset = self.dims.ndim - layer.dims.ndim
        order = np.array(self.dims.order)
        if offset <= 0:
            order = list(range(-offset)) + list(order - offset)
        else:
            order = list(order[order >= offset] - offset)
        layer.dims.order = order
        layer.dims.ndisplay = self.dims.ndisplay

        # Update the point values of the layers for the dimensions that
        # the layer has
        for axis in range(layer.dims.ndim):
            step = self.dims.step[axis + offset]
            layer.dims.set_step(axis, step)

    def _update_dims(self):
        """When layer or layer list is changed update dimesnions."""
        self.dims.ndim = self.ndim
        if len(self) == 0:
            self.dims.reset()
        else:
            wr = self._world_range
            incs = self._increments
            for i in range(self.dims.ndim):
                self.dims.set_range(i, (wr[0, i], wr[1, i], incs[i]))
