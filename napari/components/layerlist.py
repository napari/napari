import itertools
import warnings
from collections import namedtuple
from typing import List, Optional

import numpy as np

from ..layers import Image, Labels, Layer
from ..layers.utils._link_layers import get_linked_layers, layer_is_linked
from ..utils.events.containers import SelectableEventedList
from ..utils.naming import inc_name_count
from ..utils.translations import trans

Extent = namedtuple('Extent', 'data world step')


class LayerList(SelectableEventedList[Layer]):
    """List-like layer collection with built-in reordering and callback hooks.

    Parameters
    ----------
    data : iterable
        Iterable of napari.layer.Layer
    """

    def __init__(self, data=()):
        super().__init__(
            data=data,
            basetype=Layer,
            lookup={str: lambda e: e.name},
        )

        # temporary: see note in _on_selection_event
        self.selection.events.changed.connect(self._on_selection_changed)

    def _on_selection_changed(self, event):
        # This method is a temporary workaround to the fact that the Points
        # layer needs to know when its selection state changes so that it can
        # update the highlight state.  This (and the layer._on_selection
        # method) can be removed once highlighting logic has been removed from
        # the layer model.
        for layer in event.added:
            layer._on_selection(True)
        for layer in event.removed:
            layer._on_selection(False)

    def __newlike__(self, data):
        return LayerList(data)

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
        if layer is None:
            for existing_name in sorted(x.name for x in self):
                if name == existing_name:
                    name = inc_name_count(name)
        else:
            for _layer in sorted(self, key=lambda x: x.name):
                if _layer is layer:
                    continue
                if name == _layer.name:
                    name = inc_name_count(name)
        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    def insert(self, index: int, value: Layer):
        """Insert ``value`` before index."""
        new_layer = self._type_check(value)
        new_layer.name = self._coerce_name(new_layer.name)
        super().insert(index, new_layer)

        # required for deprecated layer.selected property.  remove after 0.4.9
        new_layer._deprecated_layerlist = self

    @property
    def selected(self):
        """List of selected layers."""
        warnings.warn(
            trans._(
                "'viewer.layers.selected' is deprecated and will be removed in or after v0.4.9. Please use 'viewer.layers.selection'",
                deferred=True,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        return self.selection

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
        if self[index] not in self.selection:
            self.selection.select_only(self[index])
            moving = [index]
        else:
            moving = [i for i, x in enumerate(self) if x in self.selection]
        offset = insert >= index
        self.move_multiple(moving, insert + offset)

    def unselect_all(self, ignore=None):
        """Unselects all layers expect any specified in ignore.

        Parameters
        ----------
        ignore : Layer | None
            Layer that should not be unselected if specified.
        """
        warnings.warn(
            trans._(
                "'viewer.layers.unselect_all()' is deprecated and will be removed in or after v0.4.9. Please use 'viewer.layers.selection.clear()'. To unselect everything but a set of ignored layers, use 'viewer.layers.selection.intersection_update({ignored})'",
                deferred=True,
                ignored=ignore,
            ),
            category=FutureWarning,
            stacklevel=2,
        )
        self.selection.intersection_update({ignore} if ignore else {})

    def toggle_selected_visibility(self):
        """Toggle visibility of selected layers"""
        for layer in self.selection:
            layer.visible = not layer.visible

    @property
    def _extent_world(self) -> np.ndarray:
        """Extent of layers in world coordinates.

        Default to 2D with (0, 512) min/ max values if no data is present.

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        return self._get_extent_world([layer.extent for layer in self])

    def _get_extent_world(self, layer_extent_list):
        """Extent of layers in world coordinates.

        Default to 2D with (0, 512) min/ max values if no data is present.

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        if len(self) == 0:
            min_v = [np.nan] * self.ndim
            max_v = [np.nan] * self.ndim
        else:
            extrema = [extent.world for extent in layer_extent_list]
            mins = [e[0][::-1] for e in extrema]
            maxs = [e[1][::-1] for e in extrema]

            with warnings.catch_warnings():
                # Taking the nanmin and nanmax of an axis of all nan
                # raises a warning and returns nan for that axis
                # as we have do an explict nan_to_num below this
                # behaviour is acceptable and we can filter the
                # warning
                warnings.filterwarnings(
                    'ignore',
                    message=str(
                        trans._('All-NaN axis encountered', deferred=True)
                    ),
                )
                min_v = np.nanmin(
                    list(itertools.zip_longest(*mins, fillvalue=np.nan)),
                    axis=1,
                )
                max_v = np.nanmax(
                    list(itertools.zip_longest(*maxs, fillvalue=np.nan)),
                    axis=1,
                )

        min_vals = np.nan_to_num(min_v[::-1])
        max_vals = np.copy(max_v[::-1])
        max_vals[np.isnan(max_vals)] = 511

        return np.vstack([min_vals, max_vals])

    @property
    def _step_size(self) -> np.ndarray:
        """Ideal step size between planes in world coordinates.

        Computes the best step size that allows all data planes to be
        sampled if moving through the full range of world coordinates.
        The current implementation just takes the minimum scale.

        Returns
        -------
        step_size : array, shape (D,)
        """
        return self._get_step_size([layer.extent for layer in self])

    def _get_step_size(self, layer_extent_list):
        if len(self) == 0:
            return np.ones(self.ndim)
        else:
            scales = [extent.step[::-1] for extent in layer_extent_list]
            full_scales = list(
                np.array(
                    list(itertools.zip_longest(*scales, fillvalue=np.nan))
                ).T
            )
            min_scales = np.nanmin(full_scales, axis=0)
            return min_scales[::-1]

    @property
    def extent(self) -> Extent:
        """Extent of layers in data and world coordinates."""
        extent_list = [layer.extent for layer in self]
        return Extent(
            data=None,
            world=self._get_extent_world(extent_list),
            step=self._get_step_size(extent_list),
        )

    @property
    def ndim(self) -> int:
        """Maximum dimensionality of layers.

        Defaults to 2 if no data is present.

        Returns
        -------
        ndim : int
        """
        return max((layer.ndim for layer in self), default=2)

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

        layers = list(self.selection) if selected else list(self)

        if selected:
            msg = trans._("No layers selected", deferred=True)
        else:
            msg = trans._("No layers to save", deferred=True)

        if not layers:
            warnings.warn(msg)
            return []

        return save_layers(path, layers, plugin=plugin)

    def _selection_context(self) -> dict:
        """Return context dict for current layerlist.selection"""
        return {k: v(self.selection) for k, v in _CONTEXT_KEYS.items()}


# Each key in this list is "usable" as a variable name in the the "enable_when"
# and "show_when" expressions of the napari.layers._layer_actions.LAYER_ACTIONS
#
# each value is a function that takes a LayerList.selection, and returns
# a value. LayerList._selection_context uses this dict to generate a concrete
# context object that can be passed to the
# `qt_action_context_menu.QtActionContextMenu` method to update the enabled
# and/or visible items based on the state of the layerlist.

_CONTEXT_KEYS = {
    'selection_count': lambda s: len(s),
    'all_layers_linked': lambda s: all(layer_is_linked(x) for x in s),
    'linked_layers_unselected': lambda s: len(get_linked_layers(*s) - s),
    'active_is_rgb': lambda s: getattr(s.active, 'rgb', False),
    'only_images_selected': (
        lambda s: bool(s and all(isinstance(x, Image) for x in s))
    ),
    'only_labels_selected': (
        lambda s: bool(s and all(isinstance(x, Labels) for x in s))
    ),
    'image_active': lambda s: isinstance(s.active, Image),
    'ndim': lambda s: s.active and getattr(s.active.data, 'ndim', None),
    'active_layer_shape': (
        lambda s: s.active and getattr(s.active.data, 'shape', None)
    ),
    'same_shape': (
        lambda s: len({getattr(x.data, 'shape', ()) for x in s}) == 1
    ),
}
