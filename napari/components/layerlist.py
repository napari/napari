import itertools
import warnings
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np

from ..layers import Image, Labels, Layer
from ..layers.image.image import _ImageBase
from ..layers.utils._link_layers import get_linked_layers, layer_is_linked
from ..utils._dtype import normalize_dtype
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
        existing_layers = {x.name for x in self if x is not layer}
        for i in range(len(self)):
            if name in existing_layers:
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

    def toggle_selected_visibility(self):
        """Toggle visibility of selected layers"""
        for layer in self.selection:
            layer.visible = not layer.visible

    @property
    def _extent_world(self) -> np.ndarray:
        """Extent of layers in world coordinates.

        Default to 2D with (-0.5, 511.5) min/ max values if no data is present.
        Corresponds to pixels centered at [0, ..., 511].

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        return self._get_extent_world([layer.extent for layer in self])

    def _get_min_and_max(self, mins_list, maxes_list):

        # Reverse dimensions since it is the last dimensions that are
        # displayed.
        mins_list = [mins[::-1] for mins in mins_list]
        maxes_list = [maxes[::-1] for maxes in maxes_list]

        with warnings.catch_warnings():
            # Taking the nanmin and nanmax of an axis of all nan
            # raises a warning and returns nan for that axis
            # as we have do an explicit nan_to_num below this
            # behaviour is acceptable and we can filter the
            # warning
            warnings.filterwarnings(
                'ignore',
                message=str(
                    trans._('All-NaN axis encountered', deferred=True)
                ),
            )
            min_v = np.nanmin(
                list(itertools.zip_longest(*mins_list, fillvalue=np.nan)),
                axis=1,
            )
            max_v = np.nanmax(
                list(itertools.zip_longest(*maxes_list, fillvalue=np.nan)),
                axis=1,
            )

        # 512 element default extent as documented in `_get_extent_world`
        try:
            min_v = np.nan_to_num(min_v, nan=-0.5)
            max_v = np.nan_to_num(max_v, nan=511.5)
        except TypeError:
            # In NumPy < 1.17, nan_to_num doesn't have a nan kwarg
            min_v = np.asarray(min_v)
            min_v[np.isnan(min_v)] = -0.5
            max_v = np.asarray(max_v)
            max_v[np.isnan(max_v)] = 511.5

        # switch back to original order
        return min_v[::-1], max_v[::-1]

    def _get_extent_world(self, layer_extent_list):
        """Extent of layers in world coordinates.

        Default to 2D with (-0.5, 511.5) min/ max values if no data is present.
        Corresponds to pixels centered at [0, ..., 511].

        Returns
        -------
        extent_world : array, shape (2, D)
        """
        if len(self) == 0:
            min_v = np.asarray([-0.5] * self.ndim)
            max_v = np.asarray([511.5] * self.ndim)
        else:
            extrema = [extent.world for extent in layer_extent_list]
            mins = [e[0] for e in extrema]
            maxs = [e[1] for e in extrema]
            min_v, max_v = self._get_min_and_max(mins, maxs)

        return np.vstack([min_v, max_v])

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

    def _step_size_from_scales(self, scales):
        # Reverse order so last axes of scale with different ndim are aligned
        scales = [scale[::-1] for scale in scales]
        full_scales = list(
            np.array(list(itertools.zip_longest(*scales, fillvalue=np.nan)))
        )
        # restore original order
        return np.nanmin(full_scales, axis=1)[::-1]

    def _get_step_size(self, layer_extent_list):
        if len(self) == 0:
            return np.ones(self.ndim)
        else:
            scales = [extent.step for extent in layer_extent_list]
            min_scales = self._step_size_from_scales(scales)
            return min_scales

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
    def _ranges(self) -> List[Tuple[float, float, float]]:
        """Get ranges for Dims.range in world coordinates.

        This shares some code in common with the `extent` property, but
        determines Dims.range settings for each dimension such that each
        range is aligned to pixel centers at the finest scale.
        """
        if len(self) == 0:
            return [(0, 1, 1)] * self.ndim
        else:
            # Determine minimum step size across all layers
            layer_extent_list = [layer.extent for layer in self]
            scales = [extent.step for extent in layer_extent_list]
            min_steps = self._step_size_from_scales(scales)

            # Pixel-based layers need to be offset by 0.5 * min_steps to align
            # Dims.range with pixel centers in world coordinates
            pixel_offsets = [
                0.5 * min_steps
                if isinstance(layer, _ImageBase)
                else [0] * len(min_steps)
                for layer in self
            ]

            # Non-pixel layers need an offset of the range stop by min_steps since the upper
            # limit of Dims.range is non-inclusive.
            point_offsets = [
                [0] * len(min_steps)
                if isinstance(layer, _ImageBase)
                else min_steps
                for layer in self
            ]

            # Determine world coordinate extents similarly to
            # `_get_extent_world`, but including offsets calculated above.
            extrema = [extent.world for extent in layer_extent_list]
            mins = [
                e[0] + o1[: len(e[0])] for e, o1 in zip(extrema, pixel_offsets)
            ]
            maxs = [
                e[1] + o1[: len(e[0])] + o2[: len(e[0])]
                for e, o1, o2 in zip(extrema, pixel_offsets, point_offsets)
            ]
            min_v, max_v = self._get_min_and_max(mins, maxs)

            # form range tuples, switching back to original dimension order
            return [
                (start, stop, step)
                for start, stop, step in zip(min_v, max_v, min_steps)
            ]

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


def get_active_layer_dtype(layer):
    dtype = None
    if layer.active:
        try:
            dtype = normalize_dtype(layer.active.data.dtype).__name__
        except AttributeError:
            pass
    return dtype


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
    'active_layer_dtype': get_active_layer_dtype,
}
