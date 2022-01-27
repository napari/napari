from __future__ import annotations

import itertools
import warnings
from collections import namedtuple
from typing import Iterable, List, Tuple

import numpy as np

from ...utils.context import create_context
from ...utils.context._layerlist_context import LayerListContextKeys
from ...utils.naming import inc_name_count
from ...utils.translations import trans
from ...utils.tree import Group
from ..base import Layer
from ..image.image import _ImageBase
from ..utils.layer_utils import combine_extents

Extent = namedtuple('Extent', 'data world step')


class LayerGroup(Group[Layer], Layer):
    def __init__(
        self, children: Iterable[Layer] = (), name: str = 'LayerGroup'
    ) -> None:
        Group.__init__(self, children, name=name, basetype=Layer)
        Layer.__init__(self, None, 2, name=name)
        self.refresh(None)  # TODO: why...
        self.events.connect(self._handle_child_events)
        self._ctx = create_context(self)
        if self._ctx is not None:  # happens during Viewer type creation
            self._ctx_keys = LayerListContextKeys(self._ctx)

            self.selection.events.changed.connect(self._ctx_keys.update)

    def add_group(self, index=-1):
        lg = LayerGroup()
        self.insert(index, lg)
        return lg

    def _handle_child_events(self, event):
        # event.sources[0] is the original event emitter.
        # we only want child events here
        if not hasattr(event, 'index') or event.sources[0] is self:
            return
        if event.index != () and event.type == 'thumbnail':
            self._update_thumbnail()

    def __str__(self):
        return Group.__str__(self)

    def __repr__(self):
        return Group.__repr__(self)

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

        for existing_name in sorted(
            x.name
            for x in self.traverse(with_ancestors=True)
            if x is not layer
        ):
            if name == existing_name:
                name = inc_name_count(name)
        return name

    def _update_name(self, event):
        """Coerce name of the layer in `event.layer`."""
        layer = event.source
        layer.name = self._coerce_name(layer.name, layer)

    def insert(self, index: int, value: Layer):
        """Insert ``value`` before index."""
        # temporarily disabled while we work on nested selection bug and
        # group vispy nodes.
        # once removed, uncomment "add group" button in qt_viewer_buttons
        if isinstance(value, (LayerGroup, list)):
            import os

            if not os.getenv("ALLOW_LAYERGROUPS"):
                warnings.warn(
                    "Nested layergroups not quite ready. "
                    "Enabled with env var ALLOW_LAYERGROUPS=1."
                )
                return

        new_layer = self._type_check(value)
        new_layer.name = self._coerce_name(new_layer.name)
        super().insert(index, new_layer)
        self._update_thumbnail()

    def __delitem__(self, key):
        """Remove item at `key`."""
        super().__delitem__(key)
        self._update_thumbnail()

    def _extent_data(self):
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        return combine_extents([c._extent_data() for c in self])

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

    def _get_step_size(self, layer_extent_list):
        if len(self) == 0:
            return np.ones(self.ndim)

        scales = [extent.step[::-1] for extent in layer_extent_list]
        full_scales = list(
            np.array(list(itertools.zip_longest(*scales, fillvalue=np.nan))).T
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
        return self._get_ndim()

    def _get_ndim(self):
        return max((c._get_ndim() for c in self), default=2)

    def _get_state(self):
        """LayerGroup state as a list of state dictionaries.

        Returns
        -------
        state : list
            List of layer state dictionaries.
        """
        state = [self._get_base_state()]
        if self is not None:
            for layer in self:
                state.append(layer._get_state())
        return state

    def _get_value(self, position):
        """Returns a flat list of all layer values in the layergroup
        for a given mouse position and set of indices.

        Layers in layergroup are iterated over by depth-first recursive search.

        Returns
        ----------
        value : list
            Flat list containing values of the layer data at the coord.
        """
        return [layer._get_value(position) for layer in self]

    def _set_view_slice(self):
        """Set the view for each layer given the indices to slice with."""
        for child in self:
            child._set_view_slice()

    def _set_highlight(self, force=False):
        """Render layer hightlights when appropriate."""
        for child in self:
            child._set_highlight(force=force)

    def _update_thumbnail(self, *args, **kwargs):
        if not hasattr(self, '_thumbnail_shape'):
            # layer is not finished initializing
            return
        leaves = list(self.traverse(leaves_only=True))
        if len(leaves) > 1:
            # TODO: this doesn't take blending into account...
            thumb = np.clip(
                np.sum([leaf.thumbnail for leaf in leaves], axis=0), 0, 255
            )
        elif leaves:
            thumb = leaves[0].thumbnail
        else:
            thumb = np.zeros(self._thumbnail_shape)
            thumb[..., 3] = 255
        self.thumbnail = thumb

    def refresh(self, event=None):
        """Refresh all layer data if visible."""
        if self.visible:
            for child in self:
                child.refresh()

    @property
    def data(self):
        return None

    def save(self):
        raise NotImplementedError()

    def _update_draw(self, *a, **k):
        return

    # TODO .... need to recheck stuff below here.

    @property
    def _ranges(self) -> List[Tuple[float, float, float]]:
        """Get ranges for Dims.range in world coordinates.

        This shares some code in common with the `extent` property, but
        determines Dims.range settings for each dimension such that each
        range is aligned to pixel centers at the finest scale.
        """
        if len(self) == 0:
            return [(0, 1, 1)] * self.ndim
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
        return list(zip(min_v, max_v, min_steps))

    def _step_size_from_scales(self, scales):
        # Reverse order so last axes of scale with different ndim are aligned
        scales = [scale[::-1] for scale in scales]
        full_scales = list(
            np.array(list(itertools.zip_longest(*scales, fillvalue=np.nan)))
        )
        # restore original order
        return np.nanmin(full_scales, axis=1)[::-1]

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
        min_v = np.nan_to_num(min_v, nan=-0.5)
        max_v = np.nan_to_num(max_v, nan=511.5)

        # switch back to original order
        return min_v[::-1], max_v[::-1]
