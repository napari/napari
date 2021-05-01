from __future__ import annotations

import itertools
import warnings
from collections import namedtuple
from typing import Iterable

import numpy as np

from ...utils.naming import inc_name_count
from ...utils.translations import trans
from ...utils.tree import Group
from ..base import Layer
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
        new_layer = self._type_check(value)
        new_layer.name = self._coerce_name(new_layer.name)
        super().insert(index, new_layer)

    def _extent_data(self):
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        return combine_extents([c._get_extent() for c in self])

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

    def _get_ndim(self):
        try:
            self._ndim = max([c._get_ndim() for c in self])
        except ValueError:
            self._ndim = 2
        return self._ndim

    def _get_state(self):
        """LayerGroup state as a list of state dictionaries.

        Returns
        -------
        state : list
            List of layer state dictionaries.
        """
        state = []
        state.append(self._get_base_state())
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
        import numpy as np

        self.thumbnail = np.sum(
            lay.thumbnail for lay in self.traverse(leaves_only=True)
        )
        pass

    def refresh(self, event=None):
        """Refresh all layer data if visible."""
        if self.visible:
            for child in self:
                child.refresh()

    @property
    def data(self):
        return None

    @property
    def blending(self):
        return None

    @blending.setter
    def blending(self, val):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
