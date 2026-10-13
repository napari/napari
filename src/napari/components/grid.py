from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
from pydantic import PrivateAttr, model_validator

from napari.settings._application import (
    GridHeight,
    GridSpacing,
    GridStride,
    GridWidth,
)
from napari.utils.events import EventedModel

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from napari.components.canvas import Canvas
    from napari.layers import Layer


class GridCanvas(EventedModel):
    """Grid for canvas.

    Right now the only grid mode that is still inside one canvas with one
    camera, but future grid modes could support multiple canvases.

    Attributes
    ----------
    enabled : bool
        If grid is enabled or not.
    stride : int
        Number of layers to place in each grid viewbox before moving on to
        the next viewbox. The default ordering is to place the most visible
        layer in the top left corner of the grid. A negative stride will
        cause the order in which the layers are placed in the grid to be
        reversed.
    shape : 2-tuple of int
        Number of rows and columns in the grid. A value of -1 for either or
        both of will be used the row and column numbers will trigger an
        auto calculation of the necessary grid shape to appropriately fill
        all the layers at the appropriate stride.
    spacing : float
        Spacing between grid viewboxes. If between 0 and 1, it's
        interpreted as a proportion of the size of the viewboxes.
        If equal or greater than 1, it's interpreted as screen pixels.

        .. versionadded:: 0.6.0
            ``spacing`` was added in 0.6.0.
    """

    stride: GridStride = 1
    shape: tuple[GridHeight, GridWidth] = (-1, -1)
    enabled: bool = False
    spacing: GridSpacing = 0.0
    _model_parent: 'Canvas | None' = PrivateAttr(default=None)

    def _on_parent_assigned(self) -> None:
        canvas = cast('Canvas', self._model_parent)
        canvas.events.size.connect(self._ensure_safe_spacing)
        self._ensure_safe_spacing()

    @property
    def _layers(self) -> Sequence[Layer]:
        if self._model_parent is None:
            return []
        return self._model_parent._layers

    @property
    def actual_shape(self) -> tuple[int, int]:
        """Return the actual shape of the grid.

        This will return the shape parameter, unless one of the row
        or column numbers is -1 in which case it will compute the
        optimal shape of the grid given the number of layers and
        current stride.

        If the grid is not enabled, this will return (1, 1).

        Returns
        -------
        shape : 2-tuple of int
            Number of rows and columns in the grid.
        """
        if (
            not self.enabled  # grid is off
            or not self._layers  # no self._layers
            or not self._effective_indices()  # no visible layers
        ):
            return (1, 1)

        n_row, n_column = self.shape

        # Number of viewboxes is the number of stride-groups that contain at
        # least one visible layer. Stacking within a viewbox is determined by
        # original indices (see _viewbox_groups), while groups with no visible
        # layers are omitted so hidden layers never leave empty viewboxes.
        _, occupied = self._viewbox_groups()
        n_grid_squares = len(occupied)

        if n_row == -1 and n_column == -1:
            n_column = np.ceil(np.sqrt(n_grid_squares)).astype(int)
            n_row = np.ceil(n_grid_squares / n_column).astype(int)
        elif n_row == -1:
            n_row = np.ceil(n_grid_squares / n_column).astype(int)
        elif n_column == -1:
            n_column = np.ceil(n_grid_squares / n_row).astype(int)

        n_row = max(1, n_row)
        n_column = max(1, n_column)

        return (int(n_row), int(n_column))

    def position(self, index: int) -> tuple[int, int]:
        """Return the position of a given linear index in the grid, or (-1, -1) if the layer is hidden/excluded.

        If the grid is not enabled, this will return (0, 0).

        Parameters
        ----------
        index : int
            Position of current layer in layer list.
        layers : Sequence[Layer]
            List of layers that need to be placed in the grid.

        Returns
        -------
        position : 2-tuple of int
            Row and column position of current index in the grid, or (-1, -1) if the layer is hidden/excluded.
        """
        if not self.enabled or not self._layers:
            return (0, 0)

        if index < 0 or index >= len(self._layers):
            raise ValueError(
                f'Index {index} is out of bounds for number of layers {len(self._layers)}.'
            )

        effective_indices = self._effective_indices()
        if index not in effective_indices:
            return (-1, -1)

        n_row, n_column = self.actual_shape

        # Map this layer's viewbox group to its linear position among the
        # occupied groups (groups with no visible layer are compacted away).
        group_of, occupied = self._viewbox_groups()
        adj_i = occupied.index(group_of[index])

        adj_i = adj_i % (n_row * n_column)
        i_row = adj_i // n_column
        i_column = adj_i % n_column
        # convert to python int from np int
        return (int(i_row), int(i_column))

    def contents_at(self, position: tuple[int, int]) -> tuple[int, ...]:
        """Return the indices contained in the viewbox at the given position.

        If the grid is not enabled, this will return ().

        Parameters
        ----------
        position : 2-tuple of int
            Row and column position of current index in the grid.
        Returns
        -------
        indices : tuple of int
            Position of current layer in layer list.
        """
        return tuple(
            i for i in range(len(self._layers)) if self.position(i) == position
        )

    def iter_viewboxes(
        self,
    ) -> Iterator[tuple[tuple[int, int], tuple[int, ...]]]:
        """Iterate over each viewbox and its contained indices.

        Parameters
        ----------
        layers : Sequence[Layer]
            List of layers that need to be placed in the grid.

        Yields
        -------
        position : 2-tuple of int
            Row and column position of current index in the grid.
        indices : tuple of int
            Position of current layer in layer list.
        """
        for row, col in np.ndindex(self.actual_shape):
            yield (row, col), self.contents_at((row, col))

    def _effective_indices(self) -> list[int]:
        """Return indices of layers that are active (visible) in the grid.

        Only visible layers occupy grid viewboxes, so hidden layers never
        create empty viewboxes. Stacking within a viewbox is still determined
        by each layer's original index and the stride sign.
        """
        return [i for i, layer in enumerate(self._layers) if layer.visible]

    def _viewbox_groups(self) -> tuple[dict[int, int], list[int]]:
        """Return the viewbox grouping for the given layers.

        Returns
        -------
        group_of : dict[int, int]
            Maps each layer index to its viewbox group. Positive stride uses
            contiguous ranges of original indices (``i // stride``) so toggling
            visibility never moves a layer. Negative stride follows napari's
            reference behavior of reversing the sequence, then packing
            ``stride`` layers per viewbox (``(len(layers) - 1 - i) // stride``)
            so hidden layers never collapse or re-pack visible layers.
        occupied : list[int]
            Sorted viewbox groups that contain at least one visible layer.
        """

        stride = abs(self.stride)
        n = len(self._layers)
        if self.stride > 0:
            group_of = {i: i // stride for i in range(n)}
        else:
            group_of = {i: (n - 1 - i) // stride for i in range(n)}
        visible = self._effective_indices()
        occupied = sorted({group_of[i] for i in visible})
        return group_of, occupied

    @model_validator(mode='after')
    def _validate_spacing(self):
        self._ensure_safe_spacing()
        return self

    def _ensure_safe_spacing(self):
        current_spacing = self._spacing_canvas_pixels()
        safe_spacing = self._max_safe_spacing()

        if current_spacing > safe_spacing:
            warnings.warn(
                f'Grid spacing of {current_spacing:.1f} pixels is too large and has been '
                f'reduced to {safe_spacing:.1f} pixels to prevent viewboxes from '
                'becoming too small. Consider using a smaller spacing value or '
                'increasing the canvas size.',
                UserWarning,
                stacklevel=2,
            )
            # this shouldn't cause an infinite loop cause now the spacing is fixed!
            self.spacing = safe_spacing
        return self

    @property
    def _canvas_size(self) -> tuple[int, int]:
        return getattr(self._model_parent, 'size', (800, 600))

    def _max_safe_spacing(
        self,
    ) -> int:
        """Compute the maximum "safe" spacing between viewboxes in canvas pixels.

        This value is restricted so that it does not cause viewboxes to become
        too small (<20px). If the spacing value is too large,
        then viewboxes will disappear. If viewboxes are too small than
        there will be a division by zero for zoom calculation.
        """
        # limit spacing to avoid degenerate viewboxes
        rows, cols = self.actual_shape
        canvas_width, canvas_height = self._canvas_size

        minimum_viewbox_size = 20  # pixels
        max_horizontal_spacing = (
            canvas_width - cols * minimum_viewbox_size
        ) / max(1, cols - 1)
        max_vertical_spacing = (
            canvas_height - rows * minimum_viewbox_size
        ) / max(1, rows - 1)

        max_safe_spacing = min(max_horizontal_spacing, max_vertical_spacing)
        return int(max_safe_spacing)

    def _spacing_canvas_pixels(
        self,
    ) -> int:
        """Compute the spacing between viewboxes in canvas pixels.

        If the spacing is between 0 and 1, it is interpreted as a proportion
        of the size of the individual viewboxes.
        If it is equal to or greater than 1, it is interpreted as screen pixels.
        """
        rows, cols = self.actual_shape
        canvas_width, canvas_height = self._canvas_size

        spacing = self.spacing
        if spacing >= 1:
            spacing = int(spacing)
        else:
            # percentage spacing, we need to know the pre-spacing viewbox size
            unspaced_viewbox_size = (canvas_width / cols, canvas_height / rows)
            mean_size = np.mean(unspaced_viewbox_size)
            spacing = int(spacing * mean_size)

        return spacing

    @property
    def viewbox_size(self) -> tuple[int, int]:
        """Get the size of a single viewbox (whether grid is enabled or not).

        If grid.border_width > 0, that's accounted for too.
        """
        if not self.enabled:
            return self._canvas_size

        grid_shape = np.array(self.actual_shape)
        spacing_pixels = self._spacing_canvas_pixels()
        # Now calculate actual available space
        total_gap_space = spacing_pixels * (grid_shape - 1)
        available_space = self._canvas_size - total_gap_space
        viewbox_size = available_space / grid_shape
        return tuple(viewbox_size)
