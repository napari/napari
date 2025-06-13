from collections.abc import Iterator

import numpy as np

from napari.settings._application import (
    GridHeight,
    GridSpacing,
    GridStride,
    GridWidth,
)
from napari.utils.events import EventedModel


class GridCanvas(EventedModel):
    """Grid for canvas.

    Right now the only grid mode that is still inside one canvas with one
    camera, but future grid modes could support multiple canvases.

    Attributes
    ----------
    enabled : bool
        If grid is enabled or not.
    stride : int
        Number of layers to place in each grid square before moving on to
        the next square. The default ordering is to place the most visible
        layer in the top left corner of the grid. A negative stride will
        cause the order in which the layers are placed in the grid to be
        reversed.
    shape : 2-tuple of int
        Number of rows and columns in the grid. A value of -1 for either or
        both of will be used the row and column numbers will trigger an
        auto calculation of the necessary grid shape to appropriately fill
        all the layers at the appropriate stride.
    spacing : float
        Spacing between grid layers, as a proportion of the average
        of the height and width of the extent of layers in world coordinates after slicing.
        A value of 0.0 will have the grid layers touching each other.
        Positive values will space the layers apart, and negative values
        will overlap the layers.
    border_width : int
        Width of the border separating each viewbox in pixels. Borders will be highlighted
        baes on which layers are selected in the layerlists.

        .. versionadded:: 0.6.0
            ``spacing`` was added in 0.6.0.
    """

    # fields
    # See https://github.com/pydantic/pydantic/issues/156 for why
    # these need a type: ignore comment
    stride: GridStride = 1  # type: ignore[valid-type]
    shape: tuple[GridHeight, GridWidth] = (-1, -1)  # type: ignore[valid-type]
    enabled: bool = False
    spacing: GridSpacing = 0  # type: ignore[valid-type]
    border_width: int = 0

    def actual_shape(self, nlayers: int = 1) -> tuple[int, int]:
        """Return the actual shape of the grid.

        This will return the shape parameter, unless one of the row
        or column numbers is -1 in which case it will compute the
        optimal shape of the grid given the number of layers and
        current stride.

        If the grid is not enabled, this will return (1, 1).

        Parameters
        ----------
        nlayers : int
            Number of layers that need to be placed in the grid.

        Returns
        -------
        shape : 2-tuple of int
            Number of rows and columns in the grid.
        """
        if not self.enabled:
            return (1, 1)

        if nlayers == 0:
            return (1, 1)

        n_row, n_column = self.shape
        n_grid_squares = np.ceil(nlayers / abs(self.stride)).astype(int)

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

    def position(self, index: int, nlayers: int) -> tuple[int, int]:
        """Return the position of a given linear index in grid.

        If the grid is not enabled, this will return (0, 0).

        Parameters
        ----------
        index : int
            Position of current layer in layer list.
        nlayers : int
            Number of layers that need to be placed in the grid.

        Returns
        -------
        position : 2-tuple of int
            Row and column position of current index in the grid.
        """
        if not self.enabled:
            return (0, 0)

        n_row, n_column = self.actual_shape(nlayers)

        # Adjust for forward or reverse ordering
        adj_i = nlayers - index - 1 if self.stride < 0 else index

        adj_i = adj_i // abs(self.stride)
        adj_i = adj_i % (n_row * n_column)
        i_row = adj_i // n_column
        i_column = adj_i % n_column
        # convert to python int from np int
        return (int(i_row), int(i_column))

    def contents_at(
        self, position: tuple[int, int], nlayers: int
    ) -> tuple[int, ...]:
        """Return the indices contained in the quadrant at the given position.

        If the grid is not enabled, this will return ().

        Parameters
        ----------
        position : 2-tuple of int
            Row and column position of current index in the grid.
        nlayers : int
            Number of layers that need to be placed in the grid.

        Returns
        -------
        indices : tuple of int
            Position of current layer in layer list.
        """
        if not self.enabled:
            return ()

        return tuple(
            i for i in range(nlayers) if self.position(i, nlayers) == position
        )

    def iter_quadrants(
        self, nlayers: int
    ) -> Iterator[tuple[tuple[int, int], tuple[int, ...]]]:
        """Iterate over each quadrant and its contained indices.

        Parameters
        ----------
        nlayers : int
            Number of layers that need to be placed in the grid.

        Yields
        -------
        position : 2-tuple of int
            Row and column position of current index in the grid.
        indices : tuple of int
            Position of current layer in layer list.
        """
        for row, col in np.ndindex(self.actual_shape(nlayers)):
            yield (row, col), self.contents_at((row, col), nlayers)
