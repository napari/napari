from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PointsData:
    """Class with all spatially varying points data

    This includes all points properties that can be
    specified at the individual points level and
    information about where those points are
    located in our world coordinate system.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    size : array (N, D)
        Size of the point marker for N points in D dimensions.
    edge_color : array (N, 4)
        Color of the point marker border. Numeric color values should be RGB(A).
    face_color : array (N, 4)
        Color of the point marker border. Numeric color values should be RGB(A).
    index : array (N,), optional
        Integer index of point. Useful when slicing to recover position of
        sliced data in original data.
    """

    data: np.ndarray
    size: np.ndarray
    edge_color: np.ndarray
    face_color: np.ndarray
    index: Optional[np.ndarray] = None

    @property
    def ndim(self):
        """int: Dimensionality of points data."""
        return self.data.shape[1]

    def _slice_by_index(self, index, displayed):
        """Slice points data by index of points in array

        Parameters
        ----------
        index : array
            Location in array of the points to be included in the slice.
        displayed : array
            Which dimensions are to be included in slice.

        Returns
        -------
        napari.layers.points.PointsData
            Sliced points data.
        """
        if len(index) > 0:
            full_index = np.ix_(index, displayed)
            data = self.data[full_index]
            size = self.size[full_index]
            edge_color = self.edge_color[index]
            face_color = self.face_color[index]
            if self.index is not None:
                points_index = self.index[index]
            else:
                points_index = np.arange(len(self.data))[index]
        else:
            # if no points in this slice send dummy data
            data = np.zeros((0, len(displayed)))
            size = np.zeros((0, len(displayed)))
            edge_color = np.zeros((0, 4))
            face_color = np.zeros((0, 4))
            points_index = None

        return PointsData(
            data=data,
            size=size,
            edge_color=edge_color,
            face_color=face_color,
            index=points_index,
        )
