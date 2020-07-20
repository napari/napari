import warnings
from copy import copy
from typing import Union, Sequence
import numpy as np

from ..utils.events import EmitterGroup, EventedList


class Dims:
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int, optional
        Number of dimensions.
    ndisplay : int, optional
        Number of displayed dimensions.
    order : list of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str, optional
        Dimension names.

    Attributes
    ----------
    events : EmitterGroup
        Event emitter group
    range : list of 3-tuple
        List of tuples (min, max, step), one for each dimension.
    point : list of float
        List of floats setting the current value of the range slider, one for
        each dimension.
    clip : bool
        Flag if to clip indices based on range. Needed for image-like
        layers, but prevents shape-like layers from adding new shapes
        outside their range.
    ndim : int
        Number of dimensions.
    indices : tuple of slice object
        Tuple of slice objects for slicing arrays on each dimension, one for
        each dimension.
    displayed : tuple
        List of dimensions that are displayed.
    not_displayed : tuple
        List of dimensions that are not displayed.
    displayed_order : tuple
        Order of only displayed dimensions.
    """

    def __init__(self, ndim=None, *, ndisplay=2, order=None, axis_labels=None):
        super().__init__()

        # Events:
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            axis_labels=None,
            ndim=None,
            ndisplay=None,
            order=None,
            camera=None,
        )
        self._range = EventedList()
        self._point = EventedList()
        self._order = []
        self._axis_labels = []
        self.clip = True
        self._ndisplay = 2 if ndisplay is None else ndisplay

        if ndim is None and order is None and axis_labels is None:
            ndim = self._ndisplay
        elif ndim is None and order is None:
            ndim = len(axis_labels)
        elif ndim is None and axis_labels is None:
            ndim = len(order)
        self.ndim = ndim

        if order is not None:
            if len(order) != ndim:
                raise ValueError(
                    f"Length of order must be identical to ndim."
                    f" ndim is {ndim} while order is {order}."
                )
            self._order = order
        if axis_labels is not None:
            if len(axis_labels) != ndim:
                raise ValueError(
                    f"Length of axis labels must be identical to ndim."
                    f" ndim is {ndim} while axis labels is {axis_labels}."
                )
            self._axis_labels = list(axis_labels)

    @property
    def range(self):
        """List of 3-tuple: (min, max, step size) of each dimension.
        """
        return self._range

    @property
    def max_indices(self):
        """Maximum index for each dimension (in data space).
        """
        return [((ma - st) // st) for mi, ma, st in self._range]

    @property
    def point(self):
        """List of int: value of each dimension."""
        return self._point

    @property
    def axis_labels(self):
        """List of labels for each axis."""
        return copy(self._axis_labels)

    @axis_labels.setter
    def axis_labels(self, labels):
        if self._axis_labels == labels:
            return

        if len(labels) != self.ndim:
            raise ValueError(
                f"Number of labels doesn't match number of dimensions. Number"
                f" of given labels was {len(labels)}, number of dimensions is"
                f" {self.ndim}. Note: If you wish to keep some of the "
                "dimensions unlabeled, use '' instead."
            )

        self._axis_labels = list(labels)
        for axis in range(self.ndim):
            self.events.axis_labels(axis=axis)

    @property
    def order(self):
        """List of int: Display order of dimensions."""
        return copy(self._order)

    @order.setter
    def order(self, order):
        if np.all(self._order == order):
            return

        if not len(order) == self.ndim:
            raise ValueError(
                f"Invalid ordering {order} for {self.ndim} dimensions"
            )

        self._order = order
        self.events.order()
        self.events.camera()

    @property
    def ndim(self):
        """Returns the number of dimensions.

        Returns
        -------
        ndim : int
            Number of dimensions
        """
        return len(self.point)

    @ndim.setter
    def ndim(self, ndim):
        cur_ndim = self.ndim
        if cur_ndim == ndim:
            return
        elif ndim > cur_ndim:
            for i in range(ndim - cur_ndim):
                # Range value is (min, max, step) for the entire slider
                self.range.insert(0, (0, 2, 1))
                # Point is the slider value
                self.point.insert(0, 0)

            self._order = list(range(ndim - cur_ndim)) + [
                o + ndim - cur_ndim for o in self.order
            ]
            # Append new "default" labels to existing ones
            if self._axis_labels == list(map(str, range(cur_ndim))):
                self._axis_labels = list(map(str, range(ndim)))
            else:
                self._axis_labels = (
                    list(map(str, range(ndim - cur_ndim))) + self._axis_labels
                )
        elif ndim < cur_ndim:
            for i in range(cur_ndim - ndim):
                self.range.pop(0)
                self.point.pop(0)

            self._order = self._reorder_after_dim_reduction(
                self._order[-ndim:]
            )
            self._axis_labels = self._axis_labels[-ndim:]

        # Notify listeners that the number of dimensions have changed
        self.events.ndim()

    def _reorder_after_dim_reduction(self, order):
        """Ensure current dimension order is preserved after dims are dropped.

        Parameters
        ----------
        order : list-like
            The data to reorder.

        Returns
        -------
        arr : list
            The original array with the unneeded dimension
            thrown away.
        """
        arr = np.array(order)
        arr[np.argsort(arr)] = range(len(arr))
        return arr.tolist()

    @property
    def indices(self):
        """Tuple of slice objects for slicing arrays on each dimension."""
        slice_list = []
        for axis in range(self.ndim):
            if axis in self.displayed:
                slice_list.append(slice(None))
            else:
                if self.clip:
                    p = np.clip(
                        self.point[axis],
                        np.round(self.range[axis][0]),
                        np.round(self.range[axis][1]) - 1,
                    )
                else:
                    p = self.point[axis]
                p = np.round(p / self.range[axis][2]).astype(int)
                slice_list.append(p)
        return tuple(slice_list)

    @property
    def ndisplay(self):
        """Int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self._ndisplay == ndisplay:
            return

        if ndisplay not in (2, 3):
            raise ValueError(
                f"Invalid number of dimensions to be displayed {ndisplay}"
            )

        self._ndisplay = ndisplay
        self.events.ndisplay()
        self.events.camera()

    @property
    def displayed(self):
        """Tuple: Dimensions that are displayed."""
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self):
        """Tuple: Dimensions that are not displayed."""
        return list(self.order[: -self.ndisplay])

    @property
    def displayed_order(self):
        """Tuple: Order of only displayed dimensions."""
        order = np.array(self.displayed)
        order[np.argsort(order)] = list(range(len(order)))
        return tuple(order)

    def reset(self):
        """Reset dims values to initial states."""
        for axis in range(self.ndim):
            # Range value is (min, max, step) for the entire slider
            self.range[axis] = (0, 2, 1)
            # Point is the slider value
            self.point[axis] = 0
            self.order[axis] = axis
            # Default axis labels go from "-ndim" to "-1" so new axes can easily be added
            self.axis_labels[axis] = str(axis - self.ndim)

    def set_range(self, axis: int, value: Sequence[Union[int, float]]):
        """Sets the range (min, max, step) for a given dimension.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : tuple
            Range specified as (min, max, step).
        """
        warnings.warn('To be deprecated')
        self.range[axis] = value

    def set_point(self, axis: int, value: Union[int, float]):
        """Sets the point at which to slice this dimension.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : int or float
            Value of the point.
        """
        warnings.warn('To be deprecated')
        self.point[axis] = value

    def set_axis_label(self, axis: int, label: str):
        """Sets a new axis label for the given axis.

        Parameters
        ----------
        axis : int
            Dimension index
        label : str
            Given label
        """
        axis = self._assert_axis_in_bounds(axis)
        if self.axis_labels[axis] != str(label):
            self._axis_labels[axis] = str(label)
            self.events.axis_labels(axis=axis)

    def _assert_axis_in_bounds(self, axis: int) -> int:
        """Assert a given value is inside the existing axes of the image.

        Returns
        -------
        axis : int
            The axis which was checked for validity.

        Raises
        ------
        ValueError
            The given axis index is out of bounds.
        """
        if axis not in range(-self.ndim, self.ndim):
            msg = (
                f'Axis {axis} not defined for dimensionality {self.ndim}. '
                f'Must be in [{-self.ndim}, {self.ndim}).'
            )
            raise ValueError(msg)

        return axis % self.ndim

    def _roll(self):
        """Roll order of dimensions for display."""
        self.order = np.roll(self.order, 1)

    def _transpose(self):
        """Transpose displayed dimensions."""
        order = copy(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order
