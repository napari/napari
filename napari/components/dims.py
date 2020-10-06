import warnings
from copy import copy
from typing import Sequence, Union

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
        List of tuples (min, max, step), one for each dimension. In a world
        coordinates space.
    point : list of float
        List of floats setting the current value of the range slider when in
        POINT mode, one for each dimension. In a world coordinates space.
    nsteps : tuple of int
        Number of steps available to each slider.
    ndim : int
        Number of dimensions.
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
            ndim=None,
            ndisplay=None,
            order=None,
            camera=None,
        )
        self._range = EventedList()
        self._order = ()
        self._point = EventedList()
        self._axis_labels = EventedList()
        self._scroll_progress = 0
        self.last_used = None
        self._ndisplay = 2 if ndisplay is None else ndisplay

        if ndim is None and order is None and axis_labels is None:
            ndim = self._ndisplay
        elif ndim is None and order is None:
            ndim = len(axis_labels)
        elif ndim is None and axis_labels is None:
            ndim = len(order)
        self.ndim = ndim

        if axis_labels is not None:
            self.axis_labels = axis_labels

        if order is not None:
            self.order = order

    @property
    def range(self):
        """List of 3-tuple: (min, max, step size) of each dimension.
        """
        return self._range

    @property
    def _nsteps(self):
        """Number of slider steps for each dimension."""
        return [
            int((max_val - min_val - step_size) // step_size) + 1
            for min_val, max_val, step_size in self._range
        ]

    @property
    def _current_step(self):
        """Tuple of int: value of slider position for each dimension."""
        step = [
            (value - min_val) / step_size
            for (min_val, max_val, step_size), value in zip(
                self._range, self.point
            )
        ]
        return step

    @property
    def point(self):
        """List of float: value of each dimension."""
        # The point value is computed from the current_step
        return self._point

    @point.setter
    def point(self, point):
        if list(self.point) == point:
            return

        if len(point) != self.ndim:
            raise ValueError(
                f"Number of points doesn't match number of dimensions. Number"
                f" of given points was {len(point)}, number of dimensions is"
                f" {self.ndim}."
            )

        for i, p in enumerate(list(point)):
            self._point[i] = p

    @property
    def axis_labels(self):
        """List of labels for each axis."""
        return self._axis_labels

    @axis_labels.setter
    def axis_labels(self, labels):
        if list(self.axis_labels) == labels:
            return

        if len(labels) != self.ndim:
            raise ValueError(
                f"Number of labels doesn't match number of dimensions. Number"
                f" of given labels was {len(labels)}, number of dimensions is"
                f" {self.ndim}. Note: If you wish to keep some of the "
                "dimensions unlabeled, use '' instead."
            )

        for i, ax in enumerate(list(labels)):
            self._axis_labels[i] = ax

    @property
    def order(self):
        """Tuple of int: Display order of dimensions."""
        return self._order

    @order.setter
    def order(self, order):
        if self._order == tuple(order):
            return

        if not set(order) == set(range(self.ndim)):
            raise ValueError(
                f"Invalid ordering {order} for {self.ndim} dimensions"
            )

        self._order = tuple(order)
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
                # Insert new default axis labels
                total = ndim - cur_ndim - 1
                self.axis_labels.insert(0, str(total - i))

            self._order = tuple(range(ndim - cur_ndim)) + tuple(
                [o + ndim - cur_ndim for o in self.order]
            )

            # If axis labels were previouly ordered, preserve full ordering
            if list(self.axis_labels[ndim - cur_ndim :]) == list(
                map(str, range(cur_ndim))
            ):
                for i in range(ndim - cur_ndim, ndim):
                    self._axis_labels[i] = str(i)

        elif ndim < cur_ndim:
            for i in range(cur_ndim - ndim):
                self.range.pop(0)
                self.point.pop(0)
                self.axis_labels.pop(0)
            reorder_after_dim_reduction = 1
            self._order = tuple(
                reorder_after_dim_reduction(self._order[-ndim:])
            )

        # Notify listeners that the number of dimensions have changed
        self.events.ndim()

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
            # Point is the slider value if in point mode
            self.point[axis] = 0
            # Default axis labels go from "-ndim" to "-1" so new axes can easily be added
            self.axis_labels[axis] = str(axis - self.ndim)
        self.order = tuple(range(self.ndim))

    def set_range(self, axis: int, value: Sequence[Union[int, float]]):
        """Sets the range (min, max, step) for a given dimension.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : tuple
            Range specified as (min, max, step).
        """
        warnings.warn(
            'To be deprecated, use self.range[axis] = value instead.'
        )
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
        warnings.warn(
            'To be deprecated, use self.point[axis] = value instead.'
        )
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
        warnings.warn(
            'To be deprecated, use self.axis_labels[axis] = value instead.'
        )
        self.axis_labels[axis] = label

    def set_current_step(self, axis: int, value: Union[int, float]):
        """Sets the slider step at which to slice this dimension.

        The position of the slider in world coordinates gets
        calculated from the current_step of the slider.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : int or float
            Value of the point.
        """
        step = np.round(np.clip(value, 0, self._nsteps[axis] - 1)).astype(int)

        min_val, max_val, step_size = self.range[axis]
        self.point[axis] = min_val + step_size * step

    def _increment_dims_right(self, axis: int = None):
        """Increment dimensions to the right along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        if axis is not None:
            self.set_current_step(axis, self.current_step[axis] + 1)

    def _increment_dims_left(self, axis: int = None):
        """Increment dimensions to the left along given axis, or last used axis if None

        Parameters
        ----------
        axis : int, optional
            Axis along which to increment dims, by default None
        """
        if axis is None:
            axis = self.last_used
        if axis is not None:
            self.set_current_step(axis, self.current_step[axis] - 1)

    def _roll(self):
        """Roll order of dimensions for display."""
        order = np.array(self.order)
        nsteps = np.array(self.nsteps)
        order[nsteps > 1] = np.roll(order[nsteps > 1], 1)
        self.order = list(order)

    def _transpose(self):
        """Transpose displayed dimensions."""
        order = copy(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order
