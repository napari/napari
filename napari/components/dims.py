from typing import ClassVar, Sequence, Tuple, Union

import numpy as np

from ..utils.events.dataclass import Property, dataclass


def only_2D_3D(ndisplay):
    if ndisplay not in (2, 3):
        raise ValueError(
            f"Invalid number of dimensions to be displayed {ndisplay}"
            f" must be either 2 or 3."
        )
    else:
        return ndisplay


def reorder_after_dim_reduction(order):
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


@dataclass(events=True, properties=True)
class Dims:
    """Dimensions object modeling slicing and displaying.

    Parameters
    ----------
    ndim : int, optional
        Number of dimensions
    ndisplay : int, optional
        Number of displayed dimensions.
    order : list of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    axis_labels : list of str, optional
        Dimension names

    Attributes
    ----------
    range : list of 3-tuple
        List of tuples (min, max, step), one for each dimension. In a world
        coordinates space.
    point : list of float
        List of floats setting the current value of the range slider when in
        POINT mode, one for each dimension. In a world coordinates space.
    current_step : tuple of int
        Tuple the slider position for each dims slider, in slider coordinates.
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

    ndim: int = 2
    ndisplay: Property[int, None, only_2D_3D] = 2
    last_used: int = 0

    range: Property[Tuple, None, tuple] = ()
    current_step: Property[Tuple, None, tuple] = ()
    order: Property[Tuple, None, tuple] = ()
    axis_labels: Property[Tuple, None, tuple] = ()

    _scroll_progress: ClassVar[int] = 0

    def __post_init__(self):
        max_ndim = max(
            self.ndim,
            self.ndisplay,
            len(self.axis_labels),
            len(self.order),
            len(self.range),
            len(self.current_step),
        )
        if max_ndim > self.ndim:
            self.ndim = max_ndim
        else:
            self._update(self.ndim)

    def _on_ndim_set(self, ndim):
        self._update(ndim)

    def _on_order_set(self, order):
        if not set(order) == set(range(self.ndim)):
            raise ValueError(
                f"Invalid ordering {order} for {self.ndim} dimensions"
            )

    def _on_axis_labels_set(self, axis_labels):
        if not len(axis_labels) == self.ndim:
            raise ValueError(
                f"Invalid number of axis labels {axis_labels} for {self.ndim} dimensions"
            )

    def _on_range_set(self, range_var):
        if not len(range_var) == self.ndim:
            raise ValueError(
                f"Invalid length range {range_var} for {self.ndim} dimensions"
            )

    def _update(self, ndim):
        """Update with new dimensionality.

        Parameters
        ----------
        ndim : int
            New dimensionality
        """

        if len(self.range) < ndim:
            # Range value is (min, max, step) for the entire slider
            self._range = ((0, 2, 1),) * (len(self.range) - ndim) + self.range
        if len(self.range) > ndim:
            self._range = self.range[-ndim:]

        if len(self.current_step) < ndim:
            self._current_step = (0,) * (
                len(self.current_step) - ndim
            ) + self.current_step
        if len(self.current_step) > ndim:
            self._current_step = self.current_step[-ndim:]

        self._order = tuple(range(ndim - len(self.order))) + tuple(
            o + ndim - len(self.order) for o in self.order
        )

        # Append new "default" labels to existing ones
        if self.axis_labels == tuple(map(str, range(len(self.axis_labels)))):
            self._axis_labels = tuple(map(str, range(ndim)))
        else:
            self._axis_labels = (
                tuple(map(str, range(ndim - len(self.axis_labels))))
                + self.axis_labels
            )

    @property
    def nsteps(self):
        """Number of slider steps for each dimension.
        """
        return [
            int((max_val - min_val) // step_size) + 1
            for min_val, max_val, step_size in self._range
        ]

    @property
    def point(self):
        """List of float: value of each dimension."""
        # The point value is computed from the current_step
        point = [
            min_val + step_size * value
            for (min_val, max_val, step_size), value in zip(
                self._range, self._current_step
            )
        ]
        return point

    @property
    def displayed(self):
        """Tuple: Dimensions that are displayed."""
        return self.order[-self.ndisplay :]

    @property
    def not_displayed(self):
        """Tuple: Dimensions that are not displayed."""
        return self.order[: -self.ndisplay]

    @property
    def displayed_order(self):
        """Tuple: Order of only displayed dimensions."""
        order = np.array(self.displayed)
        order[np.argsort(order)] = list(range(len(order)))
        return tuple(order)

    def reset(self):
        """Reset dims values to initial states."""
        # Don't reset axis labels
        self.range = ((0, 2, 1),) * self.ndim
        self.current_step = (0,) * self.ndim
        self.order = tuple(range(self.ndim))

    def set_range(self, axis: int, _range: Sequence[Union[int, float]]):
        """Sets the range (min, max, step) for a given dimension.

        Parameters
        ----------
        axis : int
            Dimension index.
        _range : tuple
            Range specified as (min, max, step).
        """
        axis = self._assert_axis_in_bounds(axis)
        if self.range[axis] != _range:
            full_range = list(self.range)
            full_range[axis] = _range
            self.range = full_range

    def set_point(self, axis: int, value: Union[int, float]):
        """Sets point to slice dimension in world coordinates.

        The desired point gets transformed into an integer step
        of the slider and stored in the current_step.

        Parameters
        ----------
        axis : int
            Dimension index.
        value : int or float
            Value of the point.
        """
        axis = self._assert_axis_in_bounds(axis)
        (min_val, max_val, step_size) = self._range[axis]
        raw_step = (value - min_val) / step_size
        self.set_current_step(axis, raw_step)

    def set_current_step(self, axis: int, value: int):
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
        axis = self._assert_axis_in_bounds(axis)
        step = np.round(np.clip(value, 0, self.nsteps[axis] - 1)).astype(int)

        if self._current_step[axis] != step:
            full_current_step = list(self.current_step)
            full_current_step[axis] = step
            self.current_step = full_current_step

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

    def _focus_up(self):
        """Shift focused dimension slider to be the next slider above."""
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        if self.last_used is None:
            self.last_used = sliders[-1]
        else:
            index = (sliders.index(self.last_used) + 1) % len(sliders)
            self.last_used = sliders[index]

    def _focus_down(self):
        """Shift focused dimension slider to be the next slider bellow."""
        sliders = [d for d in self.not_displayed if self.nsteps[d] > 1]
        if len(sliders) == 0:
            return

        if self.last_used is None:
            self.last_used = sliders[-1]
        else:
            index = (sliders.index(self.last_used) - 1) % len(sliders)
            self.last_used = sliders[index]

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
            full_axis_labels = list(self.axis_labels)
            full_axis_labels[axis] = str(label)
            self.axis_labels = full_axis_labels

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
        order = np.array(self.order)
        nsteps = np.array(self.nsteps)
        order[nsteps > 1] = np.roll(order[nsteps > 1], 1)
        self.order = order

    def _transpose(self):
        """Transpose displayed dimensions."""
        order = list(self.order)
        order[-2], order[-1] = order[-1], order[-2]
        self.order = order
