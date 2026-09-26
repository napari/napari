"""Map per-axis direction labels to the screen edges of a 2D view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from napari.utils.camera_orientations import (
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from napari.components.camera import Camera
    from napari.components.dims import Dims


def direction_edge_labels(
    direction_labels: Sequence[tuple[str | None, str | None] | None],
    *,
    dims: Dims,
    camera: Camera,
) -> dict[str, str] | None:
    """Which direction label faces each screen edge, for the current 2D view.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    direction_labels : sequence
        One entry per world axis: a ``(negative, positive)`` pair labeling the
        axis's decreasing and increasing world directions, or ``None`` if the
        axis is unlabeled. Either label in a pair may also be ``None``.
    dims : napari.components.Dims
        The viewer dims.
    camera : napari.components.Camera
        The viewer camera.

    Returns
    -------
    dict of str to str, or None
        The label facing each of ``'top'``, ``'bottom'``, ``'left'`` and
        ``'right'``, omitting unlabeled edges. ``None`` when
        ``dims.ndisplay != 2`` or fewer than two axes are displayed.

    Raises
    ------
    ValueError
        If ``direction_labels`` does not have one entry per dimension.

    Notes
    -----
    The displayed axes ``(vertical, horizontal)`` come from ``dims.displayed``
    and their screen directions from ``camera.orientation``, so the result
    follows axis flips and transposes.
    """
    if len(direction_labels) != dims.ndim:
        raise ValueError(
            'direction_labels must have one entry per dimension: got '
            f'{len(direction_labels)} for ndim={dims.ndim}.'
        )
    if dims.ndisplay != 2 or len(dims.displayed) != 2:
        return None

    _, vertical, horizontal = camera.orientation
    vertical_edges = (
        ('top', 'bottom')
        if vertical == VerticalAxisOrientation.DOWN
        else ('bottom', 'top')
    )
    horizontal_edges = (
        ('left', 'right')
        if horizontal == HorizontalAxisOrientation.RIGHT
        else ('right', 'left')
    )

    edges = {}
    for axis, axis_edges in zip(
        dims.displayed, (vertical_edges, horizontal_edges), strict=True
    ):
        pair = direction_labels[axis] or (None, None)
        for edge, label in zip(axis_edges, pair, strict=True):
            if label is not None:
                edges[edge] = label
    return edges
