"""Map per-axis direction labels to screen edges for a 2D display.

This is pure geometry over the viewer model: given a label for each end of
each world axis (opaque strings such as ``'L'``/``'R'`` or ``'anterior'``),
work out which label faces each screen edge under the current camera. napari
never interprets the strings, so any domain (medical LPS, a microscopy stage,
a detector frame) uses the identical mechanism.

The mapping is only defined when the world-to-screen map is a signed axis
permutation. In napari that is exactly the ``ndisplay == 2`` case: the 2D
canvas is a ``PanZoomCamera`` with no rotation and ``camera.angles`` is unused,
so the two displayed world axes map to the screen axes up to a sign taken from
``camera.orientation``. For ``ndisplay == 3`` (arbitrary rotation/perspective)
there are no unambiguous edge labels, and this returns ``None`` rather than a
guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari.utils.camera_orientations import (
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)
from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Sequence

    from napari.components.camera import Camera
    from napari.components.dims import Dims

# One (negative-end, positive-end) label pair per world axis; either end (or
# the whole pair) may be ``None`` to leave that direction unlabeled.
DirectionLabelPair = tuple[Optional[str], Optional[str]]

__all__ = [
    'DirectionLabelPair',
    'direction_edge_labels',
    'reconcile_direction_labels',
]


def direction_edge_labels(
    direction_labels: Sequence[Optional[DirectionLabelPair]] | None,
    *,
    dims: Dims,
    camera: Camera,
) -> dict[str, str] | None:
    """Which direction label faces each screen edge, for the current 2D view.

    Parameters
    ----------
    direction_labels : sequence of (str or None, str or None) or None
        One ``(negative, positive)`` label pair per world axis, indexed by
        world-axis number (same length and indexing as ``dims`` has axes). The
        pair labels the axis's decreasing and increasing *world* directions;
        either end, or a whole axis's pair, may be ``None`` to leave it
        unlabeled. ``direction_labels`` itself may be ``None`` to mean "no
        labels".
    dims : napari.components.Dims
        The viewer dims, read for ``ndisplay`` and ``displayed``.
    camera : napari.components.Camera
        The viewer camera, read for ``orientation``.

    Returns
    -------
    dict of str to str, or None
        ``None`` when the mapping is undefined (``dims.ndisplay != 2``).
        Otherwise a dict whose keys are a subset of ``{'top', 'bottom',
        'left', 'right'}`` mapping each edge to the label facing it; edges
        whose direction is unlabeled are omitted, so the dict is empty when
        nothing is labeled.

    Raises
    ------
    ValueError
        If ``direction_labels`` is not ``None`` and its length differs from
        ``dims.ndim``, if an entry is neither ``None`` nor a
        ``(negative, positive)`` pair, or if a label is neither ``None`` nor a
        string. These are checked for every axis regardless of ``ndisplay``,
        so a malformed non-displayed axis is still rejected.

    Notes
    -----
    The screen convention is taken from ``camera.orientation``, a
    ``(depth, vertical, horizontal)`` tuple. The two displayed axes are
    ``dims.displayed == (vertical_axis, horizontal_axis)``; the vertical axis
    maps to screen-y and the horizontal axis to screen-x. A
    ``VerticalAxisOrientation.DOWN`` sends the vertical axis's positive
    direction to the bottom edge (``UP`` to the top); a
    ``HorizontalAxisOrientation.RIGHT`` sends the horizontal axis's positive
    direction to the right edge (``LEFT`` to the left). The depth component
    does not affect the in-plane edges. Reducing an oblique frame to per-axis
    labels is the caller's responsibility; this function only places the
    labels it is given.
    """
    # Validate the whole sequence up front, before any early return, so the
    # ValueError contract holds regardless of ndisplay or which axes show.
    if direction_labels is not None:
        _validate_direction_labels(direction_labels, dims.ndim)

    if dims.ndisplay != 2:
        return None

    if direction_labels is None:
        return {}

    vertical_axis, horizontal_axis = dims.displayed
    _, vertical, horizontal = camera.orientation

    edges: dict[str, str] = {}

    if vertical == VerticalAxisOrientation.DOWN:
        positive_edge, negative_edge = 'bottom', 'top'
    else:
        positive_edge, negative_edge = 'top', 'bottom'
    _place_axis(
        edges,
        direction_labels[vertical_axis],
        positive_edge=positive_edge,
        negative_edge=negative_edge,
    )

    if horizontal == HorizontalAxisOrientation.RIGHT:
        positive_edge, negative_edge = 'right', 'left'
    else:
        positive_edge, negative_edge = 'left', 'right'
    _place_axis(
        edges,
        direction_labels[horizontal_axis],
        positive_edge=positive_edge,
        negative_edge=negative_edge,
    )

    return edges


def _validate_direction_labels(
    direction_labels: Sequence[Optional[DirectionLabelPair]],
    ndim: int,
) -> None:
    """Check length and the shape/type of every entry; raise on any problem."""
    if len(direction_labels) != ndim:
        raise ValueError(
            trans._(
                # NB: not ``n=`` — ``trans._`` reserves ``n`` for pluralization,
                # so a ``{n}`` placeholder crashes message construction for any
                # count != 1 (it looks for a nonexistent plural form).
                'direction_labels must have one entry per dimension: got '
                '{n_labels} for ndim={ndim}.',
                deferred=True,
                n_labels=len(direction_labels),
                ndim=ndim,
            )
        )
    for entry in direction_labels:
        if entry is None:
            continue
        # A str is a 2-length iterable, so it would unpack silently; reject it
        # explicitly along with any non-(list/tuple) or wrong-length entry.
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(
                trans._(
                    'each direction_labels entry must be None or a '
                    '(negative, positive) pair; got {entry!r}.',
                    deferred=True,
                    entry=entry,
                )
            )
        for label in entry:
            if label is not None and not isinstance(label, str):
                raise ValueError(
                    trans._(
                        'each direction label must be a string or None; got '
                        '{label!r}.',
                        deferred=True,
                        label=label,
                    )
                )


def _place_axis(
    edges: dict[str, str],
    entry: Optional[DirectionLabelPair],
    *,
    positive_edge: str,
    negative_edge: str,
) -> None:
    """Add one axis's two labels to ``edges``; ``entry`` is already validated."""
    if entry is None:
        return
    negative, positive = entry
    if negative is not None:
        edges[negative_edge] = negative
    if positive is not None:
        edges[positive_edge] = positive


def reconcile_direction_labels(
    labels: Sequence[Optional[DirectionLabelPair]],
    ndim: int,
) -> tuple[Optional[DirectionLabelPair], ...]:
    """Reconcile a stored direction-label tuple to ``ndim``.

    ``labels`` is a *suffix* (trailing) coordinate frame: it aligns with the
    highest-numbered world axes, so a caller may store labels for more axes than
    are currently present. Reconcile to exactly ``ndim`` entries the same way
    napari reindexes ``axis_labels`` when dimensionality changes -- prepend
    ``None`` for new leading axes, keep the trailing entries when reducing.

    This is meant to run at render time against a stored, *unmutated* label
    tuple, so labels survive ``dims.ndim`` changes (layer add/remove) without a
    stored value ever going stale. The result always has length ``ndim`` (empty
    for ``ndim <= 0``), so ``direction_edge_labels`` can consume it directly.
    """
    labels = tuple(labels)
    if len(labels) < ndim:
        return (None,) * (ndim - len(labels)) + labels
    # Trailing ndim entries; for ndim <= 0 this is the empty tuple.
    return labels[len(labels) - ndim :]
