"""Direction (orientation) edge-labels overlay model."""

from typing import Literal

from pydantic import field_validator

from napari.components._direction_edge_labels import DirectionLabelPair
from napari.components.overlays.base import CanvasOverlay
from napari.utils.color import ColorValue
from napari.utils.translations import trans


class DirectionLabelsOverlay(CanvasOverlay):
    """Direction labels pinned to the four edges of a 2D canvas.

    Renders the label facing each screen edge (top/bottom/left/right) for the
    per-world-axis ``labels``, under the current camera orientation. The labels
    are opaque strings, so any domain uses the same overlay: DICOM ``'L'``/``'R'``,
    GIS ``'N'``/``'S'``, microscopy ``'basal'``/``'apical'``. Nothing is drawn in
    3D or when the view does not resolve to two displayed axes.

    Attributes
    ----------
    labels : tuple of (str or None, str or None) or None
        Per-world-axis ``(negative, positive)`` direction-label pairs, aligned to
        the highest-numbered axes (a suffix frame reconciled to ``dims.ndim`` at
        render time, see ``reconcile_direction_labels``). Any end, a whole pair,
        or the whole value may be ``None`` to leave it unlabeled. Set this to
        drive the overlay from domain code.
    color : ColorValue or None
        Text color. If ``None``, a default contrasting the canvas is used.
    font_size : float
        Font size in points.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    """

    labels: tuple[DirectionLabelPair | None, ...] = ()
    color: ColorValue | None = None
    font_size: float = 10
    # This overlay attaches to all four edges, so it has no single position and
    # must not be tiled: ``position`` is pinned to the non-CanvasPosition value
    # 'free' so the canvas placement machinery skips it, and ``gridded`` is
    # pinned off (it positions in whole-canvas coordinates, not per grid cell).
    # It also has no single rectangular extent, so the background box is off.
    position: Literal['free'] = 'free'
    gridded: Literal[False] = False
    box: bool = False

    @field_validator('labels', mode='before')
    def _validate_labels(v):
        """Coerce to a tuple of (negative, positive) str/None pairs (or None).

        Length is reconciled to ndim at render time, not here (see
        ``reconcile_direction_labels``); this only checks entry shape/type.
        """
        if v is None:
            # Top-level None means "no labels".
            return ()
        try:
            entries = list(v)
        except TypeError:
            raise ValueError(
                trans._(
                    'axis direction labels must be a sequence of '
                    '(negative, positive) pairs; got {v!r}.',
                    deferred=True,
                    v=v,
                )
            ) from None
        labels: list[DirectionLabelPair | None] = []
        for entry in entries:
            if entry is None:
                labels.append(None)
                continue
            # A str is not a tuple/list, so this also rejects 'RL' (which would
            # otherwise unpack silently to ('R', 'L')).
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    trans._(
                        'each direction-labels entry must be None or a '
                        '(negative, positive) pair; got {entry!r}.',
                        deferred=True,
                        entry=entry,
                    )
                )
            pair = tuple(entry)
            if any(e is not None and not isinstance(e, str) for e in pair):
                raise ValueError(
                    trans._(
                        'each direction label must be a string or None; got '
                        '{entry!r}.',
                        deferred=True,
                        entry=entry,
                    )
                )
            labels.append(pair)
        return tuple(labels)
