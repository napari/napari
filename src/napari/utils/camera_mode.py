from __future__ import annotations

from enum import auto

from napari.utils.misc import StringEnum


class CameraMode(StringEnum):
    """Camera synchronization mode for 2D/3D display switching.

    Controls how the camera state (center, zoom, angles) is managed when
    toggling between 2D (``ndisplay=2``) and 3D (``ndisplay=3``) views.
    This enum is intentionally placed in ``napari.utils`` to avoid circular
    imports: both ``napari.components.camera`` and ``napari.settings`` need
    to reference it.
    """

    SEPARATE = auto()
    """Each ndisplay mode remembers its own center, zoom, and angles."""

    SHARED = auto()
    """Center and zoom are shared between 2D and 3D views.

    The depth (z) component is set from the dims slider on 2D→3D
    and the dims slider tracks the camera z on 3D→2D, so that the views
    can round-trip.
    """

    LEGACY = auto()
    """No caching — ``fit_to_view()`` is called on every ndisplay switch.

    This replicates the napari behavior before per-mode camera caching was
    introduced, where every ndisplay switch reset the camera zoom and center.
    """
