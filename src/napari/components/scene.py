from __future__ import annotations

from pydantic import Field

from napari.components.camera import Camera
from napari.components.overlays import (
    CoordinateGridOverlay,
    CrosshairOverlay,
    SceneAxesOverlay,
    SceneOverlay,
)
from napari.utils.events import EventedDictNamespace, EventedModel


class SceneEventedDictNamespace(EventedDictNamespace[SceneOverlay]):
    """EventedDictNamespace for Scene overlays.

    This class is used to provide type hints for the overlays attribute of the
    Scene class. It ensures that the overlays attribute is an EventedDictNamespace
    containing only SceneOverlay objects.
    """

    axes: SceneAxesOverlay


class Scene(EventedModel):
    """
    Scene evented model.

    Controls scene-related attributes, such as the camera and scene overlays.

    Attributes
    ----------
    camera: napari.components.camera.Camera
        The camera object modeling the position and view.
    overlays : SceneEventedDictNamespace
        A dictionary/namespace containing scene overlays. By default, it exposes
        publicly 'axes' and 'crosshair'.
    """

    camera: Camera = Field(default_factory=Camera, frozen=True)
    overlays: SceneEventedDictNamespace = Field(
        default_factory=lambda: SceneEventedDictNamespace(
            {
                'axes': SceneAxesOverlay(),
                'crosshair': CrosshairOverlay(),
                'coordinate_grid': CoordinateGridOverlay(),
            }
        ),
        frozen=True,
    )
