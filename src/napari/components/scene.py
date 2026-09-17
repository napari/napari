from __future__ import annotations

from pydantic import Field

from napari.components.camera import Camera
from napari.components.overlays import (
    SceneAxesOverlay,
    SceneOverlay,
)
from napari.utils.events import EventedDictNamespace, EventedModel


class Scene(EventedModel):
    """
    Scene evented model.

    Controls scene-related attributes, such as the camera and scene overlays.

    Attributes
    ----------
    camera: napari.components.camera.Camera
        The camera object modeling the position and view.
    overlays : EventedDictNamespace
        A dictionary/namespace containing scene overlays. By default, it exposes
        publicly 'axes'.
    """

    camera: Camera = Field(default_factory=Camera, frozen=True)
    # NOTE: the type ignore comment below is only there because pyrefly
    #       thinks EventedDictNamespace[SceneAxesOverlay] is not a subtype
    #       of EventedDictNamespace[SceneOverlay]. This will not be necessary
    #       as soon as we add any other overlay
    overlays: EventedDictNamespace[SceneOverlay] = Field(
        default_factory=lambda: EventedDictNamespace(
            {
                'axes': SceneAxesOverlay(),
            }
        )
    )
