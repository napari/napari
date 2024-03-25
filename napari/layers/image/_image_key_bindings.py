from __future__ import annotations

from collections.abc import Generator
from typing import Callable, Union

from app_model.types import KeyCode

import napari
from napari.layers.base._base_constants import Mode
from napari.layers.image.image import Image
from napari.layers.utils.interactivity_utils import (
    orient_plane_normal_around_cursor,
)
from napari.layers.utils.layer_utils import register_layer_action
from napari.utils.events import Event
from napari.utils.translations import trans


def register_image_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Image, description, repeatable)


@Image.bind_key(KeyCode.KeyZ, overwrite=True)
@register_image_action(trans._('Orient plane normal along z-axis'))
def orient_plane_normal_along_z(layer: Image) -> None:
    orient_plane_normal_around_cursor(layer, plane_normal=(1, 0, 0))


@Image.bind_key(KeyCode.KeyY, overwrite=True)
@register_image_action(trans._('orient plane normal along y-axis'))
def orient_plane_normal_along_y(layer: Image) -> None:
    orient_plane_normal_around_cursor(layer, plane_normal=(0, 1, 0))


@Image.bind_key(KeyCode.KeyX, overwrite=True)
@register_image_action(trans._('orient plane normal along x-axis'))
def orient_plane_normal_along_x(layer: Image) -> None:
    orient_plane_normal_around_cursor(layer, plane_normal=(0, 0, 1))


@Image.bind_key(KeyCode.KeyO, overwrite=True)
@register_image_action(trans._('orient plane normal along view direction'))
def orient_plane_normal_along_view_direction(
    layer: Image,
) -> Union[None, Generator[None, None, None]]:
    viewer = napari.viewer.current_viewer()
    if viewer is None or viewer.dims.ndisplay != 3:
        return None

    def sync_plane_normal_with_view_direction(
        event: Union[None, Event] = None
    ) -> None:
        """Plane normal syncronisation mouse callback."""
        layer.plane.normal = layer._world_to_displayed_data_ray(
            viewer.camera.view_direction, [-3, -2, -1]
        )

    # update plane normal and add callback to mouse drag
    sync_plane_normal_with_view_direction()
    viewer.camera.events.angles.connect(sync_plane_normal_with_view_direction)
    yield None
    # remove callback on key release
    viewer.camera.events.angles.disconnect(
        sync_plane_normal_with_view_direction
    )
    return None


@register_image_action(trans._('orient plane normal along view direction'))
def orient_plane_normal_along_view_direction_no_gen(layer: Image) -> None:
    viewer = napari.viewer.current_viewer()
    if viewer is None or viewer.dims.ndisplay != 3:
        return
    layer.plane.normal = layer._world_to_displayed_data_ray(
        viewer.camera.view_direction, [-3, -2, -1]
    )


@register_image_action(trans._('Transform'))
def activate_image_transform_mode(layer: Image) -> None:
    layer.mode = str(Mode.TRANSFORM)


@register_image_action(trans._('Pan/zoom'))
def activate_image_pan_zoom_mode(layer: Image) -> None:
    layer.mode = str(Mode.PAN_ZOOM)


image_fun_to_mode = [
    (activate_image_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_image_transform_mode, Mode.TRANSFORM),
]
