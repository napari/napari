from __future__ import annotations

from typing import TYPE_CHECKING

from napari.layers.base._base_constants import Mode
from napari.layers.tracks.tracks import Tracks
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def register_tracks_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(Tracks, description, repeatable)


def register_tracks_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(Tracks, description, 'mode')


@register_tracks_mode_action('Transform')
def activate_tracks_transform_mode(layer: Tracks) -> None:
    layer.mode = str(Mode.TRANSFORM)


@register_tracks_mode_action('Move camera')
def activate_tracks_pan_zoom_mode(layer: Tracks) -> None:
    layer.mode = str(Mode.PAN_ZOOM)


tracks_fun_to_mode = [
    (activate_tracks_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_tracks_transform_mode, Mode.TRANSFORM),
]
