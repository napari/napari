from napari.layers.base._base_constants import Mode
from napari.layers.tracks.tracks import Tracks


def activate_tracks_transform_mode(layer: Tracks):
    layer.mode = Mode.TRANSFORM


def activate_tracks_pan_zoom_mode(layer: Tracks):
    layer.mode = str(Mode.PAN_ZOOM)


tracks_fun_to_mode = [
    (activate_tracks_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_tracks_transform_mode, Mode.TRANSFORM),
]
