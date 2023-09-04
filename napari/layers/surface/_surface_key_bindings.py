from napari.layers.base._base_constants import Mode
from napari.layers.surface.surface import Surface


def activate_surface_transform_mode(layer: Surface):
    layer.mode = Mode.TRANSFORM


def activate_surface_pan_zoom_mode(layer: Surface):
    layer.mode = str(Mode.PAN_ZOOM)


surface_fun_to_mode = [
    (activate_surface_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_surface_transform_mode, Mode.TRANSFORM),
]
