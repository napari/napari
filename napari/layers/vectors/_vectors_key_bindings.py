from napari.layers.base._base_constants import Mode
from napari.layers.vectors.vectors import Vectors


def activate_vectors_transform_mode(layer: Vectors):
    layer.mode = str(Mode.TRANSFORM)


def activate_vectors_pan_zoom_mode(layer: Vectors):
    layer.mode = str(Mode.PAN_ZOOM)


vectors_fun_to_mode = [
    (activate_vectors_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_vectors_transform_mode, Mode.TRANSFORM),
]
