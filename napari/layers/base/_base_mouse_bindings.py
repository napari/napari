import numpy as np

from napari.components.overlays._interaction_box_constants import (
    InteractionBoxHandle,
)
from napari.utils.geometry import generate_interaction_box_vertices

# from napari.utils.transforms import Affine


def _get_interaction_box_position(box_bounds, pos):
    # generates in vispy canvas pos, so invert x and y
    top_left, bot_right = (tuple(point) for point in box_bounds.T[:, ::-1])
    vertices = generate_interaction_box_vertices(
        top_left, bot_right, handles=True
    )[:, ::-1]

    dist = np.linalg.norm(pos - vertices, axis=1)
    tolerance = dist.max() / 100
    close_to_vertex = np.isclose(dist, 0, atol=tolerance)
    if np.any(close_to_vertex):
        return InteractionBoxHandle(np.argmax(close_to_vertex))
    elif np.all((pos[::-1] >= top_left) & (pos[::-1] <= bot_right)):
        return InteractionBoxHandle.ALL
    else:
        return None


def highlight_box_handles(layer, event):
    if not layer._overlays['transform_box'].visible:
        return

    bounds = layer._display_bounding_box(event.dims_displayed)
    coordinates = layer.world_to_data(event.position)
    pos = np.array(coordinates)[event.dims_displayed]
    nearby_handle = _get_interaction_box_position(bounds, pos)
    if nearby_handle is not None:
        layer._overlays['transform_box'].selected_vertex = nearby_handle


def transform_with_box(layer, event):
    if not layer._overlays['transform_box'].visible:
        return

    bounds = layer._display_bounding_box(event.dims_displayed)
    coordinates = layer.world_to_data(event.position)
    pos = np.array(coordinates)[event.dims_displayed]
    nearby_handle = _get_interaction_box_position(bounds, pos)

    if nearby_handle is None:
        return

    # initial_transform = Affine(
    #     rotate=layer.rotate,
    #     translate=layer.translate,
    #     scale=layer.scale,
    #     shear=layer.shear,
    # ).set_slice(event.dims_displayed)
    initial_position = pos

    yield

    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        pos = np.array(coordinates)[event.dims_displayed]

        if nearby_handle == InteractionBoxHandle.ALL:
            print(f'translate by {initial_position - pos}')
            yield
        else:
            print(
                f'rescale by dragging {repr(nearby_handle)} by {initial_position - pos}'
            )
            yield

    print('done transforming!')
