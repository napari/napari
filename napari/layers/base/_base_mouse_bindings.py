import numpy as np

from napari.components.overlays._interaction_box_utils import (
    InteractionBoxHandle,
    generate_transform_box_from_layer,
    get_nearby_handle,
)
from napari.utils.transforms import Affine


def highlight_box_handles(layer, event):
    if not len(event.dims_displayed) == 2:
        return

    pos_data = layer.world_to_data(event.position)
    pos_displayed = np.array(pos_data)[event.dims_displayed]

    handle_coords = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    nearby_handle = get_nearby_handle(pos_displayed, handle_coords)

    # set the selected vertex of the box to the nearby_handle (can also be INSIDE or None)
    layer._overlays['transform_box'].selected_vertex = nearby_handle


def transform_with_box(layer, event):
    if not len(event.dims_displayed) == 2:
        return

    pos_data = layer.world_to_data(event.position)
    pos_displayed = np.array(pos_data)[event.dims_displayed]

    handle_coords = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    nearby_handle = get_nearby_handle(pos_displayed, handle_coords)

    if nearby_handle is None:
        return

    # initial layer transform so we can calculate
    initial_transform = layer._transforms.set_slice(event.dims_displayed)
    initial_affine = layer.affine.set_slice(event.dims_displayed)
    initial_position = pos_displayed
    center = np.mean(
        handle_coords[
            [InteractionBoxHandle.TOP_LEFT, InteractionBoxHandle.BOTTOM_RIGHT]
        ],
        axis=0,
    )

    yield

    while event.type == 'mouse_move':
        # same as Layer.world_to_data
        pos_data = initial_transform[1:].simplified.inverse(event.position)
        pos_displayed = np.array(pos_data)[event.dims_displayed]

        if nearby_handle == InteractionBoxHandle.INSIDE:
            offset = pos_displayed - initial_position
            new_affine = Affine(translate=offset).compose(initial_affine)
            layer.affine = layer.affine.replace_slice(
                event.dims_displayed, new_affine
            )
            yield
        elif nearby_handle == InteractionBoxHandle.ROTATION:
            initial_vector = (
                handle_coords[InteractionBoxHandle.ROTATION] - center
            )
            initial_vector /= np.linalg.norm(initial_vector)
            new_vector = pos_displayed - center
            new_vector /= np.linalg.norm(new_vector)
            angle = np.arctan2(new_vector[1], new_vector[0]) - np.arctan2(
                initial_vector[1], initial_vector[0]
            )
            # TODO: center of rotation is wrong, despite angles being correct. Need to transform it to other
            #       coordinates system?
            new_affine = (
                Affine(translate=center)
                .compose(Affine(rotate=np.rad2deg(angle)))
                .compose(Affine(translate=-center))
                .compose(initial_affine)
            )
            layer.affine = layer.affine.replace_slice(
                event.dims_displayed, new_affine
            )
            yield
        else:

            if 'Shift' in event.modifiers:
                if nearby_handle not in InteractionBoxHandle.corners():
                    raise ValueError(
                        'aspect ratio can only be blocked when resizing from a corner'
                    )
                locked_aspect_ratio = True
            else:
                locked_aspect_ratio = False

            opposite_handle = handle_coords[
                InteractionBoxHandle.opposite_handle(nearby_handle)
            ]
            initial_vec_from_opposite_handle = (
                handle_coords[nearby_handle] - opposite_handle
            )
            new_vec_from_opposite_handle = pos_displayed - opposite_handle
            scale = (
                new_vec_from_opposite_handle / initial_vec_from_opposite_handle
            )
            scale = np.nan_to_num(scale, nan=0, posinf=1, neginf=1)
            if locked_aspect_ratio:
                scale = np.linalg.norm(scale)
                scale = [scale, scale]
            new_affine = (
                Affine(translate=opposite_handle)
                .compose(Affine(scale=scale))
                .compose(Affine(translate=-opposite_handle))
                .compose(initial_affine)
            )
            layer.affine = layer.affine.replace_slice(
                event.dims_displayed, new_affine
            )
            yield
