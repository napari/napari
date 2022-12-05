import warnings

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

    # same as Layer.world_to_data
    data_to_world = layer._transforms.set_slice(
        event.dims_displayed
    ).simplified
    pos_displayed = np.array(event.position)[event.dims_displayed]

    handle_coords = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    handle_coords = data_to_world(handle_coords)
    nearby_handle = get_nearby_handle(pos_displayed, handle_coords)

    # set the selected vertex of the box to the nearby_handle (can also be INSIDE or None)
    layer._overlays['transform_box'].selected_vertex = nearby_handle


def transform_with_box(layer, event):
    if not len(event.dims_displayed) == 2:
        return

    # same as Layer.world_to_data
    data_to_world = layer._transforms.set_slice(
        event.dims_displayed
    ).simplified
    pos_displayed = np.array(event.position)[event.dims_displayed]

    initial_handle_coords = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    initial_handle_coords = data_to_world(initial_handle_coords)
    nearby_handle = get_nearby_handle(pos_displayed, initial_handle_coords)

    if nearby_handle is None:
        return

    # initial layer transform so we can calculate changes later
    initial_affine = layer.affine.set_slice(event.dims_displayed)
    initial_position = pos_displayed

    yield

    while event.type == 'mouse_move':
        pos_displayed = np.array(event.position)[event.dims_displayed]

        if nearby_handle == InteractionBoxHandle.INSIDE:
            offset = pos_displayed - initial_position
            new_affine = Affine(translate=offset).compose(initial_affine)
            layer.affine = layer.affine.replace_slice(
                event.dims_displayed, new_affine
            )
            yield
        elif nearby_handle == InteractionBoxHandle.ROTATION:
            center = np.mean(
                initial_handle_coords[
                    [
                        InteractionBoxHandle.TOP_LEFT,
                        InteractionBoxHandle.BOTTOM_RIGHT,
                    ]
                ],
                axis=0,
            )

            # calculate the angle between the center-handle vector and the center-mouse vector
            center_to_handle = (
                initial_handle_coords[InteractionBoxHandle.ROTATION] - center
            )
            center_to_handle /= np.linalg.norm(center_to_handle)
            center_to_mouse = pos_displayed - center
            center_to_mouse /= np.linalg.norm(center_to_mouse)
            angle = np.arctan2(
                center_to_mouse[1], center_to_mouse[0]
            ) - np.arctan2(center_to_handle[1], center_to_handle[0])

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

            # calculate the distance to the opposite handle (which is fixed) before and after drag
            opposite_handle = initial_handle_coords[
                InteractionBoxHandle.opposite_handle(nearby_handle)
            ]
            opposite_to_handle = (
                initial_handle_coords[nearby_handle] - opposite_handle
            )
            opposite_to_mouse = pos_displayed - opposite_handle

            # TODO: prevent shear!

            # get per-dimension scale values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scale = opposite_to_mouse / opposite_to_handle
            # infinite values (due to numerical imprecision) mean we are rescaling only
            # one dimension, so we set to 1.
            scale = np.nan_to_num(scale, posinf=1, neginf=1)

            if locked_aspect_ratio:
                scale_factor = np.linalg.norm(scale)
                scale = [scale_factor, scale_factor]

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
