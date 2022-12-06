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

    # we work in data space so we're axis aligned which simplifies calculation
    # same as Layer.world_to_data
    world_to_data = (
        layer._transforms[1:].set_slice(event.dims_displayed).inverse
    )
    pos = np.array(world_to_data(event.position))[event.dims_displayed]

    handle_coords = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    # TODO: dynamically set tolerance based on canvas size so it's not hard to pick small layer
    nearby_handle = get_nearby_handle(pos, handle_coords)

    # set the selected vertex of the box to the nearby_handle (can also be INSIDE or None)
    layer._overlays['transform_box'].selected_vertex = nearby_handle


def transform_with_box(layer, event):
    if not len(event.dims_displayed) == 2:
        return

    # we work in data space so we're axis aligned which simplifies calculation
    # same as Layer.data_to_world
    data_to_world = layer._transforms[1:].simplified.set_slice(
        event.dims_displayed
    )
    world_to_data = data_to_world.inverse
    pos = np.array(event.position)[event.dims_displayed]
    pos_data = world_to_data(pos)

    initial_handle_coords_data = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    nearby_handle = get_nearby_handle(pos_data, initial_handle_coords_data)

    if nearby_handle is None:
        return

    # now that we have the nearby handles, other calculations need
    # the world space handle positions
    initial_handle_coords = data_to_world(initial_handle_coords_data)

    # initial layer transform so we can calculate changes later
    initial_affine = layer.affine.set_slice(event.dims_displayed)
    initial_data2physical = layer._transforms['data2physical'].set_slice(
        event.dims_displayed
    )
    initial_position = pos

    # needed for resize and rotate
    center = np.mean(
        initial_handle_coords[
            [
                InteractionBoxHandle.TOP_LEFT,
                InteractionBoxHandle.BOTTOM_RIGHT,
            ]
        ],
        axis=0,
    )

    yield

    while event.type == 'mouse_move':
        pos = np.array(event.position)[event.dims_displayed]

        if nearby_handle == InteractionBoxHandle.INSIDE:
            offset = pos - initial_position
            new_affine = Affine(translate=offset).compose(initial_affine)
            layer.affine = layer.affine.replace_slice(
                event.dims_displayed, new_affine
            )
            yield
        elif nearby_handle == InteractionBoxHandle.ROTATION:

            # calculate the angle between the center-handle vector and the center-mouse vector
            center_to_handle = (
                initial_handle_coords[InteractionBoxHandle.ROTATION] - center
            )
            center_to_handle /= np.linalg.norm(center_to_handle)
            center_to_mouse = pos - center
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
            # scale transform

            if 'Shift' in event.modifiers:
                if nearby_handle not in InteractionBoxHandle.corners():
                    raise ValueError(
                        'aspect ratio can only be blocked when resizing from a corner'
                    )
                locked_aspect_ratio = True
            else:
                locked_aspect_ratio = False

            locked_center = 'Control' in event.modifiers

            # work in data space

            if locked_center:
                scaling_center = world_to_data(center)
            else:
                # opposite handle
                scaling_center = initial_handle_coords_data[
                    InteractionBoxHandle.opposite_handle(nearby_handle)
                ]

            # calculate the distance to the scaling center (which is fixed) before and after drag
            center_to_handle = (
                initial_handle_coords_data[nearby_handle] - scaling_center
            )
            center_to_mouse = world_to_data(pos) - scaling_center

            # get per-dimension scale values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scale = center_to_mouse / center_to_handle
            # infinite values (due to numerical imprecision) mean we are rescaling only
            # one dimension (mid-side handles), so we set to 1.
            scale = np.nan_to_num(scale, posinf=1, neginf=1)

            if locked_aspect_ratio:
                scale_factor = np.linalg.norm(scale)
                scale = [scale_factor, scale_factor]

            new_affine = (
                # bring layer to axis aligned space
                initial_affine.compose(initial_data2physical)
                # center opposite handle
                .compose(Affine(translate=scaling_center))
                # apply scale
                .compose(Affine(scale=scale))
                # undo all the above, backwards
                .compose(Affine(translate=-scaling_center))
                .compose(initial_data2physical.inverse)
                .compose(initial_affine.inverse)
                # compose with the original affine
                .compose(initial_affine)
            )
            layer.affine = layer.affine.replace_slice(
                event.dims_displayed, new_affine
            )
            yield
