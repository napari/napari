import warnings

import numpy as np

from napari.layers.utils.interaction_box import (
    InteractionBoxHandle,
    generate_transform_box_from_layer,
    get_nearby_handle,
)
from napari.utils.transforms import Affine
from napari.utils.translations import trans


def highlight_box_handles(layer, event):
    """
    Highlight the hovered handle of a TransformBox.
    """
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


def _translate_with_box(
    layer, initial_affine, initial_mouse_pos, mouse_pos, event
):
    offset = mouse_pos - initial_mouse_pos
    new_affine = Affine(translate=offset).compose(initial_affine)
    layer.affine = layer.affine.replace_slice(event.dims_displayed, new_affine)


def _rotate_with_box(
    layer,
    initial_affine,
    initial_mouse_pos,
    initial_handle_coords,
    initial_center,
    mouse_pos,
    event,
):
    # calculate the angle between the center-handle vector and the center-mouse vector
    center_to_handle = (
        initial_handle_coords[InteractionBoxHandle.ROTATION] - initial_center
    )
    center_to_handle /= np.linalg.norm(center_to_handle)
    center_to_mouse = mouse_pos - initial_center
    center_to_mouse /= np.linalg.norm(center_to_mouse)
    angle = np.arctan2(center_to_mouse[1], center_to_mouse[0]) - np.arctan2(
        center_to_handle[1], center_to_handle[0]
    )

    new_affine = (
        Affine(translate=initial_center)
        .compose(Affine(rotate=np.rad2deg(angle)))
        .compose(Affine(translate=-initial_center))
        .compose(initial_affine)
    )
    layer.affine = layer.affine.replace_slice(event.dims_displayed, new_affine)


def _scale_with_box(
    layer,
    initial_affine,
    initial_world_to_data,
    initial_data2physical,
    nearby_handle,
    initial_center,
    initial_handle_coords_data,
    mouse_pos,
    event,
):
    locked_aspect_ratio = False
    if 'Shift' in event.modifiers:
        if nearby_handle in InteractionBoxHandle.corners():
            locked_aspect_ratio = True
        else:
            warnings.warn(
                trans._(
                    'Aspect ratio can only be blocked when resizing from a corner',
                    deferred=True,
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    # note: we work in data space from here on!

    # if Control is held, instead of locking into place the opposite handle,
    # lock into place the center of the layer and resize around it.
    if 'Control' in event.modifiers:
        scaling_center = initial_world_to_data(initial_center)
    else:
        # opposite handle
        scaling_center = initial_handle_coords_data[
            InteractionBoxHandle.opposite_handle(nearby_handle)
        ]

    # calculate the distance to the scaling center (which is fixed) before and after drag
    center_to_handle = (
        initial_handle_coords_data[nearby_handle] - scaling_center
    )
    center_to_mouse = initial_world_to_data(mouse_pos) - scaling_center

    # get per-dimension scale values
    with warnings.catch_warnings():
        # a "divide by zero" warning is raised here when resizing along only one axis
        # (i.e: dragging the central handle of the TransformBox).
        # That's intended, because we get inf or nan, which we can then replace with 1s
        # and thus maintain the size along that axis.
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = center_to_mouse / center_to_handle
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
    layer.affine = layer.affine.replace_slice(event.dims_displayed, new_affine)


def transform_with_box(layer, event):
    """
    Translate, rescale or rotate a layer by dragging a TransformBox handle.
    """
    if not len(event.dims_displayed) == 2:
        return

    # we work in data space so we're axis aligned which simplifies calculation
    # same as Layer.data_to_world
    initial_data_to_world = layer._transforms[1:].simplified.set_slice(
        event.dims_displayed
    )
    initial_world_to_data = initial_data_to_world.inverse
    initial_mouse_pos = np.array(event.position)[event.dims_displayed]
    initial_mouse_pos_data = initial_world_to_data(initial_mouse_pos)

    initial_handle_coords_data = generate_transform_box_from_layer(
        layer, event.dims_displayed
    )
    nearby_handle = get_nearby_handle(
        initial_mouse_pos_data, initial_handle_coords_data
    )

    if nearby_handle is None:
        return

    # now that we have the nearby handles, other calculations need
    # the world space handle positions
    initial_handle_coords = initial_data_to_world(initial_handle_coords_data)

    # initial layer transform so we can calculate changes later
    initial_affine = layer.affine.set_slice(event.dims_displayed)

    # needed for rescaling
    initial_data2physical = layer._transforms['data2physical'].set_slice(
        event.dims_displayed
    )

    # needed for resize and rotate
    initial_center = np.mean(
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
        mouse_pos = np.array(event.position)[event.dims_displayed]

        if nearby_handle == InteractionBoxHandle.INSIDE:
            _translate_with_box(
                layer, initial_affine, initial_mouse_pos, mouse_pos, event
            )
            yield
        elif nearby_handle == InteractionBoxHandle.ROTATION:
            _rotate_with_box(
                layer,
                initial_affine,
                initial_mouse_pos,
                initial_handle_coords,
                initial_center,
                mouse_pos,
                event,
            )
            yield
        else:
            _scale_with_box(
                layer,
                initial_affine,
                initial_world_to_data,
                initial_data2physical,
                nearby_handle,
                initial_center,
                initial_handle_coords_data,
                mouse_pos,
                event,
            )
            yield
