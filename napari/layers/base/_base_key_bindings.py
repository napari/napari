# from app_model.types import KeyCode

# from napari.layers.base.base import Layer


# @Layer.bind_key(KeyCode.Space, overwrite=True)
# def hold_to_pan_zoom(layer):
#     """Hold to pan and zoom in the viewer."""
#     if layer._mode != layer._modeclass.PAN_ZOOM:
#         # on key press
#         prev_mode = layer.mode
#         layer.mode = layer._modeclass.PAN_ZOOM

#         yield

#         # on key release
#         layer.mode = prev_mode
