import logging

from qtpy import API_NAME
from vispy import app

# set vispy application to the appropriate qt backend
app.use_app(API_NAME)
del app

# set vispy logger to show warning and errors only
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.WARNING)


from napari._vispy.camera import VispyCamera
from napari._vispy.canvas import VispyCanvas
from napari._vispy.overlays.axes import VispyAxesOverlay
from napari._vispy.overlays.interaction_box import VispyInteractionBox
from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari._vispy.overlays.text import VispyTextOverlay
from napari._vispy.utils.quaternion import quaternion2euler
from napari._vispy.utils.visual import create_vispy_layer
