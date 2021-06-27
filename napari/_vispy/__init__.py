import logging

import qtpy
from vispy import app

# set vispy application to the appropriate qt backend
app.use_app(qtpy.API_NAME)
del app

# set vispy logger to show warning and errors only
vispy_logger = logging.getLogger('vispy')
vispy_logger.setLevel(logging.WARNING)


from .quaternion import quaternion2euler
from .utils import create_vispy_visual
from .vispy_axes_visual import VispyAxesVisual
from .vispy_camera import VispyCamera
from .vispy_canvas import VispyCanvas
from .vispy_scale_bar_visual import VispyScaleBarVisual
from .vispy_text_visual import VispyTextVisual
