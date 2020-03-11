from .qt_range_slider import QHRangeSlider, QVRangeSlider
from .event_loop import gui_qt
from ..resources import import_resources
from .. import __version__
from qtpy import API, QT_VERSION

version_string = f'{__version__}_{API}_{QT_VERSION}'
import_resources(version_string)
