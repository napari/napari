import warnings

vispy_warning = "VisPy is not yet compatible with matplotlib 2.2+"

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=vispy_warning
    )
    from .viewer import Viewer

from .view_function import view
from ._qt import gui_qt

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
