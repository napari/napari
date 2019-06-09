from .viewer import Viewer
from ._view import view
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
