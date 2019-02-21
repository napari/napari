from .util.misc import imshow, scatter
from .components import Window, Viewer

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
