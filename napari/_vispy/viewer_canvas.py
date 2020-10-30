"""ViewerCanvas class.
"""
from vispy.scene import SceneCanvas

from .utils_gl import get_max_texture_sizes


class ViewerCanvas(SceneCanvas):
    """SceneCanvas for our QtViewer class.

    Add two features to SceneCanvas. Ignore mousewheel events with
    modifiers, and get the max texture size in __init__().

    Attributes
    ----------
    max_texture_sizes : Tuple[int, int]
        The max textures sizes as a (2d, 3d) tuple.

    """

    def __init__(self, *args, **kwargs):

        # Since the base class is frozen we must create this attribute
        # before called before super().__init__().
        self.max_texture_sizes = None

        super().__init__(*args, **kwargs)

        # Call get_max_texture_sizes() so we query OpenGL right now while
        # we know a Canvas exists. The results of get_max_texture_sizes()
        # are cached, so subsequent calls are instant.
        self.max_texture_sizes = get_max_texture_sizes()

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super()._process_mouse_event(event)
