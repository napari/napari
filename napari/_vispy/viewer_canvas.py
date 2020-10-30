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
        super().__init__(*args, **kwargs)

        # Call this now so we query the OpenGL parameters while we know a
        # Canvas exists. This prevents get_max_texture_sizes() from
        # creating a temporary Canvas which can cause problems.
        #
        # If get_max_texture_sizes() is called a second time it will return
        # the same values without actually executing the body of the function.
        get_max_texture_sizes()

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super()._process_mouse_event(event)
