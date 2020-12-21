"""VispyCanvas class.
"""
from vispy.scene import SceneCanvas

from .utils_gl import get_max_texture_sizes


class VispyCanvas(SceneCanvas):
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
        # before calling super().__init__().
        self.max_texture_sizes = None

        super().__init__(*args, **kwargs)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()
        self.events.touch.connect(self._process_touch_event)

    def _process_touch_event(self, event):
        #        #import traceback
        #        #traceback.print_stack()
        if event.type == 'pinch':
            print(f"{event.type=}")
            print(f"{event.pos=}")
            print(f"{event.last_pos=}")
            print(f"{event.scale=}")
            print(f"{event.last_scale=}")
            print(f"{event.rotation=}")
        elif event.type == 'begin':
            pass
        elif event.type == 'end':
            pass

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super()._process_mouse_event(event)
