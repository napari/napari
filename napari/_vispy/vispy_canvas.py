"""VispyCanvas class.
"""
from vispy import app
from vispy.scene import SceneCanvas
from vispy.scene.node import Node

from ..utils.perf import block_timer
from .image import Image as ImageVisual
from .utils_gl import get_max_texture_sizes

canvas = None
callbacks = []

POOL_SIZE = 100


class VisualPool:
    def __init__(self, pool_parent):
        self.visuals = []
        self.pool_root = Node()

        print(f"VisualPool.__init__: creating {POOL_SIZE} visuals")
        self.visuals = self._create_visuals()
        print("VisualPool.__init__: done")

        self.pool_root.parent = pool_parent

    def _create_visuals(self):
        visuals = []
        for i in range(POOL_SIZE):
            visual = ImageVisual(None, method='auto')
            visual.parent = self.pool_root
            visual.visible = False
            visuals.append(visual)
        return visuals

    def get_node(self):
        visual = self.visuals.pop(0)
        visual.visible = True
        return visual

    def return_node(self, node):
        node.visible = True
        self.visuals.append(node)

    @property
    def size(self) -> None:
        return len(self.visuals)


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

        global canvas
        canvas = self
        # Since the base class is frozen we must create this attribute
        # before calling super().__init__().
        self.max_texture_sizes = None

        self.processing = False

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.pool = None

        super().__init__(*args, **kwargs)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

    def add_visuals(self, parent):
        self.pool = VisualPool(parent)

    def get_node(self):
        return self.pool.get_node()

    def return_node(self, node):
        self.pool.return_node(node)

    def set_pool_parent(self, parent):
        print("set_pool_parent -> START")
        self.processing = True
        self.pool.parent = parent
        self.processing = False
        print("set_pool_parent -> END")

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super()._process_mouse_event(event)

    def update(self, node=None):
        if not self.processing:
            super().update(node)

    def on_timer(self, event):
        global callbacks
        if len(callbacks) == 0:
            return
        self.processing = True
        print(f"process_callbacks: {len(callbacks)} callbacks")
        count = 0
        while True:
            try:
                callback = callbacks.pop(0)
            except IndexError:
                self.processing = False
                return
            count += 1
            print(f"process_callbacks: calling callback {count}")
            with block_timer("callback", print_time=True):
                callback()
            break
        self.update()
        self.processing = False
