"""VispyCanvas class.
"""
from weakref import WeakSet

from vispy.scene import SceneCanvas, Widget

from napari._vispy.utils.gl import get_max_texture_sizes
from napari.utils.colormaps.standardize_color import transform_color


class VispyCanvas:
    """SceneCanvas for our QtViewer class.

    Add two features to SceneCanvas. Ignore mousewheel events with
    modifiers, and get the max texture size in __init__().

    Attributes
    ----------
    max_texture_sizes : Tuple[int, int]
        The max textures sizes as a (2d, 3d) tuple.

    """

    _instances = WeakSet()

    def __init__(self, *args, **kwargs):

        # Since the base class is frozen we must create this attribute
        # before calling super().__init__().
        self.max_texture_sizes = None
        self._last_theme_color = None
        self._background_color_override = None
        self.viewer = kwargs["parent"].viewer
        self.napari_canvas = kwargs["parent"].viewer.canvas
        self.scene_canvas = SceneCanvas(*args, **kwargs)
        self._instances.add(self)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

        self.scene_canvas.events.ignore_callback_errors = False
        self.scene_canvas.context.set_depth_func('lequal')

        # Connecting events from SceneCanvas
        self.scene_canvas.events.draw.connect(self.viewer.dims.enable_play)

    @property
    def destroyed(self):
        return self.scene_canvas._backend.destroyed

    @property
    def background_color_override(self):
        return self._background_color_override

    @background_color_override.setter
    def background_color_override(self, value):
        self._background_color_override = value
        self.bgcolor = value or self._last_theme_color

    def _on_theme_change(self, event):
        self._set_theme_change(event.value)

    def _set_theme_change(self, theme: str):
        from napari.utils.theme import get_theme

        # Note 1. store last requested theme color, in case we need to reuse it
        # when clearing the background_color_override, without needing to
        # keep track of the viewer.
        # Note 2. the reason for using the `as_hex` here is to avoid
        # `UserWarning` which is emitted when RGB values are above 1
        self._last_theme_color = transform_color(
            get_theme(theme, False).canvas.as_hex()
        )[0]
        self.bgcolor = self._last_theme_color

    @property
    def bgcolor(self):
        self.scene_canvas.bgcolor.hex

    @bgcolor.setter
    def bgcolor(self, value):
        _value = self._background_color_override or value
        self.scene_canvas.bgcolor = _value

    @property
    def central_widget(self):
        """Overrides SceneCanvas.central_widget to make border_width=0"""
        if self.scene_canvas._central_widget is None:
            self.scene_canvas._central_widget = Widget(
                size=self.scene_canvas.size,
                parent=self.scene_canvas.scene,
                border_width=0,
            )
        return self.scene_canvas._central_widget

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super().scene_canvas._process_mouse_event(event)
