"""VispyCanvas class.
"""
import time
from weakref import WeakSet

from qtpy.QtCore import QSize
from vispy.scene import SceneCanvas, Widget

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events import EmitterGroup, Event
from .utils.gl import get_max_texture_sizes


class VispyCanvas(SceneCanvas):
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
        super().__init__(*args, **kwargs)
        self._instances.add(self)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

        self.events.ignore_callback_errors = False
        self.native.setMinimumSize(QSize(200, 200))
        self.context.set_depth_func('lequal')

    @property
    def destroyed(self):
        return self._backend.destroyed

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
        from ..utils.theme import get_theme

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
        SceneCanvas.bgcolor.fget(self)

    @bgcolor.setter
    def bgcolor(self, value):
        _value = self._background_color_override or value
        SceneCanvas.bgcolor.fset(self, _value)

    @property
    def central_widget(self):
        """Overrides SceneCanvas.central_widget to make border_width=0"""
        if self._central_widget is None:
            self._central_widget = Widget(
                size=self.size, parent=self.scene, border_width=0
            )
        return self._central_widget

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super()._process_mouse_event(event)


class FramerateMonitor:
    """Tracks and filters the framerate emitted from the canvas measure_fps() callback."""

    def __init__(
        self,
        fps_window: float = 0.5,
        stale_threshold: float = 0.6,
        debounce_threshold: int = 2,
    ):

        self.events = EmitterGroup(source=self, fps=Event)
        self._fps_window = fps_window
        self._debounce_counter = 0
        self._debounce_threshold = debounce_threshold
        self._last_update = time.time()
        self._stale_threshold = stale_threshold

        self._fps = 0
        self._measuring = False

        self._last_measurement_valid = False

    @property
    def fps(self) -> float:
        """The most recently measure framerate in frames per second."""
        return self._fps

    @property
    def valid(self) -> bool:
        """Flag set to True if the current fps measurement is valid."""
        return self._last_measurement_valid and not self._fps_stale()

    def _fps_stale(self):
        """Check if the too much time has elapsed since the last fps update.

        Returns
        -------
        fps_stale : bool
            Flag set to True if the time since the last update is greater
            than the _stale_threshold
        """
        return (time.time() - self._last_update) > self._stale_threshold

    def update_fps(self, fps: float):
        """Update with the most recently measured framerate.

        This only stores the new framerate if the last draw was within
        the specified stale_threshold and the debounce condition has
        been met.

        If the framerate update is valid, the fps event is emitted.

        This is generally connected to the canvas.measure_fps() callback.

        Parameters
        ----------
        fps : float
            The newly measured framerate in frames per second.
        """
        if not self._fps_stale():
            # do nothing if the last fps measurement is still valid
            return
        elif self._measuring is False:
            # if the measurement is stale, start measuring
            self._last_measurement_valid = False
            self._measuring = True
            self._debounce_counter = 0

        # debounce and update fps
        # we need to debounce because the fps average is
        # calculated over multiple calls, so the first ones
        # are not very accurate
        self._debounce_counter += 1
        if self._debounce_counter > self._debounce_threshold:
            self._fps = fps

            # update states
            self._last_measurement_valid = True
            self._measuring = False
            self._last_update = time.time()

            # emit the event
            self.events.fps(fps=self.fps)
