"""QtFrameRate widget.
"""
import math
import time
from typing import Optional

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

# Shape of the frame rate bitmap.
BITMAP_SHAPE = (20, 270, 4)

# Left to right are segments.
NUM_SEGMENTS = 30

# 16.7ms or faster is our minimum reading, one segment only.
MIN_MS = 16.17
LOG_MIN_MS = math.log10(MIN_MS)

# 100ms will be all segments lit up.
MAX_MS = 100
LOG_MAX_MS = math.log10(MAX_MS)  # 4

# Don't use pure RGB, looks better?
GREEN = (57, 252, 3, 255)
YELLOW = (252, 232, 3, 255)
RED = (252, 78, 3, 255)

SEGMENT_SPACING = BITMAP_SHAPE[1] / NUM_SEGMENTS
SEGMENT_GAP = 2
SEGMENT_WIDTH = SEGMENT_SPACING - SEGMENT_GAP

LIVE_SEGMENTS = 20

# LEDs decay from MAX to OFF. We leave them visible when off because it
# looks better, and so you can see how high the meter goes.
ALPHA_MAX = 255
ALPHA_OFF = 25

# DECAY_MS (X, Y) means if segment is at least X then keep that LED
# on for Y milliseconds. Peak means we hit that exact value,
# while active means it was a leading LED on the way to that peak.
#
# The general goals for the LED brightness is:
#
# 1) The current peak is bright and easily visible.
# 2) The LEDs leading up to the current peak are bright so that makes
#    the current peak easier to see.
# 3) The peaks decay more slowly then the leading/active LEDs. So the slow
#    frames are visible even when the framerate has improved.
#
# We want to see at a glance what the current frame rate is, but we want to
# be able to see the "recent history" of slow frames, and see the pattern
# of that slowness. Like one super slow frame, plus some medium-slow ones.
DECAY_MS = {"peak": [(0, 1000), (10, 2000), (20, 3000)], "active": [(0, 250)]}

COLORS = [(0, GREEN), (10, YELLOW), (20, RED)]


def _decay_seconds(index: int, decay_config) -> float:
    """Return duration in seconds the segment should stay on.

    This is only called during __init__ to create a lookup table, so
    it doesn't matter how fast it is.

    Parameters
    ----------
    index : int
        The LED segment index.

    Return
    ------
    float
        Time in milliseconds to stay lit up.
    """
    for limit, ms in reversed(decay_config):
        if index >= limit:
            return ms / 1000
    assert False, "DECAY_MS table is messed up?"
    return 0


def _segment_color(index: int) -> tuple:
    for limit, color in reversed(COLORS):
        if index >= limit:
            return color
    assert False, "DECAY_MS table is messed up?"
    return (0, 0, 0, 0)


def _clamp(value, low, high):
    return max(min(value, high), low)


def _get_peak(delta_seconds: float) -> None:
    """Get highest segment that should be lit up.

    Parameters
    ----------
    delta_seconds : float
        The current frame interval.
    """
    if delta_seconds <= 0:
        delta_ms = 0
        log_value = 0
    else:
        # Create log value where MIN_MS is zero.
        delta_ms = delta_seconds * 1000
        log_value = math.log10(delta_ms) - LOG_MIN_MS

    # Compute fraction [0..1] for the whole width (all segments)
    # and then return the max the semgent.
    fraction = _clamp(log_value / LOG_MAX_MS, 0, 1)
    peak = int(fraction * (NUM_SEGMENTS - 1))
    print(f"{delta_ms} -> {peak}")
    return peak


def _expire(now: float, array: np.ndarray, decay_seconds: np.ndarray) -> None:
    # TODO_OCTREE: do this with numpy?
    for index in range(NUM_SEGMENTS):
        max_time = decay_seconds[index]
        elapsed = now - array[index]
        if elapsed > max_time:
            array[index] = 0  # Segment went dark.


def _get_fraction(
    now: float, index: int, array: np.array, decay_seconds: np.ndarray,
) -> float:
    duration = now - array[index]
    decay = decay_seconds[index]
    return _clamp(duration / decay, 0, 1)


def _print_mapping():
    """Print mapping from LED segment to millseconds.

    Since we don't yet have the reverse math figured out, we print out
    the mapping by sampling. Crude but works.
    """
    previous = None
    for ms in range(0, 10000):
        segment = _get_peak(ms)
        if previous != segment:
            print(f"ms: {ms} -> {segment}")
            previous = segment


class LedState:
    """State of the LEDs in the frame rate display."""

    def __init__(self):
        self.peak = 0  # Current highest segment lit up
        count = NUM_SEGMENTS

        # Use numpy array lookups for speed, compactness.
        self._color = np.zeros((count, 4), dtype=np.uint8)
        self._decay_peak_seconds = np.zeros((count), dtype=np.float)
        self._decay_active_seconds = np.zeros((count), dtype=np.float)

        # Last time this LED segment was active, because
        # it was leading up to the current peak.
        self._last_active = np.zeros((count), dtype=np.float)

        # Last time this LED segment was fully lit up, the
        # peak was exactly on this LED
        self._last_peak = np.zeros((count), dtype=np.float)

        # Initialize.
        for i in range(count):
            self._color[i] = _segment_color(i)
            self._decay_peak_seconds[i] = _decay_seconds(i, DECAY_MS['peak'])
            self._decay_active_seconds[i] = _decay_seconds(
                i, DECAY_MS['active']
            )

    def are_idle(self) -> bool:
        """Return True if all LEDs are off.

        Return
        ------
        bool
            True if all LEDs are off.
        """
        no_active = np.all(self._last_active == 0)
        no_peak = np.all(self._last_peak == 0)
        return no_active and no_peak

    def set_peak(self, now: float, delta_seconds: float) -> None:
        """Set the peak LED based on this frame time.

        Parameters
        ----------
        now : float
            Current time in seconds.
        delta_seconds : float
            The last frame time.
        """
        # Compute the peak LED segment that should light up. Note this is
        # logarithmic so the red LEDs are much slower frames than green.
        self.peak = _get_peak(delta_seconds)

        # Mark everything before the peak as active. These light up and
        # fade but not as bright. So we can see the peaks.
        for index in range(self.peak):
            self._last_active[index] = now

        # This specific LED gets fully lit up. That way we can see this
        # peak even when the leading LEDs have faded.
        self._last_peak[self.peak] = now

    def update(self, now: float) -> None:
        """Update the LED states.

        Parameters
        ----------
        now : float
            Current time in seconds.
        """
        _expire(now, self._last_peak, self._decay_peak_seconds)
        _expire(now, self._last_active, self._decay_active_seconds)

    def get_color(self, now: float, index: int) -> np.ndarray:
        """Get color for this LED segment, including decay alpha.

        Parameters
        ----------
        now : float
            The current time in seconds.
        index : int
            The LED segment index.

        Return
        ------
        int
            Alpha in range [0..255]
        """
        # Jam the alpha right into the self._color table, faster?
        self._color[index, 3] = self._get_alpha(now, index)
        return self._color[index]

    def _get_alpha(self, now: float, index: int) -> int:
        """Get alpha for this LED segment.

        Parameters
        ----------
        now : float
            The current time in seconds.
        index : int
            The LED segment index.

        Return
        ------
        int
            Alpha in range [0..255]
        """

        def _alpha(fraction):
            # This is lerp basically
            return ALPHA_MAX - (fraction * (ALPHA_MAX - ALPHA_OFF))

        if self._last_peak[index] == 0 and self._last_active[index] == 0:
            return ALPHA_OFF  # LED is totally off.

        peak_fraction = _get_fraction(
            now, index, self._last_peak, self._decay_peak_seconds
        )
        active_fraction = _get_fraction(
            now, index, self._last_active, self._decay_active_seconds
        )

        peak_alpha = _alpha(peak_fraction)
        active_alpha = _alpha(active_fraction)

        print(f"{peak_alpha} - {active_alpha}")
        return max(peak_alpha, active_alpha)


class QtFrameRate(QLabel):
    """A frame rate label with green/yellow/red LEDs.

    The LED values are logarithmic, so a lot of detail between 60Hz and
    10Hz but then the highest LED is many seconds long.
    """

    def __init__(self):
        super().__init__()
        # The per-LED config and state.
        self.leds = LedState()

        # The last time we were updated, either from mouse move or our
        # timer.
        self._last_time: Optional[float] = None

        # The bitmap image we draw into.
        self._image = np.zeros(BITMAP_SHAPE, dtype=np.uint8)

        # We animate on camera movements, but then we run this time to animate
        # the display. When all the LEDs go off, we stop the timer to use
        # zero CPU until another camera movement.
        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._on_timer)

        _print_mapping()  # Debugging.

    def _on_timer(self):
        """Animate the LEDs."""
        now = time.time()
        self._draw(now)  # Just animate and draw, no new peak.

        # Turn off the time when there's nothing more to animate. So we use
        # zero CPU until the camera moves again.
        if self.leds.are_idle():
            self._timer.stop()

    def _draw(self, now: float) -> None:
        """Animate the LEDs.

        We turn off the time when there's nothing more to animate.
        """
        self.leds.update(now)  # Animates the LEDs.
        self._update_image(now)  # Draws our internal self._image
        self._update_bitmap()  # Writes self._image into the QLabel bitmap.

        # We always update _last_time whether this was from a camera move
        # or the timer. This is the right thing to do since in either case
        # it's marking a frame draw.
        self._last_time = now

    def on_camera_move(self) -> None:
        """Update our display with the new framerate."""

        # Only count this frame if the timer is active. This avoids
        # displaying a potentially super long frame since it might have
        # been many seconds or minutes since the last movement. Ideally we
        # could capture the draw time of that first frame, because if it's
        # slow we should show it, but there's no easy/obvious way to do
        # that today. And this will show everthing but that out of the
        # blue first frame.
        use_delta = self._timer.isActive() and self._last_time is not None
        now = time.time()

        if use_delta:
            delta_seconds = now - self._last_time
            self.leds.set_peak(now, delta_seconds)

        self._draw(now)  # Draw the whole meter.

        # Since there was activity, we need to animate the decay of what we
        # displayed. The timer will be shut off them the LEDs go idle.
        self._timer.start()

    def _update_image(self, now: float) -> None:
        """Update our self._image with the latest image.

        Parameters
        ----------
        now : float
            The current time in seconds.

        Return
        ----------
        np.ndarray
            The bit image to display.
        """
        self._image.fill(0)  # Start fresh each time.

        # Draw each segment with the right color and alpha (decay).
        for index in range(NUM_SEGMENTS):
            color = self.leds.get_color(now, index)
            self._draw_segment(index, color)

    def _draw_segment(self, segment: int, color: np.ndarray):
        """Draw one LED segment, one rectangle.

        segment : int
            Index of the segment we are drawing.
        color : np.ndarray
            Color as RGBA.
        """
        x0 = int(segment * SEGMENT_SPACING)
        x1 = int(x0 + SEGMENT_WIDTH)
        y0 = 0
        y1 = BITMAP_SHAPE[0]
        self._image[y0:y1, x0:x1] = color

    def _update_bitmap(self) -> None:
        """Update the bitmap with latest image data.

        now : float
            Current time in seconds.
        segment : int
            Current highest segment to light up.
        """
        height, width = BITMAP_SHAPE[:2]
        image = QImage(self._image, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))
