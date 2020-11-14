"""QtFrameRate widget.

Displays the framerate as colored LEDs.
"""
import math
import time
from typing import Optional

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

# Shape of the bitmap.
BITMAP_SHAPE = (20, 270, 4)

# Number of left to right LED segments.
NUM_SEGMENTS = 30

# Spacing between segments, including the gap.
SEGMENT_SPACING = BITMAP_SHAPE[1] / NUM_SEGMENTS

# Gap between segments in pixels.
SEGMENT_GAP = 2
SEGMENT_WIDTH = SEGMENT_SPACING - SEGMENT_GAP

# DECAY_MS (X, Y) means if segment is at least X then keep that LED
# on for Y milliseconds. Peak means we hit that exact value,
# while active means it was a leading LED on the way to that peak.
#
# The general goals for the LEDs are:
#
# 1) The current peak is bright and easily visible.
# 2) The LEDs leading up to the current peak are bright so that makes
#    the current peak easier to see.
# 3) The peaks decay more slowly than the leading LEDs. So the slow frames
#    are visible even when the framerate has improved. So we can
#    see the bad peaks for a while.
#
# Make yellow and green stay lit longer, so we don't miss them.
DECAY_MS = {"peak": [(0, 1000), (10, 2000), (20, 3000)], "active": [(0, 250)]}

# LEDs are colors GREEN -> YELLOW -> RED using slightly non-pure colors.
# Colors were chosen quickly and could be tweaked.
GREEN = (57, 252, 3)
YELLOW = (252, 232, 3)
RED = (252, 78, 3)

# Ten segments of each color.
LED_COLORS = [(0, GREEN), (10, YELLOW), (20, RED)]

# LEDs decay from MAX to OFF. We leave them visible when off because it
# looks better, and so you can see how high the meter goes.
ALPHA_MAX = 255
ALPHA_OFF = 25

# We should have better knobs for calibrating which framerates lead to
# which colors. But playing around this values we now get:
#
# GREEN: up to 31ms (32Hz)
# YELLOW: up to 312ms (3.1Hz)
# RED: up to 6000ms (0.16Hz)
#
# That seems pretty useful. Green is good, yellow is bad, red is awful.
#
PERFECT_MS = 16.7
LOG_BASE = 1.35


def _clamp(value, low, high):
    return max(min(value, high), low)


def _lerp(low: int, high: int, fraction: float) -> int:
    return int(low + (high - low) * fraction)


def _decay_seconds(index: int, decay_config) -> float:
    """Return duration in seconds the segment should stay on.

    Only called during init to create a lookup table, for speed.

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


def _led_color(index: int) -> tuple:
    """Return color of the given LED segment index.

    Only called during init to create a lookup table, for speed.

    Parameters
    ----------
    index : int
        The led segment index.
    """
    for limit, color in reversed(LED_COLORS):
        if index >= limit:
            return color
    assert False, "DECAY_MS table is messed up?"
    return (0, 0, 0)


def _get_peak(delta_seconds: float) -> int:
    """Get highest segment that should be lit up.

    Parameters
    ----------
    delta_seconds : float
        The current frame interval.

    Return
    ------
    int
        The peak segment that should be lit up.
    """

    if delta_seconds <= 0:
        return 0

    # Slide everything from PERFECT_MS on down to zero, or else we'd
    # waste a lot of LEDs show stuff faster than 60Hz.
    input_value = max((delta_seconds * 1000) - PERFECT_MS, 0) + 1

    log_value = math.log(input_value, LOG_BASE)
    return _clamp(int(log_value), 0, NUM_SEGMENTS - 1)


class DecayTimes:
    """Keep track of when LEDs were last lit up.

    Parameters
    ----------
    config : str
        Config is 'active' or 'peak'.
    """

    def __init__(self, config: str):
        count = NUM_SEGMENTS
        decay_ms = DECAY_MS[config]
        self._decay = np.array(
            [_decay_seconds(i, decay_ms) for i in range(count)], dtype=np.float
        )
        self._last = np.zeros((count), dtype=np.float)

    def is_off(self, index: int) -> bool:
        """Return True if this LED is off.

        Return
        ------
        bool
            True if this LED is off.
        """
        return self._last[index] == 0

    def all_off(self) -> bool:
        """Return True if all LEDs are off.

        Return
        ------
        bool
            True if all LEDs are off.
        """
        return np.all(self._last == 0)

    def mark(self, slicer: slice, now: float) -> None:
        """Mark that one or more LEDs have lit up.

        Parameters
        ----------
        slicer : slice
            The LEDs to light up.
        now : float
            The current time in seconds.
        """
        self._last[slicer] = now

    def expire(self, now: float) -> None:
        """Zero out the last time of any expired LEDs.

        Parameters
        ----------
        now : float
            The current time in seconds.
        """
        # Set self._last equal to zero for any LED which as gone dark.
        elapsed = now - self._last
        self._last[elapsed > self._decay] = 0

    def get_alpha(self, now: float) -> np.ndarray:
        """Return alpha for every LED segment.

        Alpha values will range from ALPHA_MAX if the LED was recently lit
        up to ALPHA_OFF if the LED is past the its decay period.

        Parameters
        ----------
        now : float
            The current time in seconds.

        Return
        ------
        np.ndarray
            The alpha values for each LED.
        """
        fractions = (now - self._last) / self._decay
        return np.interp(fractions, [0, 1], [ALPHA_MAX, ALPHA_OFF])


def _print_calibration():
    """Print mapping from LED segment to millseconds.

    Since we don't yet have the reverse math figured out, we print out
    the mapping by sampling. Crude but works.
    """
    print(f"LOG_BASE = {LOG_BASE}")
    previous = None
    for ms in range(0, 10000):
        segment = _get_peak(ms / 1000)
        if previous != segment:
            print(f"LED: {segment} -> {ms}ms")
            previous = segment


class LedState:
    """State of the LEDs in the frame rate display."""

    def __init__(self):

        # Active means the LED lit up because it was leading LED before the
        # peak value. While peaks means the framerate was that exact value.
        self._active = DecayTimes('active')
        self._peak = DecayTimes('peak')

        # Color at each LED index.
        self._color = np.array(
            [_led_color(i) for i in range(NUM_SEGMENTS)], dtype=np.uint8
        )

        self.peak = 0  # Current highest segment lit up

    def all_off(self) -> bool:
        """Return True if all LEDs are off.

        Return
        ------
        bool
            True if all LEDs are off.
        """
        return self._active.all_off() and self._peak.all_off()

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
        up_to_peak = slice(0, self.peak + 1)
        self._active.mark(up_to_peak, now)

        # This specific LED gets fully lit up. That way we can see this
        # peak even when the leading LEDs have faded.
        only_peak = slice(self.peak, self.peak + 1)
        self._peak.mark(only_peak, now)

    def update(self, now: float) -> None:
        """Update the LED states.

        Parameters
        ----------
        now : float
            Current time in seconds.
        """
        self._active.expire(now)
        self._peak.expire(now)

    def get_colors(self, now: float) -> np.ndarray:
        """Get current color (with alpha for our LED segments.

        Parameters
        ----------
        now : float
            The current time in seconds.

        Return
        ------
        np.ndarray
            Color (r, g, b, a)values for each LED.
        """
        # Use whichever alpha is higher (brighter).
        alpha = np.maximum(
            self._active.get_alpha(now), self._peak.get_alpha(now)
        )

        # Combine the fixed colors with the computed alpha values.
        return np.hstack((self._color, alpha.reshape(-1, 1)))


class QtFrameRate(QLabel):
    """A frame rate label with green/yellow/red LEDs.

    The LED values are logarithmic, so a lot of detail between 60Hz and
    10Hz but then the highest LED is many seconds long.
    """

    def __init__(self):
        super().__init__()

        self.leds = LedState()  # The per-LED config and state.

        # The last time we were updated, either from mouse move or our
        # timer. We update _last_time in both cases.
        self._last_time: Optional[float] = None

        # The bitmap image we draw into.
        self._image = np.zeros(BITMAP_SHAPE, dtype=np.uint8)

        # We animate on camera movements, but then we use a timer to
        # animate the display. When all the LEDs go off, we stop the timer
        # so that we use zero CPU until another camera movement.
        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_timer)

        # _print_calibration()  # Debugging.

    def _on_timer(self) -> None:
        """Animate the LEDs."""
        now = time.time()
        self._draw(now)  # Just animate and draw, no new peak.

        # Stop timer if nothing more to animation, save CPU.
        if self.leds.all_off():
            self._timer.stop()

    def _draw(self, now: float) -> None:
        """Animate the LEDs.

        Parameters
        ----------
        now : float
            The current time in seconds.
        """
        self.leds.update(now)  # Animates the LEDs.
        self._update_image(now)  # Draws our internal self._image
        self._update_bitmap()  # Writes self._image into the QLabel bitmap.

        # We always update _last_time whether this was from a camera move
        # or the timer. This is the right thing to do since in either case
        # it means a frame was drawn.
        self._last_time = now

    def on_camera_move(self) -> None:
        """Update our display to show the new framerate."""

        # Only count this frame if the timer is active. This avoids
        # displaying a potentially super long frame since it might have
        # been many seconds or minutes since the last camera movement.
        #
        # Ideally we should display the draw time of even that first frame,
        # but there's no easy/obvious way to do that today. And this will
        # show everthing except one frame.
        first_time = self._last_time is None
        use_delta = self._timer.isActive() and not first_time
        now = time.time()

        if use_delta:
            delta_seconds = now - self._last_time
            self.leds.set_peak(now, delta_seconds)

        self._draw(now)  # Draw the whole meter.

        # Since there was activity, we need to start the timer so we can
        # animate the decay of the LEDs. The timer will be shut off when
        # all the LEDs go idle.
        self._timer.start()

    def _update_image(self, now: float) -> None:
        """Update our self._image with the latest meter display.

        Parameters
        ----------
        now : float
            The current time in seconds.
        """
        self._image.fill(0)  # Start fresh each time.

        # Get colors with latest alpha values accord to decay.
        colors = self.leds.get_colors(now)

        # Draw each segment with the right color and alpha (due to decay).
        for index in range(NUM_SEGMENTS):
            x0 = int(index * SEGMENT_SPACING)
            x1 = int(x0 + SEGMENT_WIDTH)
            y0, y1 = 0, BITMAP_SHAPE[0]  # The whole height of the bitmap.
            self._image[y0:y1, x0:x1] = colors[index]

    def _update_bitmap(self) -> None:
        """Update the bitmap with latest image data."""
        height, width = BITMAP_SHAPE[:2]
        image = QImage(self._image, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))
