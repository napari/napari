"""QtFrameRate widget.
"""
import math
import time

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

# Shape of the frame rate bitmap.
BITMAP_SHAPE = (20, 200, 4)

# Left to right are segments.
NUM_SEGMENTS = 30

# 16.7ms or faster is our minimum reading, one segment only.
MIN_MS = 16.17
LOG_MIN_MS = math.log10(MIN_MS)

# 100ms will be all segments lit up.
MAX_MS = 100
LOG_MAX_MS = math.log10(MAX_MS)  # 4

GREEN = (57, 252, 3, 255)
YELLOW = (252, 232, 3, 255)
RED = (252, 78, 3, 255)

SEGMENT_SPACING = BITMAP_SHAPE[1] / NUM_SEGMENTS
SEGMENT_GAP = 2
SEGMENT_WIDTH = SEGMENT_SPACING - SEGMENT_GAP

LIVE_SEGMENTS = 20

PEAK_MS = [250, 2000, 5000]


def _peak_ms(segment: int) -> float:
    if segment < 10:
        return PEAK_MS[0]
    if segment < 20:
        return PEAK_MS[1]
    return PEAK_MS[2]


def _bar_color(segment: int) -> tuple:
    if segment < 10:
        return GREEN
    if segment < 20:
        return YELLOW
    return RED


def _clamp(value, low, high):
    return max(min(value, high), low)


class QtFrameRate(QLabel):
    """A small bitmap that shows the current frame rate."""

    def __init__(self):
        super().__init__()
        self._last_time = None
        self.peaks = np.zeros((NUM_SEGMENTS,), dtype=np.float)
        self.data = np.zeros(BITMAP_SHAPE, dtype=np.uint8)
        self.current_segment = 0

        self._bar_color = [_bar_color(i) for i in range(NUM_SEGMENTS)]
        self._peak_seconds = [_peak_ms(i) / 1000 for i in range(NUM_SEGMENTS)]

        self._timer = QTimer()
        self._timer.setSingleShot(False)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._on_timer)

    def _on_timer(self):
        if self.current_segment > 0:
            self.current_segment -= 1

        now = time.time()
        self._last_time = now

        for i in range(NUM_SEGMENTS):
            if now - self.peaks[i] > self._peak_seconds[i]:
                self.peaks[i] = 0  # done

        self._update_bitmap(now, self.current_segment)

        if np.all(self.peaks == 0) and self.current_segment == 0:
            self._timer.stop()

    def on_camera_move(self) -> None:
        now = time.time()
        if self._last_time is not None:
            if self._timer.isActive():
                delta_ms = (now - self._last_time) * 1000
                segment = self._get_segment(delta_ms)
                self.peaks[segment] = now
                self.current_segment = segment
                self._update_bitmap(now, self.current_segment)
        self._last_time = now
        self._timer.start()

    def _get_segment(self, delta_ms: float) -> None:
        """Update bars with this new interval.

        Parameters
        ----------
        delta_ms : float
            The current frame interval.
        """
        if delta_ms <= 0:
            log_value = 0
        else:
            # Create log value where MIN_MS has the value zero.
            log_value = math.log10(delta_ms) - LOG_MIN_MS

        # Compute fraction [0..1] for the whole width (all segments)
        # and then find the bar we need to increment.
        fraction = _clamp(log_value / LOG_MAX_MS, 0, 1)
        return int(fraction * (NUM_SEGMENTS - 1))

    def _remove_past_values(self) -> None:
        if len(self.past_bars) < LIVE_SEGMENTS:
            return

        # Remove the value at the front of the list.
        past_bar = self.past_bars.pop(0)
        self.values[past_bar] -= 1

        # Make sure none go under 0.
        self.values = np.clip(self.values, 0, NUM_SEGMENTS)

    def _update_image_data(self, now: float, segment: int) -> np.ndarray:
        """Return bitmap data for the display.

        Return
        ----------
        np.ndarray
            The bit image to display.
        """
        self.data.fill(0)
        for index in range(segment + 1):
            self._draw_segment(index)

        for i in range(NUM_SEGMENTS):
            if self.peaks[i] != 0:
                duration = now - self.peaks[i]
                fraction = _clamp(duration / self._peak_seconds[i], 0, 1)
                alpha = 255 - (fraction * 255)
                self._draw_segment(i, alpha)

    def _draw_segment(self, segment: int, alpha=255):
        x0 = int(segment * SEGMENT_SPACING)
        x1 = int(x0 + SEGMENT_WIDTH)
        y0 = 0
        y1 = BITMAP_SHAPE[0]
        color = self._bar_color[segment][:3] + (alpha,)
        self.data[y0:y1, x0:x1] = color

    def _update_bitmap(self, now: float, segment: int) -> None:
        """Update the bitmap with latest image data.
        """
        self._update_image_data(now, segment)
        height, width = BITMAP_SHAPE[:2]
        image = QImage(self.data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))
