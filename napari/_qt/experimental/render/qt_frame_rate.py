"""QtFrameRate widget.
"""
import math
import time

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

# Shape of the frame rate display.
BITMAP_SHAPE = (40, 200, 4)

# Left to right are bars.
NUM_BARS = 9

# Each bar has this many vertical segments.
NUM_SEGMENTS = 5

MIN_MS = 16.17
LOG_MIN_MS = math.log10(MIN_MS)

MAX_MS = 100
LOG_MAX_MS = math.log10(MAX_MS)  # 4

GREEN = (57, 252, 3, 255)
YELLOW = (252, 232, 3, 255)
RED = (252, 78, 3, 255)

SEGMENT_WIDTH = BITMAP_SHAPE[1] / NUM_BARS

SEGMENT_SPACING = BITMAP_SHAPE[0] / NUM_SEGMENTS
SEGMENT_HEIGHT = SEGMENT_SPACING - 2

LIVE_SEGMENTS = 20

BAR_COLOR = [GREEN, GREEN, GREEN, YELLOW, YELLOW, YELLOW, RED, RED, RED]


class QtFrameRate(QLabel):
    """A small bitmap that shows the current frame rate."""

    def __init__(self):
        super().__init__()
        self._last_time = None
        self.values = np.zeros((NUM_BARS,), dtype=np.int32)
        self.past_bars = []
        self.data = np.zeros(BITMAP_SHAPE, dtype=np.uint8)

    def update(self) -> None:
        """Update the frame rate display."""
        now = time.time()
        if self._last_time is not None:
            delta_ms = (now - self._last_time) * 1000
            self._update_bars(delta_ms)
            self._update_bitmap()
        self._last_time = now

    def _update_bars(self, delta_ms: float) -> None:
        """Update bars with this new interval.

        Parameters
        ----------
        delta_ms : float
            The current frame interval.
        """
        self._remove_past_values()

        def _clamp(value, low, high):
            return max(min(value, high), low)

        # Create log value where MIN_MS has the value zero.
        log_value = math.log10(delta_ms) - LOG_MIN_MS

        # Compute fraction [0..1] for the whole width (all bars)
        # and then find the bar we need to increment.
        fraction = _clamp(log_value / LOG_MAX_MS, 0, 1)
        current_bar = int(fraction * (NUM_BARS - 1))

        # Increment the bar and save off the value.
        self.values[current_bar] += 1
        self.past_bars.append(current_bar)

        # Make sure values are not too high.
        self.values = np.clip(self.values, 0, NUM_SEGMENTS)

    def _remove_past_values(self) -> None:
        if len(self.past_bars) < LIVE_SEGMENTS:
            return

        # Remove the value at the front of the list.
        past_bar = self.past_bars.pop(0)
        self.values[past_bar] -= 1

        # Make sure none go under 0.
        self.values = np.clip(self.values, 0, NUM_SEGMENTS)

    def _update_image_data(self) -> np.ndarray:
        """Return bitmap data for the display.

        Return
        ----------
        np.ndarray
            The bit image to display.
        """
        self.data.fill(0)
        for bar_index in range(NUM_BARS):
            bar_value = self.values[bar_index]
            print(f"bar: {bar_index} -> {bar_value}")
            for segment in range(bar_value + 1):
                x0 = int(bar_index * SEGMENT_WIDTH)
                x1 = int(x0 + SEGMENT_WIDTH)
                y0 = BITMAP_SHAPE[0] - int(segment * SEGMENT_SPACING)
                y1 = int(y0 + SEGMENT_HEIGHT)
                self.data[y0:y1, x0:x1] = BAR_COLOR[bar_index]

    def _update_bitmap(self) -> None:
        """Update the bitmap with latest image data.
        """
        self._update_image_data()
        height, width = BITMAP_SHAPE[:2]
        image = QImage(self.data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))
