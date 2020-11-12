"""QtFrameRate widget.
"""
import math
import time

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

# Shape of the frame rate display.
BITMAP_SHAPE = (20, 200, 4)

# Left to right are bars.
NUM_BARS = 9

# Each bar has this many verticle segments.
NUM_SEGMENTS = 10

MIN_MS = 16.17
LOG_MIN_MS = math.log10(MIN_MS)

MAX_MS = 100
LOG_MAX_MS = math.log10(MAX_MS)  # 4

GREEN = (57, 252, 3, 255)
YELLOW = (252, 232, 3, 255)
RED = (252, 78, 3, 255)

SEGMENT_WIDTH = BITMAP_SHAPE[1] / NUM_BARS
SEGMENT_HEIGHT = BITMAP_SHAPE[0] / NUM_SEGMENTS

LIVE_SEGMENTS = 10

BAR_COLOR = [GREEN, GREEN, GREEN, YELLOW, YELLOW, YELLOW, RED, RED, RED]


class QtFrameRate(QLabel):
    """A small bitmap that shows the current frame rate."""

    def __init__(self):
        super().__init__()
        self._last_time = None
        self.values = np.zeros((NUM_BARS,), dtype=np.int32)
        self.frame_counter = 0
        self.past_bars = []

    def update(self) -> None:
        """Update the frame rate display."""
        now = time.time()
        if self._last_time is not None:
            delta_seconds = now - self._last_time
            self._mark(delta_seconds * 1000)
        self._last_time = now

    def _mark(self, elapsed_ms: float) -> None:
        """Mark the interval between two frames.
        """
        self.frame_counter += 1
        data = self._get_bitmap_data(elapsed_ms)
        height, width = BITMAP_SHAPE[:2]
        image = QImage(data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))

    def _get_bitmap_data(self, elapsed_ms: float) -> np.ndarray:
        """Return bitmap data for the display.

        Parameters
        ----------
        elapsed_seconds : float
            The current frame interval.
        """
        if len(self.past_bars) > LIVE_SEGMENTS:
            past_bar = self.past_bars.pop(0)
            self.values[past_bar] -= 1
        self.values = np.clip(self.values, 0, NUM_SEGMENTS)

        def _clamp(value, low, high):
            return max(min(value, high), low)

        # So if MIN_MS is 16.7 then log_value for 16.7ms is 0
        log_value = math.log10(elapsed_ms) - LOG_MIN_MS

        # The log_fraction is [0..1] for covering all the bars.
        log_fraction = _clamp(log_value / LOG_MAX_MS, 0, 1)

        # Increment the bar for this value.
        current_bar = int(log_fraction * (NUM_BARS - 1))
        self.values[current_bar] += 1
        self.past_bars.append(current_bar)

        # Values should only be too high, but just clip.
        self.values = np.clip(self.values, 0, NUM_SEGMENTS)

        data = np.zeros(BITMAP_SHAPE, dtype=np.uint8)

        for bar_index in range(NUM_BARS):
            bar_value = self.values[bar_index]
            print(f"bar: {bar_index} -> {bar_value}")
            for segment in range(bar_value):
                x0 = int(bar_index * SEGMENT_WIDTH)
                x1 = int(x0 + SEGMENT_WIDTH)
                y0 = BITMAP_SHAPE[0] - int(segment * SEGMENT_HEIGHT)
                y1 = int(y0 + SEGMENT_HEIGHT)
                data[y0:y1, x0:x1] = BAR_COLOR[bar_index]

        return data
