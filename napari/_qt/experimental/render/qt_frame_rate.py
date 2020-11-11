"""QtFrameRate widget.
"""
import time

import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel

# Shape of the frame rate display.
BITMAP_SHAPE = (20, 200, 4)

COLOR = (227, 220, 111, 2555)  # yellow


class QtFrameRate(QLabel):
    """A small bitmap that shows the current frame rate."""

    def __init__(self):
        super().__init__()
        self._last_time = None

    def update(self) -> None:
        """Update the frame rate display."""
        now = time.time()
        if self._last_time is not None:
            self._mark(now - self._last_time)
        self._last_time = now

    def _mark(self, elapsed_seconds: float) -> None:
        """Mark the interval between two frames.
        """
        data = self._get_bitmap_data(elapsed_seconds)
        height, width = BITMAP_SHAPE[:2]
        image = QImage(data, width, height, QImage.Format_RGBA8888)
        self.setPixmap(QPixmap.fromImage(image))

    def _get_bitmap_data(self, elapsed_seconds: float) -> np.ndarray:
        """Return bitmap data for the display.

        Parameters
        ----------
        elapsed_seconds : float
            The current frame interval.
        """
        data = np.zeros(BITMAP_SHAPE, dtype=np.uint8)
        width = BITMAP_SHAPE[1]

        level = int(min(width - 1, (elapsed_seconds / 0.250) * width))
        print(f"elapsed = {elapsed_seconds} level={level}")

        # Write the view color into this rectangular regions.
        # TODO_OCTREE: must be a nicer way to index this?
        data[:, :level] = COLOR

        return data
