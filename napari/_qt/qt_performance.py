"""A dockable widget to show performance information.
"""
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QTextCursor

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QProgressBar,
)

from ..utils.perf_timers import TIMERS


class TextLog(QTextEdit):
    """Text window we can write "log" messages to.

    This is very WIP might not use this control at all.
    """

    def append(self, name, time_ms):
        """Add one line of text for this timer.
        """
        self.moveCursor(QTextCursor.End)
        self.setTextColor(Qt.red)
        self.insertPlainText(f"{time_ms:5.0f}ms {name}\n")


class QtPerformance(QWidget):
    """Dock widget to show performance metrics and info.

    This UI is totally placeholder for now. It's really just proof of
    concept that we can display information from our PerfTimers.

    What exactly we want to show here is TBD.
    """

    # Log events that take longer than this.
    LONG_EVENT_MS = 100

    def __init__(self):
        """Create our progress bar and text window.
        """
        super().__init__()
        layout = QVBoxLayout()

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(50)
        bar.setFormat("%vms")
        layout.addWidget(bar)
        self.bar = bar

        log = TextLog()
        layout.addWidget(log)
        self.log = log

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(1000)
        self.timer.start()

    def update(self):
        """Update our progress bar and log any new slow events.
        """
        update_average = None
        long_events = []

        for name, timer in TIMERS.timers.items():

            # Update request is the main "draw" event.
            if name == "UpdateRequest":
                update_average = timer.average

            # For now just log any "long" events.
            if timer.max >= self.LONG_EVENT_MS:
                long_events.append((name, timer.max))

        # Update GUI only after iteration is one, or the timers
        # will change out from under us.
        if update_average is not None:
            self.bar.setValue(timer.average)

        for name, time_ms in long_events:
            self.log.append(name, time_ms)

        # Clear all the timers since we've displayed them.
        TIMERS.clear()
