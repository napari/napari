"""QtPerformance widget to show performance information.
"""
import time

from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QTextCursor

from qtpy.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QTextEdit,
    QWidget,
)

from ..utils import perf


class TextLog(QTextEdit):
    """Text window we can write "log" messages to.

    TODO: need to limit length, erase oldest messages.
    """

    def append(self, name, time_ms):
        """Add one line of text for this timer.
        """
        self.moveCursor(QTextCursor.End)
        self.setTextColor(Qt.red)
        self.insertPlainText(f"{time_ms:5.0f}ms {name}\n")


class QtPerformance(QWidget):
    """Dockable widget to show performance info.

    Layout
    ------

    Draw Time:
    <----progress bar showing draw time---->

    Slow Events:
    <----TextLog listing slow events---->

    Uptime: <---uptime in seconds--->

    Notes
    -----

    1) The progress bar doesn't show "progress", we use it as a bar graph to
       show the average duration of recent "UpdateRequest" events. This
       is actually not the total draw time, but it's generally the biggest
       part of each frame.

    2) We log any event whose duration is longer then SLOW_EVENT_MS

    3) We show uptime so you can tell if this window is being updated at all.
    """

    # Log events that take longer than this.
    SLOW_EVENT_MS = 100

    # Update at 250ms / 4Hz for now. The more we update more alive our
    # display will look, but the more we will slow things down.
    #
    # Also for some reason our timer gets "starved out" if someone is using the
    # slider in the main window, the main window will update but we want. Maybe
    # it's being directly updating things? So the timer event is not being handled?
    UPDATE_MS = 250

    def __init__(self):
        """Create our windgets.
        """
        super().__init__()
        layout = QVBoxLayout()

        # For our "uptime" timer.
        self.start_time = time.time()

        # Label for our progress bar.
        bar_label = QLabel("Draw Time:")
        layout.addWidget(bar_label)

        # Progress bar is not used for "progress", it's just a bar graph to show
        # the "draw time", the duration of the "UpdateRequest" event.
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(50)
        bar.setFormat("%vms")
        layout.addWidget(bar)
        self.bar = bar

        # Label for our text window.
        log_label = QLabel("Slow Events:")
        layout.addWidget(log_label)

        # We log slow events to this window.
        log = TextLog()
        layout.addWidget(log)
        self.log = log

        # Uptime label. To indicate if the widget is getting updated.
        label = QLabel('')
        layout.addWidget(label)
        self.timer_label = label

        self.setLayout(layout)

        # Update us with a timer.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(self.UPDATE_MS)
        self.timer.start()

    def _get_timer_info(self):
        """Get the information from the timers that we want to display.
        """
        average = None
        long_events = []

        # We don't update any GUI/widgets while iterating over the timers.
        # Updating widgets can create immediate Qt Events which would modify the
        # timers out from under us!
        for name, timer in perf.timers.timers.items():

            # The Qt Event "UpdateRequest" is the main "draw" event, so
            # that's what we use for our progress bar.
            if name == "UpdateRequest":
                average = timer.average

            # Log any "long" events to the text window.
            if timer.max >= self.SLOW_EVENT_MS:
                long_events.append((name, timer.max))

        return average, long_events

    def update(self):
        """Update our label and progress bar and log any new slow events.
        """
        # Update our timer label.
        elapsed = time.time() - self.start_time
        self.timer_label.setText(f"Uptime: {elapsed:.2f}")

        average, long_events = self._get_timer_info()

        # Now safe to update the GUI: progress bar first.
        if average is not None:
            self.bar.setValue(average)

        # And log any new slow events.
        for name, time_ms in long_events:
            self.log.append(name, time_ms)

        # Clear all the timers since we've displayed them. They will immediately
        # start accumulating numbers for the next update.
        perf.timers.clear()
