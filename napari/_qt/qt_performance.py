"""A dockable widget to show performance information.
"""
import threading
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QTextCursor

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QProgressBar,
)

from ..utils.perf_timers import TIMERS


class Labels:
    def __init__(self):
        self.counter = QLabel("counter")
        self.draw_time = QLabel("draw_time")
        self.fps = QLabel("fps")

    def add_to_layout(self, layout):
        layout.addWidget(self.counter)
        layout.addWidget(self.draw_time)
        layout.addWidget(self.fps)

    def set_text(self, counter_str, draw_time_str, fps_str):
        self.counter.setText(counter_str)
        self.draw_time.setText(draw_time_str)
        self.fps.setText(fps_str)


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

    This UI is totally placeholder for now, just experimenting with what
    we want to show to people.
    """

    def __init__(self):
        super().__init__()
        self.counter = 1
        self.labels = Labels()

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
        print("QtPerformance thread: ", threading.get_ident())
        update_average = None
        long_events = []

        for name, timer in TIMERS.timers.items():

            # Update request is the main "draw" event.
            if name == "UpdateRequest":
                update_average = timer.average

            # For now just log any "long" events.
            if timer.max >= 100:
                long_events.append((name, timer.max))

        # Update GUI only after iteration is one, or the timers
        # will change out from under us.
        if update_average is not None:
            self.bar.setValue(timer.average)

        for name, time_ms in long_events:
            self.log.append(name, time_ms)

        # Clear all the timers since we've displayed them.
        TIMERS.clear()
