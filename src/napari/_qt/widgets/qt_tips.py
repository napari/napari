from random import sample

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari.utils.tips import NAPARI_TIPS, format_tip


class TipsWidget(QWidget):
    """
    Widget for showing tips and tricks.
    """

    def __init__(
        self,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.setLayout(QVBoxLayout())

        self.tips = sample(NAPARI_TIPS, len(NAPARI_TIPS))
        self.current_tip = -1

        self.tip = QLabel()
        self.layout().addWidget(self.tip)

        self.buttons = QHBoxLayout()
        self.layout().addLayout(self.buttons)

        self.prev = QPushButton('Previous tip')
        self.next = QPushButton('Next tip')
        self.buttons.addWidget(self.prev)
        self.buttons.addWidget(self.next)
        self.prev.pressed.connect(self.prev_tip)
        self.next.pressed.connect(self.next_tip)

        self.next_tip()

    def prev_tip(self, event=None) -> None:
        self.current_tip -= 1
        self.current_tip %= len(self.tips)
        tip = self.tips[self.current_tip]
        self.tip.setText(format_tip(tip))

    def next_tip(self, event=None) -> None:
        self.current_tip += 1
        self.current_tip %= len(self.tips)
        tip = self.tips[self.current_tip]
        self.tip.setText(format_tip(tip))
