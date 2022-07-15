from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollBar, QStyle, QStyleOptionSlider


# https://stackoverflow.com/questions/29710327/how-to-override-qscrollbar-onclick-default-behaviour
class ModifiedScrollBar(QScrollBar):
    """Modified QScrollBar that moves fully to the clicked position.

    When the user clicks on the scroll bar background area (aka, the "page
    control"), the default behavior of the QScrollBar is to move one "page"
    towards the click (rather than all the way to the clicked position).
    See: https://doc.qt.io/qt-5/qscrollbar.html
    This scroll bar modifies the mousePressEvent to move the slider position
    fully to the clicked position.
    """

    def _move_to_mouse_position(self, event):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        control = self.style().hitTestComplexControl(
            QStyle.CC_ScrollBar, opt, event.pos(), self
        )
        if control not in [
            QStyle.SC_ScrollBarAddPage,
            QStyle.SC_ScrollBarSubPage,
        ]:
            return
        # scroll here
        gr = self.style().subControlRect(
            QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarGroove, self
        )
        sr = self.style().subControlRect(
            QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarSlider, self
        )
        if self.orientation() == Qt.Horizontal:
            pos = event.pos().x()
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
            if self.layoutDirection() == Qt.RightToLeft:
                opt.upsideDown = not opt.upsideDown
        else:
            pos = event.pos().y()
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1
        self.setValue(
            QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                pos - slider_min - slider_length // 2,
                slider_max - slider_min,
                opt.upsideDown,
            )
        )

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            # dragging with the mouse button down should move the slider
            self._move_to_mouse_position(event)
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # clicking the mouse button should move slider to the clicked point
            self._move_to_mouse_position(event)
        return super().mousePressEvent(event)
