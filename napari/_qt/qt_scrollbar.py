from qtpy.QtWidgets import QScrollBar, QStyleOptionSlider, QStyle
from qtpy.QtCore import Qt


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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            control = self.style().hitTestComplexControl(
                QStyle.CC_ScrollBar, opt, event.pos(), self
            )
            if (
                control == QStyle.SC_ScrollBarAddPage
                or control == QStyle.SC_ScrollBarSubPage
            ):
                # scroll here
                gr = self.style().subControlRect(
                    QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarGroove, self
                )
                sr = self.style().subControlRect(
                    QStyle.CC_ScrollBar, opt, QStyle.SC_ScrollBarSlider, self
                )
                if self.orientation() == Qt.Horizontal:
                    pos = event.pos().x()
                    sliderLength = sr.width()
                    sliderMin = gr.x()
                    sliderMax = gr.right() - sliderLength + 1
                    if self.layoutDirection() == Qt.RightToLeft:
                        opt.upsideDown = not opt.upsideDown
                else:
                    pos = event.pos().y()
                    sliderLength = sr.height()
                    sliderMin = gr.y()
                    sliderMax = gr.bottom() - sliderLength + 1
                self.setValue(
                    QStyle.sliderValueFromPosition(
                        self.minimum(),
                        self.maximum(),
                        pos - sliderMin,
                        sliderMax - sliderMin,
                        opt.upsideDown,
                    )
                )
                return

        return super(ModifiedScrollBar, self).mousePressEvent(event)
