from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from napari.utils.logo import get_logo_path
from napari.utils.theme import get_system_theme

# This lives here in order to minimize impact of spurious imports in
# other files and so we can show the splash with a subprocess ASAP

if __name__ == '__main__':
    app = QApplication([])

    # init a set size pixmap, smaller than svg document
    size = QSize(400, 400)
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    # render logo (hardcoded for reduced performance cost)
    logo = get_logo_path(
        logo='auto', template='plain', theme=get_system_theme()
    )
    painter = QPainter(pixmap)
    renderer = QSvgRenderer(str(logo))
    renderer.render(painter)
    painter.end()

    # set up a splash screen. For some reason, a HEAVY delay is introduced by using
    # QSplashScreen. I could not track down the reason. It appears we can obtain
    # essentially the same behaviour with flags, but without the slowdown.

    splash = QWidget()
    splash.setWindowFlags(
        Qt.WindowType.SplashScreen
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
    )
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    label = QLabel(splash)
    label.setPixmap(pixmap)
    label.adjustSize()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # TODO: we can add TIPS here! see #8762
    splash.show()
    app.exec_()
    app.processEvents()
