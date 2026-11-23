import logging

from napari._qt.widgets.qt_logger import LogWidget
from napari.utils import logging as napari_logging


def test_qt_logger(qtbot):
    widget = LogWidget()
    qtbot.addWidget(widget)
    widget.show()
    logger = logging.getLogger('napari')

    with napari_logging.using_napari_log_handler('napari'):
        logger.warning('TEST WARNING')
        assert 'TEST WARNING' in widget.log_text_box.toPlainText()

    widget.hide()
