import logging

from napari.utils.logging import _LOG_STREAM, using_napari_log_handler


def test_log_stream():
    with using_napari_log_handler(''):
        logger = logging.getLogger('test_logger')
        logger.setLevel('DEBUG')
        log_msg = 'NAPARI TEST LOG MESSAGE'
        logger.debug(log_msg)

    assert log_msg in ''.join(_LOG_STREAM.get_filtered_logs_html())
