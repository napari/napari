import logging

from napari.utils._logging import LOG_STREAM, register_logger_to_napari_handler


def test_log_stream():
    with register_logger_to_napari_handler(''):
        logger = logging.getLogger('test_logger')
        logger.setLevel('DEBUG')
        log_msg = 'NAPARI TEST LOG MESSAGE'
        logger.debug(log_msg)

    assert log_msg in ''.join(LOG_STREAM.get_filtered_logs_html())
