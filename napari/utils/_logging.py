import logging

_LOG_SEPARATOR = '<NAPARI_LOG_SEPARATOR>'


class _LogStream:
    def __init__(self):
        self.logs = []

    def write(self, log_msg):
        logger, level_name, time, thread, msg = log_msg.split(_LOG_SEPARATOR)
        levels = logging.getLevelNamesMapping()
        level_value = levels[level_name]
        self.logs.append((logger, level_value, level_name, time, thread, msg))
        # TODO: actually save log to a file somewhere so it can be retrieved?

    def flush(self):
        pass


LOG_STREAM = _LogStream()
LOG_HANDLER = logging.StreamHandler(LOG_STREAM)
LOG_HANDLER.setFormatter(
    logging.Formatter(
        f'%(name)s{_LOG_SEPARATOR}%(levelname)s{_LOG_SEPARATOR}%(asctime)s{_LOG_SEPARATOR}%(threadName)s{_LOG_SEPARATOR}%(message)s'
    )
)
LOG_HANDLER.setLevel(logging.DEBUG)


def register_logger(module):
    logger = logging.getLogger(module)
    logger.addHandler(LOG_HANDLER)


def _color_from_level(level):
    if isinstance(level, str):
        level = logging.getLevelName(level)
    color = {
        logging.DEBUG: 'blue',
        logging.INFO: 'blue',
        logging.WARNING: 'orange',
        logging.ERROR: 'red',
        logging.CRITICAL: 'red',
    }
    return color[level]


def get_filtered_logs_html(level=logging.DEBUG, text_filter=''):
    if isinstance(level, str):
        level = logging.getLevelName(level)
    selected = [
        (logger_name, level_value, *others)
        for logger_name, level_value, *others in LOG_STREAM.logs
        if level_value >= level
    ]

    # TODO: fuzzy search?
    text_filter = text_filter.lower()
    filtered = [
        log
        for log in selected
        if any(text_filter in str(field).lower() for field in log)
    ]

    return ''.join(
        f'<font style="color:{_color_from_level(level_name)}">{level_name}</font> '
        f'<b>{name}</b> <font style="color:gray"><i>[{time}] ({thread})</i></font>: {msg}<br>'
        for name, _, level_name, time, thread, msg in filtered
    )
