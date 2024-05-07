import logging

# keep track of all the logging info in a stream
_LOG_SEPARATOR = '<NAPARI_LOG_SEPARATOR>'


class _LogHandler:
    def __init__(self):
        self.logs = []

    def write(self, log_msg):
        level_name, time, thread, msg = log_msg.split(_LOG_SEPARATOR)
        levels = logging.getLevelNamesMapping()
        level_value = levels[level_name]
        self.logs.append((level_value, level_name, time, thread, msg))

    def flush(self):
        pass

    def __str__(self):
        return ''.join(
            [
                f'[{time}] ({thread}) {level_name}: {msg}'
                for _, level_name, time, thread, msg in self.logs
            ]
        )

    def logs_at_level(self, level=logging.DEBUG):
        if isinstance(level, str):
            level = logging.getLevelName(level)
        return [
            (level_value, *others)
            for level_value, *others in self.logs
            if level_value >= level
        ]


def _get_custom_log_stream():
    log_stream = _LogHandler()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(
        logging.Formatter(
            f'%(levelname)s{_LOG_SEPARATOR}%(asctime)s{_LOG_SEPARATOR}%(threadName)s{_LOG_SEPARATOR}%(message)s'
        )
    )
    logger = logging.getLogger('napari')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return log_stream
