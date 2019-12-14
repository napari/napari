from qtpy.QtCore import QObject, Signal
from ..exceptions import NapariError
import traceback
import re


def camel_to_spaces(val):
    return re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-R,T-Z](?=[a-z]))", r" \1", val)


class ExceptionHandler(QObject):
    """General class to handle all raise exception errors in the GUI"""

    # error message, title, more info, detail (e.g. traceback)
    error_message = Signal(str, str, str, str)

    def __init__(self):
        super(ExceptionHandler, self).__init__()

    def handler(self, etype, value, tb):
        # etype.__module__ contains the module raising the error
        # Custom exception classes can have different behavior
        if isinstance(value, NapariError):
            self.emit_error(etype, value, tb)
        # can add custom exception handlers here ...
        else:
            # otherwise, print exception to console as usual
            traceback.print_exception(etype, value, tb)

    def emit_error(self, etype, value, tb):
        tbstring = "".join(traceback.format_exception(etype, value, tb))
        title = camel_to_spaces(etype.__name__)
        self.error_message.emit(value.msg, title, value.detail, tbstring)
