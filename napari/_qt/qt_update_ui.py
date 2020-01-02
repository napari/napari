from qtpy.QtCore import QRunnable


class QtUpdateUI(QRunnable):
    """UI Update thread, extended QThread.

    Parameters
    ----------
    fn : function
        The function that would be executed.
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.fn(*self.args, **self.kwargs)
