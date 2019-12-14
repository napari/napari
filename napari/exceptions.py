class NapariError(Exception):
    """Base class for exceptions in napari."""

    def __init__(self, msg=None, detail=""):
        if msg is None:
            msg = "An unexpected error occured in LLSpy"
        super().__init__(msg)
        self.msg = msg
        self.detail = detail
