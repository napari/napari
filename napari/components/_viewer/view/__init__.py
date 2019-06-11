import warnings


vispy_warning = "VisPy is not yet compatible with matplotlib 2.2+"
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=vispy_warning
    )
    from .main import QtViewer
