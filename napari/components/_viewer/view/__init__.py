# set vispy to use same backend as qtpy
import warnings
from qtpy import API_NAME

vispy_warning = "VisPy is not yet compatible with matplotlib 2.2+"

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=vispy_warning
    )

    from vispy import app

    app.use_app(API_NAME)
    del app

    from .main import QtViewer
