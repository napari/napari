"""The entrypoint for napari program.

This Is workaround for a problem of import path pollution
(ex. napari.py file in CDW)

The purpose of this file is to import napari.__main__ and call it.
But if it fail, try to determine if

"""

import importlib.util


def check_napari_import():
    spec = importlib.util.find_spec('napari')
    if spec is None:
        raise ImportError('napari is not installed')
    # real code here


def main():
    try:
        from napari.__main__ import main

        main()
    except ImportError:
        check_napari_import()
