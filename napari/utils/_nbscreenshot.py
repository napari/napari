def nbscreenshot(viewer, *, canvas_only=False):
    """Display napari screenshot in a jupyter notebook.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    canvas_only : bool, optional
        If True includes the napari viewer frame in the screenshot,
        otherwise just includes the canvas. By default, True.

    Returns
    -------
    NotebookScreenshot
        object with a _repr_png_ method that will show a (non-interactive)
        screenshot of the viewer.
    """
    from .._qt.utils.nbscreenshot import NotebookScreenshot

    return NotebookScreenshot(viewer=viewer, canvas_only=canvas_only)
