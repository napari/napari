import base64
from io import BytesIO

__all__ = ['nbscreenshot']


class NotebookScreenshot:
    """Display napari screenshot in the jupyter notebook.

    Functions returning an object with a _repr_png_() method
    will displayed as a rich image in the jupyter notebook.

    https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    canvas_only : bool, optional
        If True includes the napari viewer frame in the screenshot,
        otherwise just includes the canvas. By default, True.

    Examples
    --------
    ```
    import napari
    from napari.utils import nbscreenshot
    from skimage.data import chelsea

    viewer = napari.view_image(chelsea(), name='chelsea-the-cat')
    nbscreenshot(viewer)

    # screenshot just the canvas with the napari viewer framing it
    nbscreenshot(viewer, canvas_only=False)
    ```
    """

    def __init__(self, viewer, *, canvas_only=False):
        """Initialize screenshot object.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer
        canvas_only : bool, optional
            If False include the napari viewer frame in the screenshot,
            and if True then take screenshot of just the image display canvas.
            By default, False.
        """
        self.viewer = viewer
        self.canvas_only = canvas_only
        self.image = None

    def _repr_png_(self):
        """PNG representation of the viewer object for IPython.

        Returns
        -------
        In memory binary stream containing PNG screenshot image.
        """
        from imageio import imsave

        from .._qt.qt_event_loop import get_app

        get_app().processEvents()
        self.image = self.viewer.screenshot(
            canvas_only=self.canvas_only, flash=False
        )
        with BytesIO() as file_obj:
            imsave(file_obj, self.image, format='png')
            file_obj.seek(0)
            png = file_obj.read()
        return png

    def _repr_html_(self):
        png = self._repr_png_()
        url = 'data:image/png;base64,' + base64.b64encode(png).decode('utf-8')
        html = f'<img src="{url}"></img>'
        return html


nbscreenshot = NotebookScreenshot
