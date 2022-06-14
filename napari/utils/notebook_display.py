import base64
import html
from io import BytesIO
from warnings import warn

try:
    from lxml.html import document_fromstring
    from lxml.html.clean import Cleaner

    lxml_unavailable = False
except ModuleNotFoundError:
    lxml_unavailable = True

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

    def __init__(
        self,
        viewer,
        *,
        canvas_only=False,
        alt_text=None,
    ):
        """Initialize screenshot object.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer
        canvas_only : bool, optional
            If False include the napari viewer frame in the screenshot,
            and if True then take screenshot of just the image display canvas.
            By default, False.
        alt_text : str, optional
            Image description alternative text, for screenreader accessibility.
            Good alt-text describes the image and any text within the image
            in no more than three short, complete sentences.
        """
        self.viewer = viewer
        self.canvas_only = canvas_only
        self.image = None
        self.alt_text = self._clean_alt_text(alt_text)

    def _clean_alt_text(self, alt_text):
        """Clean user input to prevent script injection."""
        if alt_text is not None:
            if lxml_unavailable:
                warn(
                    'The lxml library is not installed, and is required to '
                    'sanitize alt text for napari screenshots. Alt-text '
                    'will be stripped altogether without lxml.'
                )
                return None
            # cleaner won't recognize escaped script tags, so always unescape
            # to be safe
            alt_text = html.unescape(str(alt_text))
            doc = document_fromstring(alt_text)
            alt_text = Cleaner().clean_html(doc).text_content()
        return alt_text or None

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
        _alt = html.escape(self.alt_text) if self.alt_text is not None else ''
        return f'<img src="{url}" alt="{_alt}"></img>'


nbscreenshot = NotebookScreenshot
