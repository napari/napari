from io import BytesIO


class NotebookScreenshot:
    """Display napari screenshot in the jupyter notebook.

    Functions returning an object with a _repr_png_() method
    will displayed as a rich image in the jupyter notebook.

    https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html

    Examples
    --------
    ```
    import napari
    from skimage.data import chelsea

    viewer = napari.view_image(chelsea(), name='chelsea-the-cat')
    viewer.nbscreenshot()

    # screenshot just the canvas without the napari viewer framing it
    viewer.nbscreenshot(with_viewer=False)
    ```
    """

    def __init__(self, viewer, with_viewer=True):
        """Initalize screenshot object.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer
        with_viewer : bool, optional
            If True includes the napari viewer frame in the screenshot,
            otherwise just includes the canvas. By default, True.
        """
        self.viewer = viewer
        self.with_viewer = with_viewer
        self.image = None

    def _repr_png_(self):
        """PNG representation of the viewer object for IPython.

        Returns
        -------
        In memory binary stream containing PNG screenshot image.
        """
        from imageio import imsave

        self.image = self.viewer.screenshot(with_viewer=self.with_viewer)
        file_obj = BytesIO()
        imsave(file_obj, self.image, format='png')
        file_obj.seek(0)
        return file_obj.read()
