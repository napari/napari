from io import BytesIO

import imageio


class NotebookScreenshot:
    """Display napari screenshot in the jupyter notebook.

    Examples
    --------
    ```
    import napari
    from skimage.data import chelsea

    viewer = napari.view_image(chelsea(), name='chelsea-the-cat')
    nbscreenshot(viewer)
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

    def _repr_png_(self):
        """PNG representation of the viewer object for IPython.

        Returns
        -------
        In memory binary stream containing PNG screenshot image.
        """
        image = self.viewer.screenshot(with_viewer=self.with_viewer)
        file_obj = BytesIO()
        imageio.imsave(file_obj, image, format='png')
        file_obj.seek(0)
        return file_obj.read()
