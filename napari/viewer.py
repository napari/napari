from . import __version__
from ._qt import Window
from .components import ViewerModel
from .utils import config


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string, optional
        The title of the viewer window. by default 'napari'.
    ndisplay : {2, 3}, optional
        Number of displayed dimensions. by default 2.
    order : tuple of int, optional
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3. by default None
    axis_labels : list of str, optional
        Dimension names. by default they are labeled with sequential numbers
    show : bool, optional
        Whether to show the viewer after instantiation. by default True.
    """

    # set _napari_app_id to False to avoid overwriting dock icon on windows
    # set _napari_app_id to custom string to prevent grouping different base viewer
    _napari_app_id = 'napari.napari.viewer.' + str(__version__)

    # set _napari_global_logo to control if napari logo should be set as application logo
    _napari_global_logo = True

    def __init__(
        self,
        *,
        title='napari',
        ndisplay=2,
        order=(),
        axis_labels=(),
        show=True,
    ):
        super().__init__(
            title=title,
            ndisplay=ndisplay,
            order=order,
            axis_labels=axis_labels,
        )
        self.window = Window(self, show=show)

    def update_console(self, variables):
        """Update console's namespace with desired variables.

        Parameters
        ----------
        variables : dict, str or list/tuple of str
            The variables to inject into the console's namespace.  If a dict, a
            simple update is done.  If a str, the string is assumed to have
            variable names separated by spaces.  A list/tuple of str can also
            be used to give the variable names.  If just the variable names are
            give (list/tuple/str) then the variable values looked up in the
            callers frame.
        """
        if self.window.qt_viewer.console is None:
            return
        else:
            self.window.qt_viewer.console.push(variables)

    def screenshot(self, path=None, *, canvas_only=True):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        canvas_only : bool
            If True, screenshot shows only the image display canvas, and
            if False include the napari viewer frame in the screenshot,
            By default, True.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        if canvas_only:
            image = self.window.qt_viewer.screenshot(path=path)
        else:
            image = self.window.screenshot(path=path)
        return image

    def show(self):
        """Resize, show, and raise the viewer window."""
        self.window.show()

    def close(self):
        """Close the viewer window."""
        # Remove all the layers from the viewer
        self.layers.clear()
        # Close the main window
        self.window.close()

        if config.async_loading:
            from .components.experimental.chunk import chunk_loader

            # TODO_ASYNC: Find a cleaner way to do this? This fixes some
            # tests. We are telling the ChunkLoader that this layer is
            # going away:
            # https://github.com/napari/napari/issues/1500
            for layer in self.layers:
                chunk_loader.on_layer_deleted(layer)
