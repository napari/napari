import os.path as osp

from .components._viewer.view import QtViewer
from .components import Window, Viewer
from .layers._image_layer.model import Image
from .resources import resources_dir
from .util.theme import template, palettes
from .util.misc import has_clims


class ViewerApp(Viewer):
    """Napari ndarray viewer.

    Parameters
    ----------
    *images : ndarray
        Arrays to render as image layers.
    meta : dictionary, optional
        A dictionary of metadata attributes. If multiple images are provided,
        the metadata applies to all of them.
    multichannel : bool, optional
        Whether to consider the last dimension of the image(s) as channels
        rather than spatial attributes. If not provided, napari will attempt
        to make an educated guess. If provided, and multiple images are given,
        the same value applies to all images.
    clim_range : list | array | None
        Length two list or array with the default color limit range for the
        image. If not passed will be calculated as the min and max of the
        image. Passing a value prevents this calculation which can be useful
        when working with very large datasets that are dynamically loaded.
        If provided, and multiple images are given, the same value applies to
        all images.
    **named_images : dict of str -> ndarray, optional
        Arrays to render as image layers, keyed by layer name.
    """
    def __init__(self, *images, meta=None, multichannel=None, clim_range=None,
                 title='napari', **named_images):
        super().__init__(title=title)
        self._qtviewer = QtViewer(self)
        self.window = Window(self)
        for image in images:
            self.add_image(image, meta=meta, multichannel=multichannel,
                           clim_range=clim_range)
        for name, image in named_images.items():
            self.add_image(image, meta=meta, multichannel=multichannel,
                           clim_range=clim_range, name=name)
        self.theme = 'dark'

    @property
    def theme(self):
        """string: Color theme.
        """
        if hasattr(self, '_theme'):
            return self._theme
        else:
            return None

    @theme.setter
    def theme(self, theme):
        if self.theme is not None and theme == self.theme:
            return
        self._theme = theme

        if theme not in palettes.keys():
            raise KeyError("Theme '%s' not found, options are %s." 
                           % (theme, list(palettes.keys())))

        palette = palettes[theme]

        # template and apply the primary stylesheet
        with open(osp.join(resources_dir, 'stylesheet.qss'), 'r') as f:
            raw_stylesheet = f.read()
            themed_stylesheet = template(raw_stylesheet, **palette)
        self._qtviewer.setStyleSheet(themed_stylesheet)

        # set window styles which don't use the primary stylesheet
        self.window._status_bar.setStyleSheet("""QStatusBar { background: %s;
            color: %s}""" % (palette['background'], palette['text']))
        self.window._qt_center.setStyleSheet(
            'QWidget { background: %s;}' % palette['background'])

        # set styles on clim slider
        for layer in self.layers:
            if has_clims(layer):
                layer._qt_controls.climSlider.setColors(
                    palette['foreground'], palette['highlight'])

        # set styles on dims sliders
        for slider in self._qtviewer.dims.sliders:
            slider.setColors(palette['foreground'], palette['highlight'])
