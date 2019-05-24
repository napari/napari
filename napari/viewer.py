import os.path as osp

from .components._viewer.view import QtViewer
from .components import Window, ViewerModel
from .resources import resources_dir
from .util.theme import template, palettes
from .util.misc import has_clims


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string
        The title of the viewer window.

    Attributes
    ----------
    themes : dict of str: dict of str: str
        Preset color palettes.
    """
    themes = palettes

    with open(osp.join(resources_dir, 'stylesheet.qss'), 'r') as f:
        raw_stylesheet = f.read()

    def __init__(self, title='napari'):
        super().__init__(title=title)
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer)
        self.screenshot = self.window.qt_viewer.screenshot

        self._palette = None
        self.theme = 'dark'

    @property
    def palette(self):
        """dict of str: str : Color palette with which to style the viewer.
        """
        return self._palette

    @palette.setter
    def palette(self, palette):
        if palette == self.palette:
            return

        self._palette = palette

        # template and apply the primary stylesheet
        themed_stylesheet = template(self.raw_stylesheet, **palette)
        self.window.qt_viewer.setStyleSheet(themed_stylesheet)

        # set window styles which don't use the primary stylesheet
        self.window._status_bar.setStyleSheet(
            'QStatusBar { background: %s;color: %s}'
            % (palette['background'], palette['text']))
        self.window._qt_center.setStyleSheet(
            'QWidget { background: %s;}' % palette['background'])

        # set styles on clim slider
        for layer in self.layers:
            if has_clims(layer):
                layer._qt_controls.climSlider.setColors(
                    palette['foreground'], palette['highlight'])

        # set styles on dims sliders
        for slider in self.window.qt_viewer.dims.sliders:
            slider.setColors(palette['foreground'], palette['highlight'])

    @property
    def theme(self):
        """string or None : Preset color palette.
        """
        for theme, palette in self.themes.items():
            if palette == self.palette:
                return theme

    @theme.setter
    def theme(self, theme):
        if theme == self.theme:
            return

        try:
            self.palette = self.themes[theme]
        except KeyError:
            raise ValueError(f"Theme '{theme}' not found; "
                             f"options are {list(self.themes)}.")
