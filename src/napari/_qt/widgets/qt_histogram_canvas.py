"""Qt widget for displaying histogram using vispy visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget
from vispy.scene import SceneCanvas

from napari._vispy.visuals.histogram import HistogramVisual
from napari.utils.colormaps.colormap_utils import _representative_color

if TYPE_CHECKING:
    from pydantic_extra_types.color import Color

    from napari.utils import Colormap
    from napari.utils.theme import Theme


# Default histogram canvas dimensions
_DEFAULT_CANVAS_WIDTH = 300
_DEFAULT_CANVAS_HEIGHT = 150
_DEFAULT_CANVAS_MIN_HEIGHT = 100


class QtHistogramCanvasWidget(QWidget):
    """Qt widget embedding a vispy histogram visualization.

    Parameters
    ----------
    layer : Image
        The napari Image layer to visualize.
    parent : QWidget, optional
        Parent widget.

    Attributes
    ----------
    canvas : SceneCanvas
        Vispy canvas containing the histogram visual.
    view : ViewBox
        Vispy view for the histogram.
    histogram_visual : HistogramVisual
        The vispy visual that renders the histogram.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Create vispy canvas
        self.canvas = SceneCanvas(
            size=(_DEFAULT_CANVAS_WIDTH, _DEFAULT_CANVAS_HEIGHT),
            keys=None,
        )
        self.canvas.native.setParent(self)
        self.canvas.native.setMinimumHeight(_DEFAULT_CANVAS_MIN_HEIGHT)

        from vispy.scene import ViewBox

        self.view = ViewBox(parent=self.canvas.scene)
        self.canvas.central_widget.add_widget(self.view)

        self.histogram_visual = HistogramVisual()
        self.histogram_visual.parent = self.view.scene

        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, 1), y=(0, 1), margin=0.01)  # pyrefly: ignore [missing-attribute]
        # Disable viewbox interaction to prevent accidental pan/zoom
        self.view.interactive = False

        # Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.canvas.native)
        self.setLayout(main_layout)

    def update_colormap(self, colormap: Colormap) -> None:
        rgba = _representative_color(colormap)
        rgba[3] = max(float(rgba[3]), 0.8)
        self.histogram_visual.set_style(tuple(rgba))
        self.canvas.update()

    def _on_theme_change(self, theme: Theme) -> None:
        self.canvas.bgcolor = theme.canvas.as_hex()
        self.histogram_visual.set_style(
            lut_color=self._color_to_rgba(theme.highlight, 0.95),
            axes_color=self._color_to_rgba(theme.text, 0.7),
        )
        self.canvas.update()

    @staticmethod
    def _color_to_rgba(
        color: Color, alpha: float
    ) -> tuple[float, float, float, float]:
        rgb = color.as_rgb_tuple()
        return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)

    def cleanup(self) -> None:
        """Disconnect event handlers and clean up resources."""
        self.histogram_visual.destroy()
        self.canvas.close()
