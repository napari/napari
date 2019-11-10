import numpy as np
from vispy.plot import PlotWidget
from vispy import scene
from vispy.scene.cameras import PanZoomCamera


class PanZoom1DCamera(PanZoomCamera):
    def __init__(self, axis=1, *args, **kwargs):
        self.axis = axis
        super().__init__(*args, **kwargs)

    def zoom(self, factor, center=None):
        if np.isscalar(factor):
            factor = [factor, factor]
        factor[self.axis] = 1
        return super().zoom(factor, center=center)

    def pan(self, pan):
        pan[self.axis] = 0
        self.rect = self.rect + pan


class NapariPlotWidget(PlotWidget):
    """Subclassing PlotWidget to tweak look and feel and fix #1742"""

    def _configure_2d(self, fg_color=None):
        if self._configured:
            return

        if fg_color is None:
            fg = self._fg
        else:
            fg = fg_color

        axis_kwargs = {
            'text_color': fg,
            'axis_color': fg,
            'tick_color': fg,
            'tick_width': 1,
            'tick_font_size': 8,
            'tick_label_margin': 12,
            'axis_label_margin': 50,
            'minor_tick_length': 3,
            'major_tick_length': 6,
            'axis_width': 1,
            'axis_font_size': 10,
        }

        #     c0        c1      c2      c3      c4      c5         c6
        #  r0 +---------+-------+-------+-------+-------+---------+---------+
        #     |         |                       | title |         |         |
        #  r1 |         +-----------------------+-------+---------+         |
        #     |         |                       | cbar  |         |         |
        #  r2 |         +-------+-------+-------+-------+---------+         |
        #     |         | cbar  | ylabel| yaxis |  view | cbar    | padding |
        #  r3 | padding +-------+-------+-------+-------+---------+         |
        #     |         |                       | xaxis |         |         |
        #  r4 |         +-----------------------+-------+---------+         |
        #     |         |                       | xlabel|         |         |
        #  r5 |         +-----------------------+-------+---------+         |
        #     |         |                       | cbar  |         |         |
        #  r6 |---------+-----------------------+-------+---------+---------|
        #     |                           padding                           |
        #     +---------+-----------------------+-------+---------+---------+

        # padding left
        padding_left = self.grid.add_widget(None, row=0, row_span=5, col=0)
        padding_left.width_min = 30
        padding_left.width_max = 60

        # padding right
        padding_right = self.grid.add_widget(None, row=0, row_span=5, col=6)
        padding_right.width_min = 10
        padding_right.width_max = 20

        # # padding bottom
        # padding_bottom = self.grid.add_widget(None, row=6, col=0, col_span=6)
        # padding_bottom.height_min = 20
        # padding_bottom.height_max = 40

        # row 0
        # title - column 4 to 5
        self.title_widget = self.grid.add_widget(self.title, row=0, col=4)
        self.title_widget.height_min = self.title_widget.height_max = (
            30 if self.title.text else 10
        )

        # row 1
        # colorbar - column 4 to 5
        self.cbar_top = self.grid.add_widget(None, row=1, col=4)
        self.cbar_top.height_max = 1

        # row 2
        # colorbar_left - column 1
        # ylabel - column 2
        # yaxis - column 3
        # view - column 4
        # colorbar_right - column 5
        self.cbar_left = self.grid.add_widget(None, row=2, col=1)
        self.cbar_left.width_max = 1

        self.ylabel = scene.Label("", rotation=-90)
        ylabel_widget = self.grid.add_widget(self.ylabel, row=2, col=2)
        ylabel_widget.width_max = 1

        self.yaxis = scene.AxisWidget(orientation='left', **axis_kwargs)

        yaxis_widget = self.grid.add_widget(self.yaxis, row=2, col=3)
        yaxis_widget.width_max = 35

        # row 3
        # xaxis - column 4
        self.xaxis = scene.AxisWidget(orientation='bottom', **axis_kwargs)
        xaxis_widget = self.grid.add_widget(self.xaxis, row=3, col=4)
        xaxis_widget.height_max = 20

        self.view = self.grid.add_view(
            row=2, col=4, border_color=None, bgcolor=None
        )
        self.view.camera = PanZoom1DCamera()
        self.camera = self.view.camera

        self.cbar_right = self.grid.add_widget(None, row=2, col=5)
        self.cbar_right.width_max = 1

        # row 4
        # xlabel - column 4
        self.xlabel = scene.Label("")
        xlabel_widget = self.grid.add_widget(self.xlabel, row=4, col=4)
        xlabel_widget.height_max = 10

        # row 5
        self.cbar_bottom = self.grid.add_widget(None, row=5, col=4)
        self.cbar_bottom.height_max = 1

        self._configured = True
        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)


class HistogramScene(scene.SceneCanvas):
    def __init__(self, data=None, bins=512, *args, **kwargs):
        super().__init__(*args, bgcolor='k', **kwargs)
        self.unfreeze()
        # self.data = data
        self.plot = self.central_widget.add_widget(
            NapariPlotWidget(fg_color=(1, 1, 1, 0.3))
        )
        self.plot._configure_2d()
        if data is not None:
            self.hist = self.plot.histogram(
                data.ravel(), bins, color=(1, 1, 0.9, 0.5), orientation='h'
            )
            self.hist.order = 10
        self.plot.view.camera.set_range(margin=0.005)
