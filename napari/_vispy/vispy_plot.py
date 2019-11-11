from vispy import scene
from vispy.plot import PlotWidget
from .cameras import PanZoom1DCamera


# PlotWidget inherits from scene.Widget
class NapariPlotWidget(PlotWidget):
    """Inspired by vispy.PlotWidget, with enough differences to not subclass"""

    AXIS_KWARGS = {
        'text_color': 'w',
        'axis_color': 'w',
        'tick_color': 'w',
        'tick_width': 1,
        'tick_font_size': 8,
        'tick_label_margin': 12,
        'axis_label_margin': 50,
        'minor_tick_length': 2,
        'major_tick_length': 5,
        'axis_width': 1,
        'axis_font_size': 9,
    }
    TITLE_KWARGS = {'font_size': 16, 'color': '#ff0000'}

    def __init__(
        self,
        margin=10,
        fg_color=None,
        xlabel="",
        ylabel="",
        title="",
        show_yaxis=True,
        show_xaxis=True,
        cbar_length=50,
        axis_kwargs=None,
        title_kwargs=None,
        lock_axis=None,
        **kwargs,
    ):
        self._fg = fg_color
        self.grid = None
        self.camera = None
        self.title = None
        self.title_widget = None
        self.xaxis = None
        self.yaxis = None
        self.xaxis_widget = None
        self.yaxis_widget = None
        self.show_xaxis = show_xaxis
        self.show_yaxis = show_yaxis
        self.xlabel = scene.Label(str(xlabel))
        self.ylabel = scene.Label(str(ylabel), rotation=-90)
        self.ylabel_widget = None
        self.xlabel_widget = None
        self.padding_left = None
        self.padding_bottom = None
        self.padding_right = None
        self.lock_axis = lock_axis

        self.axis_kwargs = self.AXIS_KWARGS
        if isinstance(axis_kwargs, dict):
            self.axis_kwargs.update(axis_kwargs)
        elif axis_kwargs is not None:
            raise TypeError(
                f'axis_kwargs must be a dict.  got: {type(axis_kwargs)}'
            )
        if fg_color is not None:
            self.axis_kwargs['text_color'] = fg_color
            self.axis_kwargs['axis_color'] = fg_color
            self.axis_kwargs['tick_color'] = fg_color

        self._configured = False
        self.visuals = []

        self.cbar_top = None
        self.cbar_bottom = None
        self.cbar_left = None
        self.cbar_right = None
        self.cbar_length = cbar_length

        super(PlotWidget, self).__init__(**kwargs)
        self.grid = self.add_grid(spacing=0, margin=margin)

        if isinstance(title_kwargs, dict):
            self.TITLE_KWARGS.update(title_kwargs)
        elif title_kwargs is not None:
            raise TypeError(
                f'axis_ktitle_kwargswargs must be a dict.  got: {type(title_kwargs)}'
            )

        self.title = scene.Label(str(title), **self.TITLE_KWARGS)

    def _configure_2d(self, fg_color=None):
        if self._configured:
            return

        #         c0        c1      c2      c3      c4      c5         c6
        #     +---------+-------+-------+-------+-------+---------+---------+
        #  r0 |         |                       | title |         |         |
        #     |         +-----------------------+-------+---------+         |
        #  r1 |         |                       | cbar  |         |         |
        #     | ------- +-------+-------+-------+-------+---------+ ------- |
        #  r2 | padding | cbar  | ylabel| yaxis |  view | cbar    | padding |
        #     | ------- +-------+-------+-------+-------+---------+ ------- |
        #  r3 |         |                       | xaxis |         |         |
        #     |         +-----------------------+-------+---------+         |
        #  r4 |         |                       | xlabel|         |         |
        #     |         +-----------------------+-------+---------+         |
        #  r5 |         |                       | cbar  |         |         |
        #     |---------+-----------------------+-------+---------+---------|
        #  r6 |                                 |padding|                   |
        #     +---------+-----------------------+-------+---------+---------+

        # PADDING
        self.padding_right = self.grid.add_widget(None, row=2, col=6)
        self.padding_right.width_min = 1
        self.padding_right.width_max = 5
        self.padding_bottom = self.grid.add_widget(None, row=6, col=4)
        self.padding_bottom.height_min = 1
        self.padding_bottom.height_max = 3

        # TITLE
        self.title_widget = self.grid.add_widget(self.title, row=0, col=4)
        self.title_widget.height_min = self.title_widget.height_max = (
            30 if self.title.text else 5
        )

        # COLORBARS
        self.cbar_top = self.grid.add_widget(None, row=1, col=4)
        self.cbar_top.height_max = 0
        self.cbar_left = self.grid.add_widget(None, row=2, col=1)
        self.cbar_left.width_max = 0
        self.cbar_right = self.grid.add_widget(None, row=2, col=5)
        self.cbar_right.width_max = 0
        self.cbar_bottom = self.grid.add_widget(None, row=5, col=4)
        self.cbar_bottom.height_max = 0

        # Y AXIS
        self.yaxis = scene.AxisWidget(orientation='left', **self.axis_kwargs)
        self.yaxis_widget = self.grid.add_widget(self.yaxis, row=2, col=3)
        if self.show_yaxis:
            self.yaxis_widget.width_max = 35
            self.ylabel_widget = self.grid.add_widget(
                self.ylabel, row=2, col=2
            )
            self.ylabel_widget.width_max = 10 if self.ylabel.text else 1
            self.padding_left = self.grid.add_widget(None, row=2, col=0)
            self.padding_left.width_min = 1
            self.padding_left.width_max = 60
        else:
            self.yaxis.visible = False
            self.yaxis.width_max = 1
            self.padding_left = self.grid.add_widget(
                None, row=2, col=0, col_span=3
            )
            self.padding_left.width_min = 1
            self.padding_left.width_max = 5

        # X AXIS
        self.xaxis = scene.AxisWidget(orientation='bottom', **self.axis_kwargs)
        self.xaxis_widget = self.grid.add_widget(self.xaxis, row=3, col=4)
        self.xaxis_widget.height_max = 20 if self.show_xaxis else 0
        self.xlabel_widget = self.grid.add_widget(self.xlabel, row=4, col=4)
        self.xlabel_widget.height_max = 10 if self.xlabel.text else 0

        # VIEWBOX (this has to go last, see vispy #1748)
        self.view = self.grid.add_view(
            row=2, col=4, border_color=None, bgcolor=None
        )

        if self.lock_axis is not None:
            self.view.camera = PanZoom1DCamera(self.lock_axis)
        else:
            self.view.camera = 'panzoom'
        self.camera = self.view.camera

        self._configured = True
        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

    def colorbar(
        self,
        cmap,
        position="right",
        label="",
        clim=("", ""),
        border_width=0.0,
        border_color="black",
        **kwargs,
    ):
        """Show a ColorBar

        Parameters
        ----------
        cmap : str | vispy.color.ColorMap
            Either the name of the ColorMap to be used from the standard
            set of names (refer to `vispy.color.get_colormap`),
            or a custom ColorMap object.
            The ColorMap is used to apply a gradient on the colorbar.
        position : {'left', 'right', 'top', 'bottom'}
            The position of the colorbar with respect to the plot.
            'top' and 'bottom' are placed horizontally, while
            'left' and 'right' are placed vertically
        label : str
            The label that is to be drawn with the colorbar
            that provides information about the colorbar.
        clim : tuple (min, max)
            the minimum and maximum values of the data that
            is given to the colorbar. This is used to draw the scale
            on the side of the colorbar.
        border_width : float (in px)
            The width of the border the colormap should have. This measurement
            is given in pixels
        border_color : str | vispy.color.Color
            The color of the border of the colormap. This can either be a
            str as the color's name or an actual instace of a vipy.color.Color

        Returns
        -------
        colorbar : instance of ColorBarWidget

        See also
        --------
        ColorBarWidget
        """

        self._configure_2d()

        cbar = scene.ColorBarWidget(
            orientation=position,
            label=label,
            cmap=cmap,
            clim=clim,
            border_width=border_width,
            border_color=border_color,
            **kwargs,
        )

        if cbar.orientation == "bottom":
            self.grid.remove_widget(self.cbar_bottom)
            self.cbar_bottom = self.grid.add_widget(cbar, row=5, col=4)
            self.cbar_bottom.height_max = (
                self.cbar_bottom.height_max
            ) = self.cbar_length

        elif cbar.orientation == "top":
            self.grid.remove_widget(self.cbar_top)
            self.cbar_top = self.grid.add_widget(cbar, row=1, col=4)
            self.cbar_top.height_max = (
                self.cbar_top.height_max
            ) = self.cbar_length

        elif cbar.orientation == "left":
            self.grid.remove_widget(self.cbar_left)
            self.cbar_left = self.grid.add_widget(cbar, row=2, col=1)
            self.cbar_left.width_max = (
                self.cbar_left.width_min
            ) = self.cbar_length

        else:  # cbar.orientation == "right"
            self.grid.remove_widget(self.cbar_right)
            self.cbar_right = self.grid.add_widget(cbar, row=2, col=5)
            self.cbar_right.width_max = (
                self.cbar_right.width_min
            ) = self.cbar_length

        return cbar
