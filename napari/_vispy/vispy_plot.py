import numpy as np
from vispy import scene
from vispy.plot import PlotWidget as VispyPlotWidget

from .cameras import PanZoom1DCamera


# PlotWidget inherits from scene.Widget
class PlotWidget(VispyPlotWidget):
    """Subclass of vispy.plot.PlotWidget.

    Subclassing mostly to override styles (which are not exposed in the main
    class) and to override `_configure_2d` to allow more control over the
    layout.
    """

    # default styles to use for the AxisVisuals
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
    # default styles to use for the title
    TITLE_KWARGS = {'font_size': 16, 'color': 'w'}

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
        self.xlabel = scene.Label(str(xlabel), color=fg_color or '#ccc')
        self.ylabel = scene.Label(
            str(ylabel), rotation=-90, color=fg_color or '#ccc'
        )
        self.ylabel_widget = None
        self.xlabel_widget = None
        self.padding_left = None
        self.padding_bottom = None
        self.padding_right = None
        self._locked_axis = lock_axis

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

        super(VispyPlotWidget, self).__init__(**kwargs)
        self.grid = self.add_grid(spacing=0, margin=margin)

        if isinstance(title_kwargs, dict):
            self.TITLE_KWARGS.update(title_kwargs)
        elif title_kwargs is not None:
            raise TypeError(
                f'axis_ktitle_kwargswargs must be a dict.  got: {type(title_kwargs)}'
            )

        self.title = scene.Label(str(title), **self.TITLE_KWARGS)

    def autoscale(self, axes='both'):
        # might be too slow?
        x, y = None, None
        for visual in self.visuals:
            data = None
            if hasattr(visual, '_line'):
                data = np.array(visual._line._pos).T
            elif hasattr(visual, '_markers'):
                data = np.array(visual._markers._data['a_position'])[:, :2].T
            if data is not None and len(data.shape):
                if axes in ('y', 'both'):
                    y = y if y is not None else [np.inf, -np.inf]
                    y[1] = np.maximum(data[1].max(), y[1])
                    y[0] = np.minimum(data[1].min(), y[0])
                if axes in ('x', 'both'):
                    x = x if x is not None else [np.inf, -np.inf]
                    x[1] = np.maximum(data[0].max(), x[1])
                    x[0] = np.minimum(data[0].min(), x[0])

        self.view.camera.set_range(x, y, margin=0.005)

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
            self.yaxis_widget.width_max = 30
            self.ylabel_widget = self.grid.add_widget(
                self.ylabel, row=2, col=2
            )
            self.ylabel_widget.width_max = 30 if self.ylabel.text else 1
            self.padding_left = self.grid.add_widget(None, row=2, col=0)
            self.padding_left.width_min = 1
            self.padding_left.width_max = 10
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
        self.xlabel_widget.height_max = 20 if self.xlabel.text else 0

        # VIEWBOX (this has to go last, see vispy #1748)
        self.view = self.grid.add_view(
            row=2, col=4, border_color=None, bgcolor=None
        )

        self.lock_axis(self._locked_axis)
        self._configured = True
        self.xaxis.link_view(self.view)
        self.yaxis.link_view(self.view)

    def lock_axis(self, axis):
        # work in progress
        if isinstance(axis, str):
            if axis.lower() == 'x':
                axis = 0
            elif axis.lower() == 'y':
                axis = 1
            else:
                raise ValueError("axis must be either 'x' or 'y'")
        self._locked_axis = axis
        if self._locked_axis is not None:
            self.view.camera = PanZoom1DCamera(self._locked_axis)
        else:
            self.view.camera = 'panzoom'
        self.camera = self.view.camera

    def plot(
        self,
        data,
        color=(0.53, 0.56, 0.57, 1.00),
        symbol='o',
        width=1,
        marker_size=8,
        edge_color='gray',
        face_color='gray',
        edge_width=1,
        title=None,
        xlabel=None,
        ylabel=None,
        connect='strip',
    ):
        """Plot a series of data using lines and markers

        Parameters
        ----------
        data : array | two arrays
            Arguments can be passed as ``(Y,)``, ``(X, Y)`` or
            ``np.array((X, Y))``.
        color : instance of Color
            Color of the line.
        symbol : str
            Marker symbol to use.
        width : float
            Line width.
        marker_size : float
            Marker size. If `size == 0` markers will not be shown.
        edge_color : instance of Color
            Color of the marker edge.
        face_color : instance of Color
            Color of the marker face.
        edge_width : float
            Edge width of the marker.
        title : str | None
            The title string to be displayed above the plot
        xlabel : str | None
            The label to display along the bottom axis
        ylabel : str | None
            The label to display along the left axis.
        connect : str | array
            Determines which vertices are connected by lines.

        Returns
        -------
        line : instance of LinePlot
            The line plot.

        See also
        --------
        marker_types, LinePlot
        """
        self._configure_2d()
        line = scene.LinePlot(
            data,
            connect=connect,
            color=color,
            symbol=symbol,
            width=width,
            marker_size=marker_size,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=edge_width,
        )
        self.view.add(line)
        self.visuals.append(line)

        if title is not None:
            self.title.text = title
        if xlabel is not None:
            self.xlabel.text = xlabel
        if ylabel is not None:
            self.ylabel.text = ylabel

        if data is not None:
            self.view.camera.set_range()
        return line
