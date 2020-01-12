from vispy.plot.fig import Fig as VispyFig

from .vispy_plot import PlotWidget


class Fig(VispyFig):
    """Subclas of vispy.plot.Fig mostly just to use our internal plot widget.
    """

    def __init__(
        self,
        bgcolor='k',
        size=(800, 600),
        show=True,
        keys=None,
        vsync=True,
        **kwargs,
    ):
        self._plot_widgets = []
        self._grid = None  # initialize before the freeze occurs
        super(VispyFig, self).__init__(
            bgcolor=bgcolor,
            keys=keys,
            show=show,
            size=size,
            vsync=vsync,
            **kwargs,
        )
        self._grid = self.central_widget.add_grid()
        self._grid._default_class = PlotWidget
