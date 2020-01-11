from .qt_plot_widget import QtPlotWidget
from vispy.scene import LinePlot
from numpydoc.docscrape import ClassDoc


class QtLineProfile(QtPlotWidget):
    styles = {'color': (0.53, 0.56, 0.57, 1.00), 'width': 1, 'marker_size': 0}

    def __init__(self, data=None, styles=None, **kwargs):
        """A widget that plots a 1D dataset as a line.

        Parameters
        ----------
        data : array-like
            Arguments can be passed as ``(Y,)``, ``(X, Y)``, ``(X, Y, Z)`` or
            ``np.array((X, Y))``, ``np.array((X, Y, Z))``.
        styles : dict, optional
            styles kwargs to pass to the vispy LinePlot.
        **kwargs
            all other keyword arguments will be passed to QtPlotWidget and, in
            turn, to the NapariPlotWidget.  One kwarg that is often useful for
            the QtLineProfile is `lock_axis`, which will constrain the pan/zoom
            to a single axis.
        """
        super().__init__(**kwargs)

        self.styles.update(styles or {})
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        if not hasattr(self, 'line') or (not self.line):
            self.line = LinePlot(data, **self.styles)
            self.plot.view.add(self.line)
            self.plot.view.camera.set_range(margin=0.005)
        else:
            self.line.set_data(data, marker_size=0)
            # autoscale the range
            y = (0, data.max())
            x = (0, len(data))
            self.plot.view.camera.set_range(x=x, y=y, margin=0.005)


def get_class_param_string(cls, excludes=[]):
    doc = ClassDoc(cls)
    out = []
    for param in doc.get('Parameters'):
        if param.name in excludes:
            continue
        out.append(f'{param.name} : {param.type}')
        if param.desc and ''.join(param.desc).strip():
            out += doc._str_indent(param.desc)
    return out


doc = str(QtLineProfile.__init__.__doc__)
out = get_class_param_string(LinePlot, ('data', 'name', 'parent', '**kwargs'))
out = "\n\t\t".join(['    Valid `styles` keys include:'] + out)
doc = doc.replace('**kwargs', out + '\n\t**kwargs')
QtLineProfile.__init__.__doc__ = doc
