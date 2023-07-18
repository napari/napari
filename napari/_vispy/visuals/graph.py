from vispy.visuals import LineVisual

from napari._vispy.visuals.points import PointsVisual


class GraphVisual(PointsVisual):
    def __init__(self):
        super().__init__()
        # connect='segments' indicates you need start point and end point for
        # each segment, rather than just a list of points. This mode means you
        # don't need segments to be sorted to display a line.
        self.add_subvisual(LineVisual(connect='segments'))
