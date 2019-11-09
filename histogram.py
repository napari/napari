import numpy as np
from vispy.plot import PlotWidget
from vispy.scene import SceneCanvas, Histogram as _Histogram
from vispy.scene.cameras import PanZoomCamera

# from vispy.visuals import CompoundVisual, Visual
# from vispy.scene.visuals import create_visual_node
# from vispy.color import Colormap, get_colormap
# from vispy.visuals.shaders import Function
# from vispy.visuals.colorbar import (
#     VERT_SHADER,
#     FRAG_SHADER_HORIZONTAL,
#     FRAG_SHADER_VERTICAL,
# )


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


# class RectangleGradient(Visual):
#     """
#     Visual subclass that actually renders the ColorBar.

#     Parameters
#     ----------
#      pos : tuple (x, y)
#         Position where the rectangle is to be placed with
#         respect to the center of the rectangle
#     halfdim : tuple (half_width, half_height)
#         Half the dimensions of the rectangle measured
#         from the center. That way, the total dimensions
#         of the rectangle is (x - half_width) to (x + half_width)
#         and (y - half_height) to (y + half_height)
#     cmap : str | vispy.color.ColorMap
#         Either the name of the ColorMap to be used from the standard
#         set of names (refer to `vispy.color.get_colormap`),
#         or a custom ColorMap object.
#         The ColorMap is used to apply a gradient on the rectangle.
#      orientation : {'left', 'right', 'top', 'bottom'}
#         The orientation of the rectangle, used for rendering. The
#         orientation can be thought of as the position of the label
#         relative to the color bar.
#     """

#     def __init__(self, pos, halfdim, cmap, orientation, **kwargs):

#         self._cmap = get_colormap(cmap)
#         self._pos = pos
#         self._halfdim = halfdim
#         self._orientation = orientation

#         # setup the right program shader based on color
#         if orientation == "v":
#             Visual.__init__(
#                 self, vcode=VERT_SHADER, fcode=FRAG_SHADER_HORIZONTAL, **kwargs
#             )

#         elif orientation == "h":
#             Visual.__init__(
#                 self, vcode=VERT_SHADER, fcode=FRAG_SHADER_VERTICAL, **kwargs
#             )
#         else:
#             raise ValueError(
#                 "orientation must be one of 'h', or 'v', "
#                 "not '%s'" % (orientation,)
#             )

#         tex_coords = np.array(
#             [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]], dtype=np.float32
#         )

#         glsl_map_fn = Function(self._cmap.glsl_map)

#         self.shared_program.frag['color_transform'] = glsl_map_fn
#         self.shared_program['a_texcoord'] = tex_coords.astype(np.float32)

#         self._update()

#     def _update(self):
#         """Rebuilds the shaders, and repositions the objects
#            that are used internally by the ColorBarVisual
#         """

#         x, y = self._pos
#         halfw, halfh = self._halfdim

#         # test that width and height are non-zero
#         if halfw <= 0:
#             raise ValueError(
#                 "half-width must be positive and non-zero" ", not %s" % halfw
#             )

#         # Set up the attributes that the shaders require
#         vertices = np.array(
#             [
#                 [x - halfw, y - halfh],
#                 [x + halfw, y - halfh],
#                 [x + halfw, y + halfh],
#                 # tri 2
#                 [x - halfw, y - halfh],
#                 [x + halfw, y + halfh],
#                 [x - halfw, y + halfh],
#             ],
#             dtype=np.float32,
#         )

#         self.shared_program['a_position'] = vertices

#         self.shared_program['texture2D_LUT'] = (
#             self._cmap.texture_lut()
#             if (hasattr(self._cmap, 'texture_lut'))
#             else None
#         )

#     @staticmethod
#     def _prepare_transforms(view):
#         # figure out padding by considering the entire transform
#         # on the width and height
#         program = view.view_program
#         total_transform = view.transforms.get_transform()
#         program.vert['transform'] = total_transform

#     def _prepare_draw(self, view):
#         self._draw_mode = "triangles"
#         return True


# class HistogramVisual(CompoundVisual):
#     """Visual that calculates and displays a histogram of data

#     Parameters
#     ----------
#     data : array-like
#         Data to histogram. Currently only 1D data is supported.
#     bins : int | array-like
#         Number of bins, or bin edges.
#     color : instance of Color
#         Color of the histogram.
#     orientation : {'h', 'v'}
#         Orientation of the histogram.
#     """

#     def __init__(self, data, bins=10, color='k', orientation='h'):
#         data = np.asarray(data).ravel()
#         if data.ndim != 1:
#             raise ValueError('Only 1D data currently supported')
#         if not isinstance(orientation, str) or orientation not in ('h', 'v'):
#             raise ValueError(
#                 'orientation must be "h" or "v", not %s' % (orientation,)
#             )
#         X, Y = (0, 1) if orientation == 'h' else (1, 0)

#         # do the histogramming
#         data, bin_edges = np.histogram(data, bins)
#         binwidth = bin_edges[1] - bin_edges[0]

#         cmap = Colormap([(1,) * 3, (0.2,) * 3])

#         data = 2 * data / data.max()
#         nbins = len(data)
#         binwidth = 2 / nbins
#         bar_centers = np.linspace(-1, 1 - binwidth, nbins) + binwidth / 2
#         self._colorbars = [
#             RectangleGradient(
#                 (x, y / 2 - 1), (binwidth / 2, y / 2), cmap, orientation
#             )
#             for x, y in zip(bar_centers, data)
#         ]

#         super().__init__(self._colorbars)


# Histogram = create_visual_node(HistogramVisual)


class HistogramScene(SceneCanvas):
    def __init__(
        self,
        data=None,
        bins=10,
        color=(1, 1, 0.7, 0.8),
        orientation='h',
        bgcolor='k',
        size=(800, 600),
        show=True,
        **kwargs,
    ):
        self.data = data
        super().__init__(
            bgcolor=bgcolor, keys='interactive', show=show, size=size, **kwargs
        )
        self.unfreeze()
        self.grid = self.central_widget.add_grid()
        self.view = self.grid.add_view()
        self.view.camera = PanZoom1DCamera()
        self.camera = self.view.camera
        self.camera.set_range(y=(-1, 1))
        self.hist = _Histogram(data, bins, color, orientation)
        self.view.add(self.hist)
        self.view.camera.set_range()


class MyPlotWidget(PlotWidget):
    pass


class HistogramScene2(SceneCanvas):
    def __init__(self, data, bins=255, *args, **kwargs):
        self.data = data
        super().__init__(*args, bgcolor='k', **kwargs)
        self.unfreeze()
        self.plot = self.central_widget.add_widget(MyPlotWidget())
        self.plot.histogram(
            data.ravel(), bins, color=(1, 1, 0.7, 0.7), orientation='h'
        )


if __name__ == '__main__':
    from skimage.io import imread

    data = imread('/Users/talley/Desktop/test.tif')
    # data = np.random.randn(data.size).reshape(data.shape)
    # fig = HistogramScene(data.ravel(), bins=255)

    fig = HistogramScene2(data.ravel(), bins=255)
    fig.show(run=True)
